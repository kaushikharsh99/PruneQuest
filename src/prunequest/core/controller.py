"""
Central SearchController orchestrating model adapters, benchmarks, pruners, and tracking.
"""

import os
import shutil
import time
from typing import Dict, Any, Optional, List, Tuple
import torch

from prunequest.config import PruneQuestConfig
from prunequest.models.base import BaseModelAdapter
from prunequest.models.registry import ModelRegistry
from prunequest.benchmarks.base import BaseBenchmark
from prunequest.benchmarks.registry import BenchmarkRegistry
from prunequest.pruning.base import BasePruner, PruningCandidate, PruningResult
from prunequest.pruning.layers import LayerPruner
from prunequest.core.search import BaseSearchStrategy, GreedySearchStrategy
from prunequest.core.tracking import ExperimentTracker
from prunequest.export.checkpoint import CheckpointExporter
from prunequest.export.validation import SubprocessValidator
from prunequest.reporting.metrics import compute_accuracy_metrics, clean_gpu_memory
from prunequest.exceptions import PruningError, ResourceExhaustionError


class SearchController:
    """
    Coordinates automated benchmark-guided pruning search.
    Enforces accuracy retention constraints, manages rollbacks, and validates exports.
    """

    def __init__(
        self,
        config: PruneQuestConfig,
        tracker: Optional[ExperimentTracker] = None,
        exporter: Optional[CheckpointExporter] = None,
    ):
        self.config = config
        self.run_dir = config.output_dir
        self.tracker = tracker or ExperimentTracker(self.run_dir)
        self.exporter = exporter or CheckpointExporter(os.path.join(self.run_dir, "checkpoints"))

        self.adapter: BaseModelAdapter = ModelRegistry.get_adapter(
            model_id_or_path=config.model.model_id,
            device=config.model.device,
            torch_dtype=config.model.torch_dtype,
        )

        self.benchmark: BaseBenchmark = BenchmarkRegistry.get_benchmark(
            name=config.benchmark.name,
            seed=config.benchmark.seed,
            max_new_tokens=config.benchmark.max_new_tokens,
        )

        self.pruner: BasePruner = LayerPruner(metric=config.pruning.metric)
        self.search_strategy: BaseSearchStrategy = GreedySearchStrategy()

        self.baseline_accuracy: Optional[float] = None
        self.baseline_params: Optional[int] = None
        self.accepted_checkpoints: List[Dict[str, Any]] = []

    def run_baseline(
        self,
        split: str = "smoke",
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate baseline model to establish reference performance A0.
        """
        clean_gpu_memory(self.adapter.device)
        model, tokenizer = self.adapter.load()
        meta = self.adapter.get_metadata()
        self.baseline_params = meta.total_params

        eval_res = self.benchmark.evaluate(
            model=model,
            tokenizer=tokenizer,
            split_name=split,
            max_samples=max_samples or self.config.benchmark.smoke_size,
            device=self.adapter.device,
        )

        acc_metrics = compute_accuracy_metrics(eval_res.correct, eval_res.total, baseline_acc=None)
        self.baseline_accuracy = acc_metrics["accuracy"]

        record = {
            "iteration_id": "baseline",
            "parent_id": "root",
            "pruning_type": "none",
            "operation_description": "Initial unpruned baseline model",
            "params_before": meta.total_params,
            "params_after": meta.total_params,
            "params_removed": 0,
            "param_retention_pct": 100.0,
            "stage": split,
            "sample_size": eval_res.total,
            "num_correct": eval_res.correct,
            "accuracy": acc_metrics["accuracy"],
            "accuracy_ci_lower": acc_metrics["accuracy_ci_lower"],
            "accuracy_ci_upper": acc_metrics["accuracy_ci_upper"],
            "accuracy_retention_pct": 100.0,
            "tokens_per_sec": eval_res.tokens_per_sec,
            "peak_vram_gb": eval_res.peak_vram_gb,
            "elapsed_time_sec": eval_res.total_time_sec,
            "status": "baseline",
            "checkpoint_path": self.config.model.model_id,
        }

        self.tracker.log_iteration(record)
        self.tracker.log_generations("baseline", split, eval_res.records)
        self.accepted_checkpoints.append(record)
        self.tracker.update_pareto_frontier(self.accepted_checkpoints)

        return record

    def run(
        self,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute full automated pruning search until stopping criteria.
        """
        if self.baseline_accuracy is None:
            self.run_baseline(split=self.config.benchmark.split, max_samples=self.config.benchmark.smoke_size)

        min_required_acc = self.baseline_accuracy * self.config.pruning.retain
        iteration = 0
        search_start_time = time.time()
        stopping_reason = "completed"

        while iteration < self.config.pruning.max_iterations:
            iteration += 1

            # Check time budget
            if self.config.pruning.time_budget_sec:
                if (time.time() - search_start_time) >= self.config.pruning.time_budget_sec:
                    stopping_reason = "time_budget_exhausted"
                    break

            layers = self.adapter.get_layers()
            if len(layers) <= 2:
                stopping_reason = "minimum_layers_reached"
                break

            # 1. Analyze layer redundancy
            calibration_samples = self.benchmark.data_splits.get("calibration", [])
            analysis = self.pruner.analyze(
                adapter=self.adapter,
                calibration_samples=calibration_samples,
                max_samples=32,
            )

            # 2. Propose & select candidates
            all_cands = self.pruner.propose_candidates(self.adapter, analysis, top_k=self.config.pruning.max_candidates)
            cands = self.search_strategy.select_candidates(all_cands, budget=self.config.pruning.max_candidates)

            if not cands:
                stopping_reason = "no_eligible_candidates"
                break

            best_candidate = None
            best_screen_acc = -1.0
            best_eval_res = None

            # 3. Screen candidates with safe temporary rollback
            for cand in cands:
                layer_idx = cand.target_info["layer_idx"]
                layer_backup = self.adapter.model.model.layers[layer_idx]

                prune_res = self.pruner.apply_pruning(self.adapter, cand)
                if not prune_res.validation_passed:
                    # Rollback
                    self._rollback_layer(layer_idx, layer_backup)
                    continue

                screen_res = self.benchmark.evaluate(
                    model=self.adapter.model,
                    tokenizer=self.adapter.tokenizer,
                    split_name="smoke",
                    max_samples=self.config.benchmark.smoke_size,
                    device=self.adapter.device,
                )

                # Revert candidate to test next candidate
                self._rollback_layer(layer_idx, layer_backup)

                if screen_res.accuracy > best_screen_acc:
                    best_screen_acc = screen_res.accuracy
                    best_candidate = cand
                    best_eval_res = screen_res

            if best_candidate is None:
                stopping_reason = "all_candidates_failed_validation"
                break

            # 4. Check accuracy retention against original baseline threshold
            if best_screen_acc < min_required_acc:
                stopping_reason = f"retention_threshold_reached (Best {best_screen_acc*100:.2f}% < Required {min_required_acc*100:.2f}%)"
                break

            # 5. Apply winning candidate permanently
            prune_res = self.pruner.apply_pruning(self.adapter, best_candidate)
            remaining_layers = len(self.adapter.get_layers())
            iter_id = f"iter_{iteration}_{best_candidate.candidate_id}"
            ckpt_name = f"checkpoint_layers_{remaining_layers}"

            # Free disk if low before saving
            self._ensure_disk_space()

            ckpt_dir = self.exporter.export(
                adapter=self.adapter,
                checkpoint_name=ckpt_name,
                metadata={
                    "iteration": iteration,
                    "candidate_id": best_candidate.candidate_id,
                    "accuracy": best_screen_acc,
                    "baseline_accuracy": self.baseline_accuracy,
                }
            )

            # Subprocess reload validation
            if self.config.hardware.cpu_offload_validation:
                self.adapter.model.to("cpu")
                clean_gpu_memory(self.adapter.device)
                reload_ok, reload_msg = SubprocessValidator.validate_checkpoint(ckpt_dir, device=self.adapter.device)
                self.adapter.model.to(self.adapter.device)
            else:
                reload_ok, reload_msg = True, "skipped"

            retention_pct = (best_screen_acc / self.baseline_accuracy) * 100.0 if self.baseline_accuracy > 0 else 0.0
            acc_metrics = compute_accuracy_metrics(best_eval_res.correct, best_eval_res.total, baseline_acc=self.baseline_accuracy)

            record = {
                "iteration_id": iter_id,
                "parent_id": "baseline" if iteration == 1 else f"iter_{iteration-1}",
                "pruning_type": "layer",
                "operation_description": f"Removed layer {best_candidate.target_info['layer_idx']}, remaining={remaining_layers}",
                "params_before": prune_res.params_before,
                "params_after": prune_res.params_after,
                "params_removed": prune_res.params_removed,
                "param_retention_pct": (prune_res.params_after / self.baseline_params) * 100.0,
                "stage": "smoke",
                "sample_size": best_eval_res.total,
                "num_correct": best_eval_res.correct,
                "accuracy": acc_metrics["accuracy"],
                "accuracy_ci_lower": acc_metrics["accuracy_ci_lower"],
                "accuracy_ci_upper": acc_metrics["accuracy_ci_upper"],
                "accuracy_retention_pct": retention_pct,
                "tokens_per_sec": best_eval_res.tokens_per_sec,
                "peak_vram_gb": best_eval_res.peak_vram_gb,
                "elapsed_time_sec": best_eval_res.total_time_sec,
                "status": "accepted",
                "checkpoint_path": ckpt_dir,
            }

            self.tracker.log_iteration(record)
            self.tracker.log_generations(iter_id, "smoke", best_eval_res.records)
            self.accepted_checkpoints.append(record)
            self.tracker.update_pareto_frontier(self.accepted_checkpoints)

            if progress_callback:
                progress_callback(record)

            clean_gpu_memory(self.adapter.device)

        self.tracker.export_csv()
        latest = self.tracker.get_latest_accepted() or self.accepted_checkpoints[-1]

        return {
            "status": stopping_reason,
            "best_checkpoint": latest.get("checkpoint_path"),
            "original_params": self.baseline_params,
            "final_params": latest.get("params_after"),
            "baseline_accuracy": self.baseline_accuracy,
            "final_accuracy": latest.get("accuracy"),
            "accuracy_retention_pct": latest.get("accuracy_retention_pct"),
            "iterations_completed": iteration,
            "run_dir": self.run_dir,
        }

    def _rollback_layer(self, layer_idx: int, layer_module: torch.nn.Module):
        self.adapter.model.model.layers.insert(layer_idx, layer_module)
        self.adapter.model.config.num_hidden_layers = len(self.adapter.model.model.layers)
        if hasattr(self.adapter.model.config, "layer_types") and self.adapter.model.config.layer_types is not None:
            self.adapter.model.config.layer_types.insert(layer_idx, "full_attention")
        for idx, l in enumerate(self.adapter.model.model.layers):
            l.layer_idx = idx
            if hasattr(l, "self_attn"):
                l.self_attn.layer_idx = idx

    def _ensure_disk_space(self, min_free_gb: float = 6.0):
        try:
            free_gb = shutil.disk_usage(self.run_dir).free / (1024**3)
            if free_gb < min_free_gb and len(self.accepted_checkpoints) > 1:
                # Remove older safetensors to preserve disk
                for old in self.accepted_checkpoints[:-1]:
                    p = old.get("checkpoint_path")
                    if p and os.path.exists(p):
                        sf = os.path.join(p, "model.safetensors")
                        if os.path.exists(sf):
                            os.remove(sf)
        except Exception:
            pass

    def resume(self) -> Dict[str, Any]:
        """Resume search from last accepted state."""
        latest = self.tracker.get_latest_accepted()
        if not latest or not latest.get("checkpoint_path") or not os.path.exists(latest["checkpoint_path"]):
            return self.run()

        # Update model_id to point to accepted checkpoint
        self.config.model.model_id = latest["checkpoint_path"]
        self.adapter = ModelRegistry.get_adapter(
            model_id_or_path=self.config.model.model_id,
            device=self.config.model.device,
            torch_dtype=self.config.model.torch_dtype,
        )
        return self.run()
