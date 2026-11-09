"""
Central SearchController orchestrating model adapters, evaluation suites, pruners, and tracking.
Fully decoupled from specific model families or benchmarks.
"""

import os
import shutil
import time
from typing import Dict, Any, Optional, List, Union
import torch

from prunequest.config import PruneQuestConfig
from prunequest.models.base import BaseModelAdapter
from prunequest.models.registry import ModelRegistry
from prunequest.benchmarks.base import BaseBenchmark, EvaluationResult
from prunequest.benchmarks.registry import BenchmarkRegistry
from prunequest.benchmarks.constraints import ConstraintSuite, MetricSpec
from prunequest.pruning.base import BasePruner
from prunequest.pruning.registry import PrunerRegistry
from prunequest.core.search import BaseSearchStrategy, GreedySearchStrategy
from prunequest.core.tracking import ExperimentTracker
from prunequest.export.checkpoint import CheckpointExporter
from prunequest.export.validation import SubprocessValidator
from prunequest.reporting.metrics import compute_accuracy_metrics, clean_gpu_memory, get_gpu_vram_allocated_gb


class SearchController:
    """
    Coordinates automated benchmark-guided pruning search across arbitrary models and evaluation suites.
    Enforces multi-benchmark retention constraints, manages rollbacks, and validates exports.
    """

    def __init__(
        self,
        config: PruneQuestConfig,
        adapter: Optional[BaseModelAdapter] = None,
        benchmarks: Optional[Union[List[BaseBenchmark], Dict[str, BaseBenchmark]]] = None,
        pruner: Optional[BasePruner] = None,
        search_strategy: Optional[BaseSearchStrategy] = None,
        constraints: Optional[ConstraintSuite] = None,
        tracker: Optional[ExperimentTracker] = None,
        exporter: Optional[CheckpointExporter] = None,
    ):
        self.config = config
        self.run_dir = config.output_dir
        self.tracker = tracker or ExperimentTracker(self.run_dir)
        self.exporter = exporter or CheckpointExporter(os.path.join(self.run_dir, "checkpoints"))

        self.adapter: BaseModelAdapter = adapter or ModelRegistry.get_adapter(
            model_id_or_path=config.model.model_id,
            device=config.model.device,
            torch_dtype=config.model.torch_dtype,
        )

        # Multi-benchmark initialization
        self.benchmarks: Dict[str, BaseBenchmark] = {}
        if benchmarks:
            if isinstance(benchmarks, list):
                for b in benchmarks:
                    self.benchmarks[b.name] = b
            else:
                self.benchmarks = dict(benchmarks)
        else:
            # Parse benchmarks from config (comma-separated or single)
            bench_names = [b.strip() for b in config.benchmark.name.split(",") if b.strip()]
            for b_name in bench_names:
                self.benchmarks[b_name] = BenchmarkRegistry.get_benchmark(
                    name=b_name,
                    seed=config.benchmark.seed,
                    max_new_tokens=config.benchmark.max_new_tokens,
                )

        self.pruner: BasePruner = pruner or PrunerRegistry.get_pruner(
            name=config.pruning.backend,
            metric=config.pruning.metric,
        )
        self.search_strategy: BaseSearchStrategy = search_strategy or GreedySearchStrategy()

        # Multi-benchmark constraints
        self.constraints: ConstraintSuite = constraints or ConstraintSuite()
        if not self.constraints.constraints:
            for b_name in self.benchmarks:
                self.constraints.add_constraint(
                    b_name,
                    MetricSpec(name=b_name, higher_is_better=True, retention=config.pruning.retain),
                )

        self.baselines: Dict[str, EvaluationResult] = {}
        self.baseline_accuracy: Optional[float] = None  # Primary benchmark baseline
        self.baseline_params: Optional[int] = None
        self.accepted_checkpoints: List[Dict[str, Any]] = []

    @property
    def benchmark(self) -> Optional[BaseBenchmark]:
        """Convenience property returning the primary evaluation benchmark."""
        if self.benchmarks:
            return list(self.benchmarks.values())[0]
        return None

    def _get_split_samples(self, split: str) -> Optional[int]:
        if self.config.benchmark.sample_limit is not None:
            return self.config.benchmark.sample_limit
        if split == "smoke":
            return self.config.benchmark.smoke_size
        elif split == "dev":
            return self.config.benchmark.dev_size
        elif split == "screening":
            return self.config.benchmark.screening_size
        elif split in ("full", "test"):
            return None
        return self.config.benchmark.smoke_size

    def run_baseline(
        self,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate baseline model to establish reference performance A0 across all configured benchmarks.
        """
        clean_gpu_memory(self.adapter.device)
        model, tokenizer = self.adapter.load()
        meta = self.adapter.get_metadata()
        self.baseline_params = meta.total_params

        eval_split = split or self.config.benchmark.split
        eval_samples = max_samples if max_samples is not None else self._get_split_samples(eval_split)

        combined_scores = []
        for b_name, benchmark in self.benchmarks.items():
            eval_res = benchmark.evaluate(
                model=model,
                tokenizer=tokenizer,
                split_name=eval_split,
                max_samples=eval_samples,
                device=self.adapter.device,
            )
            self.baselines[b_name] = eval_res
            combined_scores.append(eval_res.accuracy)

            record = {
                "iteration_id": "baseline" if len(self.benchmarks) == 1 else f"baseline_{b_name}",
                "parent_id": "root",
                "pruning_type": "none",
                "operation_description": f"Initial unpruned baseline for {b_name}",
                "params_before": meta.total_params,
                "params_after": meta.total_params,
                "params_removed": 0,
                "param_retention_pct": 100.0,
                "stage": eval_split,
                "sample_size": eval_res.total,
                "num_correct": eval_res.correct,
                "accuracy": eval_res.accuracy,
                "accuracy_ci_lower": eval_res.accuracy,
                "accuracy_ci_upper": eval_res.accuracy,
                "accuracy_retention_pct": 100.0,
                "tokens_per_sec": eval_res.tokens_per_sec,
                "peak_vram_gb": eval_res.peak_vram_gb,
                "elapsed_time_sec": eval_res.total_time_sec,
                "status": "baseline",
                "checkpoint_path": self.config.model.model_id,
            }
            self.tracker.log_iteration(record)
            self.tracker.log_generations(record["iteration_id"], eval_split, eval_res.records)

        # Primary baseline accuracy
        self.baseline_accuracy = combined_scores[0] if combined_scores else 0.0

        root_record = {
            "iteration_id": "baseline",
            "parent_id": "root",
            "pruning_type": "none",
            "operation_description": "Initial unpruned baseline model",
            "params_before": meta.total_params,
            "params_after": meta.total_params,
            "params_removed": 0,
            "param_retention_pct": 100.0,
            "stage": eval_split,
            "sample_size": sum(r.total for r in self.baselines.values()),
            "num_correct": sum(r.correct for r in self.baselines.values()),
            "accuracy": self.baseline_accuracy,
            "accuracy_ci_lower": self.baseline_accuracy,
            "accuracy_ci_upper": self.baseline_accuracy,
            "accuracy_retention_pct": 100.0,
            "tokens_per_sec": 0.0,
            "peak_vram_gb": get_gpu_vram_allocated_gb(self.adapter.device),
            "elapsed_time_sec": sum(r.total_time_sec for r in self.baselines.values()),
            "status": "baseline",
            "checkpoint_path": self.config.model.model_id,
        }
        self.accepted_checkpoints.append(root_record)
        self.tracker.update_pareto_frontier(self.accepted_checkpoints)

        return root_record

    def run(
        self,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute full automated pruning search until stopping criteria.
        Enforces multi-benchmark constraints and transactional acceptance.
        """
        # Restore or compute baseline
        if self.baseline_accuracy is None:
            existing_baseline = self.tracker.get_baseline()
            if existing_baseline:
                self.baseline_accuracy = existing_baseline["accuracy"]
                self.baseline_params = existing_baseline["params_after"]
                self.accepted_checkpoints = self.tracker.get_all_accepted()
                for b_name in self.benchmarks:
                    self.baselines[b_name] = EvaluationResult(
                        total=existing_baseline.get("sample_size", 16),
                        correct=existing_baseline.get("num_correct", 16),
                        accuracy=existing_baseline.get("accuracy", self.baseline_accuracy),
                    )
            else:
                self.run_baseline()

        for b_name in self.benchmarks:
            if b_name not in self.baselines:
                self.baselines[b_name] = EvaluationResult(
                    total=16,
                    correct=int(round((self.baseline_accuracy or 1.0) * 16)),
                    accuracy=self.baseline_accuracy or 1.0,
                )

        accepted_pruning_count = len([c for c in self.accepted_checkpoints if c.get("iteration_id") != "baseline"])
        iteration = accepted_pruning_count
        search_start_time = time.time()
        stopping_reason = "completed"

        val_split = self.config.benchmark.split
        val_samples = self._get_split_samples(val_split)

        screen_split = "screening" if any("screening" in b.data_splits for b in self.benchmarks.values()) else "smoke"
        screen_samples = min(self.config.benchmark.screening_size, val_samples or self.config.benchmark.screening_size)

        primary_bench = list(self.benchmarks.values())[0]

        while iteration < self.config.pruning.max_iterations:
            iteration += 1

            if self.config.pruning.time_budget_sec:
                if (time.time() - search_start_time) >= self.config.pruning.time_budget_sec:
                    stopping_reason = "time_budget_exhausted"
                    break

            layers = self.adapter.get_layers()
            if len(layers) <= 2:
                stopping_reason = "minimum_layers_reached"
                break

            # 1. Structural Importance Analysis
            calibration_samples = primary_bench.data_splits.get("calibration", [])
            analysis = self.pruner.analyze(
                adapter=self.adapter,
                calibration_samples=calibration_samples,
                max_samples=32,
            )

            # 2. Propose & Rank Candidates
            all_cands = self.pruner.propose_candidates(self.adapter, analysis, top_k=self.config.pruning.max_candidates)
            cands = self.search_strategy.select_candidates(all_cands, budget=self.config.pruning.max_candidates)

            if not cands:
                stopping_reason = "no_eligible_candidates"
                break

            best_candidate = None
            best_screen_score = -1.0

            # 3. Phase 1: Fast Screening across candidate pool
            for cand in cands:
                layer_idx = cand.target_info["layer_idx"]
                layer_backup = self.adapter.model.model.layers[layer_idx]

                prune_res = self.pruner.apply_pruning(self.adapter, cand)
                if not prune_res.validation_passed:
                    self._rollback_layer(layer_idx, layer_backup)
                    continue

                screen_res = primary_bench.evaluate(
                    model=self.adapter.model,
                    tokenizer=self.adapter.tokenizer,
                    split_name=screen_split,
                    max_samples=screen_samples,
                    device=self.adapter.device,
                )

                self._rollback_layer(layer_idx, layer_backup)

                if screen_res.accuracy > best_screen_score:
                    best_screen_score = screen_res.accuracy
                    best_candidate = cand

            if best_candidate is None:
                stopping_reason = "all_candidates_failed_validation"
                break

            # 4. Phase 2: Full Development Benchmark Validation
            best_layer_idx = best_candidate.target_info["layer_idx"]
            best_layer_backup = self.adapter.model.model.layers[best_layer_idx]
            prune_res = self.pruner.apply_pruning(self.adapter, best_candidate)

            candidate_eval_results: Dict[str, EvaluationResult] = {}
            for b_name, benchmark in self.benchmarks.items():
                cand_res = benchmark.evaluate(
                    model=self.adapter.model,
                    tokenizer=self.adapter.tokenizer,
                    split_name=val_split,
                    max_samples=val_samples,
                    device=self.adapter.device,
                )
                candidate_eval_results[b_name] = cand_res

            # Multi-benchmark constraint checking
            all_passed, details = self.constraints.check_all(candidate_eval_results, self.baselines)

            primary_res = candidate_eval_results[primary_bench.name]
            remaining_layers = len(self.adapter.get_layers())
            iter_id = f"iter_{iteration}_{best_candidate.candidate_id}"
            parent_id = "baseline" if iteration == 1 else f"iter_{iteration-1}"
            retention_pct = (primary_res.accuracy / self.baseline_accuracy * 100.0) if self.baseline_accuracy > 0 else 0.0
            acc_metrics = compute_accuracy_metrics(primary_res.correct, primary_res.total, baseline_acc=self.baseline_accuracy)

            if not all_passed:
                # Rollback candidate: failed multi-benchmark constraint!
                self._rollback_layer(best_layer_idx, best_layer_backup)
                clean_gpu_memory(self.adapter.device)

                fail_summary = "; ".join(f"{k}: {v['reason']}" for k, v in details.items() if not v["passed"])
                rej_record = {
                    "iteration_id": iter_id,
                    "parent_id": parent_id,
                    "pruning_type": "layer",
                    "operation_description": f"Rejected layer {best_layer_idx}: {fail_summary}",
                    "params_before": prune_res.params_before,
                    "params_after": prune_res.params_before,
                    "params_removed": 0,
                    "param_retention_pct": (prune_res.params_before / self.baseline_params) * 100.0,
                    "stage": val_split,
                    "sample_size": primary_res.total,
                    "num_correct": primary_res.correct,
                    "accuracy": acc_metrics["accuracy"],
                    "accuracy_ci_lower": acc_metrics["accuracy_ci_lower"],
                    "accuracy_ci_upper": acc_metrics["accuracy_ci_upper"],
                    "accuracy_retention_pct": retention_pct,
                    "tokens_per_sec": primary_res.tokens_per_sec,
                    "peak_vram_gb": primary_res.peak_vram_gb,
                    "elapsed_time_sec": primary_res.total_time_sec,
                    "status": "rejected_retention_failed",
                    "rejection_reason": fail_summary,
                    "checkpoint_path": "",
                }
                self.tracker.log_iteration(rej_record)
                stopping_reason = f"retention_constraint_failed ({fail_summary})"
                break

            # 5. Checkpoint Export
            self._ensure_disk_space()
            ckpt_name = f"checkpoint_layers_{remaining_layers}"
            ckpt_dir = self.exporter.export(
                adapter=self.adapter,
                checkpoint_name=ckpt_name,
                metadata={
                    "iteration": iteration,
                    "candidate_id": best_candidate.candidate_id,
                    "accuracy": primary_res.accuracy,
                    "baseline_accuracy": self.baseline_accuracy,
                    "retention_pct": retention_pct,
                    "benchmark_split": val_split,
                    "sample_size": primary_res.total,
                    "multi_benchmark_details": details,
                },
            )

            # 6. Subprocess Reload Validation
            if self.config.hardware.cpu_offload_validation:
                # Validate in isolated subprocess using CPU memory to eliminate VRAM contention on <=8GB GPUs
                reload_ok, reload_msg = SubprocessValidator.validate_checkpoint(ckpt_dir, device="cpu")
            else:
                reload_ok, reload_msg = True, "skipped"

            if not reload_ok:
                self._rollback_layer(best_layer_idx, best_layer_backup)
                clean_gpu_memory(self.adapter.device)

                rej_record = {
                    "iteration_id": iter_id,
                    "parent_id": parent_id,
                    "pruning_type": "layer",
                    "operation_description": f"Rejected layer {best_layer_idx}: reload failed ({reload_msg})",
                    "params_before": prune_res.params_before,
                    "params_after": prune_res.params_before,
                    "params_removed": 0,
                    "param_retention_pct": (prune_res.params_before / self.baseline_params) * 100.0,
                    "stage": val_split,
                    "sample_size": primary_res.total,
                    "num_correct": primary_res.correct,
                    "accuracy": acc_metrics["accuracy"],
                    "accuracy_ci_lower": acc_metrics["accuracy_ci_lower"],
                    "accuracy_ci_upper": acc_metrics["accuracy_ci_upper"],
                    "accuracy_retention_pct": retention_pct,
                    "tokens_per_sec": primary_res.tokens_per_sec,
                    "peak_vram_gb": primary_res.peak_vram_gb,
                    "elapsed_time_sec": primary_res.total_time_sec,
                    "status": "rejected_reload_failed",
                    "rejection_reason": f"Subprocess reload validation failed: {reload_msg}",
                    "checkpoint_path": ckpt_dir,
                }
                self.tracker.log_iteration(rej_record)
                stopping_reason = f"checkpoint_validation_failed: {reload_msg}"
                break

            # 7. Candidate Accepted and Registered
            record = {
                "iteration_id": iter_id,
                "parent_id": parent_id,
                "pruning_type": "layer",
                "operation_description": f"Removed layer {best_layer_idx}, remaining={remaining_layers}",
                "params_before": prune_res.params_before,
                "params_after": prune_res.params_after,
                "params_removed": prune_res.params_removed,
                "param_retention_pct": (prune_res.params_after / self.baseline_params) * 100.0,
                "stage": val_split,
                "sample_size": primary_res.total,
                "num_correct": primary_res.correct,
                "accuracy": acc_metrics["accuracy"],
                "accuracy_ci_lower": acc_metrics["accuracy_ci_lower"],
                "accuracy_ci_upper": acc_metrics["accuracy_ci_upper"],
                "accuracy_retention_pct": retention_pct,
                "tokens_per_sec": primary_res.tokens_per_sec,
                "peak_vram_gb": primary_res.peak_vram_gb,
                "elapsed_time_sec": primary_res.total_time_sec,
                "status": "accepted",
                "checkpoint_path": ckpt_dir,
            }

            self.tracker.log_iteration(record)
            self.tracker.log_generations(iter_id, val_split, primary_res.records)
            self.accepted_checkpoints.append(record)
            self.tracker.update_pareto_frontier(self.accepted_checkpoints)

            if progress_callback:
                progress_callback(record)

            clean_gpu_memory(self.adapter.device)

        self.tracker.export_csv()
        latest = self.tracker.get_latest_accepted()
        if not latest:
            latest = self.accepted_checkpoints[-1] if self.accepted_checkpoints else {}

        return {
            "status": stopping_reason,
            "best_checkpoint": latest.get("checkpoint_path"),
            "original_params": self.baseline_params,
            "final_params": latest.get("params_after", self.baseline_params),
            "baseline_accuracy": self.baseline_accuracy,
            "final_accuracy": latest.get("accuracy", self.baseline_accuracy),
            "accuracy_retention_pct": latest.get("accuracy_retention_pct", 100.0),
            "iterations_completed": iteration,
            "run_dir": self.run_dir,
        }

    def _rollback_layer(self, layer_idx: int, layer_module: torch.nn.Module):
        self.adapter.model.model.layers.insert(layer_idx, layer_module)
        self.adapter.model.config.num_hidden_layers = len(self.adapter.model.model.layers)
        if hasattr(self.adapter.model.config, "layer_types") and self.adapter.model.config.layer_types is not None:
            self.adapter.model.config.layer_types.insert(layer_idx, "full_attention")
        for idx, layer in enumerate(self.adapter.model.model.layers):
            layer.layer_idx = idx
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_idx = idx

    def _ensure_disk_space(self, min_free_gb: float = 6.0):
        try:
            free_gb = shutil.disk_usage(self.run_dir).free / (1024**3)
            # Never delete active checkpoints or states referenced by the tracker
            if free_gb < min_free_gb and len(self.accepted_checkpoints) > 2:
                # Remove safetensors from older non-terminal checkpoints while preserving metadata
                for old in self.accepted_checkpoints[:-2]:
                    p = old.get("checkpoint_path")
                    if p and os.path.exists(p) and p != self.config.model.model_id:
                        sf = os.path.join(p, "model.safetensors")
                        if os.path.exists(sf):
                            os.remove(sf)
        except Exception:
            pass

    def resume(self) -> Dict[str, Any]:
        """
        Resume search from last accepted state, strictly preserving original baseline accuracy
        and parameter counts across all benchmarks.
        """
        baseline_record = self.tracker.get_baseline()
        if baseline_record:
            self.baseline_accuracy = baseline_record["accuracy"]
            self.baseline_params = baseline_record["params_after"]
            self.accepted_checkpoints = self.tracker.get_all_accepted()

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
        self.adapter.load()
        return self.run()
