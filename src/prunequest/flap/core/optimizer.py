"""
Incremental FLAP-based structured pruning optimizer with transactional rollback.
"""

from typing import Dict, Any
import os
import time
import shutil
import gc
import torch

from prunequest.flap.config import FLAPConfig
from prunequest.flap.models.registry import ModelRegistry
from prunequest.flap.calibration.dataset import DatasetManager
from prunequest.flap.calibration.collector import FLAPActivationCollector
from prunequest.flap.metrics import FLAPMetricCalculator
from prunequest.flap.standardization import ScoreStandardizer
from prunequest.flap.allocation import PruningAllocator
from prunequest.flap.pruning.slicer import MLPSlicer
from prunequest.flap.pruning.manager import PruningStateManager
from prunequest.flap.evaluation.screening import ProxyScreeningEvaluator
from prunequest.flap.evaluation.gsm8k import GSM8KEvaluator
from prunequest.flap.export.exporter import CheckpointExporter
from prunequest.flap.export.validator import CheckpointValidator
from prunequest.flap.core.tracker import ExperimentTracker


class FLAPOptimizer:
    """
    Orchestrates the complete incremental FLAP pruning optimization loop.
    """

    def __init__(self, config: FLAPConfig):
        self.config = config
        self.tracker = ExperimentTracker(config.output_dir)
        self.adapter = None
        self.model = None
        self.tokenizer = None
        self.dataset_manager = DatasetManager(
            dataset_name=config.calibration.dataset_name,
            dataset_config=config.calibration.dataset_config,
            seed=config.calibration.seed,
        )
        self.splits = None
        self.importance_scores: Dict[int, torch.Tensor] = {}
        self.baseline_accuracy: float = 0.0
        self.baseline_loss: float = 0.0
        self.baseline_params: int = 0
        self.active_channels: Dict[int, list] = {}

    def initialize(self):
        """Load model, tokenizer, and dataset partitions."""
        self.adapter = ModelRegistry.create_adapter(
            model_id_or_path=self.config.model.model_id,
            device=self.config.model.device,
            torch_dtype=self.config.model.torch_dtype,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        self.model, self.tokenizer = self.adapter.load()
        self.baseline_params = self.adapter.count_parameters()

        # Initialize active channels to full intermediate width per layer
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            layers = []

        self.active_channels = {
            l_idx: list(range(layer.mlp.gate_proj.out_features))
            for l_idx, layer in enumerate(layers)
        }

        # Load disjoint data partitions
        self.splits = self.dataset_manager.load_and_partition(
            calibration_samples=self.config.calibration.num_samples,
            screening_samples=self.config.calibration.screening_samples,
            acceptance_samples=self.config.calibration.acceptance_samples,
            tokenizer=self.tokenizer,
        )

    def run_profiling(self) -> Dict[int, torch.Tensor]:
        """
        Runs calibration and computes FLAP importance scores across all layers.
        """
        calib_dir = os.path.join(self.config.output_dir, "calibration")
        os.makedirs(calib_dir, exist_ok=True)
        scores_file = os.path.join(calib_dir, "importance_scores.pt")

        if os.path.exists(scores_file):
            self.importance_scores = torch.load(scores_file, map_location="cpu", weights_only=False)
            return self.importance_scores

        collector = FLAPActivationCollector(device=self.config.model.device)
        stats = collector.collect(
            model=self.model,
            tokenizer=self.tokenizer,
            samples=self.splits["calibration"],
            max_length=self.config.calibration.max_length,
            batch_size=self.config.calibration.batch_size,
        )

        raw_scores = FLAPMetricCalculator.compute_all_scores(stats, metric=self.config.flap.metric)
        self.importance_scores = ScoreStandardizer.standardize_all(raw_scores, method=self.config.flap.standardization)

        # Save profiling artifacts
        torch.save(self.importance_scores, scores_file)
        manifest = {
            "dataset": self.config.calibration.dataset_name,
            "samples": len(self.splits["calibration"]),
            "metric": self.config.flap.metric,
            "standardization": self.config.flap.standardization,
            "layers": len(self.importance_scores),
        }
        with open(os.path.join(calib_dir, "dataset_manifest.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(manifest, f, indent=2)

        return self.importance_scores

    def establish_baseline(self) -> Dict[str, Any]:
        """
        Evaluates unpruned model baseline on screening and acceptance splits.
        """
        existing = self.tracker.get_baseline()
        if existing:
            self.baseline_accuracy = existing["accuracy"]
            self.baseline_loss = existing.get("screening_loss") or 0.0
            self.baseline_params = existing["params_after"]
            return existing

        # Proxy screening baseline
        screen_res = ProxyScreeningEvaluator.evaluate_loss(
            model=self.model,
            tokenizer=self.tokenizer,
            samples=self.splits["screening"],
            device=self.config.model.device,
        )
        self.baseline_loss = screen_res.loss

        # Mathematical accuracy baseline
        accept_res = GSM8KEvaluator.evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            samples=self.splits["acceptance"],
            device=self.config.model.device,
            batch_size=self.config.evaluation.batch_size,
            max_batch_size=self.config.evaluation.max_batch_size,
            max_new_tokens=self.config.evaluation.max_new_tokens,
            temperature=self.config.evaluation.temperature,
        )
        self.baseline_accuracy = accept_res.accuracy
        print(f"\n✔ Baseline Established: Score = {accept_res.correct}/{accept_res.total} ({accept_res.accuracy*100:.2f}%) | Screening Loss = {screen_res.loss:.3f}\n")

        baseline_record = {
            "candidate_id": "baseline",
            "step_index": 0,
            "prune_ratio": 0.0,
            "params_before": self.baseline_params,
            "params_after": self.baseline_params,
            "params_removed": 0,
            "screening_loss": screen_res.loss,
            "screening_ppl": screen_res.perplexity,
            "accuracy": accept_res.accuracy,
            "num_correct": accept_res.correct,
            "sample_size": accept_res.total,
            "retention_pct": 100.0,
            "tokens_per_sec": accept_res.tokens_per_sec,
            "peak_vram_gb": accept_res.peak_vram_gb,
            "elapsed_sec": accept_res.total_time_sec + screen_res.total_time_sec,
            "status": "accepted",
            "rejection_reason": "",
            "checkpoint_path": self.config.model.model_id,
        }
        self.tracker.log_candidate(baseline_record)
        return baseline_record

    def optimize(self, resume: bool = True) -> Dict[str, Any]:
        """
        Executes incremental adaptive pruning loop with rollback and acceptance checks.
        """
        self.initialize()
        self.establish_baseline()
        self.run_profiling()

        min_required_acc = self.baseline_accuracy * self.config.pruning.retention
        current_ratio = 0.0
        step_ratio = self.config.pruning.initial_step_ratio
        evaluations_count = 0
        step_index = 0
        search_start_time = time.time()
        stopping_reason = "completed"

        if resume:
            latest_acc = self.tracker.get_latest_accepted()
            if latest_acc and latest_acc["candidate_id"] != "baseline":
                ckpt_path = latest_acc.get("checkpoint_path")
                if ckpt_path and not os.path.exists(ckpt_path):
                    alt_path = os.path.join(self.config.output_dir, "checkpoints", os.path.basename(ckpt_path))
                    if os.path.exists(alt_path):
                        ckpt_path = alt_path

                if ckpt_path and os.path.exists(ckpt_path):
                    print(f"\n🔄 Resuming from accepted checkpoint: {ckpt_path}")
                    print(f"   Pruned: {latest_acc['prune_ratio']*100:.2f}% | Score: {latest_acc['num_correct']}/{latest_acc['sample_size']} ({latest_acc['accuracy']*100:.2f}%) | Retention: {latest_acc['retention_pct']:.2f}%\n")
                    del self.model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    self.adapter = ModelRegistry.create_adapter(
                        model_id_or_path=ckpt_path,
                        device=self.config.model.device,
                        torch_dtype=self.config.model.torch_dtype,
                        trust_remote_code=self.config.model.trust_remote_code,
                    )
                    self.model, _ = self.adapter.load()
                    current_ratio = latest_acc["prune_ratio"]
                    step_index = latest_acc["step_index"]

                    plan = (
                        PruningAllocator.allocate_uniform(self.importance_scores, current_ratio)
                        if self.config.flap.allocation == "uniform"
                        else PruningAllocator.allocate_global_adaptive(self.importance_scores, current_ratio)
                    )
                    self.active_channels = {k: list(v) for k, v in plan.keep_indices.items()}

        while current_ratio < self.config.pruning.maximum_total_ratio:
            if evaluations_count >= self.config.pruning.max_candidate_evaluations:
                stopping_reason = "max_evaluations_reached"
                break

            if step_ratio < self.config.pruning.minimum_step_ratio:
                stopping_reason = "fully_pruned (minimum_step_reached)"
                break

            if self.config.pruning.time_budget_sec:
                if (time.time() - search_start_time) >= self.config.pruning.time_budget_sec:
                    stopping_reason = "time_budget_exhausted"
                    break

            target_ratio = min(current_ratio + step_ratio, self.config.pruning.maximum_total_ratio)
            evaluations_count += 1
            cand_id = f"cand_step_{step_index+1}_ratio_{target_ratio:.4f}"

            # 1. Allocate pruning channels
            if self.config.flap.allocation == "uniform":
                plan = PruningAllocator.allocate_uniform(self.importance_scores, target_ratio)
            else:
                plan = PruningAllocator.allocate_global_adaptive(self.importance_scores, target_ratio)

            # 2. Transactional backup
            backup = PruningStateManager.backup_mlp_weights(self.model, active_channels=self.active_channels)

            # 3. Physical slicing
            _ = MLPSlicer.apply_plan(self.model, plan, active_channels=self.active_channels)
            new_params = self.adapter.count_parameters()

            # 4. Stage A: Proxy loss screening
            screen_res = ProxyScreeningEvaluator.evaluate_loss(
                model=self.model,
                tokenizer=self.tokenizer,
                samples=self.splits["screening"],
                device=self.config.model.device,
            )

            # Check if loss exploded
            loss_threshold = (self.baseline_loss * self.config.evaluation.screening_loss_factor) + self.config.evaluation.screening_loss_margin
            if screen_res.loss > loss_threshold:
                # Screening failed
                PruningStateManager.restore_mlp_weights(self.model, backup, active_channels=self.active_channels)
                del backup
                gc.collect()
                rej_rec = {
                    "candidate_id": cand_id,
                    "step_index": step_index + 1,
                    "prune_ratio": target_ratio,
                    "params_before": self.baseline_params,
                    "params_after": new_params,
                    "params_removed": self.baseline_params - new_params,
                    "screening_loss": screen_res.loss,
                    "screening_ppl": screen_res.perplexity,
                    "accuracy": 0.0,
                    "num_correct": 0,
                    "sample_size": len(self.splits["screening"]),
                    "retention_pct": 0.0,
                    "elapsed_sec": screen_res.total_time_sec,
                    "status": "rejected_screening",
                    "rejection_reason": f"Screening loss ({screen_res.loss:.3f}) exceeded threshold ({loss_threshold:.3f})",
                    "checkpoint_path": "",
                }
                self.tracker.log_candidate(rej_rec)
                step_ratio *= self.config.pruning.backoff_factor
                continue

            # 5. Stage B: Mathematical accuracy evaluation
            accept_res = GSM8KEvaluator.evaluate(
                model=self.model,
                tokenizer=self.tokenizer,
                samples=self.splits["acceptance"],
                device=self.config.model.device,
                batch_size=self.config.evaluation.batch_size,
                max_batch_size=self.config.evaluation.max_batch_size,
                max_new_tokens=self.config.evaluation.max_new_tokens,
                temperature=self.config.evaluation.temperature,
            )
            retention_pct = (accept_res.accuracy / self.baseline_accuracy * 100.0) if self.baseline_accuracy > 0 else 0.0
            print(f"\n[Candidate {cand_id}] Pruned: {target_ratio*100:.2f}% | Score: {accept_res.correct}/{accept_res.total} ({accept_res.accuracy*100:.2f}%) | Retention: {retention_pct:.2f}%\n")

            if accept_res.accuracy < min_required_acc:
                # Retention failed
                PruningStateManager.restore_mlp_weights(self.model, backup, active_channels=self.active_channels)
                del backup
                gc.collect()
                rej_rec = {
                    "candidate_id": cand_id,
                    "step_index": step_index + 1,
                    "prune_ratio": target_ratio,
                    "params_before": self.baseline_params,
                    "params_after": new_params,
                    "params_removed": self.baseline_params - new_params,
                    "screening_loss": screen_res.loss,
                    "screening_ppl": screen_res.perplexity,
                    "accuracy": accept_res.accuracy,
                    "num_correct": accept_res.correct,
                    "sample_size": accept_res.total,
                    "retention_pct": retention_pct,
                    "tokens_per_sec": accept_res.tokens_per_sec,
                    "peak_vram_gb": accept_res.peak_vram_gb,
                    "elapsed_sec": accept_res.total_time_sec + screen_res.total_time_sec,
                    "status": "rejected_retention",
                    "rejection_reason": f"Accuracy {accept_res.accuracy*100:.2f}% below retention threshold {min_required_acc*100:.2f}%",
                    "checkpoint_path": "",
                }
                self.tracker.log_candidate(rej_rec)
                step_ratio *= self.config.pruning.backoff_factor
                continue

            # 6. Stage C: Checkpoint export and subprocess reload validation
            step_index += 1
            ckpt_name = f"accepted_step_{step_index:03d}"
            metadata = {
                "step_index": step_index,
                "candidate_id": cand_id,
                "prune_ratio": target_ratio,
                "accuracy": accept_res.accuracy,
                "baseline_accuracy": self.baseline_accuracy,
                "retention_pct": retention_pct,
                "params": new_params,
                "params_pruned": self.baseline_params - new_params,
                "method": "flap",
            }
            gc.collect()
            ckpt_dir = CheckpointExporter.export(
                model=self.model,
                tokenizer=self.tokenizer,
                output_dir=self.config.output_dir,
                checkpoint_name=ckpt_name,
                metadata=metadata,
            )
            gc.collect()

            if self.config.hardware.cpu_offload_validation:
                reload_ok, reload_msg = CheckpointValidator.validate(
                    ckpt_dir,
                    max_new_tokens=self.config.evaluation.validation_max_tokens,
                    prompt=self.config.evaluation.validation_prompt,
                )
            else:
                reload_ok, reload_msg = True, "skipped"

            if not reload_ok:
                # Reload validation failed
                PruningStateManager.restore_mlp_weights(self.model, backup, active_channels=self.active_channels)
                del backup
                gc.collect()
                rej_rec = {
                    "candidate_id": cand_id,
                    "step_index": step_index,
                    "prune_ratio": target_ratio,
                    "params_before": self.baseline_params,
                    "params_after": new_params,
                    "params_removed": self.baseline_params - new_params,
                    "accuracy": accept_res.accuracy,
                    "retention_pct": retention_pct,
                    "status": "rejected_reload",
                    "rejection_reason": f"Reload validation failed: {reload_msg}",
                    "checkpoint_path": ckpt_dir,
                }
                self.tracker.log_candidate(rej_rec)
                step_ratio *= self.config.pruning.backoff_factor
                continue

            # 7. Candidate ACCEPTED!
            del backup
            gc.collect()
            current_ratio = target_ratio
            acc_rec = {
                "candidate_id": cand_id,
                "step_index": step_index,
                "prune_ratio": current_ratio,
                "params_before": self.baseline_params,
                "params_after": new_params,
                "params_removed": self.baseline_params - new_params,
                "screening_loss": screen_res.loss,
                "screening_ppl": screen_res.perplexity,
                "accuracy": accept_res.accuracy,
                "num_correct": accept_res.correct,
                "sample_size": accept_res.total,
                "retention_pct": retention_pct,
                "tokens_per_sec": accept_res.tokens_per_sec,
                "peak_vram_gb": accept_res.peak_vram_gb,
                "elapsed_sec": accept_res.total_time_sec + screen_res.total_time_sec,
                "status": "accepted",
                "rejection_reason": "",
                "checkpoint_path": ckpt_dir,
            }
            self.tracker.log_candidate(acc_rec)

            # Promote to 'best'
            best_dir = os.path.join(self.config.output_dir, "checkpoints", "best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)

            # On success with retention >= threshold, continue with configured 5% increment
            if self.config.pruning.adaptive_step and retention_pct >= (self.config.pruning.retention * 100.0):
                step_ratio = self.config.pruning.initial_step_ratio

        best_checkpoint = self.tracker.get_best_accepted()
        return {
            "stopping_reason": stopping_reason,
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_params": self.baseline_params,
            "best_checkpoint": best_checkpoint,
            "total_evaluations": evaluations_count,
            "model_id": self.config.model.model_id,
            "metric": self.config.flap.metric.upper(),
            "target": self.config.pruning.target,
            "benchmark": self.config.evaluation.benchmark.upper(),
        }
