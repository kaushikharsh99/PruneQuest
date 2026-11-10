"""
Joint Incremental Pruning Optimizer (JIPO) — Full Sweep Engine.
Implements multi-fidelity joint width & depth pruning with adaptive fallback and Pareto tracking.
"""

from typing import Dict, Any, Optional, List
import os
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from prunequest.sweep.config import JointSweepConfig
from prunequest.sweep.state import JointStateManager
from prunequest.sweep.tracker import JointExperimentTracker
from prunequest.models.registry import ModelRegistry
from prunequest.models.base import BaseModelAdapter
from prunequest.flap.calibration.dataset import DatasetManager, DatasetSample
from prunequest.flap.calibration.collector import FLAPActivationCollector
from prunequest.flap.metrics import FLAPMetricCalculator
from prunequest.flap.standardization import ScoreStandardizer
from prunequest.flap.allocation import PruningAllocator, AllocationPlan
from prunequest.flap.evaluation.screening import ProxyScreeningEvaluator
from prunequest.flap.evaluation.gsm8k import GSM8KEvaluator
from prunequest.flap.export.exporter import CheckpointExporter
from prunequest.flap.export.validator import CheckpointValidator
from prunequest.pruning.importance import compute_block_influence


class JointCandidate:
    """Represents a proposed pruning modification (joint, flap-only, or block-only)."""

    def __init__(
        self,
        candidate_type: str,  # 'joint', 'flap_only', 'block_only'
        block_idx: Optional[int],
        plan: Optional[AllocationPlan],
        estimated_params_removed: int,
        predicted_loss: float = 0.0,
    ):
        self.candidate_type = candidate_type
        self.block_idx = block_idx
        self.plan = plan
        self.estimated_params_removed = estimated_params_removed
        self.predicted_loss = predicted_loss
        self.utility_score: float = 0.0

    def compute_utility(self, epsilon: float = 1e-4, block_penalty: float = 1.0) -> float:
        """Computes ranking heuristic U(C) = Bytes Saved / (eps + Predicted Loss)."""
        bytes_saved = self.estimated_params_removed * 2  # bfloat16: 2 bytes/param
        degradation = max(self.predicted_loss, 0.0001)
        base_u = bytes_saved / (epsilon + degradation)
        if self.candidate_type == "block_only":
            base_u *= block_penalty
        self.utility_score = base_u
        return self.utility_score


class JointIncrementalOptimizer:
    """
    Executes the 3-Level Joint Incremental Pruning Optimization (JIPO) loop:
    - Level 1: Cheap importance profiling with cached invalidation
    - Level 2: Multi-fidelity cheap proxy loss screening
    - Level 3: Targeted benchmark validation with fallback and Pareto frontier tracking
    """

    def __init__(self, config: JointSweepConfig):
        self.config = config
        self.run_dir = config.output_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.tracker = JointExperimentTracker(self.run_dir)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")

        self.adapter: Optional[BaseModelAdapter] = None
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        self.calibration_samples: List[DatasetSample] = []
        self.screening_samples: List[DatasetSample] = []
        self.acceptance_samples: List[DatasetSample] = []

        self.baseline_accuracy: float = 0.0
        self.baseline_loss: float = 0.0
        self.baseline_params: int = 0

        # Profile caches
        self.cached_block_scores: Optional[List[Dict[str, Any]]] = None
        self.cached_flap_stats: Optional[Dict[int, Any]] = None
        self.cached_flap_scores: Optional[Dict[int, torch.Tensor]] = None
        self.profiles_valid: bool = False

        # Search dynamics
        self.current_flap_ratio: float = config.pruning.initial_flap_ratio
        self.total_flap_ratio_applied: float = 0.0
        self.total_blocks_removed: int = 0
        self.consecutive_block_failures: int = 0
        self.block_pruning_exhausted: bool = False
        self.active_channels: Dict[int, List[int]] = {}

    def initialize(self):
        """Loads model, tokenizer, and dataset partitions."""
        self.adapter = ModelRegistry.get_adapter(
            model_id_or_path=self.config.model.model_id,
            device=self.config.model.device,
            torch_dtype=self.config.model.torch_dtype,
        )
        self.model, self.tokenizer = self.adapter.load()

        dm = DatasetManager(
            dataset_name=self.config.calibration.dataset_name,
            dataset_config=self.config.calibration.dataset_config,
            seed=self.config.calibration.seed,
        )
        splits = dm.load_and_partition(
            calibration_samples=self.config.calibration.num_samples,
            screening_samples=self.config.calibration.screening_samples,
            acceptance_samples=self.config.calibration.acceptance_samples,
            tokenizer=self.tokenizer,
        )
        self.calibration_samples = splits["calibration"]
        self.screening_samples = splits["screening"]
        self.acceptance_samples = splits["acceptance"]

        # Initialize active channels tracking for all MLP layers
        layers = JointStateManager.get_layers(self.model)
        for l_idx, layer in enumerate(layers):
            inter_size = layer.mlp.down_proj.in_features
            self.active_channels[l_idx] = list(range(inter_size))

        self.baseline_params = sum(p.numel() for p in self.model.parameters())

    def run_baseline(self) -> Dict[str, Any]:
        """Establishes unpruned baseline for loss and GSM8K reasoning accuracy."""
        dev = self.config.model.device

        # Screening baseline loss
        screen_res = ProxyScreeningEvaluator.evaluate_loss(
            self.model, self.tokenizer, self.screening_samples, device=dev,
            max_length=self.config.calibration.max_length,
        )
        self.baseline_loss = screen_res.loss

        # Accuracy baseline
        eval_res = GSM8KEvaluator.evaluate(
            self.model, self.tokenizer, self.acceptance_samples, device=dev,
            batch_size=self.config.evaluation.batch_size,
            max_batch_size=self.config.evaluation.max_batch_size,
            max_new_tokens=self.config.evaluation.max_new_tokens,
            temperature=self.config.evaluation.temperature,
        )
        self.baseline_accuracy = eval_res.accuracy
        self.tracker.record_full_evaluation()

        record = {
            "trial_id": "baseline",
            "iteration": 0,
            "action_type": "baseline",
            "parent_id": "none",
            "params_before": self.baseline_params,
            "params_after": self.baseline_params,
            "params_removed": 0,
            "param_retention_pct": 100.0,
            "screening_loss": self.baseline_loss,
            "accuracy": self.baseline_accuracy,
            "accuracy_retention_pct": 100.0,
            "status": "accepted",
            "rejection_reason": "",
            "checkpoint_path": "unpruned_baseline",
            "elapsed_time_sec": eval_res.total_time_sec,
        }
        self.tracker.log_trial(record)
        return record

    def _refresh_profiles(self):
        """Level 1: Computes or refreshes cached Block Influence and FLAP profiles."""
        if self.profiles_valid and self.cached_block_scores and self.cached_flap_scores:
            return

        dev = self.config.model.device

        # 1. Block Influence calculation
        benchmark_samples = [
            type("Sample", (), {"question": s.question, "id": s.id})()
            for s in self.calibration_samples[:32]
        ]
        self.cached_block_scores = compute_block_influence(
            self.model, self.tokenizer, benchmark_samples,
            metric="cosine", device=dev, max_samples=32,
        )

        # 2. FLAP intermediate activations
        collector = FLAPActivationCollector(device=dev)
        self.cached_flap_stats = collector.collect(
            self.model, self.tokenizer, self.calibration_samples,
            max_length=self.config.calibration.max_length, batch_size=1,
        )
        self.cached_flap_scores = FLAPMetricCalculator.compute_all_scores(
            self.cached_flap_stats, metric=self.config.flap.metric,
        )
        self.profiles_valid = True

    def _propose_candidates(self) -> Dict[str, JointCandidate]:
        """
        Generates candidate operations:
        - C1: Joint (Block + FLAP)
        - C2: Block-only
        - C3: FLAP-only
        """
        self._refresh_profiles()
        layers = JointStateManager.get_layers(self.model)
        num_layers = len(layers)

        # Propose Block candidate
        block_idx: Optional[int] = None
        block_param_reduction = 0
        if (not self.block_pruning_exhausted and
            self.total_blocks_removed < self.config.pruning.max_block_removals and
            num_layers > self.config.pruning.min_remaining_layers and
            self.cached_block_scores):
            # Sort by lowest score (lowest cosine distance = highest redundancy)
            valid_block_rankings = [
                b for b in sorted(self.cached_block_scores, key=lambda x: x["score"])
                if 0 <= b["layer_idx"] < num_layers
            ]
            if valid_block_rankings:
                block_idx = valid_block_rankings[0]["layer_idx"]
                block_param_reduction = sum(p.numel() for p in layers[block_idx].parameters())

        # Propose FLAP candidate
        plan: Optional[AllocationPlan] = None
        flap_param_reduction = 0
        if (self.current_flap_ratio >= self.config.pruning.minimum_flap_ratio and
            (self.total_flap_ratio_applied + self.current_flap_ratio) <= self.config.pruning.maximum_total_flap_ratio and
            self.cached_flap_scores):

            std_scores = ScoreStandardizer.standardize_all(
                self.cached_flap_scores, method=self.config.flap.standardization,
            )
            if self.config.flap.allocation == "global_adaptive":
                plan = PruningAllocator.allocate_global_adaptive(std_scores, self.current_flap_ratio)
            else:
                plan = PruningAllocator.allocate_uniform(std_scores, self.current_flap_ratio)
            # Estimate param reduction for FLAP: 3 projections * hidden_size * channels_pruned
            hidden_size = self.model.config.hidden_size
            total_channels_pruned = sum(len(c) for c in plan.prune_indices.values())
            flap_param_reduction = total_channels_pruned * hidden_size * 3

        candidates: Dict[str, JointCandidate] = {}

        # 1. Joint candidate
        if block_idx is not None and plan is not None:
            candidates["joint"] = JointCandidate(
                candidate_type="joint",
                block_idx=block_idx,
                plan=plan,
                estimated_params_removed=block_param_reduction + flap_param_reduction,
                predicted_loss=0.05,
            )

        # 2. Block-only candidate
        if block_idx is not None:
            candidates["block_only"] = JointCandidate(
                candidate_type="block_only",
                block_idx=block_idx,
                plan=None,
                estimated_params_removed=block_param_reduction,
                predicted_loss=0.03,
            )

        # 3. FLAP-only candidate
        if plan is not None:
            candidates["flap_only"] = JointCandidate(
                candidate_type="flap_only",
                block_idx=None,
                plan=plan,
                estimated_params_removed=flap_param_reduction,
                predicted_loss=0.01,
            )

        return candidates

    def optimize(self, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes the Joint Incremental Pruning Optimization search.
        """
        iters_limit = max_iterations or self.config.pruning.max_iterations
        start_time = time.time()
        dev = self.config.model.device
        stopping_reason = "max_iterations_reached"

        iteration = 0
        parent_id = "baseline"

        while iteration < iters_limit:
            iteration += 1

            if self.config.pruning.time_budget_sec and (time.time() - start_time) > self.config.pruning.time_budget_sec:
                stopping_reason = "time_budget_exhausted"
                break

            candidates = self._propose_candidates()
            if not candidates:
                stopping_reason = "no_eligible_candidates"
                break

            parent_snapshot = JointStateManager.backup_state(self.model, active_channels=self.active_channels)
            iteration_committed = False

            # ================= LEVEL 2: CHEAP JOINT SCREENING =================
            joint_cand = candidates.get("joint")
            joint_passed_screen = False

            if joint_cand is not None:
                # Apply joint candidate in-memory
                JointStateManager.apply_joint_candidate(
                    self.model, joint_cand.block_idx, joint_cand.plan, active_channels=self.active_channels
                )

                # Tier 1: Teacher-forced loss screening
                screen_res = ProxyScreeningEvaluator.evaluate_loss(
                    self.model, self.tokenizer, self.screening_samples, device=dev,
                    max_length=self.config.calibration.max_length,
                )
                loss_thresh = (self.baseline_loss * self.config.pruning.screening_loss_factor) + self.config.pruning.screening_loss_margin
                joint_cand.predicted_loss = max(screen_res.loss - self.baseline_loss, 0.0)

                if screen_res.loss <= loss_thresh:
                    joint_passed_screen = True
                else:
                    self.tracker.record_screening_rejection_savings()
                    self.tracker.log_trial({
                        "trial_id": f"iter_{iteration}_joint_screen_reject",
                        "iteration": iteration,
                        "action_type": "joint",
                        "parent_id": parent_id,
                        "params_before": sum(p.numel() for p in parent_snapshot["layers_list"][0].parameters()) * len(parent_snapshot["layers_list"]),
                        "params_after": sum(p.numel() for p in self.model.parameters()),
                        "params_removed": joint_cand.estimated_params_removed,
                        "param_retention_pct": (sum(p.numel() for p in self.model.parameters()) / self.baseline_params) * 100.0,
                        "screening_loss": screen_res.loss,
                        "accuracy": 0.0,
                        "accuracy_retention_pct": 0.0,
                        "status": "rejected_screen",
                        "rejection_reason": f"Screening loss {screen_res.loss:.3f} exceeded threshold {loss_thresh:.3f}",
                    })
                    # Revert joint candidate
                    JointStateManager.restore_state(self.model, parent_snapshot, active_channels_ref=self.active_channels)

            # ================= LEVEL 3: VALIDATION =================
            if joint_passed_screen and joint_cand is not None:
                # Full GSM8K development validation for Joint candidate
                eval_res = GSM8KEvaluator.evaluate(
                    self.model, self.tokenizer, self.acceptance_samples, device=dev,
                    batch_size=self.config.evaluation.batch_size,
                    max_batch_size=self.config.evaluation.max_batch_size,
                    max_new_tokens=self.config.evaluation.max_new_tokens,
                    temperature=self.config.evaluation.temperature,
                )
                self.tracker.record_full_evaluation()
                retention_pct = (eval_res.accuracy / self.baseline_accuracy * 100.0) if self.baseline_accuracy > 0 else 0.0

                if retention_pct >= (self.config.pruning.retention * 100.0):
                    # ACCEPT JOINT CANDIDATE!
                    self.tracker.record_joint_evaluation_savings()
                    self.total_blocks_removed += 1
                    self.total_flap_ratio_applied += self.current_flap_ratio
                    self.consecutive_block_failures = 0
                    self.profiles_valid = False  # Invalidate cached profile due to block deletion

                    curr_params = sum(p.numel() for p in self.model.parameters())
                    ckpt_name = f"joint_iter_{iteration}_layers_{len(JointStateManager.get_layers(self.model))}"
                    ckpt_path = CheckpointExporter.export(
                        self.model, self.tokenizer, self.run_dir, ckpt_name,
                        metadata={"iteration": iteration, "action": "joint", "retention": retention_pct},
                    )

                    self.tracker.log_trial({
                        "trial_id": f"iter_{iteration}_joint",
                        "iteration": iteration,
                        "action_type": "joint",
                        "parent_id": parent_id,
                        "params_before": sum(p.numel() for p in parent_snapshot["layers_list"][0].parameters()) * len(parent_snapshot["layers_list"]),
                        "params_after": curr_params,
                        "params_removed": joint_cand.estimated_params_removed,
                        "param_retention_pct": (curr_params / self.baseline_params) * 100.0,
                        "screening_loss": screen_res.loss,
                        "accuracy": eval_res.accuracy,
                        "accuracy_retention_pct": retention_pct,
                        "status": "accepted",
                        "checkpoint_path": ckpt_path,
                        "elapsed_time_sec": eval_res.total_time_sec,
                    })

                    parent_id = f"iter_{iteration}_joint"
                    iteration_committed = True
                else:
                    # Joint candidate failed accuracy retention -> Rollback to parent snapshot!
                    JointStateManager.restore_state(self.model, parent_snapshot, active_channels_ref=self.active_channels)
                    self.tracker.log_trial({
                        "trial_id": f"iter_{iteration}_joint_eval_reject",
                        "iteration": iteration,
                        "action_type": "joint",
                        "parent_id": parent_id,
                        "params_before": sum(p.numel() for p in parent_snapshot["layers_list"][0].parameters()) * len(parent_snapshot["layers_list"]),
                        "params_after": sum(p.numel() for p in self.model.parameters()),
                        "params_removed": joint_cand.estimated_params_removed,
                        "param_retention_pct": (sum(p.numel() for p in self.model.parameters()) / self.baseline_params) * 100.0,
                        "screening_loss": screen_res.loss,
                        "accuracy": eval_res.accuracy,
                        "accuracy_retention_pct": retention_pct,
                        "status": "rejected_retention",
                        "rejection_reason": f"Retention {retention_pct:.1f}% below target {self.config.pruning.retention*100:.1f}%",
                    })

            # ================= LEVEL 3 FALLBACK: TEST INDIVIDUAL ALTERNATIVES =================
            if not iteration_committed:
                alternatives: List[JointCandidate] = []
                if "block_only" in candidates:
                    alternatives.append(candidates["block_only"])
                if "flap_only" in candidates:
                    alternatives.append(candidates["flap_only"])

                # Rank alternatives by Utility U(C) = Bytes / (eps + Loss), applying block penalty if failing
                block_penalty = 0.2 if self.consecutive_block_failures >= 1 else 1.0
                for alt in alternatives:
                    alt.compute_utility(block_penalty=block_penalty)

                alternatives.sort(key=lambda x: x.utility_score, reverse=True)

                for alt in alternatives:
                    JointStateManager.apply_joint_candidate(
                        self.model, alt.block_idx, alt.plan, active_channels=self.active_channels
                    )

                    # Tier 1: Screen alternative
                    alt_screen = ProxyScreeningEvaluator.evaluate_loss(
                        self.model, self.tokenizer, self.screening_samples, device=dev,
                        max_length=self.config.calibration.max_length,
                    )
                    loss_thresh = (self.baseline_loss * self.config.pruning.screening_loss_factor) + self.config.pruning.screening_loss_margin

                    if alt_screen.loss > loss_thresh:
                        self.tracker.record_screening_rejection_savings()
                        JointStateManager.restore_state(self.model, parent_snapshot, active_channels_ref=self.active_channels)
                        if alt.candidate_type == "block_only":
                            self.consecutive_block_failures += 1
                            self.block_pruning_exhausted = True
                        continue

                    # Tier 3: Benchmark validation for alternative
                    alt_eval = GSM8KEvaluator.evaluate(
                        self.model, self.tokenizer, self.acceptance_samples, device=dev,
                        batch_size=self.config.evaluation.batch_size,
                        max_batch_size=self.config.evaluation.max_batch_size,
                        max_new_tokens=self.config.evaluation.max_new_tokens,
                        temperature=self.config.evaluation.temperature,
                    )
                    self.tracker.record_full_evaluation()
                    retention_pct = (alt_eval.accuracy / self.baseline_accuracy * 100.0) if self.baseline_accuracy > 0 else 0.0

                    if retention_pct >= (self.config.pruning.retention * 100.0):
                        # ACCEPT SINGLE ALTERNATIVE!
                        curr_params = sum(p.numel() for p in self.model.parameters())
                        if alt.candidate_type == "block_only":
                            self.total_blocks_removed += 1
                            self.consecutive_block_failures = 0
                            self.profiles_valid = False  # Invalidate profiles
                        elif alt.candidate_type == "flap_only":
                            self.total_flap_ratio_applied += self.current_flap_ratio

                        ckpt_name = f"{alt.candidate_type}_iter_{iteration}"
                        ckpt_path = CheckpointExporter.export(
                            self.model, self.tokenizer, self.run_dir, ckpt_name,
                            metadata={"iteration": iteration, "action": alt.candidate_type, "retention": retention_pct},
                        )

                        self.tracker.log_trial({
                            "trial_id": f"iter_{iteration}_{alt.candidate_type}",
                            "iteration": iteration,
                            "action_type": alt.candidate_type,
                            "parent_id": parent_id,
                            "params_before": sum(p.numel() for p in parent_snapshot["layers_list"][0].parameters()) * len(parent_snapshot["layers_list"]),
                            "params_after": curr_params,
                            "params_removed": alt.estimated_params_removed,
                            "param_retention_pct": (curr_params / self.baseline_params) * 100.0,
                            "screening_loss": alt_screen.loss,
                            "accuracy": alt_eval.accuracy,
                            "accuracy_retention_pct": retention_pct,
                            "status": "accepted",
                            "checkpoint_path": ckpt_path,
                            "elapsed_time_sec": alt_eval.total_time_sec,
                        })

                        parent_id = f"iter_{iteration}_{alt.candidate_type}"
                        iteration_committed = True
                        break
                    else:
                        # Rejection of alternative
                        JointStateManager.restore_state(self.model, parent_snapshot, active_channels_ref=self.active_channels)
                        if alt.candidate_type == "block_only":
                            self.consecutive_block_failures += 1
                            self.block_pruning_exhausted = True

                # If all alternatives failed in this step -> Adaptive step reduction
                if not iteration_committed:
                    self.current_flap_ratio *= self.config.pruning.backoff_factor
                    if self.consecutive_block_failures >= 1:
                        self.block_pruning_exhausted = True

                    if self.current_flap_ratio < self.config.pruning.minimum_flap_ratio and self.block_pruning_exhausted:
                        stopping_reason = "retention_boundary_reached"
                        break

        # Select best Pareto checkpoint
        best_ckpt = self.tracker.get_best_pareto_checkpoint()
        summary = {
            "status": "completed",
            "stopping_reason": stopping_reason,
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_params": self.baseline_params,
            "best_checkpoint": best_ckpt,
            "metrics": self.tracker.get_summary_metrics(),
            "total_blocks_removed": self.total_blocks_removed,
            "total_flap_ratio_applied": self.total_flap_ratio_applied,
            "elapsed_time_sec": time.time() - start_time,
        }

        # Subprocess validation of best checkpoint if exists
        if best_ckpt and best_ckpt.get("checkpoint_path"):
            val_ok, val_msg = CheckpointValidator.validate(best_ckpt["checkpoint_path"], device=dev)
            summary["best_checkpoint_reload_verified"] = val_ok
            summary["reload_message"] = val_msg

        return summary
