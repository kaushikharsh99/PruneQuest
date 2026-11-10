"""
Joint Incremental Pruning Optimizer (JIPO) — Full Sweep Engine.
Implements multi-fidelity joint width & depth pruning with adaptive fallback and Pareto tracking.
"""

from typing import Dict, Any, Optional, List
import os
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM

from prunequest.sweep.config import JointSweepConfig
from prunequest.sweep.state import JointStateManager
from prunequest.sweep.tracker import JointExperimentTracker
from prunequest.models.registry import ModelRegistry
from prunequest.models.base import BaseModelAdapter
from prunequest.reporting.metrics import clean_gpu_memory
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
from rich.console import Console
from rich.panel import Panel

console = Console()


class JointCandidate:
    """Represents a proposed pruning modification (joint, flap-only, or block-only)."""

    def __init__(
        self,
        candidate_type: str,  # 'joint', 'flap_only', 'block_only'
        block_idx: Optional[int],
        plan: Optional[AllocationPlan],
        estimated_params_removed: int,
        predicted_loss: float = 0.0,
        flap_ratio: float = 0.0,
        cumulative_params_removed: int = 0,
        cumulative_flap_ratio: float = 0.0,
    ):
        self.candidate_type = candidate_type
        self.block_idx = block_idx
        self.plan = plan
        self.estimated_params_removed = estimated_params_removed
        self.predicted_loss = predicted_loss
        self.flap_ratio = flap_ratio
        self.cumulative_params_removed = cumulative_params_removed
        self.cumulative_flap_ratio = cumulative_flap_ratio
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
        self.failed_block_indices: set = set()
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
        existing = self.tracker.get_baseline()
        if existing:
            self.baseline_accuracy = existing["accuracy"]
            self.baseline_loss = existing.get("screening_loss", 0.0)
            self.baseline_params = existing.get("params_after", self.baseline_params)
            return existing

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
                if 0 <= b["layer_idx"] < num_layers and b["layer_idx"] not in self.failed_block_indices
            ]
            if valid_block_rankings:
                block_idx = valid_block_rankings[0]["layer_idx"]
                block_param_reduction = sum(p.numel() for p in layers[block_idx].parameters())

        # Propose FLAP candidate
        plan: Optional[AllocationPlan] = None
        flap_param_reduction = 0
        actual_flap_step = min(
            self.current_flap_ratio,
            max(0.0, self.config.pruning.maximum_total_flap_ratio - self.total_flap_ratio_applied),
        )
        target_flap_ratio = self.total_flap_ratio_applied + actual_flap_step

        if (actual_flap_step >= (self.config.pruning.minimum_flap_ratio - 1e-6) and
            target_flap_ratio <= (self.config.pruning.maximum_total_flap_ratio + 1e-6) and
            self.cached_flap_scores):

            std_scores = ScoreStandardizer.standardize_all(
                self.cached_flap_scores, method=self.config.flap.standardization,
            )
            if self.config.flap.allocation == "global_adaptive":
                plan = PruningAllocator.allocate_global_adaptive(std_scores, target_flap_ratio)
            else:
                plan = PruningAllocator.allocate_uniform(std_scores, target_flap_ratio)

            # Estimate param reduction for FLAP: 3 projections * hidden_size * channels_pruned_in_this_step
            hidden_size = self.model.config.hidden_size
            channels_to_prune_now = 0
            for l_idx, keep_idx in plan.keep_indices.items():
                if self.active_channels and l_idx in self.active_channels:
                    curr_len = len(self.active_channels[l_idx])
                    target_len = len([ch for ch in keep_idx if ch in self.active_channels[l_idx]])
                    channels_to_prune_now += max(0, curr_len - target_len)
                else:
                    channels_to_prune_now += len(plan.prune_indices.get(l_idx, []))

            if channels_to_prune_now > 0:
                flap_param_reduction = channels_to_prune_now * hidden_size * 3
            else:
                plan = None
                actual_flap_step = 0.0

        candidates: Dict[str, JointCandidate] = {}

        # Current cumulative stats before this step
        current_model_params = sum(p.numel() for p in self.model.parameters())
        current_params_removed = max(0, self.baseline_params - current_model_params)
        current_flap_applied = self.total_flap_ratio_applied

        # 1. Joint candidate
        if block_idx is not None and plan is not None:
            step_params = block_param_reduction + flap_param_reduction
            candidates["joint"] = JointCandidate(
                candidate_type="joint",
                block_idx=block_idx,
                plan=plan,
                estimated_params_removed=step_params,
                predicted_loss=0.05,
                flap_ratio=actual_flap_step,
                cumulative_params_removed=current_params_removed + step_params,
                cumulative_flap_ratio=current_flap_applied + actual_flap_step,
            )

        # 2. Block-only candidate
        if block_idx is not None:
            step_params = block_param_reduction
            candidates["block_only"] = JointCandidate(
                candidate_type="block_only",
                block_idx=block_idx,
                plan=None,
                estimated_params_removed=step_params,
                predicted_loss=0.03,
                flap_ratio=0.0,
                cumulative_params_removed=current_params_removed + step_params,
                cumulative_flap_ratio=current_flap_applied,
            )

        # 3. FLAP-only candidate
        if plan is not None:
            step_params = flap_param_reduction
            candidates["flap_only"] = JointCandidate(
                candidate_type="flap_only",
                block_idx=None,
                plan=plan,
                estimated_params_removed=step_params,
                predicted_loss=0.01,
                flap_ratio=actual_flap_step,
                cumulative_params_removed=current_params_removed + step_params,
                cumulative_flap_ratio=current_flap_applied + actual_flap_step,
            )

        return candidates

    def optimize(self, max_iterations: Optional[int] = None, resume: bool = True) -> Dict[str, Any]:
        """
        Executes the Joint Incremental Pruning Optimization search.
        """
        iters_limit = max_iterations or self.config.pruning.max_iterations
        start_time = time.time()
        dev = self.config.model.device
        stopping_reason = "max_iterations_reached"

        iteration = 0
        parent_id = "baseline"

        if resume:
            latest = self.tracker.get_latest_accepted()
            if latest and latest.get("checkpoint_path") and latest.get("checkpoint_path") != "unpruned_baseline":
                ckpt_path = latest["checkpoint_path"]
                iteration = latest["iteration"]
                parent_id = latest["trial_id"]

                console.print(
                    f"\n[bold green]↺ Found existing accepted checkpoint:[/bold green] [cyan]{ckpt_path}[/cyan]\n"
                    f"   [bold green]Resuming sweep from Iteration {iteration + 1}...[/bold green]"
                )

                # Ensure Level 1 baseline profiles are ready
                self._refresh_profiles()

                # Verify checkpoint integrity before resumption
                ok, msg = CheckpointValidator.validate(ckpt_path)
                if not ok:
                    raise ValueError(f"Cannot resume sweep: checkpoint '{ckpt_path}' failed integrity validation: {msg}")

                # Determine resumed FLAP ratio and remaining layers
                import json
                config_json_path = os.path.join(ckpt_path, "config.json")
                if os.path.exists(config_json_path):
                    with open(config_json_path, "r", encoding="utf-8") as f:
                        ckpt_cfg = json.load(f)
                else:
                    ckpt_cfg = {}

                orig_inter = getattr(self.model.config, "intermediate_size", getattr(self.model.config, "hidden_size", 0))
                curr_inter = ckpt_cfg.get("intermediate_size", orig_inter)
                resumed_flap_ratio = round(max(0.0, 1.0 - (curr_inter / orig_inter)), 4) if orig_inter > 0 else 0.0

                orig_layers = getattr(self.model.config, "num_hidden_layers", getattr(self.model.config, "n_layer", 0))
                curr_layers = ckpt_cfg.get("num_hidden_layers", orig_layers)
                self.total_blocks_removed = max(0, orig_layers - curr_layers)
                self.total_flap_ratio_applied = resumed_flap_ratio

                # Reconstruct active channels up to resumed FLAP ratio
                if resumed_flap_ratio > 0 and self.cached_flap_scores:
                    std_scores = ScoreStandardizer.standardize_all(
                        self.cached_flap_scores, method=self.config.flap.standardization,
                    )
                    if self.config.flap.allocation == "global_adaptive":
                        plan = PruningAllocator.allocate_global_adaptive(std_scores, resumed_flap_ratio)
                    else:
                        plan = PruningAllocator.allocate_uniform(std_scores, resumed_flap_ratio)
                    self.active_channels = {l_idx: list(plan.keep_indices[l_idx]) for l_idx in plan.keep_indices}

                # Check previous block pruning failures from history
                failed_blocks = [
                    r for r in self.tracker.records
                    if r.get("status") != "accepted" and r.get("action_type") in ("block", "joint")
                ]
                if failed_blocks:
                    self.block_pruning_exhausted = True
                    self.consecutive_block_failures = len(failed_blocks)

                # Load checkpoint weights onto device
                del self.model
                clean_gpu_memory(dev)
                dtype_val = getattr(torch, self.config.model.torch_dtype) if isinstance(self.config.model.torch_dtype, str) and hasattr(torch, self.config.model.torch_dtype) else torch.bfloat16
                self.model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path,
                    torch_dtype=dtype_val,
                    device_map=dev if torch.cuda.is_available() else None,
                    trust_remote_code=False,
                )
                self.model.eval()
                if self.adapter is not None:
                    self.adapter.model = self.model

                console.print(
                    f"   [green]✔ State Restored:[/green] Iteration: {iteration} | "
                    f"Active Layers: {len(JointStateManager.get_layers(self.model))} | "
                    f"Cumulative FLAP: {self.total_flap_ratio_applied * 100:.2f}% | "
                    f"Block Pruning Exhausted: {self.block_pruning_exhausted}\n"
                )

        while iteration < iters_limit:
            iteration += 1

            if self.config.pruning.time_budget_sec and (time.time() - start_time) > self.config.pruning.time_budget_sec:
                stopping_reason = "time_budget_exhausted"
                console.print("[yellow]Time budget exhausted. Stopping search.[/yellow]")
                break

            current_model_params = sum(p.numel() for p in self.model.parameters())
            current_removed = max(0, self.baseline_params - current_model_params)
            current_pct = (current_removed / self.baseline_params) * 100.0

            console.print()
            console.print(Panel(
                f"[bold cyan]Iteration {iteration}/{iters_limit}[/bold cyan] | "
                f"Active Layers: [yellow]{len(JointStateManager.get_layers(self.model))}[/yellow] | "
                f"Cumulative Pruned: [yellow]{self.total_flap_ratio_applied * 100:.2f}% FLAP[/yellow] "
                f"([yellow]-{current_removed / 1e6:.2f}M params, -{current_pct:.2f}%[/yellow]) | "
                f"Target Retention: [magenta]{self.config.pruning.retention * 100:.1f}%[/magenta] "
                f"([green]{(self.baseline_accuracy * self.config.pruning.retention * 100):.2f}% min accuracy[/green])",
                title="[bold green]JIPO Search Loop[/bold green]",
                expand=False,
            ))

            candidates = self._propose_candidates()
            if not candidates:
                stopping_reason = "no_eligible_candidates"
                console.print("[yellow]No eligible candidates remaining to propose. Stopping search.[/yellow]")
                break

            parent_snapshot = JointStateManager.backup_state(self.model, active_channels=self.active_channels)
            iteration_committed = False

            # ================= LEVEL 2: CHEAP JOINT SCREENING =================
            joint_cand = candidates.get("joint")
            joint_passed_screen = False

            if joint_cand is not None:
                b_desc = f"Layer #{joint_cand.block_idx} excision"
                f_desc = f"+{joint_cand.flap_ratio * 100:.2f}% MLP width slice"
                step_pct = (joint_cand.estimated_params_removed / self.baseline_params) * 100.0
                cum_pct = (joint_cand.cumulative_params_removed / self.baseline_params) * 100.0
                console.print(
                    f"\n[bold yellow]── Testing Candidate 1 (Joint):[/bold yellow] {b_desc} + {f_desc}\n"
                    f"   [dim]Step Reduction:[/dim] -{joint_cand.estimated_params_removed / 1e6:.2f}M params (-{step_pct:.2f}%) | "
                    f"[dim]Cumulative Target:[/dim] -{joint_cand.cumulative_params_removed / 1e6:.2f}M params (-{cum_pct:.2f}% of model, {joint_cand.cumulative_flap_ratio * 100:.2f}% FLAP)"
                )

                # Apply joint candidate in-memory
                JointStateManager.apply_joint_candidate(
                    self.model, joint_cand.block_idx, joint_cand.plan, active_channels=self.active_channels
                )

                # Tier 1: Teacher-forced loss screening
                console.print(f"   [dim]▶ Level 2: Screening loss on {len(self.screening_samples)} samples...[/dim]")
                screen_res = ProxyScreeningEvaluator.evaluate_loss(
                    self.model, self.tokenizer, self.screening_samples, device=dev,
                    max_length=self.config.calibration.max_length,
                )
                loss_thresh = (self.baseline_loss * self.config.pruning.screening_loss_factor) + self.config.pruning.screening_loss_margin
                joint_cand.predicted_loss = max(screen_res.loss - self.baseline_loss, 0.0)

                if screen_res.loss <= loss_thresh:
                    joint_passed_screen = True
                    console.print(f"   [green]✔ Passed Screening:[/green] Loss {screen_res.loss:.4f} <= threshold {loss_thresh:.4f}")
                else:
                    console.print(f"   [red]✘ Rejected by Screening:[/red] Loss {screen_res.loss:.4f} > threshold {loss_thresh:.4f} (saving expensive GSM8K evaluation)")
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
                    if joint_cand.block_idx is not None:
                        self.failed_block_indices.add(joint_cand.block_idx)

            # ================= LEVEL 3: VALIDATION =================
            if joint_passed_screen and joint_cand is not None:
                console.print(f"   [bold cyan]▶ Level 3: Full GSM8K benchmark validation ({len(self.acceptance_samples)} samples)...[/bold cyan]")
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
                    self.tracker.record_joint_evaluation_savings()
                    self.total_blocks_removed += 1
                    self.total_flap_ratio_applied += joint_cand.flap_ratio
                    self.consecutive_block_failures = 0
                    self.profiles_valid = False  # Invalidate cached profile due to block deletion

                    curr_params = sum(p.numel() for p in self.model.parameters())
                    total_removed = self.baseline_params - curr_params
                    total_pct = (total_removed / self.baseline_params) * 100.0

                    console.print(
                        f"   [bold green]✔ ACCEPTED Joint Candidate![/bold green] "
                        f"Accuracy: [green]{eval_res.accuracy * 100:.2f}%[/green] "
                        f"(Retention: [green]{retention_pct:.2f}%[/green] >= {self.config.pruning.retention * 100:.1f}%)\n"
                        f"      [green]Cumulative Pruned:[/green] -{total_removed / 1e6:.2f}M params (-{total_pct:.2f}% of baseline) | "
                        f"[green]Layers Remaining:[/green] {len(JointStateManager.get_layers(self.model))} | "
                        f"[green]Cumulative FLAP:[/green] {self.total_flap_ratio_applied * 100:.2f}%"
                    )

                    ckpt_name = f"joint_iter_{iteration}_layers_{len(JointStateManager.get_layers(self.model))}"
                    ckpt_path = CheckpointExporter.export(
                        self.model, self.tokenizer, self.run_dir, ckpt_name,
                        metadata={"iteration": iteration, "action": "joint", "retention": retention_pct},
                    )
                    console.print(f"   [green]Committed checkpoint:[/green] [cyan]{ckpt_path}[/cyan]")

                    self.tracker.log_trial({
                        "trial_id": f"iter_{iteration}_joint",
                        "iteration": iteration,
                        "action_type": "joint",
                        "parent_id": parent_id,
                        "params_before": sum(p.numel() for p in parent_snapshot["layers_list"][0].parameters()) * len(parent_snapshot["layers_list"]),
                        "params_after": curr_params,
                        "params_removed": total_removed,
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
                    console.print(
                        f"   [bold red]✘ REJECTED Joint Candidate:[/bold red] "
                        f"Accuracy: [red]{eval_res.accuracy * 100:.2f}%[/red] "
                        f"(Retention: [red]{retention_pct:.2f}%[/red] < Target: {self.config.pruning.retention * 100:.1f}%)"
                    )
                    console.print("   [cyan]↺ Transactional Rollback:[/cyan] Restored model in-memory (0 layers removed, 0 channels sliced).")
                    # Joint candidate failed accuracy retention -> Rollback to parent snapshot!
                    JointStateManager.restore_state(self.model, parent_snapshot, active_channels_ref=self.active_channels)
                    if joint_cand.block_idx is not None:
                        self.failed_block_indices.add(joint_cand.block_idx)
                        self.consecutive_block_failures += 1
                        self.block_pruning_exhausted = True
                        console.print("   [bold red]Block Pruning Locked:[/bold red] Layer removal exceeded retention threshold. Locking block pruning (0 layers removed).")

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
                if "flap_only" in candidates:
                    alternatives.append(candidates["flap_only"])
                if "block_only" in candidates and not self.block_pruning_exhausted:
                    alternatives.append(candidates["block_only"])

                if joint_cand is not None:
                    console.print("   [bold yellow]▶ Evaluating Fallback Alternatives:[/bold yellow]")
                else:
                    console.print("   [bold yellow]▶ Evaluating Proposed Candidates:[/bold yellow]")

                # Prioritize FLAP-only over block-only when joint candidate failed
                block_penalty = 0.05 if (self.consecutive_block_failures >= 1 or joint_cand is not None) else 1.0
                for alt in alternatives:
                    alt.compute_utility(block_penalty=block_penalty)

                alternatives.sort(key=lambda x: x.utility_score, reverse=True)

                for alt_idx, alt in enumerate(alternatives, start=1):
                    alt_name = "FLAP-Only" if alt.candidate_type == "flap_only" else "Block-Only"
                    alt_desc = (
                        f"+{alt.flap_ratio * 100:.2f}% MLP width slice"
                        if alt.candidate_type == "flap_only"
                        else f"Layer #{alt.block_idx} excision"
                    )
                    step_pct = (alt.estimated_params_removed / self.baseline_params) * 100.0
                    cum_pct = (alt.cumulative_params_removed / self.baseline_params) * 100.0
                    cand_header = (
                        f"Fallback Alternative {alt_idx}/{len(alternatives)}"
                        if joint_cand is not None
                        else f"Candidate {alt_idx}/{len(alternatives)}"
                    )
                    console.print(
                        f"\n   [bold yellow]── Testing {cand_header} ({alt_name}):[/bold yellow] {alt_desc}\n"
                        f"      [dim]Step Reduction:[/dim] -{alt.estimated_params_removed / 1e6:.2f}M params (-{step_pct:.2f}%) | "
                        f"[dim]Cumulative Target:[/dim] -{alt.cumulative_params_removed / 1e6:.2f}M params (-{cum_pct:.2f}% of model, {alt.cumulative_flap_ratio * 100:.2f}% FLAP)"
                    )

                    JointStateManager.apply_joint_candidate(
                        self.model, alt.block_idx, alt.plan, active_channels=self.active_channels
                    )

                    # Tier 1: Screen alternative
                    console.print(f"      [dim]▶ Running cheap loss screening on {len(self.screening_samples)} samples...[/dim]")
                    alt_screen = ProxyScreeningEvaluator.evaluate_loss(
                        self.model, self.tokenizer, self.screening_samples, device=dev,
                        max_length=self.config.calibration.max_length,
                    )
                    loss_thresh = (self.baseline_loss * self.config.pruning.screening_loss_factor) + self.config.pruning.screening_loss_margin

                    if alt_screen.loss > loss_thresh:
                        console.print(f"      [red]✘ Rejected by Screening:[/red] Loss {alt_screen.loss:.4f} > threshold {loss_thresh:.4f}")
                        self.tracker.record_screening_rejection_savings()
                        JointStateManager.restore_state(self.model, parent_snapshot, active_channels_ref=self.active_channels)
                        if alt.candidate_type == "block_only":
                            self.consecutive_block_failures += 1
                            if alt.block_idx is not None:
                                self.failed_block_indices.add(alt.block_idx)
                            self.block_pruning_exhausted = True
                            console.print("      [bold red]Block Pruning Locked:[/bold red] Layer removal exceeded screening threshold.")
                        continue

                    console.print(f"      [green]✔ Passed Screening:[/green] Loss {alt_screen.loss:.4f} <= threshold {loss_thresh:.4f}")
                    console.print(f"      [bold cyan]▶ Launching full GSM8K evaluation ({len(self.acceptance_samples)} samples)...[/bold cyan]")
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
                        curr_params = sum(p.numel() for p in self.model.parameters())
                        if alt.candidate_type == "block_only":
                            self.total_blocks_removed += 1
                            self.consecutive_block_failures = 0
                            self.profiles_valid = False  # Invalidate profiles
                        elif alt.candidate_type == "flap_only":
                            self.total_flap_ratio_applied += alt.flap_ratio

                        total_removed = self.baseline_params - curr_params
                        total_pct = (total_removed / self.baseline_params) * 100.0
                        console.print(
                            f"      [bold green]✔ ACCEPTED {alt_name} Candidate![/bold green] "
                            f"Accuracy: [green]{alt_eval.accuracy * 100:.2f}%[/green] "
                            f"(Retention: [green]{retention_pct:.2f}%[/green] >= {self.config.pruning.retention * 100:.1f}%)\n"
                            f"      [green]Cumulative Pruned:[/green] -{total_removed / 1e6:.2f}M params (-{total_pct:.2f}% of baseline) | "
                            f"[green]Cumulative FLAP:[/green] {self.total_flap_ratio_applied * 100:.2f}%"
                        )

                        ckpt_name = f"{alt.candidate_type}_iter_{iteration}"
                        ckpt_path = CheckpointExporter.export(
                            self.model, self.tokenizer, self.run_dir, ckpt_name,
                            metadata={"iteration": iteration, "action": alt.candidate_type, "retention": retention_pct},
                        )
                        console.print(f"      [green]Committed checkpoint:[/green] [cyan]{ckpt_path}[/cyan]")

                        self.tracker.log_trial({
                            "trial_id": f"iter_{iteration}_{alt.candidate_type}",
                            "iteration": iteration,
                            "action_type": alt.candidate_type,
                            "parent_id": parent_id,
                            "params_before": sum(p.numel() for p in parent_snapshot["layers_list"][0].parameters()) * len(parent_snapshot["layers_list"]),
                            "params_after": curr_params,
                            "params_removed": total_removed,
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
                        console.print(
                            f"      [bold red]✘ REJECTED {alt_name} Candidate:[/bold red] "
                            f"Accuracy: [red]{alt_eval.accuracy * 100:.2f}%[/red] "
                            f"(Retention: [red]{retention_pct:.2f}%[/red] < Target: {self.config.pruning.retention * 100:.1f}%)"
                        )
                        console.print("      [cyan]↺ Transactional Rollback:[/cyan] Restored model in-memory.")
                        # Rejection of alternative
                        JointStateManager.restore_state(self.model, parent_snapshot, active_channels_ref=self.active_channels)
                        if alt.candidate_type == "block_only":
                            self.consecutive_block_failures += 1
                            if alt.block_idx is not None:
                                self.failed_block_indices.add(alt.block_idx)
                            self.block_pruning_exhausted = True
                            console.print("      [bold red]Block Pruning Locked:[/bold red] Layer removal exceeded retention threshold.")

                # If all alternatives failed in this step -> Adaptive step reduction
                if not iteration_committed:
                    old_step = self.current_flap_ratio * 100
                    self.current_flap_ratio *= self.config.pruning.backoff_factor
                    new_step = self.current_flap_ratio * 100
                    console.print(
                        f"\n   [bold magenta]Adaptive Step Halving:[/bold magenta] "
                        f"All candidates at {old_step:.2f}% exceeded threshold. Halving FLAP step to [yellow]{new_step:.2f}%[/yellow] for next iteration."
                    )
                    if self.consecutive_block_failures >= 1:
                        self.block_pruning_exhausted = True
                        console.print("   [bold red]Block Pruning Locked:[/bold red] Layer removal exceeded threshold. No further layers will be removed.")

                    if self.current_flap_ratio < self.config.pruning.minimum_flap_ratio and self.block_pruning_exhausted:
                        stopping_reason = "retention_boundary_reached"
                        console.print("[yellow]Reached minimum FLAP ratio and block pruning exhausted. Stopping search.[/yellow]")
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
            val_ok, val_msg = CheckpointValidator.validate(best_ckpt["checkpoint_path"])
            summary["best_checkpoint_reload_verified"] = val_ok
            summary["reload_message"] = val_msg

        return summary
