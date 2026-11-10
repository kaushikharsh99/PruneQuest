"""
Unit tests for Joint Incremental Pruning Optimizer (JIPO) and Full Sweep Engine.
"""

from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from prunequest.sweep.config import JointSweepConfig, JointPruningConfig
from prunequest.sweep.state import JointStateManager
from prunequest.sweep.optimizer import JointIncrementalOptimizer, JointCandidate
from prunequest.sweep.tracker import JointExperimentTracker
from prunequest.flap.allocation import AllocationPlan


class DummyMLP(nn.Module):
    def __init__(self, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden_size=64, intermediate_size=128, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = nn.Module()
        self.self_attn.layer_idx = layer_idx
        self.mlp = DummyMLP(hidden_size, intermediate_size)


class DummyModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, num_layers=4, hidden_size=64, intermediate_size=128):
        config = PretrainedConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            layer_types=["dense"] * num_layers,
        )
        super().__init__(config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            DummyDecoderLayer(hidden_size, intermediate_size, i)
            for i in range(num_layers)
        ])


def test_joint_state_manager_backup_restore():
    model = DummyModel(num_layers=4, hidden_size=32, intermediate_size=64)
    init_params = sum(p.numel() for p in model.parameters())
    assert len(model.model.layers) == 4

    active_channels = {i: list(range(64)) for i in range(4)}
    backup = JointStateManager.backup_state(model, active_channels=active_channels)

    # 1. Apply Block removal
    removed_params = JointStateManager.remove_block(model, 1)
    assert len(model.model.layers) == 3
    assert model.config.num_hidden_layers == 3
    assert removed_params > 0

    # 2. Apply FLAP slicing on remaining layers
    keep_indices = list(range(32))  # Prune half the channels
    plan = AllocationPlan(
        prune_indices={i: list(range(32, 64)) for i in range(3)},
        keep_indices={i: keep_indices for i in range(3)},
        channels_removed_per_layer={i: 32 for i in range(3)},
        total_channels_removed=32 * 3,
        target_ratio=0.5,
    )
    sliced_params = JointStateManager.slice_mlp_channels(model, plan)
    assert sliced_params > 0
    assert model.model.layers[0].mlp.gate_proj.out_features == 32

    # 3. Restore state
    JointStateManager.restore_state(model, backup, active_channels_ref=active_channels)
    assert len(model.model.layers) == 4
    assert model.config.num_hidden_layers == 4
    assert model.model.layers[0].mlp.gate_proj.out_features == 64
    assert sum(p.numel() for p in model.parameters()) == init_params
    assert [layer.layer_idx for layer in model.model.layers] == [0, 1, 2, 3]


def test_joint_candidate_utility():
    cand = JointCandidate(
        candidate_type="joint",
        block_idx=1,
        plan=None,
        estimated_params_removed=10_000_000,
        predicted_loss=0.02,
    )
    score = cand.compute_utility()
    assert score > 0
    # Bytes saved = 10,000,000 * 2 = 20,000,000; score approx 20,000,000 / 0.0201
    assert score > 900_000_000


def test_tracker_pareto_and_savings(tmp_path):
    tracker = JointExperimentTracker(str(tmp_path))

    # Log trials
    tracker.log_trial({
        "trial_id": "trial_1",
        "params_after": 1_400_000_000,
        "accuracy": 0.85,
        "status": "accepted",
        "checkpoint_path": "ckpt_1",
    })
    tracker.record_joint_evaluation_savings()
    tracker.record_screening_rejection_savings()
    tracker.record_full_evaluation()

    # Second trial with fewer params and higher accuracy (dominates trial_1)
    tracker.log_trial({
        "trial_id": "trial_2",
        "params_after": 1_350_000_000,
        "accuracy": 0.86,
        "status": "accepted",
        "checkpoint_path": "ckpt_2",
    })

    best = tracker.get_best_pareto_checkpoint()
    assert best["trial_id"] == "trial_2"

    metrics = tracker.get_summary_metrics()
    assert metrics["evaluations_saved_by_joint"] == 1
    assert metrics["evaluations_saved_by_screening"] == 1
    assert metrics["total_evaluations_saved"] == 2
    assert metrics["full_evaluations_executed"] == 1


def test_joint_optimizer_workflow(tmp_path):
    cfg = JointSweepConfig(
        output_dir=str(tmp_path / "sweep_run"),
        pruning=JointPruningConfig(
            max_iterations=1,
            initial_flap_ratio=0.01,
            retention=0.99,
            min_remaining_layers=2,
        ),
    )

    optimizer = JointIncrementalOptimizer(cfg)
    optimizer.model = DummyModel(num_layers=4, hidden_size=32, intermediate_size=64)
    optimizer.tokenizer = MagicMock()
    optimizer.screening_samples = [MagicMock()]
    optimizer.acceptance_samples = [MagicMock()]
    optimizer.calibration_samples = [MagicMock()]
    optimizer.baseline_accuracy = 0.80
    optimizer.baseline_loss = 1.0
    optimizer.baseline_params = sum(p.numel() for p in optimizer.model.parameters())
    optimizer.active_channels = {i: list(range(64)) for i in range(4)}

    # Mock cached profiles
    optimizer.cached_block_scores = [{"layer_idx": 1, "score": 0.05}]
    optimizer.cached_flap_scores = {i: torch.ones(64) for i in range(4)}
    optimizer.profiles_valid = True

    # Mock screening to pass and GSM8K to pass (Joint Accepted)
    with patch("prunequest.sweep.optimizer.ProxyScreeningEvaluator.evaluate_loss") as mock_screen:
        mock_screen.return_value = type("ScreenRes", (), {"loss": 1.05})()
        with patch("prunequest.sweep.optimizer.GSM8KEvaluator.evaluate") as mock_eval:
            mock_eval.return_value = type("EvalRes", (), {"accuracy": 0.80, "total_time_sec": 1.0})()
            with patch("prunequest.sweep.optimizer.CheckpointExporter.export"):
                with patch("prunequest.sweep.optimizer.CheckpointValidator.validate", return_value=(True, "OK")):
                    summary = optimizer.optimize(max_iterations=1)

    assert summary["status"] == "completed"
    assert summary["total_blocks_removed"] == 1
    assert summary["metrics"]["evaluations_saved_by_joint"] == 1
