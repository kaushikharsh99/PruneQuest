"""
Unit tests for SearchController: baseline restoration, reload failure rejection, and retention enforcement.
"""

import tempfile
from unittest.mock import MagicMock, patch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from prunequest.config import PruneQuestConfig
from prunequest.models.qwen2 import Qwen2Adapter
from prunequest.benchmarks.base import EvaluationResult
from prunequest.core.controller import SearchController
from prunequest.core.tracking import ExperimentTracker


def create_synthetic_adapter():
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = Qwen2ForCausalLM(cfg)
    adapter = Qwen2Adapter("synthetic", device="cpu")
    adapter.model = model
    adapter.tokenizer = MagicMock()
    return adapter


def test_controller_baseline_preservation_on_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PruneQuestConfig(output_dir=tmpdir)
        config.pruning.max_iterations = 1
        tracker = ExperimentTracker(tmpdir)

        # Pre-seed tracker with baseline and one accepted checkpoint
        tracker.log_iteration({
            "iteration_id": "baseline",
            "params_after": 1500000000,
            "accuracy": 0.95,
            "status": "baseline",
            "checkpoint_path": "original_model",
        })
        tracker.log_iteration({
            "iteration_id": "iter_1_layer_2",
            "params_after": 1400000000,
            "accuracy": 0.94,
            "status": "accepted",
            "checkpoint_path": "checkpoint_layers_3",
        })

        controller = SearchController(config=config, tracker=tracker)
        adapter = create_synthetic_adapter()
        controller.adapter = adapter

        dummy_eval = EvaluationResult(
            total=16,
            correct=15,
            accuracy=0.9375,
            records=[],
        )
        controller.benchmark.evaluate = MagicMock(return_value=dummy_eval)

        # Resume without running baseline
        with patch.object(controller, "run_baseline") as mock_baseline, \
             patch.object(controller.tracker, "get_latest_accepted", return_value={"checkpoint_path": "dummy"}), \
             patch.object(controller.exporter, "export", return_value=tmpdir), \
             patch("prunequest.export.validation.SubprocessValidator.validate_checkpoint", return_value=(True, "ok")):
            with patch("os.path.exists", return_value=False):
                controller.run()
                # Ensure baseline was NOT re-evaluated, but restored from SQLite
                mock_baseline.assert_not_called()
                assert controller.baseline_accuracy == 0.95
                assert controller.baseline_params == 1500000000


def test_controller_rejects_on_reload_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PruneQuestConfig(output_dir=tmpdir)
        config.pruning.max_iterations = 1
        config.benchmark.split = "smoke"
        tracker = ExperimentTracker(tmpdir)

        controller = SearchController(config=config, tracker=tracker)
        adapter = create_synthetic_adapter()
        controller.adapter = adapter

        controller.baseline_accuracy = 1.0
        controller.baseline_params = adapter.count_parameters()

        dummy_eval = EvaluationResult(
            total=16,
            correct=16,
            accuracy=1.0,
            records=[],
        )
        controller.benchmark.evaluate = MagicMock(return_value=dummy_eval)

        # Mock subprocess reload validator to FAIL
        with patch("prunequest.export.validation.SubprocessValidator.validate_checkpoint", return_value=(False, "Corrupted tensor")):
            with patch.object(controller.exporter, "export", return_value=tmpdir):
                result = controller.run()

                assert "checkpoint_validation_failed" in result["status"]
                # Verify that rejection was recorded in SQLite
                df = tracker.get_history_dataframe()
                rejected = df[df["status"] == "rejected_reload_failed"]
                assert len(rejected) == 1
                assert "Corrupted tensor" in rejected.iloc[0]["rejection_reason"]


def test_controller_rejects_on_retention_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = PruneQuestConfig(output_dir=tmpdir)
        config.pruning.max_iterations = 1
        config.pruning.retain = 0.99
        config.benchmark.split = "dev"
        tracker = ExperimentTracker(tmpdir)

        controller = SearchController(config=config, tracker=tracker)
        adapter = create_synthetic_adapter()
        controller.adapter = adapter

        controller.baseline_accuracy = 1.0
        controller.baseline_params = adapter.count_parameters()

        # Screening passes, but Dev evaluation drops to 0.70 (below 0.99 threshold)
        screen_eval = EvaluationResult(
            total=16,
            correct=16,
            accuracy=1.0,
            records=[],
        )
        dev_eval = EvaluationResult(
            total=512,
            correct=358,
            accuracy=0.70,
            records=[],
        )

        def eval_side_effect(*args, **kwargs):
            if kwargs.get("split_name") in ("screening", "smoke"):
                return screen_eval
            return dev_eval

        controller.benchmark.evaluate = MagicMock(side_effect=eval_side_effect)

        result = controller.run()
        assert "retention" in result["status"]

        df = tracker.get_history_dataframe()
        rejected = df[df["status"] == "rejected_retention_failed"]
        assert len(rejected) == 1
        assert "Rejected layer" in rejected.iloc[0]["operation_description"]
        # Model should have rolled back to 4 layers
        assert len(controller.adapter.get_layers()) == 4
