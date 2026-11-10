from unittest.mock import MagicMock, patch
import torch
from prunequest.flap.config import FLAPConfig, PruningConfig, ModelConfig
from prunequest.flap.core.optimizer import FLAPOptimizer
from prunequest.flap.evaluation.base import EvaluationResult


def test_optimizer_retention_rejection(tmp_path):
    out_dir = str(tmp_path / "test_run")
    cfg = FLAPConfig(
        model=ModelConfig(model_id="test_model", device="cpu"),
        pruning=PruningConfig(initial_step_ratio=0.05, minimum_step_ratio=0.02, retention=0.99, max_candidate_evaluations=2),
        output_dir=out_dir,
    )

    optimizer = FLAPOptimizer(cfg)

    # Mock adapter and model
    mock_adapter = MagicMock()
    mock_adapter.count_parameters.return_value = 1000
    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_adapter.load.return_value = (mock_model, mock_tok)

    # Mock dataset
    optimizer.adapter = mock_adapter
    optimizer.model = mock_model
    optimizer.tokenizer = mock_tok
    optimizer.splits = {
        "calibration": [],
        "screening": [MagicMock()],
        "acceptance": [MagicMock()],
    }
    optimizer.baseline_accuracy = 1.0
    optimizer.baseline_loss = 1.0
    optimizer.baseline_params = 1000
    optimizer.importance_scores = {0: torch.ones(10)}

    with patch("prunequest.flap.models.registry.ModelRegistry.create_adapter", return_value=mock_adapter):
        with patch.object(optimizer.dataset_manager, "load_and_partition", return_value=optimizer.splits):
            with patch.object(optimizer, "run_profiling", return_value=optimizer.importance_scores):
                with patch("prunequest.flap.pruning.slicer.MLPSlicer.apply_plan"):
                    with patch("prunequest.flap.pruning.manager.PruningStateManager.backup_mlp_weights"):
                        with patch("prunequest.flap.pruning.manager.PruningStateManager.restore_mlp_weights") as mock_restore:
                            with patch("prunequest.flap.evaluation.screening.ProxyScreeningEvaluator.evaluate_loss") as mock_screen:
                                with patch("prunequest.flap.evaluation.gsm8k.GSM8KEvaluator.evaluate") as mock_eval:
                                    # Screening passes
                                    mock_screen.return_value = EvaluationResult(total=1, correct=0, accuracy=0.0, loss=1.0, perplexity=2.7)
                                    # Baseline: 1.0; Candidate evaluation: 0.80 (below 0.99 retention)
                                    mock_eval.side_effect = [
                                        EvaluationResult(total=10, correct=10, accuracy=1.0),
                                        EvaluationResult(total=10, correct=8, accuracy=0.80),
                                        EvaluationResult(total=10, correct=8, accuracy=0.80),
                                    ]

                                    summary = optimizer.optimize()

                                    assert mock_restore.called
                                    assert "minimum_step" in summary["stopping_reason"] or summary["stopping_reason"] == "max_evaluations_reached"
                                    # Verify rejection was logged
                                    records = optimizer.tracker.get_all_records()
                                    rejections = [r for r in records if r["status"] == "rejected_retention"]
                                    assert len(rejections) > 0
