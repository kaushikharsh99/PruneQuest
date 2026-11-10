from unittest.mock import MagicMock
import torch
from prunequest.flap.calibration.dataset import DatasetSample
from prunequest.flap.evaluation.screening import ProxyScreeningEvaluator


def test_proxy_screening_evaluator():
    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_out.loss.item.return_value = 1.5
    mock_model.return_value = mock_out

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102, 103]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    samples = [
        DatasetSample(id="s1", question="What is 2+2?", gold_answer="4"),
        DatasetSample(id="s2", question="What is 3*5?", gold_answer="15"),
    ]

    res = ProxyScreeningEvaluator.evaluate_loss(mock_model, mock_tokenizer, samples, device="cpu")
    assert res.total == 2
    assert res.loss == 1.5
    assert res.perplexity > 0.0
