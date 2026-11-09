import torch
from prunequest.stage1.calibration.collector import LayerActivationStats
from prunequest.stage1.flap.metrics import FLAPMetricCalculator
from prunequest.stage1.flap.standardization import ScoreStandardizer
from prunequest.stage1.flap.allocation import PruningAllocator


def test_flap_metric_wifv():
    stats = LayerActivationStats(
        layer_idx=0,
        intermediate_size=4,
        token_count=100,
        mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        variance=torch.tensor([0.1, 0.5, 0.2, 0.8]),
        mean_abs=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        down_proj_weight_norm2=torch.tensor([2.0, 1.0, 3.0, 0.5]),
    )

    scores = FLAPMetricCalculator.compute_layer_scores(stats, metric="wifv")
    # WIFV: norm2 * variance
    expected = torch.tensor([0.2, 0.5, 0.6, 0.4])
    assert torch.allclose(scores, expected, atol=1e-5)


def test_flap_metric_wifn():
    stats = LayerActivationStats(
        layer_idx=0,
        intermediate_size=4,
        token_count=100,
        mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        variance=torch.tensor([0.1, 0.5, 0.2, 0.8]),
        mean_abs=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        down_proj_weight_norm2=torch.tensor([2.0, 1.0, 3.0, 0.5]),
    )

    scores = FLAPMetricCalculator.compute_layer_scores(stats, metric="wifn")
    # WIFN: norm2 * mean_abs
    expected = torch.tensor([2.0, 2.0, 9.0, 2.0])
    assert torch.allclose(scores, expected, atol=1e-5)


def test_score_standardization():
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    z_scores = ScoreStandardizer.standardize_layer_scores(scores, method="z_score")
    assert torch.isclose(z_scores.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(z_scores.std(unbiased=False), torch.tensor(1.0), atol=1e-4)

    min_max_scores = ScoreStandardizer.standardize_layer_scores(scores, method="min_max")
    assert torch.isclose(min_max_scores.min(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(min_max_scores.max(), torch.tensor(1.0), atol=1e-4)


def test_uniform_allocation():
    scores_dict = {
        0: torch.tensor([10.0, 1.0, 5.0, 2.0]),  # lowest: 1, 3
        1: torch.tensor([0.5, 8.0, 3.0, 9.0]),   # lowest: 0, 2
    }
    plan = PruningAllocator.allocate_uniform(scores_dict, prune_ratio=0.5)
    assert plan.target_ratio == 0.5
    assert plan.prune_indices[0] == [1, 3]
    assert plan.keep_indices[0] == [0, 2]
    assert plan.prune_indices[1] == [0, 2]
    assert plan.keep_indices[1] == [1, 3]
    assert plan.total_channels_removed == 4
