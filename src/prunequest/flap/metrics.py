"""
Fluctuation-based structural importance metrics (WIFV and WIFN) from the FLAP paper.
"""

from typing import Dict
import torch

from prunequest.flap.calibration.collector import LayerActivationStats


class FLAPMetricCalculator:
    """
    Computes importance scores for intermediate MLP channels.
    Supports:
    - WIFV: Weighted Input Feature Variance: ||W_down[:, j]||_2^2 * Var(h_j)
    - WIFN: Weighted Input Feature Norm: ||W_down[:, j]||_2^2 * E[|h_j|]
    """

    @classmethod
    def compute_layer_scores(
        cls,
        stats: LayerActivationStats,
        metric: str = "wifv",
    ) -> torch.Tensor:
        """
        Computes 1D importance scores tensor of shape [intermediate_size] for a layer.
        """
        w_norm2 = stats.down_proj_weight_norm2  # [intermediate_size]

        if metric.lower() == "wifv":
            # WIFV: ||W[:, j]||_2^2 * Var(h_j)
            scores = w_norm2 * stats.variance
        elif metric.lower() == "wifn":
            # WIFN: ||W[:, j]||_2^2 * E[|h_j|]
            scores = w_norm2 * stats.mean_abs
        else:
            raise ValueError(f"Unknown FLAP metric: '{metric}'. Supported: 'wifv', 'wifn'")

        # Numerical safety: replace NaNs/Infs with zero
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores

    @classmethod
    def compute_all_scores(
        cls,
        stats_dict: Dict[int, LayerActivationStats],
        metric: str = "wifv",
    ) -> Dict[int, torch.Tensor]:
        """
        Computes importance scores for all layers.
        """
        return {
            l_idx: cls.compute_layer_scores(stats, metric=metric)
            for l_idx, stats in stats_dict.items()
        }
