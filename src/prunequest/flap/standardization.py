"""
Score standardization methods for cross-layer FLAP importance normalization.
"""

from typing import Dict
import torch


class ScoreStandardizer:
    """
    Standardizes importance scores within each layer to allow balanced comparison.
    """

    @classmethod
    def standardize_layer_scores(
        cls,
        scores: torch.Tensor,
        method: str = "z_score",
    ) -> torch.Tensor:
        if method == "none":
            return scores

        if method == "z_score":
            mean = scores.mean()
            std = scores.std(unbiased=False)
            eps = 1e-6
            return (scores - mean) / (std + eps)

        elif method == "min_max":
            min_v = scores.min()
            max_v = scores.max()
            eps = 1e-6
            return (scores - min_v) / (max_v - min_v + eps)

        raise ValueError(f"Unknown standardization method '{method}'. Supported: 'z_score', 'min_max', 'none'")

    @classmethod
    def standardize_all(
        cls,
        scores_dict: Dict[int, torch.Tensor],
        method: str = "z_score",
    ) -> Dict[int, torch.Tensor]:
        return {
            l_idx: cls.standardize_layer_scores(s, method=method)
            for l_idx, s in scores_dict.items()
        }
