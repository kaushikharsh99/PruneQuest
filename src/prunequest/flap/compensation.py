"""
FLAP bias compensation mechanism recovering mean feature map shifts.
"""

from typing import List, Optional
import torch
import torch.nn as nn



class BiasCompensator:
    """
    Computes output compensation vector for pruned channels:
    Delta_b = W_down[:, pruned_indices] @ mean[pruned_indices]
    """

    @classmethod
    def compute_layer_compensation(
        cls,
        w_down: torch.Tensor,       # [hidden_size, intermediate_size]
        mean: torch.Tensor,         # [intermediate_size]
        prune_indices: List[int],
    ) -> Optional[torch.Tensor]:
        if not prune_indices:
            return None

        # Sliced weights and means for pruned indices
        w_pruned = w_down[:, prune_indices]  # [hidden_size, len(prune_indices)]
        mean_pruned = mean[prune_indices]    # [len(prune_indices)]

        # Delta b: [hidden_size]
        delta_b = torch.matmul(w_pruned, mean_pruned)
        return delta_b

    @classmethod
    def apply_compensation_to_layer(
        cls,
        down_proj: nn.Linear,
        delta_b: torch.Tensor,
    ) -> bool:
        """
        Applies compensation to down_proj.bias.
        If down_proj.bias is None, creates a new parameter if supported.
        """
        if delta_b is None:
            return False

        device = down_proj.weight.device
        dtype = down_proj.weight.dtype
        delta_b_adapted = delta_b.to(device=device, dtype=dtype)

        if down_proj.bias is not None:
            with torch.no_grad():
                down_proj.bias.add_(delta_b_adapted)
            return True
        else:
            # Install new bias parameter
            down_proj.bias = nn.Parameter(delta_b_adapted)
            return True
