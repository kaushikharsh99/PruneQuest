"""
Physical tensor slicing for SwiGLU MLP intermediate channels.
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from prunequest.stage1.flap.allocation import AllocationPlan


class MLPSlicer:
    """
    Physically removes pruned channels from gate_proj, up_proj, and down_proj weights.
    """

    @classmethod
    def slice_mlp(
        cls,
        mlp: nn.Module,
        keep_indices: List[int],
    ) -> Dict[str, Any]:
        """
        Physically slices the linear layers in a single SwiGLU MLP.
        """
        device = mlp.gate_proj.weight.device
        idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=device)

        orig_params = (
            mlp.gate_proj.weight.numel() +
            mlp.up_proj.weight.numel() +
            mlp.down_proj.weight.numel()
        )

        with torch.no_grad():
            # 1. gate_proj: [out_features, in_features] -> slice out_features
            new_gate_w = mlp.gate_proj.weight[idx_tensor, :].clone()
            mlp.gate_proj.weight = nn.Parameter(new_gate_w)
            mlp.gate_proj.out_features = len(keep_indices)
            if mlp.gate_proj.bias is not None:
                mlp.gate_proj.bias = nn.Parameter(mlp.gate_proj.bias[idx_tensor].clone())

            # 2. up_proj: [out_features, in_features] -> slice out_features
            new_up_w = mlp.up_proj.weight[idx_tensor, :].clone()
            mlp.up_proj.weight = nn.Parameter(new_up_w)
            mlp.up_proj.out_features = len(keep_indices)
            if mlp.up_proj.bias is not None:
                mlp.up_proj.bias = nn.Parameter(mlp.up_proj.bias[idx_tensor].clone())

            # 3. down_proj: [out_features, in_features] -> slice in_features
            new_down_w = mlp.down_proj.weight[:, idx_tensor].clone()
            mlp.down_proj.weight = nn.Parameter(new_down_w)
            mlp.down_proj.in_features = len(keep_indices)

        new_params = (
            mlp.gate_proj.weight.numel() +
            mlp.up_proj.weight.numel() +
            mlp.down_proj.weight.numel()
        )

        return {
            "params_before": orig_params,
            "params_after": new_params,
            "params_removed": orig_params - new_params,
            "new_intermediate_size": len(keep_indices),
        }

    @classmethod
    def apply_plan(
        cls,
        model: PreTrainedModel,
        plan: AllocationPlan,
    ) -> Dict[str, Any]:
        """
        Applies allocation plan across all layers in the model.
        """
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError("Unsupported model architecture: layers not found.")

        total_removed_params = 0
        layer_sizes = {}

        for l_idx, keep_idx in plan.keep_indices.items():
            res = cls.slice_mlp(layers[l_idx].mlp, keep_idx)
            total_removed_params += res["params_removed"]
            layer_sizes[l_idx] = res["new_intermediate_size"]

        # If uniform, synchronize model.config.intermediate_size
        sizes_set = set(layer_sizes.values())
        if len(sizes_set) == 1:
            uniform_size = list(sizes_set)[0]
            if hasattr(model.config, "intermediate_size"):
                model.config.intermediate_size = uniform_size

        return {
            "total_params_removed": total_removed_params,
            "layer_sizes": layer_sizes,
            "is_uniform": len(sizes_set) == 1,
        }
