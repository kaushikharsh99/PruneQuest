"""
Transactional state management and lossless in-memory rollback for candidate pruning.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from transformers import PreTrainedModel


class PruningStateManager:
    """
    Saves and restores model MLP states in host RAM for fast, lossless rollback.
    """

    @classmethod
    def backup_mlp_weights(
        cls,
        model: PreTrainedModel,
        active_channels: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Clones all MLP weight tensors to CPU RAM.
        """
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError("Unsupported model architecture: layers not found.")

        backup: Dict[str, Any] = {}
        for l_idx, layer in enumerate(layers):
            mlp = layer.mlp
            backup[l_idx] = {
                "gate_w": mlp.gate_proj.weight.detach().cpu().clone(),
                "gate_out": mlp.gate_proj.out_features,
                "gate_in": mlp.gate_proj.in_features,
                "gate_b": mlp.gate_proj.bias.detach().cpu().clone() if mlp.gate_proj.bias is not None else None,

                "up_w": mlp.up_proj.weight.detach().cpu().clone(),
                "up_out": mlp.up_proj.out_features,
                "up_in": mlp.up_proj.in_features,
                "up_b": mlp.up_proj.bias.detach().cpu().clone() if mlp.up_proj.bias is not None else None,

                "down_w": mlp.down_proj.weight.detach().cpu().clone(),
                "down_out": mlp.down_proj.out_features,
                "down_in": mlp.down_proj.in_features,
                "down_b": mlp.down_proj.bias.detach().cpu().clone() if mlp.down_proj.bias is not None else None,
            }
        backup["_config_intermediate_size"] = getattr(model.config, "intermediate_size", None)
        if active_channels is not None:
            backup["_active_channels"] = {k: list(v) for k, v in active_channels.items()}
        return backup

    @classmethod
    def restore_mlp_weights(
        cls,
        model: PreTrainedModel,
        backup: Dict[str, Any],
        active_channels: Optional[Dict[int, List[int]]] = None,
    ):
        """
        Restores all MLP weight tensors from CPU backup.
        """
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError("Unsupported model architecture: layers not found.")

        for l_idx, layer in enumerate(layers):
            if l_idx not in backup:
                continue
            b = backup[l_idx]
            mlp = layer.mlp
            device = mlp.gate_proj.weight.device
            dtype = mlp.gate_proj.weight.dtype

            with torch.no_grad():
                mlp.gate_proj.weight = nn.Parameter(b["gate_w"].to(device=device, dtype=dtype))
                mlp.gate_proj.out_features = b["gate_out"]
                mlp.gate_proj.in_features = b["gate_in"]
                if b["gate_b"] is not None:
                    mlp.gate_proj.bias = nn.Parameter(b["gate_b"].to(device=device, dtype=dtype))
                else:
                    mlp.gate_proj.bias = None

                mlp.up_proj.weight = nn.Parameter(b["up_w"].to(device=device, dtype=dtype))
                mlp.up_proj.out_features = b["up_out"]
                mlp.up_proj.in_features = b["up_in"]
                if b["up_b"] is not None:
                    mlp.up_proj.bias = nn.Parameter(b["up_b"].to(device=device, dtype=dtype))
                else:
                    mlp.up_proj.bias = None

                mlp.down_proj.weight = nn.Parameter(b["down_w"].to(device=device, dtype=dtype))
                mlp.down_proj.out_features = b["down_out"]
                mlp.down_proj.in_features = b["down_in"]
                if b["down_b"] is not None:
                    mlp.down_proj.bias = nn.Parameter(b["down_b"].to(device=device, dtype=dtype))
                else:
                    mlp.down_proj.bias = None

        if backup.get("_config_intermediate_size") is not None and hasattr(model.config, "intermediate_size"):
            model.config.intermediate_size = backup["_config_intermediate_size"]

        if active_channels is not None and "_active_channels" in backup:
            active_channels.clear()
            active_channels.update({k: list(v) for k, v in backup["_active_channels"].items()})
