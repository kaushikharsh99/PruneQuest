"""
Transactional in-memory state manager for Joint (FLAP + Block) pruning operations.
Guarantees zero-loss atomic rollback across arbitrary combinations of width and depth pruning.
"""

from typing import Dict, Any, Optional, List, Tuple
from transformers import PreTrainedModel

from prunequest.flap.pruning.manager import PruningStateManager
from prunequest.flap.pruning.slicer import MLPSlicer
from prunequest.flap.allocation import AllocationPlan


class JointStateManager:
    """
    Manages in-memory atomic snapshots and lossless rollback for joint pruning candidates.
    Supports individual block removal, individual MLP slicing, and combined joint modifications.
    """

    @classmethod
    def get_layers(cls, model: PreTrainedModel):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Could not locate transformer layers in model structure.")

    @classmethod
    def backup_state(
        cls,
        model: PreTrainedModel,
        active_channels: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Creates a complete in-memory snapshot of model structure and parameters in host RAM.
        """
        layers = cls.get_layers(model)
        return {
            "num_hidden_layers": model.config.num_hidden_layers,
            "intermediate_size": getattr(model.config, "intermediate_size", None),
            "layers_list": list(layers),
            "mlp_backup": PruningStateManager.backup_mlp_weights(model, active_channels=active_channels),
            "layer_types": list(model.config.layer_types) if hasattr(model.config, "layer_types") and model.config.layer_types is not None else None,
        }

    @classmethod
    def restore_state(
        cls,
        model: PreTrainedModel,
        backup: Dict[str, Any],
        active_channels_ref: Optional[Dict[int, List[int]]] = None,
    ):
        """
        Atomically restores the model to the exact snapshot state without reloading from disk.
        """
        import torch.nn as nn
        new_module_list = nn.ModuleList(backup["layers_list"])
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = new_module_list
        elif hasattr(model, "layers"):
            model.layers = new_module_list
        layers = new_module_list

        model.config.num_hidden_layers = backup["num_hidden_layers"]
        if backup["intermediate_size"] is not None:
            model.config.intermediate_size = backup["intermediate_size"]

        if backup["layer_types"] is not None and hasattr(model.config, "layer_types"):
            model.config.layer_types = list(backup["layer_types"])

        # Re-index remaining layers for KV cache and RoPE consistency
        for idx, layer in enumerate(layers):
            layer.layer_idx = idx
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = idx

        # Restore MLP weight matrices
        PruningStateManager.restore_mlp_weights(model, backup["mlp_backup"], active_channels=active_channels_ref)

    @classmethod
    def remove_block(
        cls,
        model: PreTrainedModel,
        layer_idx: int,
        active_channels: Optional[Dict[int, List[int]]] = None,
    ) -> int:
        """
        Physically deletes a transformer block from the model decoder stack.
        Returns the number of parameters removed.
        """
        layers = cls.get_layers(model)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {len(layers)})")

        params_before = sum(p.numel() for p in model.parameters())

        del layers[layer_idx]
        model.config.num_hidden_layers = len(layers)

        if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
            if layer_idx < len(model.config.layer_types):
                del model.config.layer_types[layer_idx]

        for idx, layer in enumerate(layers):
            layer.layer_idx = idx
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = idx

        if active_channels is not None:
            new_active = {}
            new_idx = 0
            for idx in sorted(active_channels.keys()):
                if idx == layer_idx:
                    continue
                new_active[new_idx] = active_channels[idx]
                new_idx += 1
            active_channels.clear()
            active_channels.update(new_active)

        params_after = sum(p.numel() for p in model.parameters())
        return params_before - params_after

    @classmethod
    def slice_mlp_channels(
        cls,
        model: PreTrainedModel,
        plan: AllocationPlan,
        active_channels: Optional[Dict[int, List[int]]] = None,
    ) -> int:
        """
        Physically slices SwiGLU MLP intermediate projections according to the plan.
        Returns the number of parameters removed.
        """
        res = MLPSlicer.apply_plan(model, plan, active_channels=active_channels)
        return res["total_params_removed"]

    @classmethod
    def apply_joint_candidate(
        cls,
        model: PreTrainedModel,
        block_idx: Optional[int],
        plan: Optional[AllocationPlan],
        active_channels: Optional[Dict[int, List[int]]] = None,
    ) -> Tuple[int, int]:
        """
        Applies a joint candidate: optionally slicing surviving MLPs and removing a block.
        Always slices MLPs first so plan indices align with layers before block excision.
        Returns (params_removed, current_params).
        """
        params_before = sum(p.numel() for p in model.parameters())

        if plan is not None:
            cls.slice_mlp_channels(model, plan, active_channels=active_channels)

        if block_idx is not None:
            cls.remove_block(model, block_idx, active_channels=active_channels)

        params_after = sum(p.numel() for p in model.parameters())
        return (params_before - params_after), params_after
