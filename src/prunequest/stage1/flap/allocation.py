"""
Structured pruning allocation determining which intermediate channels to prune.
"""

from typing import Dict, List
from dataclasses import dataclass
import torch


@dataclass
class AllocationPlan:
    prune_indices: Dict[int, List[int]]
    keep_indices: Dict[int, List[int]]
    channels_removed_per_layer: Dict[int, int]
    total_channels_removed: int
    target_ratio: float


class PruningAllocator:
    """
    Computes pruning channel allocations given layer importance scores.
    """

    @classmethod
    def allocate_uniform(
        cls,
        scores_dict: Dict[int, torch.Tensor],
        prune_ratio: float,
    ) -> AllocationPlan:
        """
        Uniform allocation: removes the same number of lowest-scoring channels from every layer,
        preserving consistent intermediate size across all layers for 100% standard Hugging Face
        checkpoint serialization and reloading.
        """
        prune_indices: Dict[int, List[int]] = {}
        keep_indices: Dict[int, List[int]] = {}
        removed_per_layer: Dict[int, int] = {}
        total_removed = 0

        for l_idx, scores in scores_dict.items():
            dim = scores.shape[0]
            k = int(round(prune_ratio * dim))
            k = max(0, min(k, dim - 1))  # Keep at least 1 channel

            # Sort ascending: lowest score channels are pruned
            sorted_indices = torch.argsort(scores).tolist()
            to_prune = sorted(sorted_indices[:k])
            to_keep = sorted(sorted_indices[k:])

            prune_indices[l_idx] = to_prune
            keep_indices[l_idx] = to_keep
            removed_per_layer[l_idx] = len(to_prune)
            total_removed += len(to_prune)

        return AllocationPlan(
            prune_indices=prune_indices,
            keep_indices=keep_indices,
            channels_removed_per_layer=removed_per_layer,
            total_channels_removed=total_removed,
            target_ratio=prune_ratio,
        )

    @classmethod
    def allocate_global_adaptive(
        cls,
        scores_dict: Dict[int, torch.Tensor],
        prune_ratio: float,
    ) -> AllocationPlan:
        """
        Global adaptive allocation: ranks all (layer, channel) pairs globally by their standardized
        scores, allocating variable sparsity per layer.
        """
        all_entries = []
        for l_idx, scores in scores_dict.items():
            for c_idx in range(scores.shape[0]):
                all_entries.append((scores[c_idx].item(), l_idx, c_idx))

        total_channels = len(all_entries)
        k_total = int(round(prune_ratio * total_channels))

        # Sort globally ascending by score
        all_entries.sort(key=lambda x: x[0])
        selected_to_prune = set()
        for _, l_idx, c_idx in all_entries[:k_total]:
            selected_to_prune.add((l_idx, c_idx))

        prune_indices: Dict[int, List[int]] = {}
        keep_indices: Dict[int, List[int]] = {}
        removed_per_layer: Dict[int, int] = {}
        total_removed = 0

        for l_idx, scores in scores_dict.items():
            dim = scores.shape[0]
            to_prune = [c for c in range(dim) if (l_idx, c) in selected_to_prune]
            # Ensure at least 1 channel is retained per layer
            if len(to_prune) == dim:
                to_prune.pop()
            to_keep = [c for c in range(dim) if c not in set(to_prune)]

            prune_indices[l_idx] = sorted(to_prune)
            keep_indices[l_idx] = sorted(to_keep)
            removed_per_layer[l_idx] = len(to_prune)
            total_removed += len(to_prune)

        return AllocationPlan(
            prune_indices=prune_indices,
            keep_indices=keep_indices,
            channels_removed_per_layer=removed_per_layer,
            total_channels_removed=total_removed,
            target_ratio=prune_ratio,
        )
