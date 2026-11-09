"""
Pruning backend registry and factory.
"""

from typing import Dict, Type, List
from prunequest.pruning.base import BasePruner
from prunequest.pruning.layers import LayerPruner
from prunequest.exceptions import PruningError


class PrunerRegistry:
    """
    Registry for structural pruning engines.
    """

    _registry: Dict[str, Type[BasePruner]] = {
        "layer": LayerPruner,
        "block_influence": LayerPruner,
    }

    @classmethod
    def register(cls, name: str, pruner_cls: Type[BasePruner]):
        cls._registry[name.lower()] = pruner_cls

    @classmethod
    def get_pruner(cls, name: str = "layer", **kwargs) -> BasePruner:
        lower_name = name.lower()
        pruner_cls = cls._registry.get(lower_name)
        if pruner_cls is None:
            supported = ", ".join(cls._registry.keys())
            raise PruningError(f"Unsupported pruner backend '{name}'. Supported backends: {supported}.")
        return pruner_cls(**kwargs)

    @classmethod
    def list_pruners(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))
