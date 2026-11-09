"""
Model registry for dynamic adapter lookup based on architecture inspection.
"""

from typing import Type
from prunequest.stage1.models.base import BaseModelAdapter
from prunequest.stage1.models.discovery import ArchitectureInspector
from prunequest.stage1.models.qwen import Qwen2MLPAdapter
from prunequest.stage1.models.llama import LlamaMLPAdapter


class ModelRegistry:
    """
    Resolves model identifiers to appropriate model adapters.
    """

    _REGISTRY = {
        "qwen2": Qwen2MLPAdapter,
        "qwen2_moe": Qwen2MLPAdapter,
        "llama": LlamaMLPAdapter,
        "mistral": LlamaMLPAdapter,
    }

    @classmethod
    def resolve_adapter_class(cls, model_id_or_path: str, trust_remote_code: bool = False) -> Type[BaseModelAdapter]:
        info = ArchitectureInspector.inspect(model_id_or_path, trust_remote_code=trust_remote_code)
        if not info.is_supported:
            raise ValueError(f"Architecture for model '{model_id_or_path}' is not supported: {info.description}")

        adapter_cls = cls._REGISTRY.get(info.model_type)
        if adapter_cls is None:
            raise ValueError(f"No adapter registered for supported model type '{info.model_type}'")
        return adapter_cls

    @classmethod
    def create_adapter(
        cls,
        model_id_or_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> BaseModelAdapter:
        adapter_cls = cls.resolve_adapter_class(model_id_or_path, trust_remote_code=trust_remote_code)
        return adapter_cls(model_id_or_path=model_id_or_path, device=device, torch_dtype=torch_dtype)
