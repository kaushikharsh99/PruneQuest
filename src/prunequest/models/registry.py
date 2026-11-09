"""
Model adapter registry and dynamic loader.
"""

from typing import Dict, Type
from transformers import AutoConfig
from prunequest.models.base import BaseModelAdapter
from prunequest.models.qwen2 import Qwen2Adapter
from prunequest.exceptions import ModelAdapterError


class ModelRegistry:
    """
    Registry for model architecture adapters.
    """

    _registry: Dict[str, Type[BaseModelAdapter]] = {
        "qwen2": Qwen2Adapter,
        "qwen": Qwen2Adapter,
    }

    @classmethod
    def register(cls, model_type: str, adapter_cls: Type[BaseModelAdapter]):
        cls._registry[model_type.lower()] = adapter_cls

    @classmethod
    def get_adapter(
        cls,
        model_id_or_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
    ) -> BaseModelAdapter:
        """
        Detect model architecture and return initialized adapter instance.
        """
        try:
            cfg = AutoConfig.from_pretrained(model_id_or_path)
            model_type = getattr(cfg, "model_type", "").lower()
        except Exception:
            # Fallback to string matching on path or ID
            lower_name = model_id_or_path.lower()
            if "qwen2" in lower_name or "qwen" in lower_name:
                model_type = "qwen2"
            else:
                model_type = "unknown"

        adapter_cls = cls._registry.get(model_type)
        if adapter_cls is None:
            supported = ", ".join(cls._registry.keys())
            raise ModelAdapterError(
                f"Unsupported model architecture '{model_type}' for '{model_id_or_path}'. "
                f"Supported architectures: {supported}."
            )

        return adapter_cls(model_id_or_path=model_id_or_path, device=device, torch_dtype=torch_dtype)
