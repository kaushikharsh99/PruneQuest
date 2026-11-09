"""
Model adapter registry and dynamic loader.
"""

from typing import Dict, Type
from prunequest.models.base import BaseModelAdapter
from prunequest.models.discovery import ModelInspector
from prunequest.models.qwen2 import Qwen2Adapter
from prunequest.models.llama import LlamaAdapter
from prunequest.models.mistral import MistralAdapter
from prunequest.models.gemma import GemmaAdapter
from prunequest.models.phi import PhiAdapter
from prunequest.models.decoder import DecoderStackAdapter
from prunequest.exceptions import ModelAdapterError


class ModelRegistry:
    """
    Registry for model architecture adapters.
    """

    _registry: Dict[str, Type[BaseModelAdapter]] = {
        "qwen2": Qwen2Adapter,
        "qwen": Qwen2Adapter,
        "llama": LlamaAdapter,
        "llama2": LlamaAdapter,
        "llama3": LlamaAdapter,
        "mistral": MistralAdapter,
        "zephyr": MistralAdapter,
        "gemma": GemmaAdapter,
        "gemma2": GemmaAdapter,
        "phi": PhiAdapter,
        "phi3": PhiAdapter,
        "generic_decoder": DecoderStackAdapter,
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
        Detect model architecture using ModelInspector and return initialized adapter instance.
        """
        report = ModelInspector.inspect(model_id_or_path)

        adapter_name = report.adapter_name
        adapter_cls = None

        if adapter_name and adapter_name in cls._registry:
            adapter_cls = cls._registry[adapter_name]
        elif report.descriptor.model_type in cls._registry:
            adapter_cls = cls._registry[report.descriptor.model_type]
        else:
            # Check architectures list
            for arch in report.descriptor.architectures:
                arch_l = arch.lower()
                for key in cls._registry:
                    if key in arch_l:
                        adapter_cls = cls._registry[key]
                        break
                if adapter_cls:
                    break

        if adapter_cls is None:
            # Fallback string check
            lower_name = model_id_or_path.lower()
            for key, registered_cls in cls._registry.items():
                if key in lower_name:
                    adapter_cls = registered_cls
                    break

        if adapter_cls is None:
            supported = ", ".join(cls._registry.keys())
            raise ModelAdapterError(
                f"No verified model adapter for '{model_id_or_path}' (model_type='{report.descriptor.model_type}'). "
                f"Supported adapters: {supported}. "
                f"Inspection notes: {'; '.join(report.notes)}"
            )

        return adapter_cls(
            model_id_or_path=model_id_or_path,
            device=device,
            torch_dtype=torch_dtype,
        )
