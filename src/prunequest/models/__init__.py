"""
Model adapters, discovery, and registry.
"""

from prunequest.models.base import BaseModelAdapter, ModelMetadata
from prunequest.models.discovery import (
    ModelInspector,
    ModelInspectionReport,
    ArchitectureDescriptor,
    ArchitectureClass,
    CapabilityLevel,
)
from prunequest.models.decoder import DecoderStackAdapter
from prunequest.models.qwen2 import Qwen2Adapter
from prunequest.models.llama import LlamaAdapter
from prunequest.models.mistral import MistralAdapter
from prunequest.models.gemma import GemmaAdapter
from prunequest.models.phi import PhiAdapter
from prunequest.models.registry import ModelRegistry

__all__ = [
    "BaseModelAdapter",
    "ModelMetadata",
    "ModelInspector",
    "ModelInspectionReport",
    "ArchitectureDescriptor",
    "ArchitectureClass",
    "CapabilityLevel",
    "DecoderStackAdapter",
    "Qwen2Adapter",
    "LlamaAdapter",
    "MistralAdapter",
    "GemmaAdapter",
    "PhiAdapter",
    "ModelRegistry",
]
