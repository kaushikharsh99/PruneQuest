"""
Base model adapter interface and metadata structures.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Generator
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from prunequest.reporting.metrics import clean_gpu_memory
from prunequest.models.discovery import ModelInspector, ModelInspectionReport


@dataclass
class ModelMetadata:
    model_id: str
    architecture: str
    total_params: int
    trainable_params: int
    embedding_params: int
    non_embedding_params: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    dtype: str
    device: str
    supported_pruning: List[str] = field(default_factory=lambda: ["layer"])

    @property
    def total_params_million(self) -> float:
        return self.total_params / 1e6


class BaseModelAdapter(ABC):
    """
    Abstract interface for model architecture adapters.
    Provides uniform layer extraction, temporary ablation, physical pruning,
    validation, and serialization across diverse LLM families.
    """

    def __init__(self, model_id_or_path: str, device: str = "cuda:0", torch_dtype: str = "bfloat16"):
        self.model_id = model_id_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.bfloat16
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def inspect(self) -> ModelInspectionReport:
        """Inspect model architecture metadata without requiring full weight loading."""
        return ModelInspector.inspect(self.model_id)

    @abstractmethod
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer into memory."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Inspect and return architectural parameter statistics."""
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return model capability levels and supported operations."""
        report = self.inspect()
        return {
            "level": report.capability_level,
            "supported_operations": report.supported_operations,
            "notes": report.notes,
        }

    def count_parameters(self) -> int:
        """Return total parameter count of current model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    @abstractmethod
    def get_layers(self) -> torch.nn.ModuleList:
        """Return the list of transformer blocks."""
        pass

    @abstractmethod
    def prune_layer(self, layer_idx: int) -> Tuple[int, int]:
        """
        Physically delete the layer at layer_idx from the architecture.
        Returns: (params_before, params_after).
        """
        pass

    @contextmanager
    def temporary_ablation(self, operation: str, target: Any) -> Generator[Dict[str, Any], None, None]:
        """
        Transactionally ablate/prune a component for temporary evaluation,
        guaranteeing exact state restoration upon context exit even on exceptions.
        """
        if operation == "layer":
            layer_idx = int(target)
            layers = self.get_layers()
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"Invalid layer_idx {layer_idx} for {len(layers)} layers")

            layer_backup = layers[layer_idx]
            original_config_layers = getattr(self.model.config, "num_hidden_layers", len(layers))
            original_layer_types = (
                list(self.model.config.layer_types)
                if hasattr(self.model.config, "layer_types") and self.model.config.layer_types is not None
                else None
            )

            try:
                # Apply temporary pruning
                self.prune_layer(layer_idx)
                yield {"operation": operation, "target": target, "backup": layer_backup}
            finally:
                # Guaranteed exact restoration
                layers.insert(layer_idx, layer_backup)
                self.model.config.num_hidden_layers = original_config_layers
                if original_layer_types is not None and hasattr(self.model.config, "layer_types"):
                    self.model.config.layer_types = original_layer_types

                # Re-index remaining layers
                for idx, layer in enumerate(layers):
                    if hasattr(layer, "layer_idx"):
                        layer.layer_idx = idx
                    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                        layer.self_attn.layer_idx = idx
        else:
            raise NotImplementedError(f"Ablation operation '{operation}' not supported by adapter.")

    @abstractmethod
    def validate_architecture(self) -> Tuple[bool, Optional[str]]:
        """Run a forward pass with dummy inputs to verify absence of NaNs/Infs."""
        pass

    @abstractmethod
    def save_checkpoint(self, output_dir: str) -> str:
        """Save model weights in safetensors format along with tokenizer and configs."""
        pass

    def unload(self):
        """Release GPU memory references."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        clean_gpu_memory(self.device)
