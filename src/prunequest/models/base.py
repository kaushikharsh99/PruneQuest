"""
Base model adapter interface and metadata structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import os
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from prunequest.reporting.metrics import clean_gpu_memory


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

    @abstractmethod
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer into memory."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Inspect and return architectural parameter statistics."""
        pass

    @abstractmethod
    def get_layers(self) -> List[torch.nn.Module]:
        """Return the list of transformer blocks."""
        pass

    @abstractmethod
    def prune_layer(self, layer_idx: int) -> Tuple[int, int]:
        """
        Physically delete the layer at layer_idx from the architecture.
        Returns: (params_before, params_after).
        """
        pass

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
        self.model = None
        self.tokenizer = None
        clean_gpu_memory(self.device)
