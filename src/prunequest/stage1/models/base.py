"""
Base model adapter interface for structured MLP pruning.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseModelAdapter(ABC):
    """
    Abstract interface for model inspection, loading, and structured component access.
    """

    def __init__(self, model_id_or_path: str, device: str = "cuda:0", torch_dtype: str = "bfloat16"):
        self.model_id = model_id_or_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    @abstractmethod
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load pretrained model and tokenizer into memory."""
        pass

    @abstractmethod
    def get_mlp_modules(self) -> List[nn.Module]:
        """Return list of MLP modules for all transformer layers."""
        pass

    @abstractmethod
    def get_mlp_intermediate_size(self, layer_idx: int) -> int:
        """Return the current intermediate channel count for the specified layer."""
        pass

    def count_parameters(self) -> int:
        """Count total parameters in the active model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "total_params": self.count_parameters(),
        }

    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer to directory."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        self.model.save_pretrained(save_directory)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
