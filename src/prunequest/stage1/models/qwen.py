"""
Model adapter for Qwen2 and Qwen2.5 causal language models.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from prunequest.stage1.models.base import BaseModelAdapter


class Qwen2MLPAdapter(BaseModelAdapter):
    """
    Adapter for Qwen2 / Qwen2.5 models managing SwiGLU MLP layers:
    - gate_proj: Linear(hidden_size, intermediate_size, bias=False)
    - up_proj: Linear(hidden_size, intermediate_size, bias=False)
    - down_proj: Linear(intermediate_size, hidden_size, bias=False)
    """

    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = self.device if torch.cuda.is_available() and "cuda" in self.device else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=False,
        )
        if device_map is None and torch.cuda.is_available() and "cuda" in self.device:
            self.model.to(self.device)
        self.model.eval()
        return self.model, self.tokenizer

    def get_layers(self) -> nn.ModuleList:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("Could not locate transformer layers in model structure.")

    def get_mlp_modules(self) -> List[nn.Module]:
        layers = self.get_layers()
        return [layer.mlp for layer in layers]

    def get_mlp_intermediate_size(self, layer_idx: int) -> int:
        mlps = self.get_mlp_modules()
        mlp = mlps[layer_idx]
        return mlp.gate_proj.out_features
