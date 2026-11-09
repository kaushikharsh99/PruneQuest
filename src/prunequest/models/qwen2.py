"""
Qwen2 model adapter for Qwen2ForCausalLM (e.g., Qwen2.5-Math-1.5B-Instruct).
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from prunequest.models.base import BaseModelAdapter, ModelMetadata
from prunequest.reporting.metrics import clean_gpu_memory
from prunequest.exceptions import ModelAdapterError


class Qwen2Adapter(BaseModelAdapter):
    """
    Adapter for Qwen2 architecture family.
    Validates and executes structural operations on Qwen2ForCausalLM.
    """

    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        clean_gpu_memory(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device if torch.cuda.is_available() else None,
            )
            self.model.eval()
            return self.model, self.tokenizer
        except Exception as e:
            raise ModelAdapterError(f"Failed to load Qwen2 model from '{self.model_id}': {e}") from e

    def get_layers(self) -> List[torch.nn.Module]:
        if self.model is None:
            raise ModelAdapterError("Model is not loaded. Call load() first.")
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        raise ModelAdapterError("Expected Qwen2 model to have 'model.layers' ModuleList.")

    def get_metadata(self) -> ModelMetadata:
        if self.model is None:
            raise ModelAdapterError("Model is not loaded. Call load() first.")

        cfg = self.model.config
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        embed_params = 0
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            embed_params = sum(p.numel() for p in self.model.model.embed_tokens.parameters())

        return ModelMetadata(
            model_id=self.model_id,
            architecture="Qwen2ForCausalLM",
            total_params=total_params,
            trainable_params=trainable_params,
            embedding_params=embed_params,
            non_embedding_params=total_params - embed_params,
            num_layers=len(self.model.model.layers),
            hidden_size=getattr(cfg, "hidden_size", 1536),
            intermediate_size=getattr(cfg, "intermediate_size", 8960),
            num_attention_heads=getattr(cfg, "num_attention_heads", 12),
            num_key_value_heads=getattr(cfg, "num_key_value_heads", 2),
            vocab_size=getattr(cfg, "vocab_size", 151936),
            dtype=str(self.torch_dtype).replace("torch.", ""),
            device=self.device,
            supported_pruning=["layer", "mlp"],
        )

    def prune_layer(self, layer_idx: int) -> Tuple[int, int]:
        """
        Physically remove a transformer layer from model.model.layers.
        Synchronizes config.num_hidden_layers, config.layer_types, and re-indexes remaining layers.
        """
        if self.model is None:
            raise ModelAdapterError("Model is not loaded.")

        layers = self.model.model.layers
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ModelAdapterError(f"Layer index {layer_idx} out of range [0, {len(layers)}).")

        params_before = sum(p.numel() for p in self.model.parameters())

        # Physical removal
        del layers[layer_idx]

        # Config updates
        new_layer_count = len(layers)
        self.model.config.num_hidden_layers = new_layer_count

        # Transformers v5 layer_types synchronization
        if hasattr(self.model.config, "layer_types") and self.model.config.layer_types is not None:
            if layer_idx < len(self.model.config.layer_types):
                del self.model.config.layer_types[layer_idx]

        # Re-index remaining layers for KV cache and RoPE consistency
        for idx, layer in enumerate(layers):
            layer.layer_idx = idx
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_idx = idx

        params_after = sum(p.numel() for p in self.model.parameters())
        return params_before, params_after

    def validate_architecture(self) -> Tuple[bool, Optional[str]]:
        if self.model is None:
            return False, "Model is not loaded."
        try:
            device = next(self.model.parameters()).device
            dummy_input = torch.tensor([[15, 25, 35]], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = self.model(dummy_input)
                logits = outputs.logits

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return False, "Logits contain NaN or Inf values."

            expected_vocab = self.model.config.vocab_size
            if logits.shape[-1] != expected_vocab:
                return False, f"Output vocab dimension {logits.shape[-1]} != expected {expected_vocab}."

            return True, None
        except Exception as e:
            return False, f"Forward pass validation exception: {str(e)}"

    def save_checkpoint(self, output_dir: str) -> str:
        if self.model is None:
            raise ModelAdapterError("Model is not loaded.")

        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, safe_serialization=True)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        return output_dir
