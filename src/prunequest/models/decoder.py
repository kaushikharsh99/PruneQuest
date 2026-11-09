"""
Generic Decoder Stack Adapter for dense decoder-only Transformer architectures.
Standardizes layer discovery, inspection, ablation, and physical layer removal across
Qwen2, Llama, Mistral, Gemma, and Phi model families.
"""

from typing import Optional, Tuple
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from prunequest.models.base import BaseModelAdapter, ModelMetadata
from prunequest.reporting.metrics import clean_gpu_memory
from prunequest.exceptions import ModelAdapterError


class DecoderStackAdapter(BaseModelAdapter):
    """
    Unified base adapter for decoder-only architectures with repeated transformer blocks
    (e.g., model.model.layers or model.layers).
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        family_name: str = "GenericDecoder",
    ):
        super().__init__(model_id_or_path=model_id_or_path, device=device, torch_dtype=torch_dtype)
        self.family_name = family_name

    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        clean_gpu_memory(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device if torch.cuda.is_available() else None,
                trust_remote_code=False,
            )
            self.model.eval()
            return self.model, self.tokenizer
        except Exception as e:
            raise ModelAdapterError(f"Failed to load {self.family_name} model from '{self.model_id}': {e}") from e

    def get_layers(self) -> torch.nn.ModuleList:
        if self.model is None:
            raise ModelAdapterError("Model is not loaded. Call load() first.")

        # Standard Hugging Face layout: model.model.layers or model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h

        raise ModelAdapterError(f"Unable to locate decoder layers in {type(self.model).__name__}.")

    def get_metadata(self) -> ModelMetadata:
        if self.model is None:
            raise ModelAdapterError("Model is not loaded. Call load() first.")

        cfg = self.model.config
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        embed_params = 0
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            embed_params = sum(p.numel() for p in self.model.model.embed_tokens.parameters())
        elif hasattr(self.model, "get_input_embeddings"):
            inp_embed = self.model.get_input_embeddings()
            if inp_embed is not None:
                embed_params = sum(p.numel() for p in inp_embed.parameters())

        layers = self.get_layers()
        return ModelMetadata(
            model_id=self.model_id,
            architecture=self.family_name,
            total_params=total_params,
            trainable_params=trainable_params,
            embedding_params=embed_params,
            non_embedding_params=total_params - embed_params,
            num_layers=len(layers),
            hidden_size=getattr(cfg, "hidden_size", 0),
            intermediate_size=getattr(cfg, "intermediate_size", 0),
            num_attention_heads=getattr(cfg, "num_attention_heads", 0),
            num_key_value_heads=getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0)),
            vocab_size=getattr(cfg, "vocab_size", 0),
            dtype=str(self.torch_dtype).replace("torch.", ""),
            device=self.device,
            supported_pruning=["layer"],
        )

    def prune_layer(self, layer_idx: int) -> Tuple[int, int]:
        """
        Physically remove a transformer block from the decoder stack.
        Synchronizes configuration counts, layer_types list, and reindexes RoPE/cache indices.
        """
        if self.model is None:
            raise ModelAdapterError("Model is not loaded.")

        layers = self.get_layers()
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ModelAdapterError(f"Layer index {layer_idx} out of range [0, {len(layers)}).")

        params_before = sum(p.numel() for p in self.model.parameters())

        # Physical removal from nn.ModuleList
        del layers[layer_idx]

        # Synchronize config layer count
        new_layer_count = len(layers)
        self.model.config.num_hidden_layers = new_layer_count

        # Transformers v5 layer_types synchronization
        if hasattr(self.model.config, "layer_types") and self.model.config.layer_types is not None:
            if layer_idx < len(self.model.config.layer_types):
                del self.model.config.layer_types[layer_idx]

        # Re-index remaining layers for KV cache and RoPE consistency
        for idx, layer in enumerate(layers):
            layer.layer_idx = idx
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = idx

        params_after = sum(p.numel() for p in self.model.parameters())
        return params_before, params_after

    def validate_architecture(self) -> Tuple[bool, Optional[str]]:
        if self.model is None:
            return False, "Model is not loaded."
        try:
            device = next(self.model.parameters()).device
            dummy_input = torch.tensor([[10, 20, 30]], dtype=torch.long, device=device)
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

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, safe_serialization=True)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        return output_dir
