"""
Universal architecture inspection and adapter resolution for FLAP.
"""

from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class MLPArchitectureInfo:
    model_type: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    mlp_type: str  # 'swiglu' or 'standard'
    gate_proj_name: str
    up_proj_name: str
    down_proj_name: str
    has_down_proj_bias: bool
    is_supported: bool
    description: str


class ArchitectureInspector:
    """
    Inspects model configuration to dynamically detect MLP layer layout and parameters.
    """

    @classmethod
    def inspect(cls, model_id_or_path: str, trust_remote_code: bool = False) -> MLPArchitectureInfo:
        try:
            config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            raise ValueError(f"Failed to inspect configuration for '{model_id_or_path}': {e}") from e

        model_type = getattr(config, "model_type", "unknown").lower()
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
        hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 0))
        intermediate_size = getattr(config, "intermediate_size", 0)

        # Standard SwiGLU architectures: Qwen2, Llama, Mistral
        if model_type in ["qwen2", "qwen2_moe", "llama", "mistral"]:
            return MLPArchitectureInfo(
                model_type=model_type,
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                mlp_type="swiglu",
                gate_proj_name="gate_proj",
                up_proj_name="up_proj",
                down_proj_name="down_proj",
                has_down_proj_bias=False,
                is_supported=True,
                description=f"SwiGLU gated MLP architecture ({model_type.upper()}) with {num_layers} layers.",
            )

        # Gemma / Gemma-2
        if model_type in ["gemma", "gemma2"]:
            return MLPArchitectureInfo(
                model_type=model_type,
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                mlp_type="swiglu",
                gate_proj_name="gate_proj",
                up_proj_name="up_proj",
                down_proj_name="down_proj",
                has_down_proj_bias=False,
                is_supported=True,
                description=f"Gemma SwiGLU gated MLP architecture with {num_layers} layers.",
            )

        # Phi-3 / Phi-3.5
        if model_type in ["phi3"]:
            return MLPArchitectureInfo(
                model_type=model_type,
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                mlp_type="swiglu",
                gate_proj_name="gate_up_proj",  # fused
                up_proj_name="gate_up_proj",
                down_proj_name="down_proj",
                has_down_proj_bias=False,
                is_supported=False,
                description="Phi-3 fused gate_up_proj architecture (currently unsupported for separate channel slicing).",
            )

        # Fallback / Generic
        return MLPArchitectureInfo(
            model_type=model_type,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mlp_type="unknown",
            gate_proj_name="gate_proj",
            up_proj_name="up_proj",
            down_proj_name="down_proj",
            has_down_proj_bias=False,
            is_supported=False,
            description=f"Model type '{model_type}' is not yet verified for structured MLP pruning.",
        )
