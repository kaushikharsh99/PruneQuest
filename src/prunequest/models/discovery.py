"""
Model architecture discovery, classification, and capability assessment.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from transformers import AutoConfig


class ArchitectureClass(str, Enum):
    DECODER_ONLY_DENSE = "decoder_only_dense"
    DECODER_ONLY_MOE = "decoder_only_moe"
    ENCODER_DECODER = "encoder_decoder"
    ENCODER_ONLY = "encoder_only"
    HYBRID_SSM = "hybrid_ssm"
    RECURRENT_SSM = "recurrent_ssm"
    MULTIMODAL = "multimodal"
    QUANTIZED = "quantized"
    UNKNOWN = "unknown"


class CapabilityLevel(int, Enum):
    LEVEL_0_UNSUPPORTED = 0
    LEVEL_1_EVALUATION = 1
    LEVEL_2_EXPERIMENTAL_PRUNING = 2
    LEVEL_3_VERIFIED_PRUNING = 3


@dataclass
class ArchitectureDescriptor:
    model_id_or_path: str
    architecture_class: ArchitectureClass
    model_type: str
    architectures: List[str]
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    quantization_type: Optional[str] = None
    is_moe: bool = False
    is_multimodal: bool = False
    is_encoder_decoder: bool = False
    raw_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInspectionReport:
    descriptor: ArchitectureDescriptor
    capability_level: CapabilityLevel
    supported_operations: List[str]
    notes: List[str]
    adapter_name: Optional[str] = None


class ModelInspector:
    """
    Inspects model architecture and capabilities using Hugging Face configuration metadata
    without requiring heavy weight downloads upfront.
    """

    # Mapping of known model types to families
    DENSE_DECODER_TYPES = {
        "qwen2", "qwen2_moe", "llama", "mistral", "gemma", "gemma2",
        "phi", "phi3", "gpt_neox", "gpt2", "bloom", "falcon", "baichuan", "internlm2",
    }
    MOE_TYPES = {"qwen2_moe", "mixtral", "dbrx", "deepseek_v2", "deepseek_v3", "jetmoe", "olmoe"}
    ENCODER_DECODER_TYPES = {"t5", "bart", "pegasus", "marian", "mbart"}
    ENCODER_ONLY_TYPES = {"bert", "roberta", "deberta", "deberta-v2", "albert", "electra"}
    SSM_TYPES = {"mamba", "mamba2", "rwkv", "jamba"}
    MULTIMODAL_TYPES = {"llava", "qwen2_vl", "paligemma", "idefics", "idefics2", "chameleon"}

    @classmethod
    def inspect(
        cls,
        model_id_or_path: str,
        trust_remote_code: bool = False,
    ) -> ModelInspectionReport:
        try:
            config = AutoConfig.from_pretrained(
                model_id_or_path,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            # Level 0: could not load config
            desc = ArchitectureDescriptor(
                model_id_or_path=model_id_or_path,
                architecture_class=ArchitectureClass.UNKNOWN,
                model_type="unknown",
                architectures=[],
                num_hidden_layers=0,
                hidden_size=0,
                num_attention_heads=0,
                num_key_value_heads=0,
                vocab_size=0,
            )
            return ModelInspectionReport(
                descriptor=desc,
                capability_level=CapabilityLevel.LEVEL_0_UNSUPPORTED,
                supported_operations=[],
                notes=[f"Failed to load Hugging Face config: {str(e)}"],
                adapter_name=None,
            )

        model_type = getattr(config, "model_type", "unknown").lower()
        architectures = getattr(config, "architectures", []) or []
        arch_str = architectures[0].lower() if architectures else model_type

        def _get_int(cfg, key: str, default: int = 0) -> int:
            val = getattr(cfg, key, default)
            return val if isinstance(val, int) else default

        # Extract dimensions
        num_layers = _get_int(config, "num_hidden_layers", _get_int(config, "n_layer", 0))
        hidden_size = _get_int(config, "hidden_size", _get_int(config, "n_embd", 0))
        num_heads = _get_int(config, "num_attention_heads", _get_int(config, "n_head", 0))
        num_kv_heads = _get_int(config, "num_key_value_heads", num_heads)
        vocab_size = _get_int(config, "vocab_size", 0)

        # Detect quantization
        quant_config = getattr(config, "quantization_config", None)
        quant_type = None
        if quant_config and isinstance(quant_config, dict):
            quant_type = quant_config.get("quant_method", "quantized")

        # Detect Multimodal
        is_multimodal = (
            model_type in cls.MULTIMODAL_TYPES
            or any("vl" in str(a).lower() or "vision" in str(a).lower() for a in [model_type] + architectures)
            or (hasattr(config, "vision_config") and config.vision_config is not None and not hasattr(config.vision_config, "_mock_return_value"))
        )

        # Detect MoE
        is_moe = (
            model_type in cls.MOE_TYPES
            or _get_int(config, "num_experts") > 0
            or _get_int(config, "n_routed_experts") > 0
            or _get_int(config, "num_local_experts") > 0
        )

        # Detect Encoder-Decoder
        is_enc_dec = getattr(config, "is_encoder_decoder", False) or model_type in cls.ENCODER_DECODER_TYPES

        # Detect Encoder-only
        is_enc_only = model_type in cls.ENCODER_ONLY_TYPES

        # Detect SSM / Hybrid
        is_ssm = model_type in cls.SSM_TYPES or "mamba" in arch_str

        # Classify
        if is_multimodal:
            arch_class = ArchitectureClass.MULTIMODAL
        elif quant_type is not None:
            arch_class = ArchitectureClass.QUANTIZED
        elif is_moe:
            arch_class = ArchitectureClass.DECODER_ONLY_MOE
        elif is_ssm:
            arch_class = ArchitectureClass.HYBRID_SSM if "hybrid" in arch_str else ArchitectureClass.RECURRENT_SSM
        elif is_enc_dec:
            arch_class = ArchitectureClass.ENCODER_DECODER
        elif is_enc_only:
            arch_class = ArchitectureClass.ENCODER_ONLY
        elif model_type in cls.DENSE_DECODER_TYPES or "causallm" in arch_str:
            arch_class = ArchitectureClass.DECODER_ONLY_DENSE
        else:
            arch_class = ArchitectureClass.UNKNOWN

        descriptor = ArchitectureDescriptor(
            model_id_or_path=model_id_or_path,
            architecture_class=arch_class,
            model_type=model_type,
            architectures=architectures,
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            vocab_size=vocab_size,
            quantization_type=quant_type,
            is_moe=is_moe,
            is_multimodal=is_multimodal,
            is_encoder_decoder=is_enc_dec,
            raw_config=config.to_dict() if hasattr(config, "to_dict") else {},
        )

        # Determine Capability Level & Supported Operations
        supported_ops: List[str] = []
        notes: List[str] = []
        adapter_name = None

        if quant_type is not None:
            capability = CapabilityLevel.LEVEL_1_EVALUATION
            notes.append(f"Model is quantized ({quant_type}). Structural weight pruning requires unquantized source.")
            adapter_name = "quantized"
        elif arch_class == ArchitectureClass.DECODER_ONLY_DENSE:
            # Match family adapter
            if model_type in ("qwen2", "qwen2.5"):
                capability = CapabilityLevel.LEVEL_3_VERIFIED_PRUNING
                supported_ops = ["layer_pruning", "block_influence_scoring"]
                adapter_name = "qwen2"
                notes.append("Tier 1 verified structural layer pruning and GSM8K evaluation support.")
            elif model_type in ("llama", "llama2", "llama3"):
                capability = CapabilityLevel.LEVEL_3_VERIFIED_PRUNING
                supported_ops = ["layer_pruning", "block_influence_scoring"]
                adapter_name = "llama"
                notes.append("Llama decoder family verified structural layer pruning support.")
            elif model_type in ("mistral", "zephyr"):
                capability = CapabilityLevel.LEVEL_3_VERIFIED_PRUNING
                supported_ops = ["layer_pruning", "block_influence_scoring"]
                adapter_name = "mistral"
                notes.append("Mistral decoder family verified structural layer pruning support.")
            elif model_type in ("gemma", "gemma2"):
                capability = CapabilityLevel.LEVEL_2_EXPERIMENTAL_PRUNING
                supported_ops = ["layer_pruning"]
                adapter_name = "gemma"
                notes.append("Gemma decoder family experimental pruning support.")
            elif model_type in ("phi", "phi3"):
                capability = CapabilityLevel.LEVEL_2_EXPERIMENTAL_PRUNING
                supported_ops = ["layer_pruning"]
                adapter_name = "phi"
                notes.append("Phi decoder family experimental pruning support.")
            else:
                capability = CapabilityLevel.LEVEL_1_EVALUATION
                adapter_name = "generic_decoder"
                notes.append("Generic decoder model. Evaluation supported; pruning requires family adapter.")
        elif arch_class == ArchitectureClass.DECODER_ONLY_MOE:
            capability = CapabilityLevel.LEVEL_1_EVALUATION
            notes.append("Mixture of Experts model. Evaluation supported; expert pruning planned for future release.")
        elif arch_class in (ArchitectureClass.ENCODER_DECODER, ArchitectureClass.ENCODER_ONLY):
            capability = CapabilityLevel.LEVEL_1_EVALUATION
            notes.append("Non-causal Transformer architecture. Evaluation supported.")
        elif arch_class in (ArchitectureClass.HYBRID_SSM, ArchitectureClass.RECURRENT_SSM):
            capability = CapabilityLevel.LEVEL_1_EVALUATION
            notes.append("State-space / Recurrent model. Evaluation supported.")
        else:
            capability = CapabilityLevel.LEVEL_0_UNSUPPORTED
            notes.append("Unrecognized architecture.")

        return ModelInspectionReport(
            descriptor=descriptor,
            capability_level=capability,
            supported_operations=supported_ops,
            notes=notes,
            adapter_name=adapter_name,
        )
