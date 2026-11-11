"""
Automatic pruning strategy resolver for PruneQuest.
Determines optimal or compatible pruning strategy (joint, flap, block)
based on model architecture capabilities and hardware headroom.
"""

from enum import Enum
from typing import Tuple
from prunequest.models.discovery import ModelInspector, ModelInspectionReport, CapabilityLevel, ArchitectureClass


class PruningStrategy(str, Enum):
    AUTO = "auto"
    JOINT = "joint"
    FLAP = "flap"
    BLOCK = "block"


class StrategyResolver:
    """
    Inspects model architecture and resolves the optimal pruning strategy.
    Zero manual configuration required: automatically selects joint width+depth
    pruning when verified, or falls back to compatible single-method pruning.
    """

    FLAP_SUPPORTED_TYPES = {"qwen2", "qwen2.5", "llama", "llama2", "llama3", "mistral"}
    BLOCK_SUPPORTED_TYPES = {
        "qwen2", "qwen2.5", "llama", "llama2", "llama3", "mistral",
        "gemma", "gemma2", "phi", "phi3",
    }

    @classmethod
    def resolve(
        cls,
        model_id_or_path: str,
        requested_strategy: PruningStrategy = PruningStrategy.AUTO,
    ) -> Tuple[PruningStrategy, str, ModelInspectionReport]:
        """
        Resolves the pruning strategy.

        Returns:
            Tuple of (resolved_strategy, rationale, inspection_report)
        """
        report = ModelInspector.inspect(model_id_or_path)
        desc = report.descriptor

        # 1. Reject unsupported architectures (MoE, Quantized, Multimodal)
        if desc.is_moe:
            raise ValueError(
                f"Model '{model_id_or_path}' is a Mixture-of-Experts (MoE) architecture ({desc.model_type}). "
                "Structured weight pruning is only supported on dense decoder models (Qwen2, Llama, Mistral)."
            )
        if desc.quantization_type is not None:
            raise ValueError(
                f"Model '{model_id_or_path}' is pre-quantized ({desc.quantization_type}). "
                "PruneQuest structural pruning requires continuous floating-point weights (bfloat16/float16). "
                "Please supply the base unquantized checkpoint."
            )
        if desc.is_multimodal:
            raise ValueError(
                f"Model '{model_id_or_path}' is a multimodal architecture. "
                "PruneQuest currently supports causal language models."
            )
        if desc.architecture_class == ArchitectureClass.UNKNOWN and report.capability_level == CapabilityLevel.LEVEL_0_UNSUPPORTED:
            # Check if notes have useful information
            msg = "; ".join(report.notes) if report.notes else "Unknown architecture"
            raise ValueError(f"Could not inspect model '{model_id_or_path}': {msg}")

        model_type = desc.model_type.lower()
        supports_flap = model_type in cls.FLAP_SUPPORTED_TYPES
        supports_block = model_type in cls.BLOCK_SUPPORTED_TYPES

        # 2. Strategy Resolution
        if requested_strategy == PruningStrategy.AUTO:
            if supports_flap and supports_block:
                return (
                    PruningStrategy.JOINT,
                    f"Selected Joint Optimization (JIPO): Model '{desc.model_type}' has verified "
                    "support for both SwiGLU MLP width pruning and Transformer depth pruning.",
                    report,
                )
            elif supports_block:
                return (
                    PruningStrategy.BLOCK,
                    f"Selected Block Influence: Model '{desc.model_type}' supports structural depth pruning "
                    "(FLAP width pruning is currently unverified for this architecture).",
                    report,
                )
            elif supports_flap:
                return (
                    PruningStrategy.FLAP,
                    f"Selected FLAP: Model '{desc.model_type}' supports SwiGLU MLP intermediate width pruning.",
                    report,
                )
            else:
                raise ValueError(
                    f"Model architecture '{desc.model_type}' is not supported for automated pruning.\n"
                    "Verified model families: Qwen2/2.5, Llama 2/3, Mistral.\n"
                    "Experimental families: Gemma, Phi."
                )

        elif requested_strategy == PruningStrategy.JOINT:
            if not (supports_flap and supports_block):
                raise ValueError(
                    f"Joint optimization requires support for both FLAP and Block pruning. "
                    f"Model '{desc.model_type}' does not meet this requirement (FLAP: {supports_flap}, Block: {supports_block})."
                )
            return (
                PruningStrategy.JOINT,
                f"User requested Joint optimization: Verified support for '{desc.model_type}'.",
                report,
            )

        elif requested_strategy == PruningStrategy.FLAP:
            if not supports_flap:
                raise ValueError(
                    f"FLAP width pruning is not supported for architecture '{desc.model_type}'. "
                    f"Supported families: {', '.join(sorted(cls.FLAP_SUPPORTED_TYPES))}."
                )
            return (
                PruningStrategy.FLAP,
                f"User requested FLAP width pruning: Verified support for '{desc.model_type}'.",
                report,
            )

        elif requested_strategy == PruningStrategy.BLOCK:
            if not supports_block:
                raise ValueError(
                    f"Block Influence depth pruning is not supported for architecture '{desc.model_type}'. "
                    f"Supported families: {', '.join(sorted(cls.BLOCK_SUPPORTED_TYPES))}."
                )
            return (
                PruningStrategy.BLOCK,
                f"User requested Block Influence depth pruning: Verified support for '{desc.model_type}'.",
                report,
            )

        raise ValueError(f"Unknown strategy: '{requested_strategy}'")
