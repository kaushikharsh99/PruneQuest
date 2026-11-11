"""
Model discovery and inspection API routes for PruneQuest Studio.
"""

from typing import List, Dict, Any
from fastapi import APIRouter
from prunequest.models.discovery import ModelInspector, CapabilityLevel
from prunequest.studio.schemas import ModelInspectRequest, ModelInspectResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])

POPULAR_MODELS: List[Dict[str, Any]] = [
    {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen 2.5 0.5B Instruct",
        "family": "Qwen2",
        "params": "0.49B",
        "min_vram_gb": 3.0,
        "recommended": True,
    },
    {
        "model_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "name": "Qwen 2.5 Math 1.5B Instruct",
        "family": "Qwen2",
        "params": "1.54B",
        "min_vram_gb": 4.5,
        "recommended": True,
    },
    {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama 3.2 1B Instruct",
        "family": "Llama",
        "params": "1.23B",
        "min_vram_gb": 4.0,
        "recommended": True,
    },
    {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "name": "Llama 3.2 3B Instruct",
        "family": "Llama",
        "params": "3.21B",
        "min_vram_gb": 7.5,
        "recommended": False,
    },
    {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "name": "Mistral 7B v0.1",
        "family": "Mistral",
        "params": "7.24B",
        "min_vram_gb": 14.0,
        "recommended": False,
    },
]


@router.get("/popular")
def get_popular_models():
    """Returns curated list of popular verified models."""
    return POPULAR_MODELS


@router.post("/inspect", response_model=ModelInspectResponse)
def inspect_model(req: ModelInspectRequest):
    """
    Inspects model architecture, parameters, and compatibility without downloading weights.
    """
    try:
        report = ModelInspector.inspect(req.model_id)
        desc = report.descriptor
        raw_cfg = desc.raw_config or {}

        # Parameter approximation
        num_layers = desc.num_hidden_layers
        hidden_size = desc.hidden_size
        intermediate_size = raw_cfg.get("intermediate_size") or (hidden_size * 4 if hidden_size else None)
        vocab_size = desc.vocab_size

        estimated_params = None
        if num_layers and hidden_size and vocab_size:
            # Standard transformer formula: embeddings + layers * (attn + mlp)
            layer_params = 4 * (hidden_size ** 2) + (3 * hidden_size * intermediate_size if intermediate_size else 8 * hidden_size ** 2)
            estimated_params = vocab_size * hidden_size + num_layers * layer_params

        weights_gb = round(estimated_params * 2 / 1e9, 2) if estimated_params else None
        is_supported = report.capability_level in (
            CapabilityLevel.LEVEL_2_EXPERIMENTAL_PRUNING,
            CapabilityLevel.LEVEL_3_VERIFIED_PRUNING,
        )

        recommendation = None
        if not is_supported:
            recommendation = (
                f"Model architecture '{desc.model_type}' is assessed at capability {report.capability_level.name}. "
                "Supported families include Qwen2/2.5, Llama-2/3, Mistral, Gemma, and Phi."
            )

        return ModelInspectResponse(
            model_id=req.model_id,
            supported=is_supported,
            architecture=desc.model_type,
            tier=report.capability_level.name,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=desc.num_attention_heads,
            estimated_params=estimated_params,
            estimated_weights_gb=weights_gb,
            recommendations=recommendation,
        )
    except Exception as e:
        return ModelInspectResponse(
            model_id=req.model_id,
            supported=False,
            architecture="unknown",
            tier="LEVEL_0_UNSUPPORTED",
            error=str(e),
            recommendations=f"Could not inspect model config: {e}. Check that model ID is correct and accessible.",
        )
