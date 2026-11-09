"""
Unit tests for LayerPruner using synthetic model adapter.
"""

import pytest
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from prunequest.models.qwen2 import Qwen2Adapter
from prunequest.pruning.layers import LayerPruner


@pytest.fixture
def synthetic_adapter():
    cfg = Qwen2Config(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = Qwen2ForCausalLM(cfg)
    adapter = Qwen2Adapter("dummy", device="cpu")
    adapter.model = model
    return adapter


def test_layer_pruner_analysis_and_candidates(synthetic_adapter):
    pruner = LayerPruner()
    analysis = pruner.analyze(synthetic_adapter)
    assert "layer_rankings" in analysis
    assert len(analysis["layer_rankings"]) == 4

    cands = pruner.propose_candidates(synthetic_adapter, analysis, top_k=2)
    assert len(cands) == 2
    assert cands[0].pruning_type == "layer"
    assert "layer_idx" in cands[0].target_info


def test_layer_pruner_apply(synthetic_adapter):
    pruner = LayerPruner()
    analysis = pruner.analyze(synthetic_adapter)
    cands = pruner.propose_candidates(synthetic_adapter, analysis, top_k=1)

    result = pruner.apply_pruning(synthetic_adapter, cands[0])
    assert result.validation_passed is True
    assert result.params_removed > 0
    assert len(synthetic_adapter.get_layers()) == 3
