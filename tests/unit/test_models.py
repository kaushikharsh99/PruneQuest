"""
Unit tests for model adapter using a lightweight synthetic Qwen2 model.
"""

import pytest
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer

from prunequest.models.qwen2 import Qwen2Adapter
from prunequest.models.registry import ModelRegistry


@pytest.fixture
def synthetic_qwen2():
    cfg = Qwen2Config(
        vocab_size=600,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def test_qwen2_adapter_inspection(synthetic_qwen2):
    adapter = Qwen2Adapter(model_id_or_path="dummy", device="cpu")
    adapter.model = synthetic_qwen2
    meta = adapter.get_metadata()

    assert meta.num_layers == 4
    assert meta.architecture == "Qwen2ForCausalLM"
    assert meta.total_params > 0
    assert meta.hidden_size == 64


def test_qwen2_adapter_prune_layer(synthetic_qwen2):
    adapter = Qwen2Adapter(model_id_or_path="dummy", device="cpu")
    adapter.model = synthetic_qwen2

    init_layers = len(adapter.get_layers())
    assert init_layers == 4

    p_before, p_after = adapter.prune_layer(1)
    assert p_after < p_before
    assert len(adapter.get_layers()) == 3
    assert adapter.model.config.num_hidden_layers == 3

    # Check layer re-indexing
    for idx, layer in enumerate(adapter.get_layers()):
        assert layer.layer_idx == idx
        assert layer.self_attn.layer_idx == idx

    valid, err = adapter.validate_architecture()
    assert valid is True
    assert err is None


def test_model_registry_resolution():
    adapter = ModelRegistry.get_adapter("Qwen/Qwen2.5-Math-1.5B-Instruct", device="cpu")
    assert isinstance(adapter, Qwen2Adapter)
