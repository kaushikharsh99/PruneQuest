"""
Unit tests for Llama and Mistral model adapters with synthetic architectures.
"""

from unittest.mock import MagicMock
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM

from prunequest.models.llama import LlamaAdapter
from prunequest.models.mistral import MistralAdapter


def test_llama_adapter_physical_pruning():
    cfg = LlamaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = LlamaForCausalLM(cfg)
    adapter = LlamaAdapter("synthetic_llama", device="cpu")
    adapter.model = model
    adapter.tokenizer = MagicMock()

    layers = adapter.get_layers()
    assert len(layers) == 4

    p_before, p_after = adapter.prune_layer(1)
    assert p_after < p_before
    assert len(adapter.get_layers()) == 3
    assert adapter.model.config.num_hidden_layers == 3

    valid, err = adapter.validate_architecture()
    assert valid is True
    assert err is None


def test_mistral_adapter_temporary_ablation():
    cfg = MistralConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = MistralForCausalLM(cfg)
    adapter = MistralAdapter("synthetic_mistral", device="cpu")
    adapter.model = model
    adapter.tokenizer = MagicMock()

    assert len(adapter.get_layers()) == 3

    with adapter.temporary_ablation("layer", 1):
        assert len(adapter.get_layers()) == 2
        assert adapter.model.config.num_hidden_layers == 2

    # Guaranteed exact restoration upon exiting context manager
    assert len(adapter.get_layers()) == 3
    assert adapter.model.config.num_hidden_layers == 3
