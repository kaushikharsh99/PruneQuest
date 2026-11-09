"""
Unit tests for model architecture discovery and capability assessment.
"""

from unittest.mock import MagicMock, patch
from prunequest.models.discovery import (
    ModelInspector,
    ArchitectureClass,
    CapabilityLevel,
)


def create_mock_config(**kwargs):
    mock_cfg = MagicMock()
    mock_cfg.vision_config = None
    mock_cfg.quantization_config = None
    mock_cfg.is_encoder_decoder = False
    mock_cfg.num_experts = 0
    mock_cfg.n_routed_experts = 0
    mock_cfg.num_local_experts = 0
    for k, v in kwargs.items():
        setattr(mock_cfg, k, v)
    return mock_cfg


def test_discovery_qwen2():
    mock_cfg = create_mock_config(
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        num_hidden_layers=28,
        hidden_size=1536,
        num_attention_heads=12,
        num_key_value_heads=2,
        vocab_size=151936,
    )

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
        report = ModelInspector.inspect("Qwen/Qwen2.5-Math-1.5B-Instruct")
        assert report.descriptor.architecture_class == ArchitectureClass.DECODER_ONLY_DENSE
        assert report.capability_level == CapabilityLevel.LEVEL_3_VERIFIED_PRUNING
        assert report.adapter_name == "qwen2"
        assert "layer_pruning" in report.supported_operations


def test_discovery_llama():
    mock_cfg = create_mock_config(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32000,
    )

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
        report = ModelInspector.inspect("meta-llama/Llama-2-7b")
        assert report.descriptor.architecture_class == ArchitectureClass.DECODER_ONLY_DENSE
        assert report.capability_level == CapabilityLevel.LEVEL_3_VERIFIED_PRUNING
        assert report.adapter_name == "llama"


def test_discovery_quantized():
    mock_cfg = create_mock_config(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=32,
        quantization_config={"quant_method": "awq", "bits": 4},
    )

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
        report = ModelInspector.inspect("TheBloke/Llama-2-7B-AWQ")
        assert report.descriptor.architecture_class == ArchitectureClass.QUANTIZED
        assert report.capability_level == CapabilityLevel.LEVEL_1_EVALUATION
        assert "quantized" in report.notes[0].lower()


def test_discovery_moe():
    mock_cfg = create_mock_config(
        model_type="mixtral",
        architectures=["MixtralForCausalLM"],
        num_hidden_layers=32,
        num_local_experts=8,
    )

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
        report = ModelInspector.inspect("mistralai/Mixtral-8x7B-v0.1")
        assert report.descriptor.architecture_class == ArchitectureClass.DECODER_ONLY_MOE
        assert report.capability_level == CapabilityLevel.LEVEL_1_EVALUATION
