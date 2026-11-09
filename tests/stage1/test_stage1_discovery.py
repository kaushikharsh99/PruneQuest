from unittest.mock import MagicMock, patch
from prunequest.stage1.models.discovery import ArchitectureInspector, MLPArchitectureInfo
from prunequest.stage1.models.registry import ModelRegistry
from prunequest.stage1.models.qwen import Qwen2MLPAdapter
from prunequest.stage1.models.llama import LlamaMLPAdapter


def test_inspect_qwen2():
    mock_config = MagicMock()
    mock_config.model_type = "qwen2"
    mock_config.num_hidden_layers = 28
    mock_config.hidden_size = 1536
    mock_config.intermediate_size = 8960

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
        info = ArchitectureInspector.inspect("Qwen/Qwen2.5-Math-1.5B-Instruct")
        assert isinstance(info, MLPArchitectureInfo)
        assert info.model_type == "qwen2"
        assert info.num_layers == 28
        assert info.intermediate_size == 8960
        assert info.mlp_type == "swiglu"
        assert info.is_supported is True


def test_inspect_llama():
    mock_config = MagicMock()
    mock_config.model_type = "llama"
    mock_config.num_hidden_layers = 32
    mock_config.hidden_size = 4096
    mock_config.intermediate_size = 11008

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
        info = ArchitectureInspector.inspect("meta-llama/Llama-2-7b-hf")
        assert info.is_supported is True
        assert info.mlp_type == "swiglu"


def test_inspect_unsupported():
    mock_config = MagicMock()
    mock_config.model_type = "t5"
    mock_config.num_hidden_layers = 12

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
        info = ArchitectureInspector.inspect("google/t5-base")
        assert info.is_supported is False


def test_model_registry():
    mock_config = MagicMock()
    mock_config.model_type = "qwen2"
    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
        cls = ModelRegistry.resolve_adapter_class("Qwen/Qwen2.5-Math-1.5B-Instruct")
        assert cls == Qwen2MLPAdapter

    mock_config.model_type = "llama"
    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
        cls = ModelRegistry.resolve_adapter_class("meta-llama/Llama-2-7b-hf")
        assert cls == LlamaMLPAdapter
