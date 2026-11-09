"""
Unit tests for PruneQuest Python API and CheckpointExporter.
"""

import tempfile
from pathlib import Path
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from prunequest.api import PruneQuest
from prunequest.models.qwen2 import Qwen2Adapter
from prunequest.export.checkpoint import CheckpointExporter


def test_api_initialization():
    pq = PruneQuest(
        model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        benchmark="gsm8k",
        retain=0.99,
        device="cpu",
    )
    assert pq.config.model.model_id == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    assert pq.config.pruning.retain == 0.99
    assert pq.config.model.device == "cpu"


def test_checkpoint_exporter():
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = Qwen2ForCausalLM(cfg)
    adapter = Qwen2Adapter("synthetic", device="cpu")
    adapter.model = model

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = CheckpointExporter(checkpoints_dir=tmpdir)
        target_dir = exporter.export(adapter, checkpoint_name="ckpt_test", metadata={"test_run": True})

        out_path = Path(target_dir)
        assert out_path.exists()
        assert (out_path / "config.json").exists()
        assert (out_path / "model.safetensors").exists() or (out_path / "pytorch_model.bin").exists()
        assert (out_path / "prunequest_metadata.json").exists()
