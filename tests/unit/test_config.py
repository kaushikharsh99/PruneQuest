"""
Unit tests for configuration validation and serialization.
"""

import tempfile
import pytest
from prunequest.config import PruneQuestConfig, ModelConfig, PruningConfig


def test_default_config():
    cfg = PruneQuestConfig()
    assert cfg.model.model_id == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    assert cfg.benchmark.name == "gsm8k"
    assert cfg.pruning.retain == 0.99
    assert cfg.pruning.strategy == "greedy"


def test_config_yaml_serialization():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        cfg = PruneQuestConfig(
            model=ModelConfig(model_id="test/model"),
            pruning=PruningConfig(retain=0.95, max_iterations=2),
        )
        cfg.to_yaml(tmp.name)

        loaded = PruneQuestConfig.from_yaml(tmp.name)
        assert loaded.model.model_id == "test/model"
        assert loaded.pruning.retain == 0.95
        assert loaded.pruning.max_iterations == 2


def test_config_validation():
    with pytest.raises(Exception):
        # retain must be <= 1.0 and >= 0.0
        PruningConfig(retain=1.5)
