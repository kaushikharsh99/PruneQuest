"""
Configuration management using Pydantic and PyYAML.
"""

from typing import Optional
from pydantic import BaseModel, Field
import yaml
import os


class ModelConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"
    trust_remote_code: bool = False


class BenchmarkConfig(BaseModel):
    name: str = "gsm8k"
    split: str = "smoke"
    seed: int = 42
    calibration_size: int = 256
    dev_size: int = 512
    smoke_size: int = 16
    screening_size: int = 64
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False


class PruningConfig(BaseModel):
    strategy: str = "greedy"
    backend: str = "layer"
    metric: str = "block_influence"
    max_iterations: int = 5
    max_candidates: int = 3
    retain: float = Field(default=0.99, ge=0.0, le=1.0, description="Minimum fraction of baseline accuracy to retain")
    time_budget_sec: Optional[int] = None
    dry_run: bool = False


class HardwareConfig(BaseModel):
    max_vram_gb: float = 5.5
    clear_cache: bool = True
    cpu_offload_validation: bool = True


class PruneQuestConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    output_dir: str = "runs/prunequest_run"
    reports_dir: str = "reports"

    @classmethod
    def from_yaml(cls, path: str) -> "PruneQuestConfig":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)
