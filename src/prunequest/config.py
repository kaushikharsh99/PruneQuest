"""
Configuration management using Pydantic and PyYAML.
Enforces strict schema validation, type checking, and boundary constraints.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
import yaml
import os


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(default="Qwen/Qwen2.5-Math-1.5B-Instruct", min_length=1)
    torch_dtype: str = "bfloat16"
    device: str = "cuda:0"
    trust_remote_code: bool = False

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if not (v_clean.startswith("cuda") or v_clean == "cpu"):
            raise ValueError(f"Invalid device '{v}'. Must be 'cpu' or start with 'cuda' (e.g. 'cuda:0').")
        return v


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="gsm8k", min_length=1)
    split: str = Field(default="dev", description="Evaluation split (defaults to 'dev' for safe retention measurement)")
    seed: int = 42
    calibration_size: int = Field(default=256, gt=0)
    dev_size: int = Field(default=512, gt=0)
    smoke_size: int = Field(default=16, gt=0)
    screening_size: int = Field(default=64, gt=0)
    sample_limit: Optional[int] = Field(default=None, gt=0)
    max_new_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    do_sample: bool = False

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        allowed = {"dev", "acceptance", "smoke", "screening", "test", "full", "calibration"}
        if v.strip().lower() not in allowed:
            raise ValueError(f"Unknown benchmark split '{v}'. Allowed splits: {sorted(allowed)}")
        return v.strip().lower()


class PruningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = "greedy"
    backend: str = "layer"
    metric: str = "block_influence"
    max_iterations: int = Field(default=5, gt=0)
    max_candidates: int = Field(default=3, gt=0)
    retain: float = Field(default=0.99, ge=0.0, le=1.0, description="Minimum fraction of baseline accuracy to retain")
    time_budget_sec: Optional[int] = Field(default=None, gt=0)
    dry_run: bool = False


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_vram_gb: float = Field(default=5.5, gt=0.0)
    clear_cache: bool = True
    cpu_offload_validation: bool = True


class PruneQuestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    output_dir: str = Field(default="runs/prunequest_run", min_length=1)
    reports_dir: str = Field(default="reports", min_length=1)

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
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
