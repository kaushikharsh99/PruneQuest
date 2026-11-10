"""
Configuration schema for FLAP-based incremental structured pruning.
Enforces strict boundary checks and forbids unknown configuration keys.
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


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(default="openai/gsm8k", min_length=1)
    dataset_config: str = "main"
    split: str = "train"
    seed: int = 42
    num_samples: int = Field(default=128, gt=0)
    screening_samples: int = Field(default=64, gt=0)
    acceptance_samples: int = Field(default=512, gt=0)
    max_length: int = Field(default=512, gt=0)
    batch_size: int = Field(default=1, gt=0)


class FLAPMetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(default="wifv", description="Fluctuation metric: 'wifv' (variance) or 'wifn' (norm/mean)")
    standardization: str = Field(default="z_score", description="Score standardization across layers: 'z_score', 'min_max', or 'none'")
    allocation: str = Field(default="uniform", description="Pruning allocation: 'uniform' or 'global_adaptive'")
    compensation: str = Field(default="none", description="Bias compensation mode: 'none' or 'bias'")


class PruningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = "mlp_intermediate"
    initial_step_ratio: float = Field(default=0.05, gt=0.0, le=1.0)
    minimum_step_ratio: float = Field(default=0.006, gt=0.0, le=1.0)
    maximum_total_ratio: float = Field(default=0.25, gt=0.0, le=1.0)
    adaptive_step: bool = True
    backoff_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    retention: float = Field(default=0.99, ge=0.0, le=1.0, description="Minimum relative accuracy retention")
    max_candidate_evaluations: int = Field(default=30, gt=0)
    time_budget_sec: Optional[int] = Field(default=None, gt=0)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str = Field(default="gsm8k", min_length=1)
    policy: str = Field(default="measured", description="Evaluation policy: 'measured' or 'conservative'")
    batch_size: Optional[int] = Field(default=None, gt=0, description="Explicit evaluation batch size")
    max_batch_size: int = Field(default=64, gt=0, description="Upper bound for auto-detected batch size")
    max_new_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    do_sample: bool = False
    screening_loss_factor: float = Field(default=2.5, gt=0.0)
    screening_loss_margin: float = Field(default=1.0, ge=0.0)
    validation_max_tokens: int = Field(default=5, gt=0)
    validation_prompt: str = Field(default="The sum of 10 and 20 is", min_length=1)


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_vram_gb: float = Field(default=5.5, gt=0.0)
    clear_cache: bool = True
    cpu_offload_validation: bool = True


class FLAPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    flap: FLAPMetricConfig = Field(default_factory=FLAPMetricConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    output_dir: str = Field(default="runs/qwen_math_flap", min_length=1)
    reports_dir: str = Field(default="reports", min_length=1)

    @classmethod
    def from_yaml(cls, path: str) -> "FLAPConfig":
        resolved_path = path
        if not os.path.exists(resolved_path):
            repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidate = os.path.join(repo_dir, path)
            if os.path.exists(candidate):
                resolved_path = candidate
            else:
                raise FileNotFoundError(f"Config file not found at '{path}' or '{candidate}'")
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
