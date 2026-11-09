"""
Configuration schema for Stage 1 FLAP-based incremental structured pruning.
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


class CalibrationConfig(BaseModel):
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    seed: int = 42
    num_samples: int = 128
    screening_samples: int = 64
    acceptance_samples: int = 512
    max_length: int = 512
    batch_size: int = 1


class FLAPConfig(BaseModel):
    metric: str = Field(default="wifv", description="Fluctuation metric: 'wifv' (variance) or 'wifn' (norm/mean)")
    standardization: str = Field(default="z_score", description="Score standardization across layers: 'z_score', 'min_max', or 'none'")
    allocation: str = Field(default="uniform", description="Pruning allocation: 'uniform' (same channels per layer for 100% standard HF compatibility) or 'global_adaptive'")
    compensation: str = Field(default="none", description="Bias compensation mode: 'none' (pure physical dense pruning) or 'bias' (if architecture/wrapper supports downstream bias)")


class PruningConfig(BaseModel):
    target: str = "mlp_intermediate"
    initial_step_ratio: float = 0.05
    minimum_step_ratio: float = 0.006
    maximum_total_ratio: float = 0.25
    adaptive_step: bool = True
    backoff_factor: float = 0.5
    retention: float = Field(default=0.99, ge=0.0, le=1.0, description="Minimum relative accuracy retention")
    max_candidate_evaluations: int = 30
    time_budget_sec: Optional[int] = None


class EvaluationConfig(BaseModel):
    benchmark: str = "gsm8k"
    policy: str = Field(default="measured", description="Evaluation policy: 'measured' or 'conservative'")
    batch_size: Optional[int] = Field(default=None, description="Explicit evaluation batch size (e.g. 32, 64). If None, auto-detected.")
    max_batch_size: int = Field(default=64, description="Upper bound for auto-detected batch size (e.g. 32 or 64)")
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False


class HardwareConfig(BaseModel):
    max_vram_gb: float = 5.5
    clear_cache: bool = True
    cpu_offload_validation: bool = True


class Stage1Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    flap: FLAPConfig = Field(default_factory=FLAPConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    output_dir: str = "runs/qwen_math_stage1"
    reports_dir: str = "reports"

    @classmethod
    def from_yaml(cls, path: str) -> "Stage1Config":
        resolved_path = path
        if not os.path.exists(resolved_path):
            # Search relative to package root if path is not found in CWD
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
