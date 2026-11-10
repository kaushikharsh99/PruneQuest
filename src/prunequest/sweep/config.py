"""
Configuration schema for Joint Incremental Pruning Optimizer (JIPO) Full Sweep.
Enforces strict boundary checks and forbids unknown configuration keys.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
import yaml
import os

from prunequest.flap.config import ModelConfig, CalibrationConfig, FLAPMetricConfig, EvaluationConfig, HardwareConfig


class JointPruningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_flap_ratio: float = Field(default=0.05, gt=0.0, le=1.0, description="Initial channel pruning step ratio per iteration")
    minimum_flap_ratio: float = Field(default=0.002, gt=0.0, le=1.0, description="Minimum channel pruning step ratio before terminating FLAP")
    maximum_total_flap_ratio: float = Field(default=0.25, gt=0.0, le=1.0, description="Maximum cumulative MLP width pruning ratio")
    backoff_factor: float = Field(default=0.5, gt=0.0, lt=1.0, description="Reduction factor applied to FLAP step on rejection")
    max_block_removals: int = Field(default=4, ge=0, description="Maximum total transformer blocks to excise")
    min_remaining_layers: int = Field(default=4, gt=0, description="Minimum transformer layers that must be retained")
    retention: float = Field(default=0.95, ge=0.0, le=1.0, description="Minimum benchmark accuracy retention ratio")
    max_iterations: int = Field(default=10, gt=0, description="Maximum joint optimization iterations")
    time_budget_sec: Optional[int] = Field(default=None, gt=0, description="Maximum search time in seconds")
    beam_width: int = Field(default=1, gt=0, description="Search beam width")
    screening_loss_factor: float = Field(default=2.5, gt=0.0, description="Screening loss multiplier threshold")
    screening_loss_margin: float = Field(default=0.8, ge=0.0, description="Screening loss additive margin")


class JointSweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    flap: FLAPMetricConfig = Field(default_factory=FLAPMetricConfig)
    pruning: JointPruningConfig = Field(default_factory=JointPruningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    output_dir: str = Field(default="runs/full_sweep", min_length=1)
    reports_dir: str = Field(default="reports", min_length=1)

    @classmethod
    def from_yaml(cls, path: str) -> "JointSweepConfig":
        resolved_path = path
        if not os.path.exists(resolved_path):
            repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidate = os.path.join(repo_dir, path)
            if os.path.exists(candidate):
                resolved_path = candidate
            else:
                raise FileNotFoundError(f"Configuration file not found at '{path}' or '{candidate}'")
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
