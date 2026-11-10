"""
Configuration schema for Joint Incremental Pruning Optimizer (JIPO) Full Sweep.
"""

from typing import Optional
from pydantic import BaseModel, Field
import yaml
import os

from prunequest.flap.config import ModelConfig, CalibrationConfig, FLAPMetricConfig, EvaluationConfig, HardwareConfig


class JointPruningConfig(BaseModel):
    initial_flap_ratio: float = Field(default=0.05, description="Initial channel pruning step ratio per iteration (e.g. 5%)")
    minimum_flap_ratio: float = Field(default=0.002, description="Minimum channel pruning step ratio before terminating FLAP")
    maximum_total_flap_ratio: float = Field(default=0.25, description="Maximum cumulative MLP width pruning ratio")
    backoff_factor: float = Field(default=0.5, description="Reduction factor applied to FLAP step on rejection")
    max_block_removals: int = Field(default=4, description="Maximum total transformer blocks to excise")
    min_remaining_layers: int = Field(default=4, description="Minimum transformer layers that must be retained")
    retention: float = Field(default=0.95, ge=0.0, le=1.0, description="Minimum benchmark accuracy retention ratio (e.g. 0.95 for 95%)")
    max_iterations: int = Field(default=10, description="Maximum joint optimization iterations")
    time_budget_sec: Optional[int] = Field(default=None, description="Maximum search time in seconds")
    beam_width: int = Field(default=1, description="Search beam width (1 = greedy)")
    screening_loss_factor: float = Field(default=2.5, description="Screening loss multiplier threshold")
    screening_loss_margin: float = Field(default=0.8, description="Screening loss additive margin")


class JointSweepConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    flap: FLAPMetricConfig = Field(default_factory=FLAPMetricConfig)
    pruning: JointPruningConfig = Field(default_factory=JointPruningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    output_dir: str = "runs/full_sweep"
    reports_dir: str = "reports"

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
