"""
Data models for experiment run management, inspection, and comparison.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class RunSummary:
    run_id: str
    model_id: str
    strategy: str
    benchmark: str
    status: str
    compression_pct: float
    accuracy_retention_pct: float
    best_checkpoint: Optional[str]
    created_at: str
    resumable: bool
    run_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunDetails:
    run_id: str
    schema_version: int
    model_id: str
    strategy: str
    benchmark: str
    preset: str
    device: str
    status: str
    resumable: bool
    retention_target_pct: float
    original_parameters: int
    final_parameters: int
    parameters_pruned: int
    compression_pct: float
    baseline_accuracy: float
    final_accuracy: float
    accuracy_retention_pct: float
    best_checkpoint: Optional[str]
    reload_verified: bool
    elapsed_time_sec: float
    created_at: str
    run_dir: str
    reports_dir: str
    resolved_config_path: Optional[str] = None
    stopping_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunComparison:
    run_a: RunDetails
    run_b: RunDetails
    is_compatible: bool
    compatibility_notes: List[str]
    param_reduction_diff: int
    compression_diff_pct: float
    accuracy_diff: float
    retention_diff_pct: float
    elapsed_time_diff_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_a": self.run_a.to_dict(),
            "run_b": self.run_b.to_dict(),
            "is_compatible": self.is_compatible,
            "compatibility_notes": self.compatibility_notes,
            "param_reduction_diff": self.param_reduction_diff,
            "compression_diff_pct": self.compression_diff_pct,
            "accuracy_diff": self.accuracy_diff,
            "retention_diff_pct": self.retention_diff_pct,
            "elapsed_time_diff_sec": self.elapsed_time_diff_sec,
        }
