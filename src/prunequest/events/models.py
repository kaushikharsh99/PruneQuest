"""
Structured Optimization Event Models for PruneQuest.
Provides stable, typed event representation for observation, logging, and reporting.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from typing import Dict, Any, Optional
import uuid


class EventCategory:
    RUN_STARTED = "run_started"
    PREFLIGHT_COMPLETED = "preflight_completed"
    MODEL_LOADING_STARTED = "model_loading_started"
    MODEL_LOADED = "model_loaded"
    BASELINE_STARTED = "baseline_started"
    BASELINE_COMPLETED = "baseline_completed"
    PROFILING_STARTED = "profiling_started"
    PROFILING_PROGRESS = "profiling_progress"
    PROFILING_COMPLETED = "profiling_completed"
    CANDIDATE_PROPOSED = "candidate_proposed"
    CANDIDATE_SCREENING_STARTED = "candidate_screening_started"
    CANDIDATE_SCREENING_COMPLETED = "candidate_screening_completed"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_PROGRESS = "validation_progress"
    VALIDATION_COMPLETED = "validation_completed"
    CANDIDATE_ACCEPTED = "candidate_accepted"
    CANDIDATE_REJECTED = "candidate_rejected"
    ROLLBACK_COMPLETED = "rollback_completed"
    CHECKPOINT_EXPORT_STARTED = "checkpoint_export_started"
    CHECKPOINT_VERIFIED = "checkpoint_verified"
    CHECKPOINT_REJECTED = "checkpoint_rejected"
    REPORT_GENERATED = "report_generated"
    RUN_PAUSING = "run_pausing"
    RUN_INTERRUPTED = "run_interrupted"
    RUN_RESUMED = "run_resumed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

    # Dataset onboarding events (UX Release C)
    DATASET_INSPECTION_STARTED = "dataset_inspection_started"
    DATASET_INSPECTION_COMPLETED = "dataset_inspection_completed"
    FIELD_MAPPING_RESOLVED = "field_mapping_resolved"
    TASK_TYPE_RESOLVED = "task_type_resolved"
    SCORER_RESOLVED = "scorer_resolved"
    DATASET_PARTITIONING_STARTED = "dataset_partitioning_started"
    DATASET_PARTITIONING_COMPLETED = "dataset_partitioning_completed"
    BENCHMARK_VALIDATION_COMPLETED = "benchmark_validation_completed"
    DATASET_VALIDATION_FAILED = "dataset_validation_failed"


class EventPhase:
    PREFLIGHT = "preflight"
    MODEL_LOADING = "model_loading"
    BASELINE = "baseline"
    PROFILING = "profiling"
    SCREENING = "screening"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    REPORTING = "reporting"
    IDLE = "idle"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass
class OptimizationEvent:
    event_id: str
    run_id: str
    sequence: int
    timestamp: str
    event_type: str
    phase: str
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        run_id: str,
        sequence: int,
        event_type: str,
        phase: str,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> "OptimizationEvent":
        return cls(
            event_id=event_id or f"evt_{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            sequence=sequence,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            phase=phase,
            payload=payload or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationEvent":
        return cls(
            event_id=data.get("event_id", ""),
            run_id=data.get("run_id", ""),
            sequence=data.get("sequence", 0),
            timestamp=data.get("timestamp", ""),
            event_type=data.get("event_type", ""),
            phase=data.get("phase", ""),
            payload=data.get("payload") or {},
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OptimizationEvent":
        return cls.from_dict(json.loads(json_str))
