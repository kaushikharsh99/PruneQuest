"""
Durable local job manager for PruneQuest Studio.
Handles single-GPU concurrency, background thread execution,
safe cancellation, event replay for SSE, and job status tracking.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import threading
import time
from typing import Dict, Any, List, Optional
import uuid

from prunequest.events.models import OptimizationEvent, EventCategory, EventPhase
from prunequest.orchestrator.optimizer import UnifiedOptimizer
from prunequest.orchestrator.presets import OptimizationPreset
from prunequest.orchestrator.strategy import PruningStrategy


class JobConflictError(Exception):
    """Raised when attempting to start a new job while another job is active on the GPU."""
    pass


@dataclass
class StudioJob:
    job_id: str
    run_id: str
    status: str  # pending, running, cancelling, completed, failed, interrupted
    request_params: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    events: List[OptimizationEvent] = field(default_factory=list)
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        self._cancel_event = threading.Event()
        self._listeners: List[asyncio.Queue] = []
        self._lock = threading.Lock()

    def cancel(self):
        self._cancel_event.set()
        with self._lock:
            if self.status in ("pending", "running"):
                self.status = "cancelling"

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def add_event(self, event: OptimizationEvent):
        with self._lock:
            self.events.append(event)
            # Update fast progress snapshot
            if event.event_type in (EventCategory.VALIDATION_COMPLETED, EventCategory.CANDIDATE_SCREENING_COMPLETED):
                self.progress.update({
                    "current_phase": str(event.phase),
                    "measured_retention": event.payload.get("accuracy_retention", 0.0),
                    "best_compression": event.payload.get("best_compression", 0.0),
                    "step": event.payload.get("iteration", 0),
                })
            elif event.event_type == EventCategory.CANDIDATE_ACCEPTED:
                self.progress["last_decision"] = "accepted"
            elif event.event_type == EventCategory.CANDIDATE_REJECTED:
                self.progress["last_decision"] = "rejected"
            elif event.event_type == EventCategory.ROLLBACK_COMPLETED:
                self.progress["last_decision"] = "rollback"

        # Dispatch to async listeners if any
        for q in list(self._listeners):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def add_listener(self, queue: asyncio.Queue):
        with self._lock:
            self._listeners.append(queue)

    def remove_listener(self, queue: asyncio.Queue):
        with self._lock:
            if queue in self._listeners:
                self._listeners.remove(queue)


class JobManager:
    """
    Singleton manager tracking all studio jobs and enforcing single-GPU exclusivity.
    """

    _instance: Optional["JobManager"] = None
    _class_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, runs_root: str = "runs"):
        if self._initialized:
            return
        self.runs_root = runs_root
        self._active_job: Optional[StudioJob] = None
        self._jobs: Dict[str, StudioJob] = {}
        self._state_lock = threading.Lock()
        self._initialized = True

    def get_active_job(self) -> Optional[StudioJob]:
        with self._state_lock:
            if self._active_job and self._active_job.status in ("pending", "running", "cancelling"):
                return self._active_job
            return None

    def get_job(self, job_id_or_run_id: str) -> Optional[StudioJob]:
        with self._state_lock:
            if job_id_or_run_id in self._jobs:
                return self._jobs[job_id_or_run_id]
            for job in self._jobs.values():
                if job.run_id == job_id_or_run_id:
                    return job
            return None

    def list_jobs(self) -> List[StudioJob]:
        with self._state_lock:
            return list(self._jobs.values())

    def reset(self):
        """Resets active job and history for testing and state cleanup."""
        with self._state_lock:
            if self._active_job:
                self._active_job.cancel()
            self._active_job = None
            self._jobs.clear()

    def start_job(self, params: Dict[str, Any]) -> StudioJob:
        with self._state_lock:
            # Check single GPU concurrency
            if self._active_job and self._active_job.status in ("pending", "running", "cancelling"):
                raise JobConflictError(
                    f"An optimization job ('{self._active_job.job_id}') is already active on the device. "
                    "Studio enforces single-GPU exclusivity to prevent out-of-memory crashes."
                )

            job_id = f"job_{uuid.uuid4().hex[:8]}"
            model_id = params.get("model_id", "model")
            clean_model = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_id = f"{clean_model}_{timestamp}"

            output_dir = params.get("output_dir") or os.path.join(self.runs_root, run_id)
            params["output_dir"] = output_dir

            job = StudioJob(
                job_id=job_id,
                run_id=run_id,
                status="pending",
                request_params=params,
            )
            self._jobs[job_id] = job
            self._active_job = job

            # Spawn dedicated execution thread
            thread = threading.Thread(
                target=self._execute_job,
                args=(job,),
                daemon=True,
                name=f"StudioWorker-{job_id}",
            )
            thread.start()
            return job

    def cancel_job(self, job_id_or_run_id: str) -> bool:
        job = self.get_job(job_id_or_run_id)
        if not job:
            return False
        job.cancel()
        return True

    def _execute_job(self, job: StudioJob):
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.status = "running"
        params = job.request_params

        # Progress callback to stream events into the StudioJob
        def _on_event(event: OptimizationEvent):
            job.add_event(event)

        try:
            res = UnifiedOptimizer.optimize(
                model=params.get("model_id"),
                benchmark=params.get("benchmark", "gsm8k"),
                retain=float(params.get("retain", 0.99)),
                preset=OptimizationPreset(params.get("preset", "balanced")),
                strategy=PruningStrategy(params.get("strategy", "auto")),
                device=params.get("device", "auto"),
                output=params.get("output_dir"),
                batch_size=params.get("batch_size"),
                max_iterations=params.get("max_iterations"),
                dataset=params.get("dataset"),
                input_column=params.get("input_column"),
                target_column=params.get("target_column"),
                metric=params.get("metric"),
                calibration_dataset=params.get("calibration_dataset"),
                no_progress=True,
                on_progress=_on_event,
                cancel_check=job.is_cancelled,
            )
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.result = res.to_dict()

            if job.is_cancelled() or res.status == "interrupted":
                job.status = "interrupted"
            elif res.status == "failed":
                job.status = "failed"
            else:
                job.status = "completed"

        except Exception as e:
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.error = str(e)
            job.status = "interrupted" if job.is_cancelled() else "failed"
            # Emit run failure event
            fail_event = OptimizationEvent.create(
                run_id=job.run_id,
                sequence=len(job.events) + 1,
                event_type=EventCategory.RUN_FAILED,
                phase=EventPhase.INTERRUPTED if job.is_cancelled() else EventPhase.VALIDATION,
                payload={"error": str(e)},
            )
            job.add_event(fail_event)
        finally:
            with self._state_lock:
                if self._active_job == job:
                    self._active_job = None
