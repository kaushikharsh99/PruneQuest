"""
Optimization and preflight API routes for PruneQuest Studio.
"""

from fastapi import APIRouter, HTTPException
from prunequest.orchestrator.preflight import PreflightChecker
from prunequest.orchestrator.presets import OptimizationPreset
from prunequest.orchestrator.strategy import PruningStrategy
from prunequest.studio.schemas import (
    OptimizationPreflightRequest,
    OptimizationPreflightResponse,
    StartOptimizationRequest,
    StartOptimizationResponse,
)
from prunequest.studio.workers import JobManager, JobConflictError

router = APIRouter(prefix="/api/v1/optimization", tags=["optimization"])


@router.post("/preflight", response_model=OptimizationPreflightResponse)
def preflight_check(req: OptimizationPreflightRequest):
    """
    Evaluates hardware capacity, disk space, and model architecture before starting optimization.
    """
    try:
        plan = PreflightChecker.plan(
            model_id=req.model_id,
            benchmark=req.benchmark,
            retain=req.retain,
            preset=OptimizationPreset(req.preset),
            strategy=PruningStrategy(req.strategy),
            device=req.device,
            output_dir=req.output_dir or f"runs/{req.model_id.replace('/', '_')}",
            dataset=req.dataset,
            input_column=req.input_column,
            target_column=req.target_column,
            metric=req.metric,
            calibration_dataset=req.calibration_dataset,
        )

        disk_sufficient = plan.free_disk_gb >= plan.disk_needed_gb
        return OptimizationPreflightResponse(
            valid=True,
            model_id=plan.model_id,
            architecture=plan.model_type,
            strategy=plan.strategy.value,
            device=plan.device,
            weights_gb=plan.weights_gb,
            estimated_peak_memory_gb=plan.working_memory_gb,
            hardware_memory_gb=plan.free_vram_gb,
            preset=plan.preset.value,
            target_retention=plan.retention,
            max_iterations=plan.acceptance_samples,
            batch_size=32,
            disk_space_available_gb=plan.free_disk_gb,
            disk_space_required_gb=plan.disk_needed_gb,
            disk_sufficient=disk_sufficient,
            warnings=plan.warnings,
            errors=[],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/start", response_model=StartOptimizationResponse)
def start_optimization(req: StartOptimizationRequest):
    """
    Enqueues and starts a background optimization job with single-GPU concurrency safety.
    """
    mgr = JobManager()
    try:
        job = mgr.start_job(req.dict())
        return StartOptimizationResponse(
            job_id=job.job_id,
            run_id=job.run_id,
            status=job.status,
            message="Optimization job successfully started.",
        )
    except JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/active")
def get_active_optimization():
    """Returns the currently active job if any."""
    mgr = JobManager()
    job = mgr.get_active_job()
    if not job:
        return {"active": False, "job": None}
    return {
        "active": True,
        "job": {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "status": job.status,
            "model_id": job.request_params.get("model_id"),
            "benchmark": job.request_params.get("benchmark"),
            "created_at": job.created_at,
            "started_at": job.started_at,
            "progress": job.progress,
            "error": job.error,
        },
    }
