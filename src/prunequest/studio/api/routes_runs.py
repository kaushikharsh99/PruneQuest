"""
Runs management and real-time SSE streaming API routes for PruneQuest Studio.
"""

import asyncio
import json
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from prunequest.runs import RunManager
from prunequest.studio.workers import JobManager, JobConflictError

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])


@router.get("")
def list_runs(
    status: Optional[str] = None,
    model: Optional[str] = None,
    sort: str = "newest",
    limit: Optional[int] = None,
):
    """Lists previous and active runs with filtering and sorting."""
    runs = RunManager.list_runs(status=status, model=model, sort=sort, limit=limit)
    return [r.to_dict() for r in runs]


@router.get("/compare")
def compare_runs(run_a: str = Query(...), run_b: str = Query(...)):
    """Compares metrics, compression, and retention between two runs."""
    try:
        comp = RunManager.compare_runs(run_a, run_b)
        return comp.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{run_id}")
def get_run(run_id: str):
    """Returns detailed metrics, manifest metadata, and execution history for a run."""
    mgr = JobManager()
    active_job = mgr.get_job(run_id)

    try:
        details = RunManager.get_run(run_id)
        data = details.to_dict()
        if active_job and active_job.status in ("running", "pending", "cancelling"):
            data["status"] = active_job.status
            data["progress"] = active_job.progress
        return data
    except Exception as e:
        if active_job:
            return {
                "run_id": active_job.run_id,
                "status": active_job.status,
                "model_id": active_job.request_params.get("model_id"),
                "benchmark": active_job.request_params.get("benchmark"),
                "preset": active_job.request_params.get("preset"),
                "device": active_job.request_params.get("device"),
                "progress": active_job.progress,
                "created_at": active_job.created_at,
                "error": active_job.error,
            }
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found: {e}")


@router.get("/{run_id}/report")
def get_run_report(run_id: str):
    """Retrieves generated Markdown summary report if present."""
    try:
        run_dir = RunManager.resolve_run_dir(run_id)
        summary_path = os.path.join(run_dir, "reports", "summary.md")
        if not os.path.exists(summary_path):
            raise HTTPException(status_code=404, detail="Summary report not found for this run.")
        with open(summary_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"run_id": run_id, "summary_markdown": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str):
    """Requests safe cancellation of an active optimization run."""
    mgr = JobManager()
    success = mgr.cancel_job(run_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"No active job found for '{run_id}'.")
    return {"status": "cancelling", "run_id": run_id, "message": "Cancellation requested."}


@router.post("/{run_id}/resume")
def resume_run(run_id: str):
    """Safely resumes an interrupted run from its last valid checkpoint."""
    mgr = JobManager()
    try:
        run_dir = RunManager.resolve_run_dir(run_id)
        details = RunManager.get_run(run_dir)
        cfg_path = os.path.join(run_dir, "resolved_config.yaml")

        params = {
            "model_id": details.model_id,
            "benchmark": details.benchmark,
            "retain": details.retention_target_pct / 100.0,
            "preset": details.preset,
            "strategy": details.strategy,
            "device": details.device,
            "output_dir": run_dir,
            "config": cfg_path if os.path.exists(cfg_path) else None,
            "resume": True,
        }
        job = mgr.start_job(params)
        return {
            "job_id": job.job_id,
            "run_id": job.run_id,
            "status": job.status,
            "message": f"Successfully resumed run '{run_id}'.",
        }
    except JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{run_id}/events")
async def stream_events(
    run_id: str,
    request: Request,
    last_event_id: Optional[int] = Query(None),
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Streams real-time events and replays missed events based on Last-Event-ID.
    """
    mgr = JobManager()
    active_job = mgr.get_job(run_id)

    # Resolve starting event ID
    header_last_id = request.headers.get("last-event-id")
    if header_last_id and header_last_id.isdigit():
        start_id = int(header_last_id)
    elif last_event_id is not None:
        start_id = last_event_id
    else:
        start_id = 0

    async def event_generator():
        # 1. Replay historical events
        historical_events = []
        if active_job:
            with active_job._lock:
                historical_events = list(active_job.events)
        else:
            try:
                historical_events = RunManager.get_run_logs(run_id)
            except Exception:
                pass

        event_counter = 0
        for ev in historical_events:
            event_counter += 1
            if event_counter > start_id:
                data_str = json.dumps(ev.to_dict())
                yield f"id: {event_counter}\nevent: message\ndata: {data_str}\n\n"

        # 2. If the job is active, stream live events
        if active_job and active_job.status in ("pending", "running", "cancelling"):
            q = asyncio.Queue()
            active_job.add_listener(q)
            try:
                while active_job.status in ("pending", "running", "cancelling") or not q.empty():
                    # Handle client disconnect
                    if await request.is_disconnected():
                        break

                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=2.0)
                        event_counter += 1
                        data_str = json.dumps(ev.to_dict())
                        yield f"id: {event_counter}\nevent: message\ndata: {data_str}\n\n"
                    except asyncio.TimeoutError:
                        # Periodic keepalive ping
                        yield ": ping\n\n"
            finally:
                active_job.remove_listener(q)

        # 3. Final completion event
        yield f"id: {event_counter + 1}\nevent: end\ndata: {{\"status\": \"stream_ended\"}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
