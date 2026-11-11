"""
Tests for Studio SSE event streaming and replay.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from prunequest.orchestrator.optimizer import OptimizationResult
from prunequest.studio.server import create_app
from prunequest.studio.workers import JobManager
from prunequest.events.models import OptimizationEvent, EventCategory, EventPhase


def test_sse_event_streaming_and_replay():
    app = create_app()
    client = TestClient(app)

    mgr = JobManager(runs_root="runs")
    mgr.reset()

    with patch("prunequest.studio.workers.job_manager.UnifiedOptimizer.optimize") as mock_opt:
        mock_opt.return_value = OptimizationResult(
            status="completed",
            best_checkpoint=None,
            original_params=1000,
            final_params=800,
            params_pruned=200,
            compression_ratio=1.25,
            baseline_accuracy=0.8,
            final_accuracy=0.8,
            accuracy_retention=1.0,
            strategy_used="joint",
            iterations_completed=1,
            run_dir="runs/test_stream",
        )
        job = mgr.start_job({"model_id": "test/stream_model", "output_dir": "runs/test_stream"})

        # Add mock events to job
        ev1 = OptimizationEvent.create(
            run_id=job.run_id,
            sequence=1,
            event_type=EventCategory.RUN_STARTED,
            phase=EventPhase.PREFLIGHT,
            payload={"model_id": "test/stream_model"},
        )
        ev2 = OptimizationEvent.create(
            run_id=job.run_id,
            sequence=2,
            event_type=EventCategory.BASELINE_COMPLETED,
            phase=EventPhase.BASELINE,
            payload={"accuracy": 0.85},
        )
        job.add_event(ev1)
        job.add_event(ev2)
        job.status = "completed"

        # Request SSE stream
        res = client.get(f"/api/v1/runs/{job.run_id}/events")
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        text = res.text
        assert "run_started" in text or "RUN_STARTED" in text
        assert "baseline_completed" in text or "BASELINE_COMPLETED" in text

        # Test replay starting from event ID 1 (should only return event 2)
        res_replay = client.get(f"/api/v1/runs/{job.run_id}/events?last_event_id=1")
        assert res_replay.status_code == 200
        text_replay = res_replay.text
        assert "baseline_completed" in text_replay or "BASELINE_COMPLETED" in text_replay
