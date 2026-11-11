"""
Tests for Studio JobManager: concurrency, cancellation, and execution lifecycle.
"""

import time
from unittest.mock import patch
from prunequest.orchestrator.optimizer import OptimizationResult
from prunequest.studio.workers import JobManager, JobConflictError
import pytest


def test_job_manager_single_gpu_exclusivity():
    mgr = JobManager(runs_root="runs")
    mgr.reset()

    # Mock UnifiedOptimizer.optimize to run for a short duration
    with patch("prunequest.studio.workers.job_manager.UnifiedOptimizer.optimize") as mock_opt:
        def fake_optimize(*args, **kwargs):
            time.sleep(0.3)
            return OptimizationResult(
                status="completed",
                best_checkpoint="runs/test/best",
                original_params=1000,
                final_params=800,
                params_pruned=200,
                compression_ratio=1.25,
                baseline_accuracy=0.8,
                final_accuracy=0.8,
                accuracy_retention=1.0,
                strategy_used="joint",
                iterations_completed=1,
                run_dir="runs/test",
            )
        mock_opt.side_effect = fake_optimize

        job1 = mgr.start_job({"model_id": "test/model", "output_dir": "runs/test_job_1"})
        assert job1.status in ("pending", "running")

        # Starting a second concurrent job must raise JobConflictError
        with pytest.raises(JobConflictError):
            mgr.start_job({"model_id": "test/model2", "output_dir": "runs/test_job_2"})

        # Wait for job1 to conclude
        time.sleep(0.5)
        assert job1.status == "completed"
        assert mgr.get_active_job() is None


def test_job_manager_cancellation():
    mgr = JobManager(runs_root="runs")
    mgr.reset()

    with patch("prunequest.studio.workers.job_manager.UnifiedOptimizer.optimize") as mock_opt:
        def fake_optimize(*args, **kwargs):
            cancel_check = kwargs.get("cancel_check")
            # Emulate polling cancel_check
            for _ in range(50):
                if cancel_check and cancel_check():
                    return OptimizationResult(
                        status="interrupted",
                        best_checkpoint="runs/test_cancel/best",
                        original_params=1000,
                        final_params=900,
                        params_pruned=100,
                        compression_ratio=1.1,
                        baseline_accuracy=0.8,
                        final_accuracy=0.8,
                        accuracy_retention=1.0,
                        strategy_used="joint",
                        iterations_completed=1,
                        run_dir="runs/test_cancel",
                    )
                time.sleep(0.02)
            return OptimizationResult(status="completed", best_checkpoint=None, original_params=0, final_params=0, params_pruned=0, compression_ratio=1.0, baseline_accuracy=0.0, final_accuracy=0.0, accuracy_retention=1.0, strategy_used="joint", iterations_completed=0, run_dir="")

        mock_opt.side_effect = fake_optimize

        job = mgr.start_job({"model_id": "test/cancel_model", "output_dir": "runs/test_cancel"})
        time.sleep(0.05)
        assert job.status in ("pending", "running")

        # Request cancellation
        success = mgr.cancel_job(job.job_id)
        assert success is True
        assert job.is_cancelled() is True

        time.sleep(0.3)
        assert job.status == "interrupted"
        assert mgr.get_active_job() is None
