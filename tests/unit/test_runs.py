"""
Unit tests for PruneQuest Run Management (UX Release B).
Tests RunManager discovery, inspection, sorting/filtering, comparison, logs, and RunLock.
"""

import os
import json
import tempfile
import pytest
from prunequest.runs import RunManager, RunLock, RunLockError
from prunequest.events.models import EventCategory
from tests.fixtures.runs import create_mock_run


def test_run_manager_discovery_and_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 distinct runs
        create_mock_run(tmpdir, "run_1", model_id="Qwen/Qwen2.5-0.5B-Instruct", status="completed", compression_pct=10.0, timestamp="2026-11-11T10:00:00Z")
        create_mock_run(tmpdir, "run_2", model_id="meta-llama/Llama-3.2-1B-Instruct", status="interrupted", compression_pct=25.0, timestamp="2026-11-11T11:00:00Z")
        create_mock_run(tmpdir, "run_3", model_id="Qwen/Qwen2.5-Math-1.5B-Instruct", status="completed", compression_pct=15.0, timestamp="2026-11-11T12:00:00Z")

        all_runs = RunManager.list_runs(runs_root=tmpdir, sort="newest")
        assert len(all_runs) == 3
        # Check sorting by newest
        assert all_runs[0].run_id == "run_3"
        assert all_runs[2].run_id == "run_1"

        # Check sorting by compression
        comp_sorted = RunManager.list_runs(runs_root=tmpdir, sort="compression")
        assert comp_sorted[0].run_id == "run_2"  # 25%
        assert comp_sorted[1].run_id == "run_3"  # 15%
        assert comp_sorted[2].run_id == "run_1"  # 10%

        # Check filtering by status
        interrupted_runs = RunManager.list_runs(runs_root=tmpdir, status="interrupted")
        assert len(interrupted_runs) == 1
        assert interrupted_runs[0].run_id == "run_2"

        # Check filtering by model
        qwen_runs = RunManager.list_runs(runs_root=tmpdir, model="qwen")
        assert len(qwen_runs) == 2


def test_run_manager_get_run_details():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_mock_run(tmpdir, "test_inspect_run", compression_pct=18.72, retention_pct=99.21)

        # Inspect via run_id
        details = RunManager.get_run("test_inspect_run", runs_root=tmpdir)
        assert details.run_id == "test_inspect_run"
        assert details.compression_pct == 18.72
        assert details.accuracy_retention_pct == 99.21
        assert details.reload_verified is True
        assert details.schema_version == 2

        # Inspect via absolute path
        details_path = RunManager.get_run(run_dir)
        assert details_path.run_id == "test_inspect_run"


def test_run_manager_compare_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_mock_run(tmpdir, "run_gsm_1", benchmark="gsm8k", compression_pct=10.0, retention_pct=99.0)
        create_mock_run(tmpdir, "run_gsm_2", benchmark="gsm8k", compression_pct=15.0, retention_pct=98.5)
        create_mock_run(tmpdir, "run_arc_1", benchmark="arc_easy", compression_pct=20.0, retention_pct=97.0)

        # Compare compatible runs
        cmp_compat = RunManager.compare_runs("run_gsm_1", "run_gsm_2", runs_root=tmpdir)
        assert cmp_compat.is_compatible is True
        assert len(cmp_compat.compatibility_notes) == 0
        assert cmp_compat.compression_diff_pct == 5.0  # 15 - 10

        # Compare incompatible runs (different benchmarks)
        cmp_incompat = RunManager.compare_runs("run_gsm_1", "run_arc_1", runs_root=tmpdir)
        assert cmp_incompat.is_compatible is False
        assert any("Incompatible benchmarks" in note for note in cmp_incompat.compatibility_notes)


def test_run_manager_logs():
    with tempfile.TemporaryDirectory() as tmpdir:
        create_mock_run(tmpdir, "run_logs_test")

        logs = RunManager.get_run_logs("run_logs_test", runs_root=tmpdir)
        assert len(logs) == 3

        # Filter by phase
        val_logs = RunManager.get_run_logs("run_logs_test", phase="validation", runs_root=tmpdir)
        assert len(val_logs) == 1
        assert val_logs[0].event_type == EventCategory.CANDIDATE_ACCEPTED


def test_run_lock_concurrency_and_stale_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = os.path.join(tmpdir, "locked_run")

        # 1. Acquire lock
        lock1 = RunLock(run_dir)
        lock1.acquire()
        assert os.path.exists(lock1.lock_file)

        # 2. Acquiring with same active process is allowed or re-entrant
        # Simulate active external process holding lock
        lock_file_path = os.path.join(run_dir, ".run.lock")
        with open(lock_file_path, "w") as f:
            # PID 1 (init/systemd) is always alive on Linux
            json.dump({"pid": 1, "acquired_at": "2026-11-11T10:00:00Z"}, f)

        lock2 = RunLock(run_dir)
        with pytest.raises(RunLockError) as exc:
            lock2.acquire()
        assert "is locked by active process PID 1" in str(exc.value)

        # 3. Simulate dead process PID (PID 99999999 is dead)
        with open(lock_file_path, "w") as f:
            json.dump({"pid": 99999999, "acquired_at": "2026-11-11T10:00:00Z"}, f)

        # Dead PID stale lock should be reclaimed safely without raising RunLockError
        lock3 = RunLock(run_dir)
        lock3.acquire()
        assert lock3._acquired is True
        lock3.release()
        assert not os.path.exists(lock_file_path)
