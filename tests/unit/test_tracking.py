"""
Unit tests for ExperimentTracker, Pareto updates, and CSV export.
"""

import os
import tempfile
from prunequest.core.tracking import ExperimentTracker


def test_tracker_logging_and_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir, experiment_name="test_exp")

        tracker.log_iteration({
            "iteration_id": "baseline",
            "params_after": 1500000000,
            "accuracy": 0.90,
            "status": "baseline",
        })

        tracker.log_iteration({
            "iteration_id": "iter_1",
            "params_after": 1400000000,
            "accuracy": 0.88,
            "status": "accepted",
        })

        csv_file = tracker.export_csv()
        assert os.path.exists(csv_file)

        df = tracker.get_history_dataframe()
        assert len(df) == 2

        baseline = tracker.get_baseline()
        assert baseline is not None
        assert baseline["iteration_id"] == "baseline"
        assert baseline["accuracy"] == 0.90
        assert baseline["params_after"] == 1500000000

        all_acc = tracker.get_all_accepted()
        assert len(all_acc) == 2


def test_pareto_frontier_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)
        checkpoints = [
            {"iteration_id": "c1", "params_after": 1000, "accuracy": 0.95},
            {"iteration_id": "c2", "params_after": 800, "accuracy": 0.90},
            {"iteration_id": "inferior", "params_after": 900, "accuracy": 0.85},
        ]
        tracker.update_pareto_frontier(checkpoints)

        import sqlite3
        with sqlite3.connect(tracker.db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT iteration_id FROM pareto_frontier").fetchall()
            ids = {r[0] for r in rows}
            assert "c1" in ids
            assert "c2" in ids
            assert "inferior" not in ids
