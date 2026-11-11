"""
Unit tests for the PruneQuest Unified Reporting Engine (UX Release B).
Tests offline report regeneration, Markdown formats, CSV candidate history,
and true non-dominated Pareto frontier computations.
"""

import os
import tempfile
from prunequest.runs import RunManager
from prunequest.reporting import UnifiedReportGenerator
from tests.fixtures.runs import create_mock_run


def test_pareto_frontier_calculation():
    # Test true non-dominated frontier:
    # Point A: 1,000,000 params, 95% acc (baseline)
    # Point B: 900,000 params, 94.5% acc (accepted) -> non-dominated
    # Point C: 850,000 params, 93.0% acc (accepted) -> non-dominated
    # Point D: 850,000 params, 91.0% acc (dominated by C because same params but lower acc)
    # Point E: 900,000 params, 94.0% acc (dominated by B because same params but lower acc)

    records = [
        {"params_after": 1000000, "accuracy": 0.950, "status": "baseline"},
        {"params_after": 900000, "accuracy": 0.945, "status": "accepted"},
        {"params_after": 900000, "accuracy": 0.940, "status": "rejected"},
        {"params_after": 850000, "accuracy": 0.930, "status": "accepted"},
        {"params_after": 850000, "accuracy": 0.910, "status": "rejected"},
    ]

    pareto = UnifiedReportGenerator._compute_pareto_frontier(records)
    # Pareto set should contain Point A (1000k, 0.95), Point B (900k, 0.945), and Point C (850k, 0.930)
    assert len(pareto) == 3
    accuracies = [p["accuracy"] for p in pareto]
    assert 0.950 in accuracies
    assert 0.945 in accuracies
    assert 0.930 in accuracies
    # Dominated points must not appear
    assert 0.940 not in accuracies
    assert 0.910 not in accuracies


def test_offline_report_regeneration():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_mock_run(tmpdir, "report_regen_run", compression_pct=20.0, retention_pct=98.5)

        run_details = RunManager.get_run(run_dir)
        reports = UnifiedReportGenerator.generate_all_reports(run_details)

        # Check generated artifacts
        assert os.path.exists(reports["summary"])
        assert os.path.exists(reports["detailed"])
        assert os.path.exists(reports["history_csv"])

        with open(reports["summary"], "r", encoding="utf-8") as f:
            summary_content = f.read()
        assert "PruneQuest — Optimization Summary" in summary_content
        assert "-20.00%" in summary_content
        assert "AutoModelForCausalLM.from_pretrained" in summary_content

        with open(reports["detailed"], "r", encoding="utf-8") as f:
            detailed_content = f.read()
        assert "PruneQuest — Detailed Experiment Report" in detailed_content
        assert "Optimizer Efficiency Accounting" in detailed_content

        with open(reports["history_csv"], "r", encoding="utf-8") as f:
            csv_content = f.read()
        assert "iteration_id" in csv_content
