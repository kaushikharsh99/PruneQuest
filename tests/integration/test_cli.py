"""
Integration tests for CLI commands using Typer CliRunner.
Verifies all public commands, option aliases, config overrides, and error exits.
"""

import os
import re
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from prunequest.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PruneQuest" in result.stdout
    assert "doctor" in result.stdout
    assert "inspect" in result.stdout
    assert "run" in result.stdout
    assert "benchmarks" in result.stdout


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "PruneQuest version" in result.stdout


def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "PruneQuest Environment Diagnostics" in result.stdout
    assert "CUDA" in result.stdout
    assert "Python Version" in result.stdout
    assert "Verified Model Architectures" in result.stdout


def test_cli_benchmarks_list():
    result = runner.invoke(app, ["benchmarks", "list"])
    assert result.exit_code == 0
    assert "gsm8k" in result.stdout

    # Test documented --limit option
    result_limited = runner.invoke(app, ["benchmarks", "list", "--limit", "1"])
    assert result_limited.exit_code == 0
    assert "gsm8k" in result_limited.stdout


def test_cli_benchmarks_search():
    result = runner.invoke(app, ["benchmarks", "search", "gsm8k"])
    assert result.exit_code == 0
    assert "gsm8k" in result.stdout


def test_cli_benchmarks_validate_and_preview():
    spec_data = {
        "version": 1,
        "name": "dummy_test_benchmark",
        "dataset": {
            "path": "json",
            "data_files": "dummy.jsonl",
        },
        "prompt_template": "Question: {question}\nAnswer:",
        "scoring": {
            "metric": "exact_match",
        },
        "splits": {
            "dev": {"samples": 10},
            "test": {"samples": 20},
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.safe_dump(spec_data, f)
        spec_path = f.name

    try:
        # Test documented --spec option
        res_val = runner.invoke(app, ["benchmarks", "validate", "--spec", spec_path])
        assert res_val.exit_code == 0
        assert "is valid" in res_val.stdout

        # Test positional argument fallback
        res_val_pos = runner.invoke(app, ["benchmarks", "validate", spec_path])
        assert res_val_pos.exit_code == 0

        # Test non-existent file exits 1
        res_val_missing = runner.invoke(app, ["benchmarks", "validate", "--spec", "non_existent.yaml"])
        assert res_val_missing.exit_code != 0

        # Mock sample preview
        with patch("prunequest.benchmarks.spec.BenchmarkSpecCompiler.preview_spec") as mock_preview:
            mock_preview.return_value = [
                {"id": "test_0", "rendered_prompt": "Q: 1+1?", "gold_answer": "2"}
            ]
            res_prev = runner.invoke(app, ["benchmarks", "preview", "--spec", spec_path, "--limit", "1"])
            assert res_prev.exit_code == 0
            assert "Q: 1+1?" in res_prev.stdout
    finally:
        if os.path.exists(spec_path):
            os.remove(spec_path)


def test_cli_inspect():
    result = runner.invoke(app, ["inspect", "--model", "Qwen/Qwen2.5-Math-1.5B-Instruct"])
    assert result.exit_code == 0
    assert "Architecture Class" in result.stdout
    assert "Capability Level" in result.stdout


def test_cli_flap_subcommands():
    # Test FLAP doctor
    res_doc = runner.invoke(app, ["flap", "doctor"])
    assert res_doc.exit_code == 0
    assert "FLAP Pruning Environment Diagnostics" in res_doc.stdout

    # Test FLAP inspect
    res_insp = runner.invoke(app, ["flap", "inspect", "--model", "Qwen/Qwen2.5-Math-1.5B-Instruct"])
    assert res_insp.exit_code == 0
    assert "MLP Architecture Specifications" in res_insp.stdout

    # Test FLAP test with invalid split
    res_test_invalid = runner.invoke(app, ["flap", "test", "--checkpoint", "dummy_ckpt", "--split", "invalid_split"])
    assert res_test_invalid.exit_code != 0
    assert "Unknown split 'invalid_split'" in res_test_invalid.stdout
    assert "Available splits:" in res_test_invalid.stdout


def test_cli_run_config_precedence_and_resolved_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.yaml")
        run_output = os.path.join(tmpdir, "runs")

        cfg_content = {
            "model": {"model_id": "original/model", "device": "cpu"},
            "pruning": {"retain": 0.90, "max_iterations": 2},
            "output_dir": run_output,
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg_content, f)

        # Mock SearchController.run so we do not load real models
        with patch("prunequest.cli.SearchController") as mock_sc_cls:
            mock_sc = MagicMock()
            mock_sc_cls.return_value = mock_sc
            mock_sc.run.return_value = {
                "status": "completed",
                "original_params": 1000,
                "final_params": 900,
                "baseline_accuracy": 0.95,
                "final_accuracy": 0.94,
                "accuracy_retention_pct": 98.9,
                "iterations_completed": 1,
                "best_checkpoint": "/dummy/best",
            }
            mock_sc.tracker.get_history_dataframe.return_value = MagicMock(empty=True)

            # Override model via CLI --model
            result = runner.invoke(app, [
                "run",
                "--config", config_path,
                "--model", "override/model",
                "--output", run_output,
            ])
            assert result.exit_code == 0

            # Verify that SearchController received the overridden config
            call_cfg = mock_sc_cls.call_args[0][0]
            assert call_cfg.model.model_id == "override/model"
            assert call_cfg.pruning.retain == 0.90  # Preserved from config file

            # Verify resolved_config.yaml was saved
            resolved_file = os.path.join(run_output, "resolved_config.yaml")
            assert os.path.exists(resolved_file)
            with open(resolved_file) as f:
                saved_yaml = yaml.safe_load(f)
            assert saved_yaml["model"]["model_id"] == "override/model"


def test_cli_optimize_help():
    result = runner.invoke(app, ["optimize", "--help"])
    assert result.exit_code == 0
    clean_stdout = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.stdout)
    assert "benchmark" in clean_stdout
    assert "retain" in clean_stdout
    assert "preset" in clean_stdout
    assert "strategy" in clean_stdout
    assert "dry-run" in clean_stdout or "dry_run" in clean_stdout


def test_cli_optimize_dry_run():
    result = runner.invoke(app, [
        "optimize",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--dry-run",
        "--retain", "0.98",
        "--preset", "quick",
    ])
    assert result.exit_code == 0
    assert "Optimization Plan (Preflight)" in result.stdout
    assert "Preflight completed. No model weights modified." in result.stdout


def test_cli_optimize_mocked_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("prunequest.cli.UnifiedOptimizer.optimize") as mock_opt:
            mock_res = MagicMock()
            mock_res.status = "completed"
            mock_opt.return_value = mock_res

            result = runner.invoke(app, [
                "optimize",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--benchmark", "gsm8k",
                "--retain", "0.95",
                "--preset", "balanced",
                "--strategy", "joint",
                "--output", tmpdir,
            ])
            assert result.exit_code == 0
            mock_opt.assert_called_once()
            call_kwargs = mock_opt.call_args[1]
            assert call_kwargs["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
            assert call_kwargs["benchmark"] == "gsm8k"
            assert call_kwargs["retain"] == 0.95
            assert call_kwargs["preset"] == "balanced"
            assert call_kwargs["strategy"] == "joint"
            assert call_kwargs["output"] == tmpdir


def test_cli_runs_commands():
    from tests.fixtures.runs import create_mock_run
    with tempfile.TemporaryDirectory() as tmpdir:
        create_mock_run(tmpdir, "run_alpha", benchmark="gsm8k", compression_pct=15.0)
        create_mock_run(tmpdir, "run_beta", benchmark="gsm8k", compression_pct=22.0)

        # 1. runs list (terminal & json)
        res_list = runner.invoke(app, ["runs", "list", "--runs-dir", tmpdir], env={"COLUMNS": "140"})
        assert res_list.exit_code == 0
        assert "run_alpha" in res_list.stdout
        assert "run_beta" in res_list.stdout

        res_list_json = runner.invoke(app, ["runs", "list", "--runs-dir", tmpdir, "--json"])
        assert res_list_json.exit_code == 0
        import json
        data = json.loads(res_list_json.stdout)
        assert len(data) == 2

        # 2. runs show (terminal & json)
        res_show = runner.invoke(app, ["runs", "show", "run_alpha", "--runs-dir", tmpdir])
        assert res_show.exit_code == 0
        assert "PruneQuest Run Details — run_alpha" in res_show.stdout

        res_show_json = runner.invoke(app, ["runs", "show", "run_alpha", "--runs-dir", tmpdir, "--json"])
        assert res_show_json.exit_code == 0
        show_data = json.loads(res_show_json.stdout)
        assert show_data["run_id"] == "run_alpha"
        assert show_data["compression_pct"] == 15.0

        # 3. runs compare (terminal & json)
        res_cmp = runner.invoke(app, ["runs", "compare", "run_alpha", "run_beta", "--runs-dir", tmpdir])
        assert res_cmp.exit_code == 0
        assert "PruneQuest Experiment Comparison" in res_cmp.stdout

        res_cmp_json = runner.invoke(app, ["runs", "compare", "run_alpha", "run_beta", "--runs-dir", tmpdir, "--json"])
        assert res_cmp_json.exit_code == 0
        cmp_data = json.loads(res_cmp_json.stdout)
        assert cmp_data["is_compatible"] is True

        # 4. runs logs (terminal & json)
        res_logs = runner.invoke(app, ["runs", "logs", "run_alpha", "--runs-dir", tmpdir])
        assert res_logs.exit_code == 0
        assert "PruneQuest Event Logs" in res_logs.stdout

        res_logs_json = runner.invoke(app, ["runs", "logs", "run_alpha", "--runs-dir", tmpdir, "--json"])
        assert res_logs_json.exit_code == 0
        log_lines = [json.loads(line) for line in res_logs_json.stdout.strip().split("\n") if line.strip()]
        assert len(log_lines) == 3


def test_cli_report_regeneration():
    from tests.fixtures.runs import create_mock_run
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = create_mock_run(tmpdir, "report_cli_test")
        res = runner.invoke(app, ["report", run_dir])
        assert res.exit_code == 0
        assert "Reports and visualizations generated successfully" in res.stdout
        assert os.path.exists(os.path.join(run_dir, "reports", "summary.md"))
        assert os.path.exists(os.path.join(run_dir, "reports", "detailed_report.md"))


