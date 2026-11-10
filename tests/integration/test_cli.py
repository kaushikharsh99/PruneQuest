"""
Integration tests for CLI commands using Typer CliRunner.
Verifies all public commands, option aliases, config overrides, and error exits.
"""

import os
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
