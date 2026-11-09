"""
Integration tests for CLI commands using Typer CliRunner.
"""

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


def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "PruneQuest" in result.stdout
    assert "Diagnostics" in result.stdout


def test_cli_benchmarks():
    result = runner.invoke(app, ["benchmarks", "list"])
    assert result.exit_code == 0
    assert "gsm8k" in result.stdout


def test_cli_inspect():
    result = runner.invoke(app, ["inspect", "--model", "Qwen/Qwen2.5-Math-1.5B-Instruct"])
    assert result.exit_code == 0
    assert "Architecture Class" in result.stdout
    assert "Capability Level" in result.stdout
