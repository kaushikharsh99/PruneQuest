from typer.testing import CliRunner
from prunequest.stage1.cli import app

runner = CliRunner()


def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "Stage 1" in res.stdout
    assert "doctor" in res.stdout
    assert "inspect" in res.stdout
    assert "profile" in res.stdout
    assert "optimize" in res.stdout
    assert "test" in res.stdout


def test_cli_doctor():
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0
    assert "Diagnostics" in res.stdout
