"""
Packaging and installation tests.
Verifies entry points, imports, and clean packaging metadata.
"""

import sys
import subprocess
import tempfile
from prunequest import __version__, PruneQuest
from prunequest.config import PruneQuestConfig
from prunequest.core.controller import SearchController
from prunequest.models.discovery import ModelInspector
from prunequest.benchmarks.registry import BenchmarkRegistry
from prunequest.benchmarks.scoring import verify_gsm8k_answer
from prunequest.flap.core.optimizer import FLAPOptimizer
from prunequest.sweep.optimizer import JointIncrementalOptimizer


def test_public_api_symbols():
    assert __version__ == "0.3.0"
    assert PruneQuest is not None
    assert PruneQuestConfig is not None
    assert SearchController is not None
    assert ModelInspector is not None
    assert BenchmarkRegistry is not None
    assert verify_gsm8k_answer is not None
    assert FLAPOptimizer is not None
    assert JointIncrementalOptimizer is not None


def test_cli_entry_points():
    res1 = subprocess.run([sys.executable, "-m", "prunequest.cli", "--help"], capture_output=True, text=True)
    assert res1.returncode == 0
    assert "PruneQuest" in res1.stdout

    res2 = subprocess.run([sys.executable, "-m", "prunequest.flap.cli", "--help"], capture_output=True, text=True)
    assert res2.returncode == 0
    assert "FLAP" in res2.stdout


def test_wheel_buildable():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", tmpdir, "."],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "prunequest-0.3.0-py3-none-any.whl" in res.stdout or any(
            f.endswith(".whl") for f in subprocess.os.listdir(tmpdir)
        )
