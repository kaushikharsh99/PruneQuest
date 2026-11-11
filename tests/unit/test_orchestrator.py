"""
Unit tests for the PruneQuest UX Release A Orchestrator.
Tests strategy auto-resolution, presets, hardware inspection, preflight planning,
zero-YAML configuration building, standardized artifacts, and the high-level Python API.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from prunequest.orchestrator.presets import (
    OptimizationPreset,
    PresetConfig,
    get_preset_config,
)
from prunequest.orchestrator.hardware import HardwareInspector
from prunequest.orchestrator.strategy import StrategyResolver, PruningStrategy
from prunequest.orchestrator.config_builder import AutoConfigBuilder
from prunequest.orchestrator.preflight import PreflightChecker, PreflightPlan
from prunequest.orchestrator.optimizer import UnifiedOptimizer, OptimizationResult
from prunequest import optimize
from prunequest.sweep.config import JointSweepConfig
from prunequest.flap.config import FLAPConfig
from prunequest.config import PruneQuestConfig


def test_presets():
    quick = get_preset_config(OptimizationPreset.QUICK)
    assert isinstance(quick, PresetConfig)
    assert quick.calibration_samples == 32
    assert quick.screening_samples == 16
    assert quick.acceptance_samples == 64

    balanced = get_preset_config("balanced")
    assert balanced.calibration_samples == 128
    assert balanced.screening_samples == 64
    assert balanced.acceptance_samples == 256

    thorough = get_preset_config(OptimizationPreset.THOROUGH)
    assert thorough.calibration_samples == 256
    assert thorough.screening_samples == 128
    assert thorough.acceptance_samples == 512

    with pytest.raises(ValueError):
        get_preset_config("ultra_hyper_mode")


def test_hardware_inspector():
    sys_info = HardwareInspector.inspect()
    assert hasattr(sys_info, "system_ram_gb")
    assert hasattr(sys_info, "free_disk_gb")
    assert hasattr(sys_info, "cuda_available")

    resolved_cpu = HardwareInspector.resolve_device("cpu")
    assert resolved_cpu == "cpu"

    resolved_auto = HardwareInspector.resolve_device("auto")
    assert resolved_auto.startswith("cuda:") or resolved_auto in ("cpu", "mps")

    mem_est = HardwareInspector.estimate_memory_needs(500_000_000)
    assert mem_est["weights_gb"] > 0.5
    assert mem_est["working_memory_gb"] > mem_est["weights_gb"]
    assert mem_est["disk_needed_gb"] > 0.0


def test_strategy_resolver():
    # Test Auto resolution for Qwen2 (decoder with SwiGLU support -> JOINT)
    strat, reason, insp = StrategyResolver.resolve("Qwen/Qwen2.5-0.5B-Instruct", PruningStrategy.AUTO)
    assert strat == PruningStrategy.JOINT
    assert "qwen2" in insp.descriptor.model_type.lower()

    # Test explicit strategy override
    strat_block, _, _ = StrategyResolver.resolve("Qwen/Qwen2.5-0.5B-Instruct", PruningStrategy.BLOCK)
    assert strat_block == PruningStrategy.BLOCK

    strat_flap, _, _ = StrategyResolver.resolve("Qwen/Qwen2.5-0.5B-Instruct", PruningStrategy.FLAP)
    assert strat_flap == PruningStrategy.FLAP

    # Test rejection of quantized / MoE models
    with pytest.raises(ValueError) as exc:
        StrategyResolver.resolve("mistralai/Mixtral-8x7B-v0.1", PruningStrategy.AUTO)
    assert "Mixture-of-Experts" in str(exc.value)

    with pytest.raises(ValueError) as exc_awq:
        StrategyResolver.resolve("TheBloke/Llama-2-7B-AWQ", PruningStrategy.AUTO)
    assert "quantized" in str(exc_awq.value).lower()


def test_auto_config_builder():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build Joint config (zero-YAML)
        joint_cfg, strat, preset_cfg, dev, out_dir = AutoConfigBuilder.build(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            benchmark="gsm8k",
            retain=0.98,
            preset="quick",
            strategy="joint",
            device="cpu",
            output_dir=tmpdir,
        )
        assert isinstance(joint_cfg, JointSweepConfig)
        assert strat == PruningStrategy.JOINT
        assert joint_cfg.pruning.retention == 0.98
        assert joint_cfg.calibration.num_samples == 32
        assert joint_cfg.model.device == "cpu"

        # Build FLAP config (zero-YAML)
        flap_cfg, strat_f, _, _, _ = AutoConfigBuilder.build(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            benchmark="gsm8k",
            retain=0.95,
            preset="balanced",
            strategy="flap",
            device="cpu",
            output_dir=tmpdir,
        )
        assert isinstance(flap_cfg, FLAPConfig)
        assert strat_f == PruningStrategy.FLAP
        assert flap_cfg.pruning.retention == 0.95
        assert flap_cfg.calibration.num_samples == 128

        # Build Block config (zero-YAML)
        block_cfg, strat_b, _, _, _ = AutoConfigBuilder.build(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            benchmark="gsm8k",
            retain=0.99,
            preset="thorough",
            strategy="block",
            device="cpu",
            output_dir=tmpdir,
        )
        assert isinstance(block_cfg, PruneQuestConfig)
        assert strat_b == PruningStrategy.BLOCK
        assert block_cfg.pruning.retain == 0.99
        assert block_cfg.benchmark.sample_limit == 512


def test_preflight_checker():
    plan = PreflightChecker.plan(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        benchmark="gsm8k",
        retain=0.99,
        preset="balanced",
        strategy="auto",
        device="cpu",
        output_dir="runs/test_run",
    )
    assert isinstance(plan, PreflightPlan)
    assert plan.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert plan.strategy == PruningStrategy.JOINT
    assert plan.preset == OptimizationPreset.BALANCED
    assert plan.weights_gb > 0.0

    # Rendering plan does not raise exception
    PreflightChecker.render_plan(plan)


def test_optimize_api_dry_run():
    # Test high-level Python API with dry_run=True
    result = optimize(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        benchmark="gsm8k",
        retain=0.99,
        dry_run=True,
    )
    assert isinstance(result, OptimizationResult)
    assert result.status == "dry_run_completed"
    assert result.strategy_used == "joint"
    assert result.best_checkpoint is None


def test_unified_optimizer_mocked_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock JointIncrementalOptimizer to test lifecycle and standardized artifact promotion
        mock_joint_optimizer = MagicMock()
        mock_joint_optimizer.optimize.return_value = {
            "status": "completed",
            "baseline_params": 1000000,
            "baseline_accuracy": 0.90,
            "best_checkpoint": {
                "checkpoint_path": os.path.join(tmpdir, "checkpoints", "step_1"),
                "params_after": 850000,
                "accuracy": 0.895,
                "accuracy_retention_pct": 99.4,
            },
            "metrics": {"full_evaluations_executed": 3},
        }

        # Create mock checkpoint source directory with files
        ckpt_src = os.path.join(tmpdir, "checkpoints", "step_1")
        os.makedirs(ckpt_src, exist_ok=True)
        with open(os.path.join(ckpt_src, "config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(ckpt_src, "model.safetensors"), "w") as f:
            f.write("mock_weights")

        with patch("prunequest.sweep.optimizer.JointIncrementalOptimizer", return_value=mock_joint_optimizer), \
             patch("prunequest.flap.export.validator.CheckpointValidator.validate", return_value=(True, "Reload test passed")):

            result = UnifiedOptimizer.optimize(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                benchmark="gsm8k",
                retain=0.99,
                preset="quick",
                strategy="joint",
                device="cpu",
                output=tmpdir,
                dry_run=False,
            )

            assert result.status == "completed"
            assert result.original_params == 1000000
            assert result.final_params == 850000
            assert result.params_pruned == 150000
            assert result.reload_verified is True

            # Verify standardized artifacts exist in tmpdir
            canonical_best = os.path.join(tmpdir, "best")
            assert os.path.exists(canonical_best)
            assert os.path.exists(os.path.join(canonical_best, "config.json"))
            assert os.path.exists(os.path.join(canonical_best, "model.safetensors"))

            manifest_file = os.path.join(tmpdir, "manifest.json")
            assert os.path.exists(manifest_file)
            with open(manifest_file) as f:
                manifest_data = json.load(f)
            assert manifest_data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
            assert manifest_data["results"]["final_parameters"] == 850000

            results_file = os.path.join(tmpdir, "results.json")
            assert os.path.exists(results_file)
            with open(results_file) as f:
                res_data = json.load(f)
            assert res_data["status"] == "completed"
            assert res_data["parameters_pruned"] == 150000

            report_file = os.path.join(tmpdir, "report.md")
            assert os.path.exists(report_file)
            with open(report_file) as f:
                report_content = f.read()
            assert "PruneQuest Optimization Report" in report_content
