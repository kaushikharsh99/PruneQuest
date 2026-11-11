"""
Zero-YAML automated configuration builder for PruneQuest.
Translates high-level user parameters (model, benchmark, retention, preset, strategy)
into robust, validated internal configuration objects.
"""

import os
from datetime import datetime
from typing import Optional, Any
from prunequest.orchestrator.presets import get_preset_config, OptimizationPreset, PresetConfig
from prunequest.orchestrator.strategy import StrategyResolver, PruningStrategy
from prunequest.orchestrator.hardware import HardwareInspector

from prunequest.sweep.config import JointSweepConfig, JointPruningConfig
from prunequest.flap.config import (
    FLAPConfig,
    ModelConfig as FLAPModelConfig,
    CalibrationConfig,
    EvaluationConfig,
    PruningConfig as FLAPPruningConfig,
)
from prunequest.config import (
    PruneQuestConfig,
    ModelConfig as BlockModelConfig,
    BenchmarkConfig,
    PruningConfig as BlockPruningConfig,
)


class AutoConfigBuilder:
    """
    Constructs concrete workflow configurations from user inputs and presets.
    Enforces strict CLI > Config File > Default precedence.
    """

    @classmethod
    def generate_run_dir(cls, model_id: str) -> str:
        clean_model = model_id.split("/")[-1].replace(".", "_").replace("-", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"runs/{clean_model}_{timestamp}"

    @classmethod
    def build(
        cls,
        model_id: str,
        benchmark: str = "gsm8k",
        retain: float = 0.99,
        preset: OptimizationPreset | str = OptimizationPreset.BALANCED,
        strategy: PruningStrategy | str = PruningStrategy.AUTO,
        device: str = "auto",
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_iterations: Optional[int] = None,
        seed: int = 42,
        dataset: Optional[str] = None,
        input_column: Optional[str] = None,
        target_column: Optional[str] = None,
        metric: Optional[str] = None,
        calibration_dataset: Optional[str] = None,
    ) -> tuple[Any, PruningStrategy, PresetConfig, str, str]:
        """
        Builds and returns the concrete configuration object along with metadata.

        Returns:
            Tuple of (config_obj, resolved_strategy, preset_config, resolved_device, output_dir)
        """
        preset_cfg = get_preset_config(preset)
        resolved_device = HardwareInspector.resolve_device(device)
        resolved_output = output_dir or cls.generate_run_dir(model_id)
        os.makedirs(resolved_output, exist_ok=True)

        # Handle custom dataset onboarding (UX Release C)
        effective_benchmark = benchmark
        calib_data_name = calibration_dataset or ("openai/gsm8k" if benchmark == "gsm8k" else benchmark)

        if dataset:
            from prunequest.datasets.inspector import DatasetInspector
            from prunequest.datasets.resolver import DatasetSourceResolver
            from prunequest.datasets.spec_builder import BenchmarkSpecBuilder

            ds_report = DatasetInspector.inspect(
                dataset,
                explicit_input=input_column,
                explicit_target=target_column,
                explicit_metric=metric,
            )
            resolved_src = DatasetSourceResolver.resolve(dataset)
            clean_name = os.path.basename(dataset).split(".")[0] or "custom_dataset"

            spec = BenchmarkSpecBuilder.build(
                name=clean_name,
                source=resolved_src.file_format or resolved_src.source_type.value,
                path=resolved_src.path_or_id,
                input_col=ds_report.input_column or "input",
                target_col=ds_report.target_column or "target",
                task_type=ds_report.task_type,
                scorer=ds_report.proposed_scorer,
                higher_is_better=ds_report.higher_is_better,
                retention=retain,
                revision=resolved_src.revision,
            )
            spec_path = os.path.join(resolved_output, "resolved_spec.yaml")
            BenchmarkSpecBuilder.save_spec(spec, spec_path)
            effective_benchmark = spec_path
            calib_data_name = calibration_dataset or (dataset if resolved_src.is_local else "openai/gsm8k")

        # Resolve strategy
        if isinstance(strategy, str):
            strategy = PruningStrategy(strategy.lower())
        resolved_strategy, reason, inspection = StrategyResolver.resolve(model_id, strategy)

        # Build strategy-specific configuration
        if resolved_strategy == PruningStrategy.JOINT:
            if config_path and os.path.exists(config_path):
                cfg = JointSweepConfig.from_yaml(config_path)
            else:
                cfg = JointSweepConfig(
                    model=FLAPModelConfig(
                        model_id=model_id,
                        device=resolved_device,
                        torch_dtype="bfloat16",
                    ),
                    calibration=CalibrationConfig(
                        dataset_name=calib_data_name,
                        num_samples=preset_cfg.calibration_samples,
                        screening_samples=preset_cfg.screening_samples,
                        acceptance_samples=preset_cfg.acceptance_samples,
                        seed=seed,
                    ),
                    pruning=JointPruningConfig(
                        retention=retain,
                        max_iterations=max_iterations or preset_cfg.max_iterations,
                        initial_flap_ratio=preset_cfg.initial_flap_ratio,
                        minimum_flap_ratio=preset_cfg.minimum_flap_ratio,
                        screening_loss_factor=preset_cfg.screening_loss_factor,
                        time_budget_sec=preset_cfg.time_budget_sec,
                    ),
                    evaluation=EvaluationConfig(
                        benchmark=effective_benchmark,
                        batch_size=batch_size or 1,
                    ),
                    output_dir=resolved_output,
                    reports_dir=os.path.join(resolved_output, "reports"),
                )
            # Apply explicit overrides
            cfg.model.model_id = model_id
            cfg.model.device = resolved_device
            cfg.pruning.retention = retain
            cfg.output_dir = resolved_output
            if max_iterations:
                cfg.pruning.max_iterations = max_iterations
            return cfg, resolved_strategy, preset_cfg, resolved_device, resolved_output

        elif resolved_strategy == PruningStrategy.FLAP:
            if config_path and os.path.exists(config_path):
                cfg = FLAPConfig.from_yaml(config_path)
            else:
                cfg = FLAPConfig(
                    model=FLAPModelConfig(
                        model_id=model_id,
                        device=resolved_device,
                        torch_dtype="bfloat16",
                    ),
                    calibration=CalibrationConfig(
                        dataset_name=calib_data_name,
                        num_samples=preset_cfg.calibration_samples,
                        screening_samples=preset_cfg.screening_samples,
                        acceptance_samples=preset_cfg.acceptance_samples,
                        seed=seed,
                    ),
                    pruning=FLAPPruningConfig(
                        retention=retain,
                        max_candidate_evaluations=preset_cfg.max_candidate_evaluations,
                        initial_step_ratio=preset_cfg.initial_flap_ratio,
                        minimum_step_ratio=preset_cfg.minimum_flap_ratio,
                    ),
                    evaluation=EvaluationConfig(
                        benchmark=effective_benchmark,
                        batch_size=batch_size or 1,
                        screening_loss_factor=preset_cfg.screening_loss_factor,
                    ),
                    output_dir=resolved_output,
                    reports_dir=os.path.join(resolved_output, "reports"),
                )
            cfg.model.model_id = model_id
            cfg.model.device = resolved_device
            cfg.pruning.retention = retain
            cfg.output_dir = resolved_output
            return cfg, resolved_strategy, preset_cfg, resolved_device, resolved_output

        elif resolved_strategy == PruningStrategy.BLOCK:
            if config_path and os.path.exists(config_path):
                cfg = PruneQuestConfig.from_yaml(config_path)
            else:
                cfg = PruneQuestConfig(
                    model=BlockModelConfig(
                        model_id=model_id,
                        device=resolved_device,
                        torch_dtype="bfloat16",
                    ),
                    benchmark=BenchmarkConfig(
                        name=effective_benchmark,
                        split="dev",
                        sample_limit=preset_cfg.acceptance_samples,
                        seed=seed,
                    ),
                    pruning=BlockPruningConfig(
                        retain=retain,
                        max_iterations=max_iterations or preset_cfg.max_iterations,
                        max_candidates=preset_cfg.max_candidate_evaluations,
                        time_budget_sec=preset_cfg.time_budget_sec,
                    ),
                    output_dir=resolved_output,
                    reports_dir=os.path.join(resolved_output, "reports"),
                )
            cfg.model.model_id = model_id
            cfg.model.device = resolved_device
            cfg.pruning.retain = retain
            cfg.output_dir = resolved_output
            return cfg, resolved_strategy, preset_cfg, resolved_device, resolved_output

        raise ValueError(f"Unsupported resolved strategy: '{resolved_strategy}'")
