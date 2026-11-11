"""
Preflight feasibility inspection, plan computation, and dry-run rendering for PruneQuest.
Inspects hardware, model architecture metadata, and benchmark requirements
without downloading full model weights upfront.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from prunequest.orchestrator.strategy import StrategyResolver, PruningStrategy
from prunequest.orchestrator.presets import OptimizationPreset, get_preset_config
from prunequest.orchestrator.hardware import HardwareInspector


@dataclass
class PreflightPlan:
    model_id: str
    model_type: str
    architecture_class: str
    layers: int
    hidden_size: int
    intermediate_size: int
    benchmark: str
    strategy: PruningStrategy
    strategy_reason: str
    preset: OptimizationPreset
    retention: float
    device: str
    device_name: str
    free_vram_gb: float
    weights_gb: float
    working_memory_gb: float
    disk_needed_gb: float
    free_disk_gb: float
    output_dir: str
    calibration_samples: int
    screening_samples: int
    acceptance_samples: int
    dataset: Optional[str] = None
    dataset_format: Optional[str] = None
    dataset_task_type: Optional[str] = None
    dataset_scorer: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class PreflightChecker:
    """
    Computes preflight feasibility assessment and resource plan.
    Supports dry-run preview mode.
    """

    @classmethod
    def plan(
        cls,
        model_id: str,
        benchmark: str = "gsm8k",
        retain: float = 0.99,
        preset: OptimizationPreset | str = OptimizationPreset.BALANCED,
        strategy: PruningStrategy | str = PruningStrategy.AUTO,
        device: str = "auto",
        output_dir: str = "runs/preview",
        dataset: Optional[str] = None,
        input_column: Optional[str] = None,
        target_column: Optional[str] = None,
        metric: Optional[str] = None,
        calibration_dataset: Optional[str] = None,
    ) -> PreflightPlan:
        if isinstance(strategy, str):
            strategy = PruningStrategy(strategy.lower())
        if isinstance(preset, str):
            preset = OptimizationPreset(preset.lower())

        preset_cfg = get_preset_config(preset)
        resolved_strat, reason, report = StrategyResolver.resolve(model_id, strategy)
        hw = HardwareInspector.inspect()
        resolved_device = HardwareInspector.resolve_device(device)

        # Inspect dataset if provided
        dataset_fmt = None
        dataset_task = None
        dataset_scorer = None
        warnings: List[str] = []

        if dataset:
            from prunequest.datasets.inspector import DatasetInspector
            try:
                ds_report = DatasetInspector.inspect(
                    dataset,
                    explicit_input=input_column,
                    explicit_target=target_column,
                    explicit_metric=metric,
                )
                dataset_fmt = ds_report.format
                dataset_task = ds_report.task_type
                dataset_scorer = ds_report.proposed_scorer
                if ds_report.warnings:
                    warnings.extend(ds_report.warnings)
            except Exception as e:
                warnings.append(f"Dataset preflight warning: {e}")

        desc = report.descriptor
        num_layers = desc.num_hidden_layers or 28
        hidden_size = desc.hidden_size or 1536
        inter_size = (
            desc.raw_config.get("intermediate_size", 0)
            if hasattr(desc, "raw_config") and isinstance(desc.raw_config, dict)
            else 0
        )
        if inter_size == 0:
            inter_size = int(hidden_size * 2.7)

        # Approximate parameter estimate
        # Attention: 4 * hidden^2 per layer; MLP: 3 * hidden * intermediate per layer
        params_per_layer = (4 * hidden_size * hidden_size) + (3 * hidden_size * inter_size)
        total_params = params_per_layer * num_layers

        mem_needs = HardwareInspector.estimate_memory_needs(total_params)

        warnings: List[str] = []
        if resolved_device.startswith("cuda"):
            if hw.free_vram_gb < mem_needs["weights_gb"]:
                warnings.append(
                    f"Available GPU VRAM ({hw.free_vram_gb:.1f} GB) is less than estimated model weights "
                    f"({mem_needs['weights_gb']:.1f} GB). Consider using CPU or reducing batch size."
                )
        if hw.free_disk_gb < mem_needs["disk_needed_gb"] * 2:
            warnings.append(
                f"Free disk space ({hw.free_disk_gb:.1f} GB) is low. Ensure at least {mem_needs['disk_needed_gb'] * 2:.1f} GB is free."
            )

        return PreflightPlan(
            model_id=model_id,
            model_type=desc.model_type,
            architecture_class=desc.architecture_class.value,
            layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            benchmark=benchmark,
            strategy=resolved_strat,
            strategy_reason=reason,
            preset=preset,
            retention=retain,
            device=resolved_device,
            device_name=hw.device_name if resolved_device.startswith("cuda") else "CPU",
            free_vram_gb=hw.free_vram_gb,
            weights_gb=mem_needs["weights_gb"],
            working_memory_gb=mem_needs["working_memory_gb"],
            disk_needed_gb=mem_needs["disk_needed_gb"],
            free_disk_gb=hw.free_disk_gb,
            output_dir=output_dir,
            calibration_samples=preset_cfg.calibration_samples,
            screening_samples=preset_cfg.screening_samples,
            acceptance_samples=preset_cfg.acceptance_samples,
            dataset=dataset,
            dataset_format=dataset_fmt,
            dataset_task_type=dataset_task,
            dataset_scorer=dataset_scorer,
            warnings=warnings,
        )

    @classmethod
    def render_plan(cls, plan: PreflightPlan, console: Optional[Console] = None):
        console = console or Console()

        header = (
            "[bold cyan]PruneQuest — Optimization Plan (Preflight)[/bold cyan]\n"
            "[dim]Autonomous, Benchmark-Guided LLM Structural Pruning[/dim]"
        )
        console.print(Panel(header, border_style="cyan"))

        table = Table(title="[bold green]Configuration & Strategy Summary[/bold green]")
        table.add_column("Parameter", style="cyan")
        table.add_column("Resolved Value", style="magenta")

        table.add_row("Model ID", plan.model_id)
        table.add_row("Architecture Class", f"{plan.model_type} ({plan.architecture_class})")
        table.add_row("Dimensions", f"{plan.layers} Layers | Hidden: {plan.hidden_size} | MLP: {plan.intermediate_size}")
        if plan.dataset:
            table.add_row("Dataset Objective", f"{plan.dataset} ({plan.dataset_format})")
            table.add_row("Task & Scorer", f"{plan.dataset_task_type} | {plan.dataset_scorer}")
        else:
            table.add_row("Benchmark Objective", plan.benchmark.upper())
        table.add_row("Accuracy Retention", f"[bold green]{plan.retention * 100:.1f}%[/bold green] (Measured on Holdout Dev)")
        table.add_row("Preset", f"{plan.preset.value.upper()} (Calib: {plan.calibration_samples}, Screen: {plan.screening_samples}, Dev: {plan.acceptance_samples})")
        table.add_row("Pruning Strategy", f"[bold yellow]{plan.strategy.value.upper()}[/bold yellow] ({plan.strategy_reason})")
        table.add_row("Compute Device", f"{plan.device} ({plan.device_name})")
        table.add_row("Est. Model Memory", f"~{plan.weights_gb:.2f} GB (bfloat16) [Working: ~{plan.working_memory_gb:.2f} GB]")
        table.add_row("Output Directory", plan.output_dir)

        console.print(table)

        if plan.warnings:
            for w in plan.warnings:
                console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {w}")

        console.print("[dim]Preflight completed. No model weights modified.[/dim]\n")
