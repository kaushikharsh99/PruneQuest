"""
Public Python API for PruneQuest.
Provides the unified optimize() function and backward-compatible PruneQuest controller.
"""

from dataclasses import dataclass
from typing import Optional, Any, Callable
from prunequest.config import PruneQuestConfig, ModelConfig, BenchmarkConfig, PruningConfig
from prunequest.core.controller import SearchController
from prunequest.reporting.reports import ReportGenerator
from prunequest.orchestrator.optimizer import UnifiedOptimizer, OptimizationResult
from prunequest.orchestrator.presets import OptimizationPreset
from prunequest.orchestrator.strategy import PruningStrategy
from prunequest.events import OptimizationEvent


@dataclass
class PruneQuestResult:
    status: str
    best_checkpoint: Optional[str]
    original_params: int
    final_params: int
    baseline_accuracy: float
    final_accuracy: float
    accuracy_retention_pct: float
    iterations_completed: int
    run_dir: str
    report_path: Optional[str] = None


def optimize(
    model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    benchmark: str = "gsm8k",
    retain: float = 0.99,
    preset: OptimizationPreset | str = "balanced",
    strategy: PruningStrategy | str = "auto",
    device: str = "auto",
    output: Optional[str] = None,
    config: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = True,
    batch_size: Optional[int] = None,
    max_iterations: Optional[int] = None,
    no_progress: bool = False,
    json_output: bool = False,
    json_events: bool = False,
    on_progress: Optional[Callable[[OptimizationEvent], None]] = None,
) -> OptimizationResult:
    """
    Universal one-command Python interface for LLM pruning.
    Automatically inspects model architecture, checks hardware, resolves
    compatible pruning strategies (Joint, FLAP, Block), enforces quality retention,
    and exports a verified, reloadable checkpoint.

    Example:
        >>> from prunequest import optimize
        >>> res = optimize("Qwen/Qwen2.5-Math-1.5B-Instruct", benchmark="gsm8k", retain=0.99)
        >>> print(res.best_checkpoint, res.compression_ratio, res.accuracy_retention)
    """
    return UnifiedOptimizer.optimize(
        model=model,
        benchmark=benchmark,
        retain=retain,
        preset=preset,
        strategy=strategy,
        device=device,
        output=output,
        config=config,
        dry_run=dry_run,
        resume=resume,
        batch_size=batch_size,
        max_iterations=max_iterations,
        no_progress=no_progress,
        json_output=json_output,
        json_events=json_events,
        on_progress=on_progress,
    )


class PruneQuest:
    """
    Object-oriented entry point for programmatic Python use of PruneQuest.

    Example:
        >>> from prunequest import PruneQuest
        >>> quest = PruneQuest(
        ...     model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        ...     benchmark="gsm8k",
        ...     retain=0.99,
        ... )
        >>> result = quest.run()
        >>> print(result.best_checkpoint)
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        benchmark: str = "gsm8k",
        benchmarks: Optional[Any] = None,
        retain: float = 0.99,
        max_iterations: int = 4,
        max_candidates: int = 3,
        split: str = "smoke",
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        output_dir: str = "runs/prunequest_run",
        reports_dir: str = "reports",
        time_budget_sec: Optional[int] = None,
        config_path: Optional[str] = None,
    ):
        if config_path:
            self.config = PruneQuestConfig.from_yaml(config_path)
        else:
            bench_target = benchmark
            if benchmarks:
                if isinstance(benchmarks, list):
                    bench_target = ",".join(benchmarks)
                else:
                    bench_target = str(benchmarks)

            self.config = PruneQuestConfig(
                model=ModelConfig(model_id=model, device=device, torch_dtype=torch_dtype),
                benchmark=BenchmarkConfig(name=bench_target, split=split),
                pruning=PruningConfig(retain=retain, max_iterations=max_iterations, max_candidates=max_candidates, time_budget_sec=time_budget_sec),
                output_dir=output_dir,
                reports_dir=reports_dir,
            )

        self.controller = SearchController(self.config)

    def run(self) -> PruneQuestResult:
        """Execute automated pruning workflow."""
        raw_res = self.controller.run()

        # Generate report and plots
        tracker = self.controller.tracker
        df = tracker.get_history_dataframe()
        reporter = ReportGenerator(self.config.reports_dir)
        plots = reporter.generate_plots(df)
        report_path = reporter.generate_report(df, raw_res, plots)

        return PruneQuestResult(
            status=raw_res.get("status", "unknown"),
            best_checkpoint=raw_res.get("best_checkpoint"),
            original_params=raw_res.get("original_params", 0),
            final_params=raw_res.get("final_params", 0),
            baseline_accuracy=raw_res.get("baseline_accuracy", 0.0),
            final_accuracy=raw_res.get("final_accuracy", 0.0),
            accuracy_retention_pct=raw_res.get("accuracy_retention_pct", 0.0),
            iterations_completed=raw_res.get("iterations_completed", 0),
            run_dir=self.config.output_dir,
            report_path=report_path,
        )
