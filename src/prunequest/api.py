"""
Public Python API for PruneQuest.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from prunequest.config import PruneQuestConfig, ModelConfig, BenchmarkConfig, PruningConfig, HardwareConfig
from prunequest.core.controller import SearchController
from prunequest.reporting.reports import ReportGenerator
from prunequest.core.tracking import ExperimentTracker


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


class PruneQuest:
    """
    Main entry point for programmatic Python use of PruneQuest.

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
        retain: float = 0.99,
        max_iterations: int = 4,
        max_candidates: int = 3,
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
            self.config = PruneQuestConfig(
                model=ModelConfig(model_id=model, device=device, torch_dtype=torch_dtype),
                benchmark=BenchmarkConfig(name=benchmark),
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

    def baseline(self) -> Dict[str, Any]:
        """Establish benchmark baseline without pruning."""
        return self.controller.run_baseline()

    def resume(self) -> PruneQuestResult:
        """Resume an interrupted pruning experiment."""
        raw_res = self.controller.resume()
        df = self.controller.tracker.get_history_dataframe()
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
