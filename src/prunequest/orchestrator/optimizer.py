"""
Unified Optimization Orchestrator for PruneQuest.
Orchestrates end-to-end model pruning across Joint (JIPO), FLAP (Width), and Block (Depth)
strategies with automatic configuration, preflight checking, standardized outputs,
live event streaming, robust interruption recovery, and independent reload verification.
"""

from dataclasses import dataclass
import json
import os
import shutil
import sys
import time
from typing import Optional, Dict, Any, Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prunequest.orchestrator.strategy import PruningStrategy
from prunequest.orchestrator.presets import OptimizationPreset
from prunequest.orchestrator.config_builder import AutoConfigBuilder
from prunequest.orchestrator.preflight import PreflightChecker
from prunequest.flap.export.validator import CheckpointValidator
from prunequest.events import EventBus, EventLogger, EventCategory, EventPhase, OptimizationEvent
from prunequest.ui import LiveProgressDashboard, PlainProgressRenderer, JsonStreamRenderer
from prunequest.runs import RunLock, RunManager
from prunequest.reporting import UnifiedReportGenerator


@dataclass
class OptimizationResult:
    status: str
    best_checkpoint: Optional[str]
    original_params: int
    final_params: int
    params_pruned: int
    compression_ratio: float
    baseline_accuracy: float
    final_accuracy: float
    accuracy_retention: float
    strategy_used: str
    iterations_completed: int
    run_dir: str
    run_id: str = ""
    report_path: Optional[str] = None
    manifest_path: Optional[str] = None
    reload_verified: bool = False
    elapsed_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "best_checkpoint": self.best_checkpoint,
            "original_params": self.original_params,
            "final_params": self.final_params,
            "params_pruned": self.params_pruned,
            "compression_ratio": self.compression_ratio,
            "baseline_accuracy": self.baseline_accuracy,
            "final_accuracy": self.final_accuracy,
            "accuracy_retention": self.accuracy_retention,
            "strategy_used": self.strategy_used,
            "iterations_completed": self.iterations_completed,
            "run_dir": self.run_dir,
            "report_path": self.report_path,
            "manifest_path": self.manifest_path,
            "reload_verified": self.reload_verified,
            "elapsed_time_sec": self.elapsed_time_sec,
        }


class UnifiedOptimizer:
    """
    Main orchestration engine executing the one-command optimization lifecycle.
    Shared by both the CLI and Python API.
    """

    @classmethod
    def optimize(
        cls,
        model: str,
        benchmark: str = "gsm8k",
        retain: float = 0.99,
        preset: OptimizationPreset | str = OptimizationPreset.BALANCED,
        strategy: PruningStrategy | str = PruningStrategy.AUTO,
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
        console: Optional[Console] = None,
        on_progress: Optional[Callable[[OptimizationEvent], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        dataset: Optional[str] = None,
        input_column: Optional[str] = None,
        target_column: Optional[str] = None,
        metric: Optional[str] = None,
        calibration_dataset: Optional[str] = None,
    ) -> OptimizationResult:
        console = console or Console()
        start_time = time.time()

        # Step 1: Preflight Planning
        resolved_run_dir = output or AutoConfigBuilder.generate_run_dir(model)
        run_id = os.path.basename(resolved_run_dir)

        plan = PreflightChecker.plan(
            model_id=model,
            benchmark=benchmark,
            retain=retain,
            preset=preset,
            strategy=strategy,
            device=device,
            output_dir=resolved_run_dir,
            dataset=dataset,
            input_column=input_column,
            target_column=target_column,
            metric=metric,
            calibration_dataset=calibration_dataset,
        )

        if dry_run:
            if not json_output:
                PreflightChecker.render_plan(plan, console=console)
            res = OptimizationResult(
                status="dry_run_completed",
                run_id=run_id,
                best_checkpoint=None,
                original_params=int(plan.weights_gb * 1e9 / 2),
                final_params=int(plan.weights_gb * 1e9 / 2),
                params_pruned=0,
                compression_ratio=1.0,
                baseline_accuracy=0.0,
                final_accuracy=0.0,
                accuracy_retention=1.0,
                strategy_used=plan.strategy.value,
                iterations_completed=0,
                run_dir=plan.output_dir,
                reload_verified=False,
                elapsed_time_sec=time.time() - start_time,
            )
            if json_output:
                print(json.dumps(res.to_dict(), indent=2))
            return res

        # Step 2: Auto-Config Building
        cfg, resolved_strat, preset_cfg, resolved_dev, run_dir = AutoConfigBuilder.build(
            model_id=model,
            benchmark=benchmark,
            retain=retain,
            preset=preset,
            strategy=strategy,
            device=device,
            output_dir=output,
            config_path=config,
            batch_size=batch_size,
            max_iterations=max_iterations,
            dataset=dataset,
            input_column=input_column,
            target_column=target_column,
            metric=metric,
            calibration_dataset=calibration_dataset,
        )

        run_id = os.path.basename(run_dir)
        os.makedirs(run_dir, exist_ok=True)

        # File locking to prevent concurrent modifications on the same directory
        run_lock = RunLock(run_dir)
        run_lock.acquire()

        # Initialize Event Bus & Event Logger
        event_bus = EventBus(run_id)
        events_jsonl_path = os.path.join(run_dir, "events.jsonl")
        event_logger = EventLogger(events_jsonl_path)
        event_bus.subscribe(event_logger.handle_event)

        # Wire user callback if provided
        if on_progress:
            event_bus.subscribe(on_progress)

        if cancel_check:
            def _check_cancellation(ev: OptimizationEvent):
                if cancel_check():
                    raise KeyboardInterrupt("Optimization cancelled by user request.")
            event_bus.subscribe(_check_cancellation)

        # Progress UI selection
        dashboard: Optional[LiveProgressDashboard] = None
        if json_events:
            event_bus.subscribe(JsonStreamRenderer(sys.stdout).handle_event)
        elif no_progress or not console.is_terminal or json_output or os.environ.get("NO_COLOR"):
            if not json_output:
                event_bus.subscribe(PlainProgressRenderer(sys.stderr).handle_event)
        else:
            dashboard = LiveProgressDashboard(console=console)
            event_bus.subscribe(dashboard.on_event)
            dashboard.start()

        # Emit initial lifecycle events
        event_bus.emit(
            EventCategory.RUN_STARTED,
            EventPhase.PREFLIGHT,
            {
                "model_id": model,
                "benchmark": benchmark,
                "strategy": resolved_strat.value,
                "device": resolved_dev,
                "retention": retain,
                "preset": preset_cfg.name,
                "output_dir": run_dir,
            },
        )

        if dataset:
            event_bus.emit(
                EventCategory.DATASET_INSPECTION_STARTED,
                EventPhase.PREFLIGHT,
                {"source": dataset},
            )
            event_bus.emit(
                EventCategory.FIELD_MAPPING_RESOLVED,
                EventPhase.PREFLIGHT,
                {"input_column": input_column, "target_column": target_column},
            )
            event_bus.emit(
                EventCategory.TASK_TYPE_RESOLVED,
                EventPhase.PREFLIGHT,
                {"task_type": plan.dataset_task_type},
            )
            event_bus.emit(
                EventCategory.SCORER_RESOLVED,
                EventPhase.PREFLIGHT,
                {"scorer": plan.dataset_scorer},
            )
            event_bus.emit(
                EventCategory.DATASET_PARTITIONING_COMPLETED,
                EventPhase.PREFLIGHT,
                {
                    "calibration_samples": preset_cfg.calibration_samples,
                    "screening_samples": preset_cfg.screening_samples,
                    "acceptance_samples": preset_cfg.acceptance_samples,
                },
            )

        event_bus.emit(EventCategory.PREFLIGHT_COMPLETED, EventPhase.PREFLIGHT)

        # Save resolved configuration
        resolved_cfg_file = os.path.join(run_dir, "resolved_config.yaml")
        if hasattr(cfg, "to_yaml"):
            cfg.to_yaml(resolved_cfg_file)

        raw_res: Dict[str, Any] = {}
        best_ckpt_source: Optional[str] = None
        interrupted = False

        try:
            # Step 3: Strategy Execution with Event Logging
            event_bus.emit(EventCategory.MODEL_LOADING_STARTED, EventPhase.MODEL_LOADING)

            if resolved_strat == PruningStrategy.JOINT:
                from prunequest.sweep.optimizer import JointIncrementalOptimizer
                optimizer = JointIncrementalOptimizer(cfg)
                optimizer.initialize()
                event_bus.emit(
                    EventCategory.MODEL_LOADED,
                    EventPhase.MODEL_LOADING,
                    {"total_params": optimizer.baseline_params},
                )
                event_bus.emit(EventCategory.BASELINE_STARTED, EventPhase.BASELINE)
                optimizer.run_baseline()
                event_bus.emit(
                    EventCategory.BASELINE_COMPLETED,
                    EventPhase.BASELINE,
                    {"accuracy": optimizer.baseline_accuracy, "total_params": optimizer.baseline_params},
                )
                event_bus.emit(EventCategory.PROFILING_COMPLETED, EventPhase.PROFILING)

                # Search
                raw_res = optimizer.optimize(max_iterations=max_iterations, resume=resume)

                best_dict = raw_res.get("best_checkpoint") or {}
                best_ckpt_source = best_dict.get("checkpoint_path")
                orig_params = raw_res.get("baseline_params", optimizer.baseline_params)
                final_params = best_dict.get("params_after", orig_params)
                base_acc = raw_res.get("baseline_accuracy", optimizer.baseline_accuracy)
                final_acc = best_dict.get("accuracy", base_acc)

            elif resolved_strat == PruningStrategy.FLAP:
                from prunequest.flap.core.optimizer import FLAPOptimizer
                optimizer = FLAPOptimizer(cfg)
                optimizer.initialize()
                event_bus.emit(
                    EventCategory.MODEL_LOADED,
                    EventPhase.MODEL_LOADING,
                    {"total_params": optimizer.baseline_params},
                )
                event_bus.emit(EventCategory.BASELINE_STARTED, EventPhase.BASELINE)
                optimizer.establish_baseline()
                event_bus.emit(
                    EventCategory.BASELINE_COMPLETED,
                    EventPhase.BASELINE,
                    {"accuracy": optimizer.baseline_accuracy, "total_params": optimizer.baseline_params},
                )
                event_bus.emit(EventCategory.PROFILING_COMPLETED, EventPhase.PROFILING)

                raw_res = optimizer.optimize(resume=resume)

                best_ckpt_source = raw_res.get("best_checkpoint")
                orig_params = raw_res.get("original_params", optimizer.baseline_params)
                final_params = raw_res.get("final_params", orig_params)
                base_acc = raw_res.get("baseline_accuracy", optimizer.baseline_accuracy)
                final_acc = raw_res.get("final_accuracy", base_acc)

            elif resolved_strat == PruningStrategy.BLOCK:
                from prunequest.core.controller import SearchController
                controller = SearchController(cfg)
                event_bus.emit(EventCategory.MODEL_LOADED, EventPhase.MODEL_LOADING)
                event_bus.emit(EventCategory.BASELINE_STARTED, EventPhase.BASELINE)
                if controller.baseline_accuracy is None:
                    controller.run_baseline()
                event_bus.emit(
                    EventCategory.BASELINE_COMPLETED,
                    EventPhase.BASELINE,
                    {
                        "accuracy": controller.baseline_accuracy or 0.0,
                        "total_params": controller.baseline_params or 0,
                    },
                )
                event_bus.emit(EventCategory.PROFILING_COMPLETED, EventPhase.PROFILING)
                raw_res = controller.run(progress_callback=progress_callback)

                best_ckpt_source = raw_res.get("best_checkpoint")
                orig_params = raw_res.get("original_params", controller.baseline_params or 0)
                final_params = raw_res.get("final_params", orig_params)
                base_acc = raw_res.get("baseline_accuracy", controller.baseline_accuracy or 0.0)
                final_acc = raw_res.get("final_accuracy", base_acc)

            else:
                raise ValueError(f"Unknown strategy: '{resolved_strat}'")

        except KeyboardInterrupt:
            interrupted = True
            event_bus.request_interruption()
            event_bus.emit(
                EventCategory.RUN_INTERRUPTED,
                EventPhase.INTERRUPTED,
                {"best_checkpoint": best_ckpt_source or "unpruned_baseline"},
            )
            raw_res = {
                "status": "interrupted",
                "stopping_reason": "keyboard_interrupt",
            }
            orig_params = getattr(plan, "weights_gb", 1.0) * 1e9 / 2
            final_params = orig_params
            base_acc = 0.0
            final_acc = 0.0
        finally:
            if dashboard:
                dashboard.stop()

        # Step 4: Atomic Checkpoint Promotion (staging -> runs/<run_dir>/best)
        canonical_best = os.path.join(run_dir, "best")
        staging_best = os.path.join(run_dir, ".staging_best")
        reload_verified = False

        if best_ckpt_source and os.path.exists(best_ckpt_source) and best_ckpt_source != "unpruned_baseline":
            event_bus.emit(EventCategory.CHECKPOINT_EXPORT_STARTED, EventPhase.CHECKPOINT, {"source": best_ckpt_source})
            if os.path.abspath(best_ckpt_source) != os.path.abspath(canonical_best):
                if os.path.exists(staging_best):
                    shutil.rmtree(staging_best, ignore_errors=True)
                os.makedirs(staging_best, exist_ok=True)
                for item in os.listdir(best_ckpt_source):
                    s = os.path.join(best_ckpt_source, item)
                    d = os.path.join(staging_best, item)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)

                # Atomic move to canonical_best
                if os.path.exists(canonical_best):
                    shutil.rmtree(canonical_best, ignore_errors=True)
                os.rename(staging_best, canonical_best)

            # Subprocess reload validation
            ok, msg = CheckpointValidator.validate(canonical_best)
            reload_verified = ok
            if ok:
                event_bus.emit(
                    EventCategory.CHECKPOINT_VERIFIED,
                    EventPhase.CHECKPOINT,
                    {"checkpoint_path": canonical_best, "message": msg},
                )
            else:
                event_bus.emit(
                    EventCategory.CHECKPOINT_REJECTED,
                    EventPhase.CHECKPOINT,
                    {"checkpoint_path": canonical_best, "error": msg},
                )
        else:
            canonical_best = None

        # Step 5: Save manifest.json (Schema v2) and results.json
        params_pruned = max(0, int(orig_params - final_params))
        comp_ratio = round(final_params / orig_params, 4) if orig_params > 0 else 1.0
        acc_retention = round(final_acc / base_acc, 4) if base_acc > 0 else 1.0
        elapsed_sec = round(time.time() - start_time, 2)
        run_status = "interrupted" if interrupted else raw_res.get("status", "completed")

        results_data = {
            "status": run_status,
            "run_id": run_id,
            "model_id": model,
            "benchmark": benchmark,
            "strategy": resolved_strat.value,
            "preset": preset_cfg.name,
            "original_parameters": int(orig_params),
            "final_parameters": int(final_params),
            "parameters_pruned": params_pruned,
            "compression_ratio": comp_ratio,
            "parameters_pruned_pct": round((params_pruned / orig_params) * 100, 2) if orig_params > 0 else 0.0,
            "baseline_accuracy": base_acc,
            "final_accuracy": final_acc,
            "accuracy_retention_pct": round(acc_retention * 100, 2),
            "retention_target_pct": round(retain * 100, 2),
            "best_checkpoint": canonical_best,
            "reload_verified": reload_verified,
            "elapsed_time_sec": elapsed_sec,
            "stopping_reason": raw_res.get("stopping_reason"),
        }

        if dataset:
            results_data["dataset"] = {
                "source": dataset,
                "format": plan.dataset_format,
                "task_type": plan.dataset_task_type,
                "scorer": plan.dataset_scorer,
            }

        results_json_path = os.path.join(run_dir, "results.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        manifest_data = {
            "schema_version": 2,
            "run_id": run_id,
            "status": run_status,
            "resumable": run_status in ("interrupted", "running"),
            "model_id": model,
            "strategy": resolved_strat.value,
            "benchmark": benchmark,
            "preset": preset_cfg.name,
            "retention_target": retain,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": resolved_dev,
            "checkpoint_directory": canonical_best,
            "resolved_config_path": resolved_cfg_file,
            "results": results_data,
        }
        if dataset:
            manifest_data["dataset"] = results_data["dataset"]
        manifest_json_path = os.path.join(run_dir, "manifest.json")
        with open(manifest_json_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Step 6: Generate Unified Reports & Plots
        run_details = RunManager.get_run(run_dir)
        reports = UnifiedReportGenerator.generate_all_reports(run_details)
        report_md_path = reports.get("summary", os.path.join(run_dir, "reports", "summary.md"))
        root_report_md = os.path.join(run_dir, "report.md")
        if os.path.exists(report_md_path):
            shutil.copy2(report_md_path, root_report_md)
        event_bus.emit(EventCategory.REPORT_GENERATED, EventPhase.REPORTING, {"reports": reports})

        # Final completion event
        if not interrupted:
            event_bus.emit(
                EventCategory.RUN_COMPLETED,
                EventPhase.COMPLETED,
                {"best_checkpoint": canonical_best, "compression_ratio": comp_ratio},
            )

        # Release file lock & flush logger
        event_logger.close()
        run_lock.release()

        # Step 7: Render Summary or Handle JSON output
        result_obj = OptimizationResult(
            status=run_status,
            run_id=run_id,
            best_checkpoint=canonical_best,
            original_params=int(orig_params),
            final_params=int(final_params),
            params_pruned=params_pruned,
            compression_ratio=comp_ratio,
            baseline_accuracy=base_acc,
            final_accuracy=final_acc,
            accuracy_retention=acc_retention,
            strategy_used=resolved_strat.value,
            iterations_completed=raw_res.get("iterations_completed", raw_res.get("metrics", {}).get("full_evaluations_executed", 1)),
            run_dir=run_dir,
            report_path=report_md_path,
            manifest_path=manifest_json_path,
            reload_verified=reload_verified,
            elapsed_time_sec=elapsed_sec,
        )

        if json_output:
            print(json.dumps(result_obj.to_dict(), indent=2))
        elif interrupted:
            console.print(Panel(
                f"[bold yellow]Optimization Interrupted[/bold yellow]\n\n"
                f"Last verified checkpoint preserved:\n"
                f"[cyan]{canonical_best or 'None accepted'}[/cyan]\n\n"
                f"Resume search at any time:\n"
                f"[bold green]prunequest resume {run_dir}[/bold green]",
                border_style="yellow",
            ))
        else:
            table = Table(title="[bold green]PruneQuest — Optimization Complete[/bold green]")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Run ID", run_id)
            table.add_row("Model", model)
            if dataset:
                table.add_row("Evaluation Dataset", f"{dataset} ({plan.dataset_format})")
                table.add_row("Task & Scorer", f"{plan.dataset_task_type} | {plan.dataset_scorer}")
            else:
                table.add_row("Benchmark", benchmark.upper())
            table.add_row("Strategy Used", resolved_strat.value.upper())
            table.add_row("Original Parameters", f"{int(orig_params):,} ({orig_params/1e6:.1f}M)")
            table.add_row("Optimized Parameters", f"{int(final_params):,} ({final_params/1e6:.1f}M)")
            pruned_pct = (params_pruned / orig_params) * 100.0 if orig_params > 0 else 0.0
            table.add_row("Parameters Pruned", f"[bold green]-{params_pruned:,} (-{pruned_pct:.1f}%)[/bold green]")
            table.add_row("Original Accuracy", f"{base_acc * 100:.2f}%")
            table.add_row("Optimized Accuracy", f"{final_acc * 100:.2f}%")
            table.add_row("Measured Retention", f"[bold green]{acc_retention * 100:.2f}%[/bold green] (Target: {retain * 100:.1f}%)")
            reload_str = "[bold green]PASSED[/bold green]" if reload_verified else "[yellow]NOT VERIFIED[/yellow]"
            table.add_row("Checkpoint Reload", reload_str)
            table.add_row("Best Model Path", f"[cyan]{canonical_best or 'None accepted'}[/cyan]")
            table.add_row("Summary Report", f"[dim]{report_md_path}[/dim]")
            table.add_row("Elapsed Time", f"{elapsed_sec:.1f}s")

            console.print("\n")
            console.print(table)
            console.print(f"[bold green]✔ Successfully exported optimized model to:[/bold green] [cyan]{canonical_best or run_dir}[/cyan]\n")

        return result_obj
