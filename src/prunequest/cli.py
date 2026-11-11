"""
PruneQuest Command-Line Interface (Typer + Rich).
"""

import os
import sys
import signal
import platform
from typing import Optional

# Prevent BrokenPipeError / Python traceback when piping CLI output to head, tail, or grep
try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (AttributeError, ValueError):
    pass
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import torch

from prunequest.config import (
    PruneQuestConfig,
    ModelConfig,
    BenchmarkConfig,
    PruningConfig,
)
from prunequest.models.registry import ModelRegistry
from prunequest.models.discovery import ModelInspector
from prunequest.benchmarks.registry import BenchmarkRegistry
from prunequest.benchmarks.spec import BenchmarkSpecCompiler
from prunequest.benchmarks.lm_eval import LMEvalBenchmarkAdapter, LM_EVAL_AVAILABLE
from prunequest.core.controller import SearchController
from prunequest import __version__
from prunequest.reporting.reports import ReportGenerator
from prunequest.core.tracking import ExperimentTracker
from prunequest.export.validation import SubprocessValidator
from prunequest.reporting.metrics import clean_gpu_memory
from prunequest.orchestrator.optimizer import UnifiedOptimizer
from prunequest.runs import RunManager
from prunequest.reporting import UnifiedReportGenerator
from prunequest.datasets import DatasetInspector, DatasetPreviewer

from prunequest.flap.cli import app as flap_app

app = typer.Typer(
    name="prunequest",
    help="Autonomous, Benchmark-Guided LLM Structural Pruning Framework.",
    add_completion=False,
)

block_app = typer.Typer(help="Block-based Structured Depth / Layer Pruning Framework.")

app.add_typer(flap_app, name="flap-based", help="FLAP-based Structured Width / Channel Pruning.")
app.add_typer(flap_app, name="flap", help="Alias for flap-based pruning.")

app.add_typer(block_app, name="block-based", help="Block-based Structured Depth / Layer Pruning.")
app.add_typer(block_app, name="block", help="Alias for block-based pruning.")

benchmarks_app = typer.Typer(help="Discover, inspect, and validate benchmarks.")
app.add_typer(benchmarks_app, name="benchmarks")

runs_app = typer.Typer(help="Manage, inspect, compare, and view logs of optimization runs.")
app.add_typer(runs_app, name="runs")

datasets_app = typer.Typer(help="Universal dataset inspection, validation, and preview.")
app.add_typer(datasets_app, name="datasets")

console = Console()


@app.command()
def version():
    """Display PruneQuest version and core system information."""
    console.print(f"[bold cyan]PruneQuest[/bold cyan] version [bold green]v{__version__}[/bold green]")


@app.command()
def doctor():
    """Verify hardware environment, PyTorch, CUDA, memory, and dependencies."""
    table = Table(title="[bold blue]PruneQuest Environment Diagnostics[/bold blue]")
    table.add_column("Component", style="cyan")
    table.add_column("Status / Value", style="magenta")

    # Platform & Python
    table.add_row("Python Version", f"{platform.python_version()} ({sys.executable})")
    table.add_row("Operating System", f"{platform.system()} {platform.release()} ({platform.machine()})")
    table.add_row("PyTorch Version", str(torch.__version__))

    # CUDA & GPU Diagnostics
    cuda_available = torch.cuda.is_available()
    table.add_row("CUDA Available", f"[green]{cuda_available}[/green]" if cuda_available else "[yellow]False (CPU Mode)[/yellow]")
    if cuda_available:
        table.add_row("Device Name", torch.cuda.get_device_name(0))
        table.add_row("Device Count", str(torch.cuda.device_count()))
        free_mem, total_mem = torch.cuda.mem_get_info()
        table.add_row("Free VRAM", f"{free_mem / (1024**3):.2f} GB")
        table.add_row("Total VRAM", f"{total_mem / (1024**3):.2f} GB")
        table.add_row("CUDA Runtime Version", str(torch.version.cuda))
    else:
        table.add_row("Hardware Mode", "[yellow]CPU Only (Compatible with architecture discovery, CPU validation, and lightweight unit testing)[/yellow]")

    # Core Dependencies
    core_deps = [
        "transformers",
        "accelerate",
        "safetensors",
        "datasets",
        "math_verify",
        "pydantic",
        "rich",
        "typer",
    ]
    for pkg in core_deps:
        try:
            m = __import__(pkg)
            v = getattr(m, "__version__", "Installed")
            table.add_row(f"Core Dep: {pkg}", f"[green]{v}[/green]")
        except ImportError:
            table.add_row(f"Core Dep: {pkg}", "[red]Not Installed[/red]")

    # Optional Evaluation Extras
    if LM_EVAL_AVAILABLE:
        try:
            import lm_eval
            v = getattr(lm_eval, "__version__", "Installed")
            table.add_row("Optional Extra: lm-eval", f"[green]{v} (Evaluation extra active)[/green]")
            task_count = len(LMEvalBenchmarkAdapter.list_available_tasks())
            table.add_row("LM-Eval Tasks Discovered", f"[green]{task_count:,} tasks available[/green]")
        except Exception:
            table.add_row("Optional Extra: lm-eval", "[green]Installed[/green]")
    else:
        table.add_row(
            "Optional Extra: lm-eval",
            "[yellow]Not Installed (Optional: install with pip install 'prunequest[eval]')[/yellow]"
        )

    # Writable Directories
    cwd_writable = os.access(".", os.W_OK)
    table.add_row("Workspace Directory", f"{os.path.abspath('.')} ([green]Writable[/green]" if cwd_writable else "[red]Read-Only[/red])")

    hf_cache = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(hf_cache, exist_ok=True)
    hf_writable = os.access(hf_cache, os.W_OK)
    table.add_row("Hugging Face Cache", f"{hf_cache} ([green]Writable[/green]" if hf_writable else "[red]Read-Only[/red])")

    # Capability Matrix Summary
    table.add_row("Verified Model Architectures", "Qwen2 / 2.5 (Level 3), Llama 2 / 3 (Level 3), Mistral (Level 3)")
    table.add_row("Experimental Architectures", "Gemma / Gemma-2 (Level 2), Phi-2 / Phi-3 (Level 2)")
    table.add_row("Supported Pruning Methods", "Block Influence (Layer), FLAP (MLP Intermediate), JIPO (Full Sweep)")

    console.print(table)


@app.command()
@block_app.command(name="inspect")
def inspect(
    model: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", "--model", "-m", help="Hugging Face model ID or directory"),
    load_weights: bool = typer.Option(False, "--load-weights", help="Load full weights to inspect actual tensors"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Device to inspect model on"),
):
    """Inspect model architecture, parameters, capability level, and memory estimation."""
    console.print(Panel(f"[bold cyan]Inspecting Model Architecture:[/bold cyan] {model}"))

    with console.status("[bold green]Inspecting model metadata...[/bold green]"):
        report = ModelInspector.inspect(model)

    desc = report.descriptor
    table = Table(title="[bold blue]Model Architecture & Capability Assessment[/bold blue]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Model ID / Path", desc.model_id_or_path)
    table.add_row("Architecture Class", desc.architecture_class.value)
    table.add_row("Model Type", desc.model_type)
    table.add_row("Architectures", ", ".join(desc.architectures) if desc.architectures else "N/A")
    table.add_row("Capability Level", f"Level {report.capability_level.value} ({report.capability_level.name})")
    table.add_row("Resolved Adapter", report.adapter_name or "None")
    table.add_row("Transformer Layers", str(desc.num_hidden_layers))
    table.add_row("Hidden Size", str(desc.hidden_size))
    table.add_row("Attention Heads (Q)", str(desc.num_attention_heads))
    table.add_row("Attention Heads (KV)", str(desc.num_key_value_heads))
    table.add_row("Vocabulary Size", str(desc.vocab_size))
    if desc.quantization_type:
        table.add_row("Quantization", f"[yellow]{desc.quantization_type}[/yellow]")
    table.add_row("Supported Pruning Operations", ", ".join(report.supported_operations) or "Evaluation Only")
    if report.notes:
        table.add_row("Notes", "; ".join(report.notes))

    console.print(table)

    if load_weights:
        console.print("[dim]Loading model weights to count exact parameters...[/dim]")
        adapter = ModelRegistry.get_adapter(model_id_or_path=model, device=device)
        adapter.load()
        meta = adapter.get_metadata()
        console.print(f"[bold green]Total Parameters:[/bold green] {meta.total_params:,} ({meta.total_params_million:.2f}M)")
        adapter.unload()


@benchmarks_app.command("list")
def list_benchmarks_cmd(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of benchmarks displayed"),
):
    """List native benchmarks and popular evaluation tasks."""
    bench_list = BenchmarkRegistry.list_benchmarks()
    table = Table(title="[bold blue]Supported Benchmark Adapters[/bold blue]")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Metric", style="green")
    table.add_column("Direction", style="yellow")

    items = list(bench_list.items())
    if limit is not None and limit > 0:
        items = items[:limit]

    for name, props in items:
        table.add_row(
            name,
            props.get("source", "native"),
            props.get("metric_name", "acc"),
            props.get("metric_direction", "higher"),
        )

    console.print(table)


@benchmarks_app.command("search")
def search_benchmarks_cmd(query: str = typer.Argument(..., help="Search query for benchmarks")):
    """Search thousands of LM Evaluation Harness tasks and native benchmarks."""
    results = BenchmarkRegistry.search_benchmarks(query)
    table = Table(title=f"[bold blue]Benchmark Search Results for '{query}'[/bold blue]")
    table.add_column("Task Name", style="cyan")

    for t in results[:50]:
        table.add_row(t)

    console.print(table)
    if len(results) > 50:
        console.print(f"[dim]Showing 50 of {len(results)} matches.[/dim]")


@benchmarks_app.command("info")
def info_benchmark_cmd(task_name: str = typer.Argument(..., help="Task name to inspect")):
    """Get metadata for an LM Evaluation Harness benchmark task."""
    info = LMEvalBenchmarkAdapter.get_task_info(task_name)
    table = Table(title=f"[bold blue]Task Metadata: {task_name}[/bold blue]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    for k, v in info.items():
        table.add_row(k, str(v))

    console.print(table)


@benchmarks_app.command("validate")
def validate_benchmark_spec(
    spec_file: Optional[str] = typer.Argument(None, help="Path to BenchmarkSpec YAML file"),
    spec: Optional[str] = typer.Option(None, "--spec", "-s", help="Path to BenchmarkSpec YAML file"),
):
    """Validate a declarative BenchmarkSpec YAML file."""
    path = spec or spec_file
    if not path:
        console.print("[bold red]Error: Missing benchmark specification file. Provide as argument or via --spec.[/bold red]")
        raise typer.Exit(1)
    if not os.path.exists(path):
        console.print(f"[bold red]Error: Specification file not found: {path}[/bold red]")
        raise typer.Exit(1)

    valid, errors = BenchmarkSpecCompiler.validate_spec(path)
    if valid:
        console.print(f"[bold green]✔ Specification '{path}' is valid![/bold green]")
    else:
        console.print(f"[bold red]✘ Specification '{path}' is invalid:[/bold red]")
        for err in errors:
            console.print(f"  - {err}")
        raise typer.Exit(1)


@benchmarks_app.command("preview")
def preview_benchmark_spec(
    spec_file: Optional[str] = typer.Argument(None, help="Path to BenchmarkSpec YAML file"),
    spec: Optional[str] = typer.Option(None, "--spec", "-s", help="Path to BenchmarkSpec YAML file"),
    count: int = typer.Option(3, "--count", "-n", help="Number of preview samples"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Alias for --count"),
):
    """Preview rendered prompts and target extraction for a BenchmarkSpec."""
    path = spec or spec_file
    if not path:
        console.print("[bold red]Error: Missing benchmark specification file. Provide as argument or via --spec.[/bold red]")
        raise typer.Exit(1)
    if not os.path.exists(path):
        console.print(f"[bold red]Error: Specification file not found: {path}[/bold red]")
        raise typer.Exit(1)

    effective_count = limit if limit is not None else count
    try:
        samples = BenchmarkSpecCompiler.preview_spec(path, count=effective_count)
        table = Table(title=f"[bold blue]Benchmark Preview: {path}[/bold blue]")
        table.add_column("ID", style="cyan")
        table.add_column("Rendered Prompt", style="white")
        table.add_column("Gold Target", style="green")

        for s in samples:
            table.add_row(s["id"], s["rendered_prompt"], s["gold_answer"])

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to preview '{path}': {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
@block_app.command(name="baseline")
def baseline(
    model: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", "--model", "-m"),
    benchmark: str = typer.Option("gsm8k", "--benchmark", "--benchmarks", "-b", help="One or more comma-separated benchmarks"),
    split: str = typer.Option("dev", "--split", "-s", help="dev, acceptance, smoke, screening, or test"),
    samples: Optional[int] = typer.Option(None, "--samples", "-n", help="Sample count limit"),
    device: str = typer.Option("cuda:0", "--device", "-d"),
    output: str = typer.Option("runs/prunequest_baseline", "--output", "--output-dir", "-o"),
):
    """Evaluate original unpruned model across one or more benchmarks."""
    console.print(Panel(f"[bold cyan]Establishing Baseline for {model} on {benchmark} ({split} split)[/bold cyan]"))

    cfg = PruneQuestConfig(
        model=ModelConfig(model_id=model, device=device),
        benchmark=BenchmarkConfig(name=benchmark, split=split, sample_limit=samples),
        output_dir=output,
    )
    controller = SearchController(cfg)
    res = controller.run_baseline(split=split, max_samples=samples)

    table = Table(title="[bold green]Baseline Measurement Results[/bold green]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Samples Evaluated", str(res["sample_size"]))
    table.add_row("Correct Responses", str(res["num_correct"]))
    table.add_row("Primary Accuracy", f"{res['accuracy']*100:.2f}%")
    table.add_row("Peak VRAM", f"{res['peak_vram_gb']:.2f} GB")

    console.print(table)


@app.command()
@block_app.command(name="run")
def run(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Pretrained model ID or path"),
    benchmark: Optional[str] = typer.Option(None, "--benchmark", "--benchmarks", "-b", help="One or more comma-separated benchmarks"),
    retain: Optional[float] = typer.Option(None, "--retain", "-r", help="Minimum accuracy retention ratio [0.0 - 1.0]"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", "-i", help="Maximum pruning iterations"),
    split: Optional[str] = typer.Option(None, "--split", "-s", help="dev, acceptance, smoke, or test"),
    samples: Optional[int] = typer.Option(None, "--samples", "-n", help="Sample limit for benchmark evaluation"),
    output: Optional[str] = typer.Option(None, "--output", "--output-dir", "-o", help="Run directory"),
    reports: Optional[str] = typer.Option(None, "--reports", help="Reports directory"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Inference device"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
):
    """Execute complete autonomous, benchmark-guided pruning loop."""
    # Enforce Precedence: Explicit CLI Overrides > Configuration File > Application Defaults
    if config and os.path.exists(config):
        cfg = PruneQuestConfig.from_yaml(config)
        if model is not None:
            cfg.model.model_id = model
        if device is not None:
            cfg.model.device = device
        if benchmark is not None:
            cfg.benchmark.name = benchmark
        if split is not None:
            cfg.benchmark.split = split
        if samples is not None:
            cfg.benchmark.sample_limit = samples
        if retain is not None:
            cfg.pruning.retain = retain
        if max_iterations is not None:
            cfg.pruning.max_iterations = max_iterations
        if output is not None:
            cfg.output_dir = output
        if reports is not None:
            cfg.reports_dir = reports
    else:
        cfg = PruneQuestConfig(
            model=ModelConfig(
                model_id=model or "Qwen/Qwen2.5-Math-1.5B-Instruct",
                device=device or "cuda:0",
            ),
            benchmark=BenchmarkConfig(
                name=benchmark or "gsm8k",
                split=split or "dev",
                sample_limit=samples,
            ),
            pruning=PruningConfig(
                retain=retain if retain is not None else 0.99,
                max_iterations=max_iterations if max_iterations is not None else 3,
            ),
            output_dir=output or "runs/prunequest_run",
            reports_dir=reports or "reports",
        )

    # Safe Defaults Inspection and User Guidance
    if cfg.benchmark.split == "smoke" or (cfg.benchmark.sample_limit and cfg.benchmark.sample_limit <= 32):
        console.print(
            f"[bold yellow]Warning: This run uses only {cfg.benchmark.sample_limit or cfg.benchmark.smoke_size} examples "
            f"({cfg.benchmark.split} split).\nThis is suitable for a smoke test, not for reliable accuracy-retention claims.[/bold yellow]\n"
        )

    # Save Resolved Configuration for Reproducibility
    os.makedirs(cfg.output_dir, exist_ok=True)
    resolved_path = os.path.join(cfg.output_dir, "resolved_config.yaml")
    cfg.to_yaml(resolved_path)

    effective_eval_size = cfg.benchmark.sample_limit or (cfg.benchmark.dev_size if cfg.benchmark.split == "dev" else cfg.benchmark.smoke_size)
    console.print(Panel(
        f"[bold cyan]PruneQuest v{__version__} — Block Influence Pruning[/bold cyan]\n"
        f"Model: [green]{cfg.model.model_id}[/green] | Method: [yellow]Block Influence[/yellow]\n"
        f"Benchmark: [yellow]{cfg.benchmark.name}[/yellow] (Split: [magenta]{cfg.benchmark.split}[/magenta], Samples: [magenta]{effective_eval_size}[/magenta])\n"
        f"Retention Target: [magenta]{cfg.pruning.retain * 100:.1f}%[/magenta] | Device: [cyan]{cfg.model.device}[/cyan] | Output: [green]{cfg.output_dir}[/green]\n"
        f"Resolved Config: [dim]{resolved_path}[/dim]"
    ))

    controller = SearchController(cfg)

    def on_step(record):
        rprint(f"[bold green]✔ Accepted Iteration:[/bold green] {record['iteration_id']} | "
               f"Params: {record['params_after']/1e6:.2f}M | "
               f"Accuracy: {record['accuracy']*100:.2f}% (Retention: {record['accuracy_retention_pct']:.1f}%)")

    res = controller.run(progress_callback=on_step)

    df = controller.tracker.get_history_dataframe()
    reporter = ReportGenerator(cfg.reports_dir)
    plots = reporter.generate_plots(df)
    report_file = reporter.generate_report(df, res, plots)

    console.print("\n")
    summary_table = Table(title="[bold green]Pruning Search Execution Summary[/bold green]")
    summary_table.add_column("Result Field", style="cyan")
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Status", res["status"])
    summary_table.add_row("Original Parameters", f"{res['original_params']:,}")
    summary_table.add_row("Final Parameters", f"{res['final_params']:,}")
    if res.get("original_params") and res.get("final_params"):
        pruned_count = res["original_params"] - res["final_params"]
        summary_table.add_row("Parameters Pruned", f"{pruned_count:,} ({pruned_count/1e6:.2f}M)")
    summary_table.add_row("Baseline Accuracy", f"{res['baseline_accuracy']*100:.2f}%")
    summary_table.add_row("Final Accuracy", f"{res['final_accuracy']*100:.2f}%")
    summary_table.add_row("Accuracy Retained", f"{res['accuracy_retention_pct']:.2f}%")
    summary_table.add_row("Iterations Completed", str(res["iterations_completed"]))
    summary_table.add_row("Best Exported Checkpoint", str(res["best_checkpoint"]))
    summary_table.add_row("Report Path", report_file)
    summary_table.add_row("Resolved Config Saved", resolved_path)

    console.print(summary_table)


@app.command(name="optimize")
def optimize_cmd(
    model: str = typer.Argument(..., help="Hugging Face model ID or local directory to optimize"),
    benchmark: str = typer.Option("gsm8k", "--benchmark", "-b", help="Benchmark guiding pruning (e.g. gsm8k, arc_easy)"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-D", help="Custom evaluation dataset path (.jsonl, .csv, .parquet) or Hugging Face dataset ID"),
    input_column: Optional[str] = typer.Option(None, "--input-column", help="Explicit input/prompt column name"),
    target_column: Optional[str] = typer.Option(None, "--target-column", help="Explicit target/gold answer column name"),
    metric: Optional[str] = typer.Option(None, "--metric", help="Explicit evaluation metric (exact_match, normalized_match, numeric_exact_match, math_verify, multiple_choice)"),
    calibration_dataset: Optional[str] = typer.Option(None, "--calibration-dataset", help="Optional separate calibration dataset path or ID"),
    retain: float = typer.Option(0.99, "--retain", "-r", help="Target minimum accuracy retention (e.g. 0.99 for 99%)"),
    preset: str = typer.Option("balanced", "--preset", "-p", help="Optimization budget preset: quick | balanced | thorough"),
    strategy: str = typer.Option("auto", "--strategy", "-s", help="Pruning strategy: auto | joint | flap | block"),
    device: str = typer.Option("auto", "--device", "-d", help="Hardware device: auto | cuda:0 | cpu | mps"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Custom run output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run preflight inspection only without downloading or loading weights"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume interrupted optimization if run directory exists"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Optional YAML config path to override settings"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size override for evaluation"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Maximum search iterations override"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable interactive live dashboard and use plain text output"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON summary on completion"),
    json_events: bool = typer.Option(False, "--json-events", help="Stream structured JSON Lines events in real time"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal console output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose console output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """
    Autonomous one-command LLM pruning.
    Finds the smallest verified model satisfying your retention threshold.
    """
    import logging
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        logging.basicConfig(level=logging.INFO)

    cmd_console = Console(quiet=quiet or json_output)
    try:
        res = UnifiedOptimizer.optimize(
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
            console=cmd_console,
            dataset=dataset,
            input_column=input_column,
            target_column=target_column,
            metric=metric,
            calibration_dataset=calibration_dataset,
        )
        if res.status == "failed":
            raise typer.Exit(1)
    except Exception as e:
        if debug:
            raise
        if not json_output:
            cmd_console.print(f"[bold red]Optimization failed:[/bold red] {e}")
        raise typer.Exit(1)


@runs_app.command(name="list")
def runs_list(
    limit: Optional[int] = typer.Option(20, "--limit", "-l", help="Maximum number of runs to display"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status (completed, interrupted, running, failed)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model name substring"),
    sort: str = typer.Option("newest", "--sort", help="Sort order: newest | oldest | compression | retention"),
    json_output: bool = typer.Option(False, "--json", help="Output runs as machine-readable JSON"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Root directory containing experiment runs"),
):
    """List prior optimization experiments without loading model weights."""
    import json
    runs = RunManager.list_runs(runs_root=runs_dir, limit=limit, status=status, model=model, sort=sort)
    if json_output:
        print(json.dumps([r.to_dict() for r in runs], indent=2))
        return

    if not runs:
        console.print(f"[yellow]No runs found in '{runs_dir}' matching criteria.[/yellow]")
        return

    table = Table(title="[bold green]PruneQuest Experiments[/bold green]")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Model", style="white")
    table.add_column("Strategy", style="yellow")
    table.add_column("Benchmark", style="blue")
    table.add_column("Compression", style="magenta")
    table.add_column("Retention", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Created", style="dim")

    for r in runs:
        status_color = "green" if r.status == "completed" else "yellow" if r.status == "interrupted" else "red" if r.status == "failed" else "cyan"
        table.add_row(
            r.run_id,
            r.model_id.split("/")[-1],
            r.strategy.upper(),
            r.benchmark.upper(),
            f"{r.compression_pct:.1f}%",
            f"{r.accuracy_retention_pct:.1f}%",
            f"[{status_color}]{r.status.upper()}[/{status_color}]",
            r.created_at[:19].replace("T", " ") if r.created_at else "—",
        )
    console.print(table)


@runs_app.command(name="show")
def runs_show(
    run_id: str = typer.Argument(..., help="Run ID or path to run directory"),
    json_output: bool = typer.Option(False, "--json", help="Output run details as JSON"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Root directory containing experiment runs"),
):
    """Inspect detailed provenance and metrics for an optimization run without loading models."""
    import json
    try:
        run = RunManager.get_run(run_id, runs_root=runs_dir)
    except Exception as e:
        console.print(f"[bold red]Failed to inspect run '{run_id}': {e}[/bold red]")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(run.to_dict(), indent=2))
        return

    table = Table(title=f"[bold green]PruneQuest Run Details — {run.run_id}[/bold green]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Run ID", run.run_id)
    table.add_row("Model ID", run.model_id)
    table.add_row("Benchmark", run.benchmark.upper())
    table.add_row("Strategy", run.strategy.upper())
    table.add_row("Preset", run.preset.upper())
    table.add_row("Device", run.device)
    table.add_row("Status", run.status.upper())
    table.add_row("Resumable", str(run.resumable))
    table.add_row("Target Retention", f"{run.retention_target_pct:.1f}%")
    table.add_row("Original Parameters", f"{run.original_parameters:,}")
    table.add_row("Final Parameters", f"{run.final_parameters:,}")
    table.add_row("Parameters Pruned", f"{run.parameters_pruned:,} ({run.compression_pct:.2f}%)")
    table.add_row("Baseline Accuracy", f"{run.baseline_accuracy*100:.2f}%")
    table.add_row("Final Accuracy", f"{run.final_accuracy*100:.2f}%")
    table.add_row("Accuracy Retention", f"{run.accuracy_retention_pct:.2f}%")
    table.add_row("Reload Verified", "[bold green]PASSED[/bold green]" if run.reload_verified else "[yellow]NOT VERIFIED[/yellow]")
    table.add_row("Best Checkpoint", run.best_checkpoint or "None accepted")
    table.add_row("Created At", run.created_at)
    table.add_row("Elapsed Time", f"{run.elapsed_time_sec:.1f}s")
    table.add_row("Run Directory", run.run_dir)

    console.print(table)


@runs_app.command(name="compare")
def runs_compare(
    run_a: str = typer.Argument(..., help="Run ID or path to first run"),
    run_b: str = typer.Argument(..., help="Run ID or path to second run"),
    json_output: bool = typer.Option(False, "--json", help="Output comparison as JSON"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Root directory containing experiment runs"),
):
    """Compare two optimization experiments and evaluate consistency."""
    import json
    try:
        cmp = RunManager.compare_runs(run_a, run_b, runs_root=runs_dir)
    except Exception as e:
        console.print(f"[bold red]Failed to compare runs: {e}[/bold red]")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(cmp.to_dict(), indent=2))
        return

    if not cmp.is_compatible:
        for note in cmp.compatibility_notes:
            console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {note}")

    table = Table(title="[bold green]PruneQuest Experiment Comparison[/bold green]")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Run A: {cmp.run_a.run_id}", style="white")
    table.add_column(f"Run B: {cmp.run_b.run_id}", style="white")
    table.add_column("Difference (B vs A)", style="magenta")

    table.add_row("Model", cmp.run_a.model_id.split("/")[-1], cmp.run_b.model_id.split("/")[-1], "—")
    table.add_row("Strategy", cmp.run_a.strategy.upper(), cmp.run_b.strategy.upper(), "—")
    table.add_row("Benchmark", cmp.run_a.benchmark.upper(), cmp.run_b.benchmark.upper(), "—")
    table.add_row("Final Parameters", f"{cmp.run_a.final_parameters:,}", f"{cmp.run_b.final_parameters:,}", f"{cmp.param_reduction_diff:+,}")
    table.add_row("Compression", f"{cmp.run_a.compression_pct:.2f}%", f"{cmp.run_b.compression_pct:.2f}%", f"{cmp.compression_diff_pct:+.2f}%")
    table.add_row("Baseline Accuracy", f"{cmp.run_a.baseline_accuracy*100:.2f}%", f"{cmp.run_b.baseline_accuracy*100:.2f}%", f"{(cmp.run_b.baseline_accuracy - cmp.run_a.baseline_accuracy)*100:+.2f}%")
    table.add_row("Final Accuracy", f"{cmp.run_a.final_accuracy*100:.2f}%", f"{cmp.run_b.final_accuracy*100:.2f}%", f"{cmp.accuracy_diff*100:+.2f}%")
    table.add_row("Accuracy Retention", f"{cmp.run_a.accuracy_retention_pct:.2f}%", f"{cmp.run_b.accuracy_retention_pct:.2f}%", f"{cmp.retention_diff_pct:+.2f}%")
    table.add_row("Elapsed Time", f"{cmp.run_a.elapsed_time_sec:.1f}s", f"{cmp.run_b.elapsed_time_sec:.1f}s", f"{cmp.elapsed_time_diff_sec:+.1f}s")
    table.add_row("Reload Verified", str(cmp.run_a.reload_verified), str(cmp.run_b.reload_verified), "—")

    console.print(table)


@runs_app.command(name="logs")
def runs_logs(
    run_id: str = typer.Argument(..., help="Run ID or path to run directory"),
    phase: Optional[str] = typer.Option(None, "--phase", "-p", help="Filter logs by phase"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter logs by event category"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum event logs to show"),
    json_output: bool = typer.Option(False, "--json", help="Output log events as JSON Lines"),
    runs_dir: str = typer.Option("runs", "--runs-dir", help="Root directory containing experiment runs"),
):
    """View structured event logs for an optimization run."""
    try:
        events = RunManager.get_run_logs(run_id, phase=phase, category=category, limit=limit, runs_root=runs_dir)
    except Exception as e:
        console.print(f"[bold red]Failed to read logs for '{run_id}': {e}[/bold red]")
        raise typer.Exit(1)

    if json_output:
        for evt in events:
            print(evt.to_json())
        return

    if not events:
        console.print(f"[yellow]No logs found for run '{run_id}' matching criteria.[/yellow]")
        return

    table = Table(title=f"[bold green]PruneQuest Event Logs — {run_id}[/bold green]")
    table.add_column("Seq", style="dim", justify="right")
    table.add_column("Timestamp", style="dim")
    table.add_column("Phase", style="cyan")
    table.add_column("Event Type", style="yellow")
    table.add_column("Details", style="white")

    for evt in events:
        ts = evt.timestamp[11:19] if len(evt.timestamp) >= 19 else evt.timestamp
        summary_payload = ", ".join(f"{k}={v}" for k, v in (evt.payload or {}).items() if k not in ("raw", "checkpoint_path"))[:60]
        table.add_row(
            str(evt.sequence),
            ts,
            evt.phase.upper(),
            evt.event_type,
            summary_payload,
        )

    console.print(table)


# =========================================================================
# Datasets Command Group (UX Release C: Universal Dataset Onboarding)
# =========================================================================

@datasets_app.command(name="inspect")
def datasets_inspect(
    source: str = typer.Argument(..., help="Path to local file (.jsonl, .csv, .parquet) or Hugging Face dataset ID"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Dataset configuration or subset name"),
    revision: Optional[str] = typer.Option("main", "--revision", help="Git revision or branch"),
    split: Optional[str] = typer.Option(None, "--split", "-s", help="Specific split to inspect"),
    input_column: Optional[str] = typer.Option(None, "--input-column", help="Explicit input column override"),
    target_column: Optional[str] = typer.Option(None, "--target-column", help="Explicit target column override"),
    metric: Optional[str] = typer.Option(None, "--metric", help="Explicit evaluation metric override"),
    json_output: bool = typer.Option(False, "--json", help="Output inspection report as pure JSON"),
):
    """Inspect dataset source, detect fields, infer task type and verified scorer without loading model weights."""
    try:
        report = DatasetInspector.inspect(
            source=source,
            config=config,
            revision=revision,
            split=split,
            explicit_input=input_column,
            explicit_target=target_column,
            explicit_metric=metric,
        )
    except Exception as e:
        console.print(f"[bold red]Dataset inspection failed:[/bold red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(report.model_dump_json(indent=2))
        return

    table = Table(title=f"[bold green]PruneQuest Dataset Inspection — {source}[/bold green]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Source", report.source)
    table.add_row("Source Type", report.source_type.upper())
    table.add_row("Format", report.format.upper())
    table.add_row("Estimated Samples", f"{report.total_samples:,}" if report.total_samples else "Unknown")
    table.add_row("Detected Splits", ", ".join(report.detected_splits) if report.detected_splits else "None")
    table.add_row("Columns", ", ".join(report.columns))
    table.add_row("Resolved Input", report.input_column or "[red]Unresolved[/red]")
    table.add_row("Resolved Target", report.target_column or "[red]Unresolved[/red]")
    conf_color = "green" if report.mapping_confidence == "high" else "yellow" if report.mapping_confidence == "ambiguous" else "red"
    table.add_row("Mapping Confidence", f"[{conf_color}]{report.mapping_confidence.upper()}[/{conf_color}]")
    table.add_row("Detected Task", report.task_type.upper())
    dir_str = "Higher is better" if report.higher_is_better else "Lower is better"
    table.add_row("Proposed Scorer", f"{report.proposed_scorer} ({dir_str})")

    console.print(table)

    if report.warnings:
        for w in report.warnings:
            console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {w}")

    if report.sample_preview:
        console.print("\n")
        console.print(DatasetPreviewer.build_preview_table(report.sample_preview))


@datasets_app.command(name="preview")
def datasets_preview(
    source: str = typer.Argument(..., help="Dataset path or Hugging Face identifier"),
    limit: int = typer.Option(3, "--limit", "-l", help="Number of preview samples to render"),
    input_column: Optional[str] = typer.Option(None, "--input-column", help="Explicit input column override"),
    target_column: Optional[str] = typer.Option(None, "--target-column", help="Explicit target column override"),
    template: str = typer.Option("{{ input }}", "--template", help="Prompt template (must contain '{{ input }}')"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="Optional system instruction"),
    json_output: bool = typer.Option(False, "--json", help="Output preview items as JSON"),
):
    """Preview rendered model prompts and reference answers from a dataset."""
    import json
    try:
        previews = DatasetInspector.preview(
            source=source,
            limit=limit,
            explicit_input=input_column,
            explicit_target=target_column,
            prompt_template=template,
            system_prompt=system_prompt,
        )
    except Exception as e:
        console.print(f"[bold red]Dataset preview failed:[/bold red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(previews, indent=2))
        return

    console.print(DatasetPreviewer.build_preview_table(previews))


@datasets_app.command(name="validate")
def datasets_validate(
    source: str = typer.Argument(..., help="Dataset path or Hugging Face identifier"),
    input_column: Optional[str] = typer.Option(None, "--input-column", help="Explicit input column override"),
    target_column: Optional[str] = typer.Option(None, "--target-column", help="Explicit target column override"),
    metric: Optional[str] = typer.Option(None, "--metric", help="Explicit evaluation metric override"),
    calibration_size: int = typer.Option(128, "--calibration-size", help="Calibration sample count"),
    screening_size: int = typer.Option(64, "--screening-size", help="Screening sample count"),
    dev_size: int = typer.Option(512, "--dev-size", help="Development sample count"),
    seed: int = typer.Option(42, "--seed", help="Random seed for partitioning"),
    json_output: bool = typer.Option(False, "--json", help="Output validation report as JSON"),
):
    """Validate dataset schema, columns, scorer compatibility, and partition disjointness."""
    val = DatasetInspector.validate(
        source=source,
        explicit_input=input_column,
        explicit_target=target_column,
        explicit_metric=metric,
        calibration_size=calibration_size,
        screening_size=screening_size,
        dev_size=dev_size,
        seed=seed,
    )

    if json_output:
        print(val.model_dump_json(indent=2))
        if not val.is_valid:
            raise typer.Exit(1)
        return

    if val.is_valid:
        console.print(Panel(
            f"[bold green]✔ Dataset Validation PASSED: {source}[/bold green]\n"
            f"[dim]Task: {val.task_type.upper()} | Scorer: {val.scorer} | "
            f"Input: {val.input_column} | Target: {val.target_column}[/dim]\n"
            f"[dim]Partitions: {val.partition_counts}[/dim]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]✗ Dataset Validation FAILED: {source}[/bold red]\n"
            + "\n".join(f"[red]• {err}[/red]" for err in val.errors),
            border_style="red",
        ))
        raise typer.Exit(1)


@app.command()
@block_app.command(name="resume")
def resume(
    run_dir: str = typer.Argument(..., help="Path to previous run directory containing experiments.db or manifest.json"),
):
    """Resume an interrupted pruning search from the last valid checkpoint."""
    console.print(Panel(f"[bold cyan]Resuming PruneQuest Search from:[/bold cyan] {run_dir}"))

    if not os.path.exists(run_dir):
        console.print(f"[bold red]Run directory does not exist: {run_dir}[/bold red]")
        raise typer.Exit(1)

    import json
    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            model_id = manifest_data.get("model_id")
            strategy = manifest_data.get("results", {}).get("strategy", "auto")
            resolved_cfg = manifest_data.get("resolved_config_path")
            res = UnifiedOptimizer.optimize(
                model=model_id,
                strategy=strategy,
                config=resolved_cfg if (resolved_cfg and os.path.exists(resolved_cfg)) else None,
                output=run_dir,
                resume=True,
                console=console,
            )
            console.print("[bold green]✔ Resumed run finished successfully![/bold green]")
            return
        except Exception as e:
            console.print(f"[bold red]Failed to resume search from manifest: {e}[/bold red]")
            raise typer.Exit(1)

    tracker = ExperimentTracker(run_dir)
    latest = tracker.get_latest_accepted()
    if not latest:
        console.print("[bold red]No accepted checkpoints found in run directory to resume from.[/bold red]")
        raise typer.Exit(1)

    # Load resolved config if available, otherwise reconstruct
    resolved_cfg_file = os.path.join(run_dir, "resolved_config.yaml")
    if os.path.exists(resolved_cfg_file):
        cfg = PruneQuestConfig.from_yaml(resolved_cfg_file)
    else:
        cfg = PruneQuestConfig(output_dir=run_dir)

    controller = SearchController(cfg, tracker=tracker)
    try:
        res = controller.resume()
    except Exception as e:
        console.print(f"[bold red]Failed to resume search: {e}[/bold red]")
        raise typer.Exit(1)

    console.print("[bold green]✔ Resumed run finished successfully![/bold green]")
    console.print(f"Final params: {res['final_params']:,} | Final accuracy: {res['final_accuracy']*100:.2f}%")


@app.command()
@block_app.command(name="report")
def report(
    run_dir: str = typer.Argument(..., help="Path to run directory containing manifest.json or experiments.db"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Directory to save report and plots (defaults to <run_dir>/reports)"),
):
    """Generate or regenerate Markdown reports, CSV history, and publication-grade plots without loading models."""
    if not os.path.exists(run_dir):
        console.print(f"[bold red]Run directory does not exist: {run_dir}[/bold red]")
        raise typer.Exit(1)

    try:
        run_details = RunManager.get_run(run_dir)
        reports = UnifiedReportGenerator.generate_all_reports(run_details, output_dir=output)
        summary_path = reports.get("summary")
        console.print("[bold green]✔ Reports and visualizations generated successfully![/bold green]")
        console.print(f"Summary Report: [cyan]{summary_path}[/cyan]")
        console.print(f"Detailed Report: [cyan]{reports.get('detailed')}[/cyan]")
        console.print(f"Candidate History CSV: [cyan]{reports.get('history_csv')}[/cyan]")
        return
    except Exception:
        pass

    # Legacy fallback for old block-based runs with only experiments.db
    tracker = ExperimentTracker(run_dir)
    df = tracker.get_history_dataframe()
    if df.empty:
        console.print("[bold red]No experiment records found.[/bold red]")
        raise typer.Exit(1)

    latest = tracker.get_latest_accepted() or {}
    res = {
        "status": "completed",
        "best_checkpoint": latest.get("checkpoint_path"),
        "original_params": df.iloc[0].get("params_before", 0),
        "final_params": latest.get("params_after", 0),
        "baseline_accuracy": df.iloc[0].get("accuracy", 0.0),
        "final_accuracy": latest.get("accuracy", 0.0),
        "accuracy_retention_pct": latest.get("accuracy_retention_pct", 100.0),
        "iterations_completed": len(df),
        "run_dir": run_dir,
    }

    target_out = output or os.path.join(run_dir, "reports")
    reporter = ReportGenerator(target_out)
    plots = reporter.generate_plots(df)
    report_file = reporter.generate_report(df, res, plots)
    console.print(f"[bold green]✔ Report and figures generated at: {report_file}[/bold green]")


@app.command()
@block_app.command(name="export")
def export(
    checkpoint: str = typer.Argument(..., help="Path to source checkpoint"),
    output: str = typer.Option(..., "--output", "-o", help="Target output directory"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Perform standalone reload check"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Device for validation"),
):
    """Export and independently validate a pruned model checkpoint."""
    console.print(f"[bold cyan]Exporting Checkpoint:[/bold cyan] {checkpoint} -> {output}")
    adapter = ModelRegistry.get_adapter(model_id_or_path=checkpoint, device=device)
    adapter.load()
    adapter.save_checkpoint(output)
    adapter.unload()

    if validate:
        console.print("[bold yellow]Validating exported checkpoint in fresh subprocess...[/bold yellow]")
        clean_gpu_memory(device)
        ok, msg = SubprocessValidator.validate_checkpoint(output, device=device)
        if ok:
            console.print("[bold green]✔ Subprocess reload validation PASSED![/bold green]")
        else:
            console.print(f"[bold red]✘ Subprocess reload validation FAILED: {msg}[/bold red]")
            raise typer.Exit(1)


@app.command(name="full-sweep")
@app.command(name="full-pass")
def full_sweep(
    ctx: typer.Context,
    config: str = typer.Option("configs/full_sweep.yaml", "--config", "-c", help="Path to full sweep config YAML"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Target model ID or path override"),
    retain: Optional[float] = typer.Option(None, "--retain", "-r", help="Target retention ratio override (e.g. 0.99)"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", "-i", help="Maximum joint pruning iterations"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Evaluation batch size override"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Inference device override"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory override"),
    flap_step: Optional[float] = typer.Option(None, "--flap-step", "-s", help="Initial FLAP step ratio override (e.g. 0.05)"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from latest accepted checkpoint if available"),
):
    """Execute complete Full Sweep / Full Pass combining FLAP-based and Block-based pruning via JIPO."""
    from prunequest.sweep.config import JointSweepConfig
    from prunequest.sweep.optimizer import JointIncrementalOptimizer

    cfg = JointSweepConfig.from_yaml(config)
    # Apply explicit CLI overrides
    if model is not None:
        cfg.model.model_id = model
    if retain is not None:
        cfg.pruning.retention = retain
    if max_iterations is not None:
        cfg.pruning.max_iterations = max_iterations
    if batch_size is not None:
        cfg.evaluation.batch_size = batch_size
    if device is not None:
        cfg.model.device = device
    if output is not None:
        cfg.output_dir = output
    if flap_step is not None:
        cfg.pruning.initial_flap_ratio = flap_step

    # Save Resolved Configuration
    os.makedirs(cfg.output_dir, exist_ok=True)
    resolved_path = os.path.join(cfg.output_dir, "resolved_config.yaml")
    cfg.to_yaml(resolved_path)

    console.print(Panel(
        f"[bold cyan]PruneQuest Joint Incremental Pruning Optimizer (JIPO) — Full Sweep[/bold cyan]\n"
        f"Target: [green]{cfg.model.model_id}[/green] | Min Retention: [magenta]{cfg.pruning.retention*100:.1f}%[/magenta]\n"
        f"Initial FLAP Step: [yellow]{cfg.pruning.initial_flap_ratio*100:.1f}%[/yellow] | Max Blocks: [yellow]{cfg.pruning.max_block_removals}[/yellow] | Output: [green]{cfg.output_dir}[/green]\n"
        f"Resolved Config Saved: [dim]{resolved_path}[/dim]"
    ))

    optimizer = JointIncrementalOptimizer(cfg)
    console.print("[bold green]Loading model and initializing dataset partitions...[/bold green]")
    optimizer.initialize()

    console.print("[bold blue]Establishing unpruned baseline...[/bold blue]")
    baseline_rec = optimizer.run_baseline()
    console.print(f"[bold green]✔ Baseline Established:[/bold green] Accuracy: {baseline_rec['accuracy']*100:.2f}% | Loss: {baseline_rec['screening_loss']:.4f} | Params: {baseline_rec['params_before']/1e6:.2f}M")

    console.print("[bold cyan]Launching Joint Multi-Fidelity Pruning Search...[/bold cyan]")
    summary = optimizer.optimize(resume=resume)

    # Display Summary Table
    table = Table(title="[bold green]JIPO Full Sweep Execution Summary[/bold green]")
    table.add_column("Result Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Status", str(summary.get("status")))
    table.add_row("Stopping Reason", str(summary.get("stopping_reason")))
    table.add_row("Baseline Accuracy", f"{summary.get('baseline_accuracy', 0.0)*100:.2f}%")
    table.add_row("Baseline Parameters", f"{summary.get('baseline_params', 0):,}")

    best = summary.get("best_checkpoint")
    if best:
        table.add_row("Final Compressed Parameters", f"{best.get('params_after', 0):,}")
        pruned_cnt = summary.get("baseline_params", 0) - best.get("params_after", 0)
        table.add_row("Total Parameters Pruned", f"{pruned_cnt:,} ({pruned_cnt/1e6:.2f}M)")
        table.add_row("Final Accuracy", f"{best.get('accuracy', 0.0)*100:.2f}%")
        table.add_row("Accuracy Retention", f"{best.get('accuracy_retention_pct', 100.0):.2f}%")
        table.add_row("Best Pareto Checkpoint", str(best.get("checkpoint_path")))
    else:
        table.add_row("Best Pareto Checkpoint", "None accepted")

    metrics = summary.get("metrics", {})
    table.add_row("Blocks Removed", str(summary.get("total_blocks_removed", 0)))
    table.add_row("Total FLAP Ratio Applied", f"{summary.get('total_flap_ratio_applied', 0.0)*100:.2f}%")
    table.add_row("Total Full Evaluations Executed", str(metrics.get("full_evaluations_executed", 0)))
    table.add_row("Evaluations Saved by Joint Grouping", f"[bold green]{metrics.get('evaluations_saved_by_joint', 0)}[/bold green]")
    table.add_row("Evaluations Saved by Cheap Screening", f"[bold green]{metrics.get('evaluations_saved_by_screening', 0)}[/bold green]")
    table.add_row("Total Evaluations Saved", f"[bold green]{metrics.get('total_evaluations_saved', 0)}[/bold green]")
    table.add_row("Pareto Frontier Checkpoints", str(metrics.get("pareto_points", 0)))
    table.add_row("Resolved Config Saved", resolved_path)

    console.print(table)


@app.command("studio")
def studio_command(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host interface to bind"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port number (defaults to 8765 or first free port)"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the browser automatically"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for local development"),
):
    """Launch PruneQuest Studio local web application."""
    from prunequest.studio import launch_studio
    launch_studio(host=host, port=port, no_browser=no_browser, reload=reload)


def main():
    try:
        app()
    except BrokenPipeError:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
