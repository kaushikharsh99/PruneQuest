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


@app.command()
@block_app.command(name="resume")
def resume(
    run_dir: str = typer.Argument(..., help="Path to previous run directory containing experiments.db"),
):
    """Resume an interrupted pruning search from the last valid checkpoint."""
    console.print(Panel(f"[bold cyan]Resuming PruneQuest Search from:[/bold cyan] {run_dir}"))

    if not os.path.exists(run_dir):
        console.print(f"[bold red]Run directory does not exist: {run_dir}[/bold red]")
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
    run_dir: str = typer.Argument(..., help="Path to run directory containing experiments.db"),
    output: str = typer.Option("reports", "--output", "-o", help="Directory to save report and plots"),
):
    """Generate Markdown report and publication-grade plots from SQLite database."""
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

    reporter = ReportGenerator(output)
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
