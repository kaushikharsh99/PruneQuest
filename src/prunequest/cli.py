"""
PruneQuest Command-Line Interface (Typer + Rich).
"""

import os
from typing import Optional
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
from prunequest.reporting.reports import ReportGenerator
from prunequest.core.tracking import ExperimentTracker
from prunequest.export.validation import SubprocessValidator
from prunequest.reporting.metrics import clean_gpu_memory

app = typer.Typer(
    name="prunequest",
    help="Autonomous, Benchmark-Guided LLM Structural Pruning Framework.",
    add_completion=False,
)

benchmarks_app = typer.Typer(help="Discover, inspect, and validate benchmarks.")
app.add_typer(benchmarks_app, name="benchmarks")

console = Console()


@app.command()
def version():
    """Display PruneQuest version and core system information."""
    from prunequest import __version__
    console.print(f"[bold cyan]PruneQuest[/bold cyan] version [bold green]v{__version__}[/bold green]")


@app.command()
def doctor():
    """Verify hardware environment, PyTorch, CUDA, and dependencies."""
    table = Table(title="[bold blue]PruneQuest Environment Diagnostics[/bold blue]")
    table.add_column("Component", style="cyan")
    table.add_column("Status / Value", style="magenta")

    table.add_row("CUDA Available", f"[green]{torch.cuda.is_available()}[/green]")
    if torch.cuda.is_available():
        table.add_row("Device Name", torch.cuda.get_device_name(0))
        table.add_row("Device Count", str(torch.cuda.device_count()))
        free_mem, total_mem = torch.cuda.mem_get_info()
        table.add_row("Free VRAM", f"{free_mem / (1024**3):.2f} GB")
        table.add_row("Total VRAM", f"{total_mem / (1024**3):.2f} GB")
        table.add_row("CUDA Version", str(torch.version.cuda))
    else:
        table.add_row("Device Name", "[yellow]CPU Only[/yellow]")

    deps = [
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "datasets",
        "math_verify",
        "lm_eval",
        "pydantic",
        "rich",
        "typer",
    ]
    for pkg in deps:
        try:
            m = __import__(pkg)
            v = getattr(m, "__version__", "Installed")
            table.add_row(f"Dependency: {pkg}", f"[green]{v}[/green]")
        except ImportError:
            table.add_row(f"Dependency: {pkg}", "[yellow]Not Installed[/yellow]")

    if LM_EVAL_AVAILABLE:
        try:
            task_count = len(LMEvalBenchmarkAdapter.list_available_tasks())
            table.add_row("LM-Eval Tasks Discovered", f"[green]{task_count:,} tasks[/green]")
        except Exception:
            table.add_row("LM-Eval Tasks Discovered", "[yellow]Failed to enumerate[/yellow]")

    table.add_row("Supported Model Families", "Qwen2/2.5, Llama-2/3, Mistral, Gemma, Phi")
    table.add_row("Default Pruning Backend", "Block Influence Transformer Layer Pruner")

    console.print(table)


@app.command()
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
def list_benchmarks_cmd():
    """List native benchmarks and popular evaluation tasks."""
    bench_list = BenchmarkRegistry.list_benchmarks()
    table = Table(title="[bold blue]Supported Benchmark Adapters[/bold blue]")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Metric", style="green")
    table.add_column("Direction", style="yellow")

    for name, props in bench_list.items():
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
def validate_benchmark_spec(spec_file: str = typer.Argument(..., help="Path to BenchmarkSpec YAML file")):
    """Validate a declarative BenchmarkSpec YAML file."""
    valid, errors = BenchmarkSpecCompiler.validate_spec(spec_file)
    if valid:
        console.print(f"[bold green]✔ Specification '{spec_file}' is valid![/bold green]")
    else:
        console.print(f"[bold red]✘ Specification '{spec_file}' is invalid:[/bold red]")
        for err in errors:
            console.print(f"  - {err}")


@benchmarks_app.command("preview")
def preview_benchmark_spec(
    spec_file: str = typer.Argument(..., help="Path to BenchmarkSpec YAML file"),
    count: int = typer.Option(3, "--count", "-n", help="Number of preview samples"),
):
    """Preview rendered prompts and target extraction for a BenchmarkSpec."""
    try:
        samples = BenchmarkSpecCompiler.preview_spec(spec_file, count=count)
        table = Table(title=f"[bold blue]Benchmark Preview: {spec_file}[/bold blue]")
        table.add_column("ID", style="cyan")
        table.add_column("Rendered Prompt", style="white")
        table.add_column("Gold Target", style="green")

        for s in samples:
            table.add_row(s["id"], s["rendered_prompt"], s["gold_answer"])

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to preview '{spec_file}': {e}[/bold red]")


@app.command()
def baseline(
    model: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", "--model", "-m"),
    benchmark: str = typer.Option("gsm8k", "--benchmark", "--benchmarks", "-b", help="One or more comma-separated benchmarks"),
    split: str = typer.Option("smoke", "--split", "-s", help="smoke, screening, dev, or full"),
    samples: Optional[int] = typer.Option(16, "--samples", "-n", help="Sample count limit"),
    device: str = typer.Option("cuda:0", "--device", "-d"),
    output: str = typer.Option("runs/prunequest_baseline", "--output", "-o"),
):
    """Evaluate original unpruned model across one or more benchmarks."""
    console.print(Panel(f"[bold cyan]Establishing Baseline for {model} on {benchmark} ({split} split)[/bold cyan]"))

    cfg = PruneQuestConfig(
        model=ModelConfig(model_id=model, device=device),
        benchmark=BenchmarkConfig(name=benchmark, split=split, smoke_size=samples or 16),
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
def run(
    model: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", "--model", "-m", help="Pretrained model ID or path"),
    benchmark: str = typer.Option("gsm8k", "--benchmark", "--benchmarks", "-b", help="One or more comma-separated benchmarks"),
    retain: float = typer.Option(0.99, "--retain", "-r", help="Minimum accuracy retention ratio [0.0 - 1.0]"),
    max_iterations: int = typer.Option(3, "--max-iterations", "-i", help="Maximum pruning iterations"),
    split: str = typer.Option("smoke", "--split", "-s", help="smoke, dev, or full"),
    samples: int = typer.Option(16, "--samples", "-n", help="Sample limit for screening"),
    output: str = typer.Option("runs/prunequest_run", "--output", "-o", help="Run directory"),
    reports: str = typer.Option("reports", "--reports", help="Reports directory"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Inference device"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
):
    """Execute complete autonomous, benchmark-guided pruning loop."""
    console.print(Panel(
        f"[bold cyan]PruneQuest v0.2.0 — Universal Pruning Framework[/bold cyan]\n"
        f"Target: [green]{model}[/green] | Benchmark(s): [yellow]{benchmark}[/yellow] | Min Retention: [magenta]{retain*100:.1f}%[/magenta]"
    ))

    if config and os.path.exists(config):
        cfg = PruneQuestConfig.from_yaml(config)
    else:
        cfg = PruneQuestConfig(
            model=ModelConfig(model_id=model, device=device),
            benchmark=BenchmarkConfig(name=benchmark, split=split, smoke_size=samples),
            pruning=PruningConfig(retain=retain, max_iterations=max_iterations),
            output_dir=output,
            reports_dir=reports,
        )

    controller = SearchController(cfg)

    def on_step(record):
        rprint(f"[bold green]✔ Accepted Iteration:[/bold green] {record['iteration_id']} | "
               f"Params: {record['params_after']/1e6:.2f}M | "
               f"Accuracy: {record['accuracy']*100:.2f}% (Retention: {record['accuracy_retention_pct']:.1f}%)")

    res = controller.run(progress_callback=on_step)

    df = controller.tracker.get_history_dataframe()
    reporter = ReportGenerator(reports)
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

    console.print(summary_table)


@app.command()
def resume(
    run_dir: str = typer.Argument(..., help="Path to previous run directory containing experiments.db"),
):
    """Resume an interrupted pruning search from the last valid checkpoint."""
    console.print(Panel(f"[bold cyan]Resuming PruneQuest Search from:[/bold cyan] {run_dir}"))

    tracker = ExperimentTracker(run_dir)
    latest = tracker.get_latest_accepted()
    if not latest:
        console.print("[bold red]No accepted checkpoints found in run directory to resume from.[/bold red]")
        raise typer.Exit(1)

    cfg = PruneQuestConfig(output_dir=run_dir)
    controller = SearchController(cfg, tracker=tracker)
    res = controller.resume()

    console.print("[bold green]✔ Resumed run finished successfully![/bold green]")
    console.print(f"Final params: {res['final_params']:,} | Final accuracy: {res['final_accuracy']*100:.2f}%")


@app.command()
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


if __name__ == "__main__":
    app()
