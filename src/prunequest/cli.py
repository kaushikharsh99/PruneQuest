"""
Professional Typer & Rich Command-Line Interface for PruneQuest.
"""

import os
import sys
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from prunequest.config import PruneQuestConfig, ModelConfig, BenchmarkConfig, PruningConfig
from prunequest.models.registry import ModelRegistry
from prunequest.benchmarks.registry import BenchmarkRegistry
from prunequest.core.controller import SearchController
from prunequest.reporting.reports import ReportGenerator
from prunequest.core.tracking import ExperimentTracker
from prunequest.export.validation import SubprocessValidator
from prunequest.reporting.metrics import clean_gpu_memory

app = typer.Typer(
    name="prunequest",
    help="PruneQuest: Autonomous, Benchmark-Guided LLM Structural Pruning Framework.",
    add_completion=False,
)
console = Console()


@app.command()
def doctor():
    """Inspect environment capabilities, hardware, and dependency status."""
    import platform
    import torch
    import transformers

    table = Table(title="[bold green]PruneQuest System Diagnostics[/bold green]", box=None)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status / Version", style="magenta")

    table.add_row("Operating System", f"{platform.system()} {platform.release()} ({platform.machine()})")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Transformers", transformers.__version__)

    cuda_avail = torch.cuda.is_available()
    table.add_row("CUDA Available", "[green]True[/green]" if cuda_avail else "[red]False[/red]")

    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
        table.add_row("GPU Device", gpu_name)
        table.add_row("GPU VRAM Total", f"{total_vram:.2f} GB")
        table.add_row("GPU VRAM Free", f"{free_vram:.2f} GB")
    else:
        table.add_row("GPU Device", "None (CPU Execution Only)")

    # System RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        table.add_row("System RAM", f"{ram_gb:.2f} GB")
    except ImportError:
        table.add_row("System RAM", "Available (psutil not installed)")

    # Optional packages
    for pkg in ["math_verify", "lm_eval", "safetensors", "datasets", "accelerate"]:
        try:
            m = __import__(pkg)
            v = getattr(m, "__version__", "Installed")
            table.add_row(f"Dependency: {pkg}", f"[green]{v}[/green]")
        except ImportError:
            table.add_row(f"Dependency: {pkg}", "[yellow]Not Installed[/yellow]")

    table.add_row("Supported Architectures", "Qwen2ForCausalLM (Qwen2 / Qwen2.5)")
    table.add_row("Supported Benchmarks", "GSM8K (Zero-Shot CoT with Math-Verify)")
    table.add_row("Supported Backends", "Transformer Layer Pruner (ShortGPT Block Influence)")

    console.print(table)


@app.command()
def inspect(
    model: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", "--model", "-m", help="Hugging Face model ID or directory"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Device to inspect model on"),
):
    """Inspect model architecture, parameters, layers, and memory estimation."""
    console.print(Panel(f"[bold cyan]Inspecting Model Architecture:[/bold cyan] {model}"))

    with console.status("[bold green]Loading model metadata...[/bold green]"):
        adapter = ModelRegistry.get_adapter(model_id_or_path=model, device=device)
        adapter.load()
        meta = adapter.get_metadata()

    table = Table(title="[bold blue]Model Metadata Summary[/bold blue]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Architecture", meta.architecture)
    table.add_row("Total Parameters", f"{meta.total_params:,} ({meta.total_params_million:.2f}M)")
    table.add_row("Embedding Parameters", f"{meta.embedding_params:,}")
    table.add_row("Non-Embedding Parameters", f"{meta.non_embedding_params:,}")
    table.add_row("Transformer Layers", str(meta.num_layers))
    table.add_row("Hidden Size", str(meta.hidden_size))
    table.add_row("Intermediate Size", str(meta.intermediate_size))
    table.add_row("Attention Heads (Q)", str(meta.num_attention_heads))
    table.add_row("Attention Heads (KV)", str(meta.num_key_value_heads))
    table.add_row("Vocabulary Size", str(meta.vocab_size))
    table.add_row("Weight Precision", meta.dtype)
    table.add_row("Supported Pruning", ", ".join(meta.supported_pruning))

    est_weight_vram = (meta.total_params * 2) / (1024**3)  # FP16/BF16 ~ 2 bytes per param
    table.add_row("Estimated Weight VRAM", f"~{est_weight_vram:.2f} GB")

    console.print(table)
    adapter.unload()


@app.command()
def benchmarks():
    """List supported benchmark adapters and their configurations."""
    bench_list = BenchmarkRegistry.list_benchmarks()
    table = Table(title="[bold blue]Supported Benchmark Adapters[/bold blue]")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Direction", style="green")
    table.add_column("Requires Generation", style="yellow")

    for name, props in bench_list.items():
        table.add_row(
            name,
            props["metric_name"],
            props["metric_direction"],
            str(props["requires_generation"]),
        )

    console.print(table)


@app.command()
def baseline(
    model: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", "--model", "-m"),
    benchmark: str = typer.Option("gsm8k", "--benchmark", "-b"),
    split: str = typer.Option("smoke", "--split", "-s", help="smoke, screening, dev, or test"),
    samples: Optional[int] = typer.Option(16, "--samples", "-n", help="Number of benchmark samples"),
    device: str = typer.Option("cuda:0", "--device", "-d"),
    output: str = typer.Option("runs/prunequest_baseline", "--output", "-o"),
):
    """Evaluate original unpruned model to establish baseline performance A0."""
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

    table.add_row("Sample Count", str(res["sample_size"]))
    table.add_row("Correct Responses", str(res["num_correct"]))
    table.add_row("Accuracy", f"{res['accuracy']*100:.2f}%")
    table.add_row("Wilson 95% CI", f"[{res['accuracy_ci_lower']:.3f}, {res['accuracy_ci_upper']:.3f}]")
    table.add_row("Inference Speed", f"{res['tokens_per_sec']:.1f} tokens/sec")
    table.add_row("Peak VRAM", f"{res['peak_vram_gb']:.2f} GB")

    console.print(table)


@app.command()
def run(
    model: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", "--model", "-m", help="Pretrained model"),
    benchmark: str = typer.Option("gsm8k", "--benchmark", "-b", help="Evaluation benchmark"),
    retain: float = typer.Option(0.99, "--retain", "-r", help="Minimum accuracy retention ratio [0.0 - 1.0]"),
    max_iterations: int = typer.Option(3, "--max-iterations", "-i", help="Maximum pruning iterations"),
    samples: int = typer.Option(16, "--samples", "-n", help="Screening sample size"),
    output: str = typer.Option("runs/prunequest_run", "--output", "-o", help="Run directory"),
    reports: str = typer.Option("reports", "--reports", help="Reports directory"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Inference device"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
):
    """Execute complete autonomous, benchmark-guided pruning loop."""
    console.print(Panel(f"[bold cyan]PruneQuest v0.1.0 — Autonomous Pruning Engine[/bold cyan]\n"
                        f"Target: [green]{model}[/green] | Benchmark: [yellow]{benchmark}[/yellow] | Min Retention: [magenta]{retain*100:.1f}%[/magenta]"))

    if config and os.path.exists(config):
        cfg = PruneQuestConfig.from_yaml(config)
    else:
        cfg = PruneQuestConfig(
            model=ModelConfig(model_id=model, device=device),
            benchmark=BenchmarkConfig(name=benchmark, split="smoke", smoke_size=samples),
            pruning=PruningConfig(retain=retain, max_iterations=max_iterations),
            output_dir=output,
            reports_dir=reports,
        )

    controller = SearchController(cfg)

    # Progress tracking display
    def on_step(record):
        rprint(f"[bold green]✔ Accepted Iteration:[/bold green] {record['iteration_id']} | "
               f"Params: {record['params_after']/1e6:.2f}M | "
               f"Accuracy: {record['accuracy']*100:.2f}% (Retention: {record['accuracy_retention_pct']:.1f}%)")

    res = controller.run(progress_callback=on_step)

    # Export report and plots
    df = controller.tracker.get_history_dataframe()
    reporter = ReportGenerator(reports)
    plots = reporter.generate_plots(df)
    report_file = reporter.generate_report(df, res, plots)

    console.print("\n")
    summary_table = Table(title="[bold green]Pruning Search Execution Summary[/bold green]")
    summary_table.add_column("Result Field", style="cyan")
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Search Status", res["status"])
    summary_table.add_row("Baseline Accuracy", f"{res['baseline_accuracy']*100:.2f}%")
    summary_table.add_row("Final Accuracy", f"{res['final_accuracy']*100:.2f}%")
    summary_table.add_row("Accuracy Retention", f"{res['accuracy_retention_pct']:.2f}%")
    summary_table.add_row("Original Parameters", f"{res['original_params']:,}")
    summary_table.add_row("Final Parameters", f"{res['final_params']:,}")
    summary_table.add_row("Parameters Pruned", f"{res['original_params'] - res['final_params']:,}")
    summary_table.add_row("Best Checkpoint", str(res["best_checkpoint"]))
    summary_table.add_row("Report Generated", report_file)

    console.print(summary_table)


@app.command()
def resume(
    run_dir: str = typer.Argument(..., help="Path to run directory to resume"),
):
    """Resume an interrupted pruning experiment from its last accepted checkpoint."""
    console.print(Panel(f"[bold cyan]Resuming PruneQuest Experiment from:[/bold cyan] {run_dir}"))
    db_file = os.path.join(run_dir, "experiments.db")
    if not os.path.exists(db_file):
        console.print(f"[bold red]Error:[/bold red] experiments.db not found in {run_dir}.")
        raise typer.Exit(code=1)

    tracker = ExperimentTracker(run_dir)
    latest = tracker.get_latest_accepted()
    if not latest:
        console.print("[yellow]No accepted checkpoints found in run database to resume.[/yellow]")
        raise typer.Exit(code=1)

    rprint(f"Resuming from checkpoint: [green]{latest['checkpoint_path']}[/green] ({latest['params_after']/1e6:.2f}M params)")

    cfg = PruneQuestConfig(output_dir=run_dir)
    controller = SearchController(cfg, tracker=tracker)
    res = controller.resume()

    console.print(f"[bold green]Resumed run complete:[/bold green] Best checkpoint: {res.get('best_checkpoint')}")


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="Path to run directory containing experiments.db"),
    reports_dir: str = typer.Option("reports", "--reports", "-r", help="Reports destination directory"),
):
    """Generate publication plots and markdown report from an existing run directory."""
    tracker = ExperimentTracker(run_dir)
    df = tracker.get_history_dataframe()
    if df.empty:
        console.print(f"[yellow]No records found in {run_dir}/experiments.db.[/yellow]")
        raise typer.Exit(code=1)

    reporter = ReportGenerator(reports_dir)
    plots = reporter.generate_plots(df)

    latest = tracker.get_latest_accepted() or df.iloc[-1].to_dict()
    summary = {
        "status": "completed",
        "baseline_accuracy": df[df["status"] == "baseline"]["accuracy"].iloc[0] if (df["status"] == "baseline").any() else 1.0,
        "final_accuracy": latest.get("accuracy", 0.0),
        "accuracy_retention_pct": latest.get("accuracy_retention_pct", 100.0),
        "original_params": int(df.iloc[0]["params_before"]),
        "final_params": int(latest.get("params_after", 0)),
        "best_checkpoint": latest.get("checkpoint_path", ""),
    }
    rep_path = reporter.generate_report(df, summary, plots)

    console.print(f"[bold green]✔ Report generated at:[/bold green] {rep_path}")
    for name, p in plots.items():
        console.print(f"  - Plot ({name}): {p}")


@app.command()
def export(
    checkpoint_dir: str = typer.Argument(..., help="Path to pruned checkpoint to validate"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Validation device"),
):
    """Validate standalone loading and generation of an exported checkpoint."""
    console.print(Panel(f"[bold cyan]Validating Exported Checkpoint:[/bold cyan] {checkpoint_dir}"))
    ok, msg = SubprocessValidator.validate_checkpoint(checkpoint_dir, device=device)
    if ok:
        console.print(f"[bold green]✔ VALIDATION PASSED:[/bold green] {msg}")
    else:
        console.print(f"[bold red]✘ VALIDATION FAILED:[/bold red] {msg}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
