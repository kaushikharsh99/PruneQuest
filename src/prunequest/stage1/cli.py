"""
CLI commands for Stage 1 FLAP-based incremental structured pruning.
"""

from typing import Optional
import os
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from prunequest.stage1.config import Stage1Config
from prunequest.stage1.models.discovery import ArchitectureInspector
from prunequest.stage1.models.registry import ModelRegistry
from prunequest.stage1.core.optimizer import FLAPOptimizer
from prunequest.stage1.reporting.reporter import ReportGenerator
from prunequest.stage1.calibration.dataset import DatasetManager
from prunequest.stage1.evaluation.gsm8k import GSM8KEvaluator
from prunequest.stage1.export.validator import CheckpointValidator

app = typer.Typer(help="Stage 1: FLAP-Based Incremental Structured MLP Pruning Framework")
console = Console()


@app.command()
def doctor():
    """Verify CUDA, PyTorch, and environment readiness."""
    import torch
    import transformers

    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "None (CPU)"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_ok else 0.0

    table = Table(title="[bold green]Stage 1 System Diagnostics[/bold green]")
    table.add_column("Component", style="cyan")
    table.add_column("Status / Value", style="magenta")

    table.add_row("CUDA Available", str(cuda_ok))
    table.add_row("GPU Device", device_name)
    table.add_row("VRAM Capacity", f"{vram_gb:.2f} GB" if cuda_ok else "N/A")
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("Transformers Version", transformers.__version__)
    table.add_row("FLAP Target", "SwiGLU Intermediate MLP Channels")

    console.print(table)


@app.command()
def inspect(
    model: str = typer.Option("Qwen/Qwen2.5-Math-1.5B-Instruct", "--model", "-m", help="Model path or Hugging Face ID"),
):
    """Inspect model architecture layout and MLP configuration."""
    console.print(Panel(f"[bold cyan]Inspecting Model Architecture: {model}[/bold cyan]"))
    info = ArchitectureInspector.inspect(model)

    table = Table(title="[bold green]MLP Architecture Specifications[/bold green]")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Model Type", info.model_type)
    table.add_row("Transformer Layers", str(info.num_layers))
    table.add_row("Hidden Size", str(info.hidden_size))
    table.add_row("Intermediate Size", str(info.intermediate_size))
    table.add_row("MLP Type", info.mlp_type)
    table.add_row("Gate Projection", info.gate_proj_name)
    table.add_row("Up Projection", info.up_proj_name)
    table.add_row("Down Projection", info.down_proj_name)
    table.add_row("Down Projection Bias", str(info.has_down_proj_bias))
    table.add_row("Supported for Stage 1", "[green]Yes[/green]" if info.is_supported else "[red]No[/red]")
    table.add_row("Description", info.description)

    console.print(table)


@app.command()
def profile(
    config: str = typer.Option("configs/qwen_math.yaml", "--config", "-c", help="Path to config YAML"),
):
    """Run FLAP calibration and compute layer importance scores."""
    cfg = Stage1Config.from_yaml(config)
    console.print(Panel(f"[bold cyan]Profiling Model via FLAP ({cfg.flap.metric.upper()}): {cfg.model.model_id}[/bold cyan]"))

    optimizer = FLAPOptimizer(cfg)
    optimizer.initialize()
    scores = optimizer.run_profiling()

    console.print(f"[bold green]✔ Successfully profiled {len(scores)} layers![/bold green]")
    table = Table(title="[bold blue]Sample Layer Importance Statistics[/bold blue]")
    table.add_column("Layer Index", style="cyan")
    table.add_column("Intermediate Channels", style="white")
    table.add_column("Score Mean", style="green")
    table.add_column("Score Min / Max", style="magenta")

    for l_idx in sorted(scores.keys())[:5]:
        s = scores[l_idx]
        table.add_row(str(l_idx), str(s.shape[0]), f"{s.mean().item():.4f}", f"{s.min().item():.4f} / {s.max().item():.4f}")

    console.print(table)


@app.command()
def optimize(
    config: str = typer.Option("configs/qwen_math.yaml", "--config", "-c", help="Path to config YAML"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Evaluation batch size (e.g. 32, 64)"),
    max_batch_size: Optional[int] = typer.Option(None, "--max-batch-size", help="Max auto-detected batch size (default 64)"),
):
    """Execute complete incremental FLAP pruning optimization loop."""
    cfg = Stage1Config.from_yaml(config)
    if batch_size is not None:
        cfg.evaluation.batch_size = batch_size
    if max_batch_size is not None:
        cfg.evaluation.max_batch_size = max_batch_size
    console.print(Panel(
        f"[bold cyan]Stage 1 Incremental FLAP Pruning Optimizer[/bold cyan]\n"
        f"Model: [green]{cfg.model.model_id}[/green] | Target Retention: [magenta]{cfg.pruning.retention*100:.1f}%[/magenta]\n"
        f"Batch Size: [yellow]{cfg.evaluation.batch_size or f'Auto (max {cfg.evaluation.max_batch_size})'}[/yellow]"
    ))

    optimizer = FLAPOptimizer(cfg)
    summary = optimizer.optimize()

    history = optimizer.tracker.get_all_records()
    report_file = os.path.join(cfg.output_dir, "report.md")
    ReportGenerator.generate_markdown(summary, history, report_file)

    console.print("\n")
    best = summary.get("best_checkpoint") or {}
    table = Table(title="[bold green]Stage 1 Pruning Execution Summary[/bold green]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Stopping Reason", str(summary.get("stopping_reason")))
    table.add_row("Baseline Accuracy", f"{summary.get('baseline_accuracy', 0.0)*100:.2f}%")
    table.add_row("Best Compressed Accuracy", f"{best.get('accuracy', 0.0)*100:.2f}%")
    table.add_row("Relative Retention", f"{best.get('retention_pct', 100.0):.2f}%")
    table.add_row("Original Parameters", f"{summary.get('baseline_params', 0):,}")
    table.add_row("Final Parameters", f"{best.get('params_after', 0):,}")
    pruned = summary.get("baseline_params", 0) - best.get("params_after", 0)
    table.add_row("Parameters Pruned", f"{pruned:,} ({pruned/1e6:.2f}M)")
    table.add_row("Best Checkpoint", str(best.get("checkpoint_path")))
    table.add_row("Report Generated", report_file)

    console.print(table)


@app.command()
def test(
    checkpoint: str = typer.Argument(..., help="Path to exported checkpoint"),
    split: str = typer.Option("acceptance", "--split", "-s", help="acceptance, screening, or calibration"),
    samples: int = typer.Option(50, "--samples", "-n", help="Number of samples to evaluate"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Evaluation device"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Evaluation batch size"),
):
    """Independently evaluate an exported checkpoint on GSM8K partition."""
    console.print(Panel(f"[bold cyan]Testing Checkpoint: {checkpoint} ({split} split)[/bold cyan]"))

    # Verify reloadability first
    ok, msg = CheckpointValidator.validate(checkpoint)
    if not ok:
        console.print(f"[bold red]✘ Subprocess reload failed: {msg}[/bold red]")
        raise typer.Exit(1)
    console.print("[bold green]✔ Standalone subprocess reload verified successfully![/bold green]")

    adapter = ModelRegistry.create_adapter(checkpoint, device=device)
    model, tokenizer = adapter.load()

    dm = DatasetManager()
    splits = dm.load_and_partition(acceptance_samples=samples, tokenizer=tokenizer)
    test_samples = splits.get(split, splits["acceptance"])[:samples]

    res = GSM8KEvaluator.evaluate(model, tokenizer, test_samples, device=device, batch_size=batch_size)

    table = Table(title="[bold green]Independent Test Results[/bold green]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Samples", str(res.total))
    table.add_row("Correct Answers", str(res.correct))
    table.add_row("Accuracy", f"{res.accuracy*100:.2f}%")
    table.add_row("Tokens/Second", f"{res.tokens_per_sec:.2f}")
    table.add_row("Peak VRAM", f"{res.peak_vram_gb:.2f} GB")

    console.print(table)


@app.command("final-eval")
def final_eval(
    checkpoint: str = typer.Argument(..., help="Path to best checkpoint"),
    samples: Optional[int] = typer.Option(None, "--samples", "-n", help="Number of official test samples (default: all 1,319)"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Evaluation device"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Evaluation batch size"),
):
    """Evaluate final selected model on held-out official GSM8K test split."""
    console.print(Panel(f"[bold cyan]Final Evaluation on Official GSM8K Test Split: {checkpoint}[/bold cyan]"))

    adapter = ModelRegistry.create_adapter(checkpoint, device=device)
    model, tokenizer = adapter.load()

    dm = DatasetManager()
    splits = dm.load_and_partition(tokenizer=tokenizer)
    test_samples = splits["test"]
    if samples is not None:
        test_samples = test_samples[:samples]

    res = GSM8KEvaluator.evaluate(model, tokenizer, test_samples, device=device, batch_size=batch_size)

    table = Table(title="[bold green]Official GSM8K Test Results[/bold green]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Test Set Size", str(res.total))
    table.add_row("Correct Answers", str(res.correct))
    table.add_row("Test Accuracy", f"{res.accuracy*100:.2f}%")
    table.add_row("Tokens/Second", f"{res.tokens_per_sec:.2f}")

    console.print(table)
