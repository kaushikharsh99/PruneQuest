"""
CLI commands for FLAP-based incremental structured pruning.
"""

from typing import Optional
import os
import sys
import signal
import platform
import typer

# Prevent BrokenPipeError / Python traceback when piping CLI output to head, tail, or grep
try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (AttributeError, ValueError):
    pass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from prunequest.flap.config import FLAPConfig
from prunequest.flap.models.discovery import ArchitectureInspector
from prunequest.flap.models.registry import ModelRegistry
from prunequest.flap.core.optimizer import FLAPOptimizer
from prunequest.flap.reporting.reporter import ReportGenerator
from prunequest.flap.calibration.dataset import DatasetManager
from prunequest.flap.evaluation.gsm8k import GSM8KEvaluator
from prunequest.flap.export.validator import CheckpointValidator

app = typer.Typer(help="FLAP-Based Incremental Structured MLP Pruning Framework")
console = Console()


@app.command()
def doctor():
    """Verify CUDA, PyTorch, hardware headroom, and environment readiness."""
    import torch
    import transformers

    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "None (CPU)"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_ok else 0.0

    table = Table(title="[bold green]FLAP Pruning Environment Diagnostics[/bold green]")
    table.add_column("Component", style="cyan")
    table.add_column("Status / Value", style="magenta")

    table.add_row("Python Version", f"{platform.python_version()} ({sys.executable})")
    table.add_row("CUDA Available", f"[green]{cuda_ok}[/green]" if cuda_ok else "[yellow]False (CPU Mode)[/yellow]")
    table.add_row("GPU Device", device_name)
    table.add_row("VRAM Capacity", f"{vram_gb:.2f} GB" if cuda_ok else "N/A")
    table.add_row("PyTorch Version", str(torch.__version__))
    table.add_row("Transformers Version", str(transformers.__version__))
    table.add_row("FLAP Target", "SwiGLU Intermediate MLP Channels")
    table.add_row("Verified Architectures", "Qwen2 / 2.5, Llama 2 / 3, Mistral")

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
    table.add_row("Supported for FLAP Pruning", "[green]Yes[/green]" if info.is_supported else "[red]No[/red]")
    table.add_row("Description", info.description)

    console.print(table)


@app.command()
def profile(
    config: str = typer.Option("configs/qwen_math_flap.yaml", "--config", "-c", help="Path to config YAML"),
):
    """Run FLAP calibration and compute layer importance scores."""
    cfg = FLAPConfig.from_yaml(config)
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
    config: str = typer.Option("configs/qwen_math_flap.yaml", "--config", "-c", help="Path to config YAML"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Target model override"),
    retain: Optional[float] = typer.Option(None, "--retain", "-r", help="Target retention ratio override (e.g. 0.99)"),
    max_evaluations: Optional[int] = typer.Option(None, "--max-evaluations", "-i", help="Max candidate evaluations"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Evaluation batch size override"),
    max_batch_size: Optional[int] = typer.Option(None, "--max-batch-size", help="Max auto-detected batch size (default 64)"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device override"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory override"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from last accepted checkpoint if available"),
):
    """Execute complete incremental FLAP pruning optimization loop."""
    cfg = FLAPConfig.from_yaml(config)

    # Apply explicit CLI overrides (CLI > Config > Default)
    if model is not None:
        cfg.model.model_id = model
    if retain is not None:
        cfg.pruning.retention = retain
    if max_evaluations is not None:
        cfg.pruning.max_candidate_evaluations = max_evaluations
    if batch_size is not None:
        cfg.evaluation.batch_size = batch_size
    if max_batch_size is not None:
        cfg.evaluation.max_batch_size = max_batch_size
    if device is not None:
        cfg.model.device = device
    if output is not None:
        cfg.output_dir = output

    # Save Resolved Configuration
    os.makedirs(cfg.output_dir, exist_ok=True)
    resolved_path = os.path.join(cfg.output_dir, "resolved_config.yaml")
    cfg.to_yaml(resolved_path)

    console.print(Panel(
        f"[bold cyan]FLAP Incremental Pruning Optimizer[/bold cyan]\n"
        f"Model: [green]{cfg.model.model_id}[/green] | Target Retention: [magenta]{cfg.pruning.retention*100:.1f}%[/magenta]\n"
        f"Batch Size: [yellow]{cfg.evaluation.batch_size or f'Auto (max {cfg.evaluation.max_batch_size})'}[/yellow] | Resume: [cyan]{resume}[/cyan]\n"
        f"Resolved Config Saved: [dim]{resolved_path}[/dim]"
    ))

    optimizer = FLAPOptimizer(cfg)
    summary = optimizer.optimize(resume=resume)

    history = optimizer.tracker.get_all_records()
    report_file = os.path.join(cfg.output_dir, "report.md")
    ReportGenerator.generate_markdown(summary, history, report_file)

    console.print("\n")
    best = summary.get("best_checkpoint") or {}
    table = Table(title="[bold green]FLAP Pruning Execution Summary[/bold green]")
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
    table.add_row("Resolved Config Saved", resolved_path)

    console.print(table)


@app.command()
def resume(
    run_dir: str = typer.Argument("runs/qwen_math_flap", help="Path to run directory containing experiments.db and checkpoints"),
    run_dir_opt: Optional[str] = typer.Option(None, "--run-dir", "--output", "-o", help="Run directory override"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config YAML (optional)"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Evaluation batch size"),
    max_batch_size: Optional[int] = typer.Option(None, "--max-batch-size", help="Max auto-detected batch size (default 64)"),
):
    """Resume an interrupted FLAP pruning optimization from the last accepted checkpoint."""
    target_dir = run_dir_opt or run_dir
    if not os.path.exists(target_dir):
        console.print(f"[bold red]Run directory not found: {target_dir}[/bold red]")
        raise typer.Exit(1)

    if config is None:
        resolved_cfg = os.path.join(target_dir, "resolved_config.yaml")
        cand_config = os.path.join(target_dir, "config.yaml")
        if os.path.exists(resolved_cfg):
            config = resolved_cfg
        elif os.path.exists(cand_config):
            config = cand_config
        else:
            config = "configs/qwen_math_flap.yaml"

    cfg = FLAPConfig.from_yaml(config)
    cfg.output_dir = target_dir
    if batch_size is not None:
        cfg.evaluation.batch_size = batch_size
    if max_batch_size is not None:
        cfg.evaluation.max_batch_size = max_batch_size

    console.print(Panel(
        f"[bold cyan]Resuming FLAP Pruning Optimizer[/bold cyan]\n"
        f"Run Directory: [green]{target_dir}[/green] | Config: [yellow]{config}[/yellow]\n"
        f"Batch Size: [yellow]{cfg.evaluation.batch_size or f'Auto (max {cfg.evaluation.max_batch_size})'}[/yellow]"
    ))

    optimizer = FLAPOptimizer(cfg)
    summary = optimizer.optimize(resume=True)

    history = optimizer.tracker.get_all_records()
    report_file = os.path.join(cfg.output_dir, "report.md")
    ReportGenerator.generate_markdown(summary, history, report_file)

    console.print("\n")
    best = summary.get("best_checkpoint") or {}
    table = Table(title="[bold green]FLAP Pruning Execution Summary[/bold green]")
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
    checkpoint: Optional[str] = typer.Argument(None, help="Path to exported checkpoint"),
    checkpoint_opt: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Path to exported checkpoint"),
    split: str = typer.Option("acceptance", "--split", "-s", help="Split to evaluate: acceptance, screening, calibration, or test"),
    samples: int = typer.Option(50, "--samples", "-n", help="Number of samples to evaluate"),
    device: str = typer.Option("cuda:0", "--device", "-d", help="Evaluation device"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Evaluation batch size"),
):
    """Independently evaluate an exported checkpoint on GSM8K partition."""
    target_checkpoint = checkpoint_opt or checkpoint
    if not target_checkpoint:
        console.print("[bold red]Error: Must specify checkpoint path as argument or via --checkpoint[/bold red]")
        raise typer.Exit(1)

    if not os.path.exists(target_checkpoint):
        alt = os.path.join(os.path.dirname(target_checkpoint), "checkpoints", os.path.basename(target_checkpoint))
        alt2 = os.path.join(target_checkpoint, "checkpoints", "best")
        if os.path.exists(alt):
            target_checkpoint = alt
        elif os.path.exists(alt2):
            target_checkpoint = alt2

    VALID_SPLITS = ("calibration", "screening", "acceptance", "test")
    if split not in VALID_SPLITS:
        console.print(
            f"[bold red]Error: Unknown split '{split}'.[/bold red]\n\n"
            f"Available splits:\n"
            f"  calibration\n"
            f"  screening\n"
            f"  acceptance\n"
            f"  test"
        )
        raise typer.Exit(1)

    console.print(Panel(f"[bold cyan]Testing Checkpoint: {target_checkpoint} ({split} split)[/bold cyan]"))

    # Verify reloadability first
    ok, msg = CheckpointValidator.validate(target_checkpoint)
    if not ok:
        console.print(f"[bold red]✘ Subprocess reload failed: {msg}[/bold red]")
        raise typer.Exit(1)
    console.print("[bold green]✔ Standalone subprocess reload verified successfully![/bold green]")

    adapter = ModelRegistry.create_adapter(target_checkpoint, device=device)
    model, tokenizer = adapter.load()

    dm = DatasetManager()
    splits = dm.load_and_partition(acceptance_samples=samples, tokenizer=tokenizer)
    test_samples = splits[split][:samples]

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
