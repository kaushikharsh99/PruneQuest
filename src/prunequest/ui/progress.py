"""
Live Progress Presentation Layer for PruneQuest.
Provides rich interactive dashboards, plain non-TTY line renderers, and JSON streaming.
"""

import sys
import os
import time
from typing import Optional, Set
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from prunequest.events.models import OptimizationEvent, EventCategory, EventPhase


class LiveProgressDashboard:
    """
    Rich interactive live display for terminal sessions.
    Decoupled from optimization math; updates state strictly via OptimizationEvents.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        refresh_per_second: float = 4.0,
    ):
        self.console = console or Console()
        self.refresh_rate = refresh_per_second

        # Run metadata
        self.model_id = "—"
        self.benchmark = "—"
        self.strategy = "—"
        self.device = "—"
        self.retention_target = 0.99
        self.preset = "—"

        # Checkpoints & Status
        self.completed_milestones: Set[str] = set()
        self.current_phase = EventPhase.PREFLIGHT
        self.current_status_msg = "Initializing..."

        # Progress counters
        self.candidate_curr = 0
        self.candidate_total = 0
        self.sample_curr = 0
        self.sample_total = 0

        # Metrics
        self.original_params = 0
        self.best_params = 0
        self.best_compression_pct = 0.0
        self.best_retention_pct = 100.0
        self.accepted_count = 0
        self.rejected_count = 0
        self.best_checkpoint = "None accepted yet"
        self.peak_vram_gb: Optional[float] = None

        self.start_time = time.time()
        self._live: Optional[Live] = None
        self._is_running = False

    def start(self):
        if not self._is_running and self.console.is_terminal and not os.environ.get("NO_COLOR"):
            self._live = Live(
                self._render_view(),
                console=self.console,
                refresh_per_second=self.refresh_rate,
                transient=False,
            )
            self._live.start()
            self._is_running = True

    def stop(self):
        if self._is_running and self._live:
            self._live.update(self._render_view())
            self._live.stop()
            self._is_running = False

    def on_event(self, event: OptimizationEvent):
        payload = event.payload or {}
        cat = event.event_type

        # Update metadata
        if cat == EventCategory.RUN_STARTED:
            self.model_id = payload.get("model_id", self.model_id)
            self.benchmark = payload.get("benchmark", self.benchmark).upper()
            self.strategy = payload.get("strategy", self.strategy).upper()
            self.device = payload.get("device", self.device)
            self.retention_target = payload.get("retention", self.retention_target)
            self.preset = payload.get("preset", self.preset).upper()

        elif cat == EventCategory.PREFLIGHT_COMPLETED:
            self.completed_milestones.add("Hardware & Preflight verified")
            self.current_phase = EventPhase.MODEL_LOADING
            self.current_status_msg = "Loading model architecture..."

        elif cat == EventCategory.MODEL_LOADED:
            self.completed_milestones.add("Model weights loaded")
            self.original_params = payload.get("total_params", self.original_params)
            self.best_params = self.original_params
            self.current_phase = EventPhase.BASELINE
            self.current_status_msg = "Evaluating unpruned baseline..."

        elif cat == EventCategory.BASELINE_COMPLETED:
            self.completed_milestones.add("Baseline established")
            acc = payload.get("accuracy", 0.0)
            self.current_phase = EventPhase.PROFILING
            self.current_status_msg = f"Baseline: {acc * 100:.2f}% accuracy. Profiling layer sensitivities..."

        elif cat == EventCategory.PROFILING_COMPLETED:
            self.completed_milestones.add("Importance profiling completed")
            self.current_phase = EventPhase.SCREENING
            self.current_status_msg = "Searching candidate pruning operations..."

        elif cat == EventCategory.CANDIDATE_PROPOSED:
            self.candidate_curr = payload.get("candidate_index", self.candidate_curr)
            self.candidate_total = payload.get("max_candidates", self.candidate_total)
            op = payload.get("operation_description", "")
            self.current_phase = EventPhase.SCREENING
            self.current_status_msg = f"Screening candidate {self.candidate_curr}/{self.candidate_total}: {op}"

        elif cat == EventCategory.CANDIDATE_SCREENING_COMPLETED:
            passed = payload.get("passed", False)
            loss = payload.get("loss", 0.0)
            if passed:
                self.current_phase = EventPhase.VALIDATION
                self.current_status_msg = f"Candidate passed screening (loss: {loss:.4f}). Validating on dev split..."
            else:
                self.current_status_msg = f"Candidate rejected during screening (loss: {loss:.4f}). Rolling back..."

        elif cat == EventCategory.VALIDATION_PROGRESS:
            self.sample_curr = payload.get("sample_curr", 0)
            self.sample_total = payload.get("sample_total", 0)

        elif cat == EventCategory.CANDIDATE_ACCEPTED:
            self.accepted_count += 1
            self.best_params = payload.get("params_after", self.best_params)
            self.best_compression_pct = payload.get("compression_pct", self.best_compression_pct)
            self.best_retention_pct = payload.get("accuracy_retention_pct", self.best_retention_pct)
            self.best_checkpoint = payload.get("checkpoint_path", self.best_checkpoint)
            self.current_status_msg = f"Candidate {self.candidate_curr} accepted ({self.best_compression_pct:.2f}% pruned, {self.best_retention_pct:.2f}% retention)"

        elif cat == EventCategory.CANDIDATE_REJECTED:
            self.rejected_count += 1
            reason = payload.get("reason", "retention below threshold")
            self.current_status_msg = f"Candidate {self.candidate_curr} rejected: {reason}. Rolling back."

        elif cat == EventCategory.CHECKPOINT_EXPORT_STARTED:
            self.current_phase = EventPhase.CHECKPOINT
            self.current_status_msg = "Exporting best model to canonical directory..."

        elif cat == EventCategory.CHECKPOINT_VERIFIED:
            self.completed_milestones.add("Subprocess reload verification passed")
            self.current_status_msg = "Reload verification succeeded."

        elif cat == EventCategory.RUN_COMPLETED:
            self.current_phase = EventPhase.COMPLETED
            self.current_status_msg = "Optimization complete."

        elif cat == EventCategory.RUN_INTERRUPTED:
            self.current_phase = EventPhase.INTERRUPTED
            self.current_status_msg = "Optimization safely interrupted. Checkpoint preserved."

        # Peak VRAM if present
        if "peak_vram_gb" in payload:
            self.peak_vram_gb = payload["peak_vram_gb"]

        # Re-render if live
        if self._is_running and self._live:
            self._live.update(self._render_view())

    def _render_view(self) -> Panel:
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))

        # 1. Config Grid
        grid = Table.grid(padding=(0, 2))
        grid.add_column("Key", style="dim", justify="right")
        grid.add_column("Val", style="bold")
        grid.add_column("Key2", style="dim", justify="right")
        grid.add_column("Val2", style="bold")

        grid.add_row("Model", self.model_id, "Benchmark", self.benchmark)
        grid.add_row("Strategy", self.strategy, "Target Retention", f"{self.retention_target * 100:.2f}%")
        grid.add_row("Device", self.device, "Preset", self.preset)

        # 2. Checklist
        milestone_lines = []
        for m in [
            "Hardware & Preflight verified",
            "Model weights loaded",
            "Baseline established",
            "Importance profiling completed",
            "Subprocess reload verification passed",
        ]:
            if m in self.completed_milestones:
                milestone_lines.append(f"[bold green]✔[/bold green] {m}")

        checklist_text = Text.from_markup("\n".join(milestone_lines) if milestone_lines else "[dim]Initializing environment...[/dim]")

        # 3. Current Phase / Progress
        phase_style = "bold cyan" if self.current_phase not in (EventPhase.COMPLETED, EventPhase.INTERRUPTED) else "bold green"
        phase_header = f"[{phase_style}]Phase:[/{phase_style}] [bold]{self.current_phase.upper()}[/bold] — {self.current_status_msg}"

        progress_elements = [Text.from_markup(phase_header)]

        if self.sample_total > 0 and self.current_phase == EventPhase.VALIDATION:
            pct = int((self.sample_curr / max(1, self.sample_total)) * 100)
            bar_len = 24
            filled = int((pct / 100.0) * bar_len)
            bar_str = "█" * filled + "░" * (bar_len - filled)
            progress_elements.append(
                Text.from_markup(f"  [green]{bar_str}[/green]  {self.sample_curr}/{self.sample_total} examples ({pct}%)")
            )

        # 4. Metrics Table
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column("Label", style="dim")
        metrics_table.add_column("Value", style="bold magenta")

        metrics_table.add_row("Best compression", f"{self.best_compression_pct:.2f}%")
        metrics_table.add_row("Measured retention", f"{self.best_retention_pct:.2f}%")
        metrics_table.add_row("Accepted candidates", f"[green]{self.accepted_count}[/green]")
        metrics_table.add_row("Rejected candidates", f"[yellow]{self.rejected_count}[/yellow]")
        if self.peak_vram_gb:
            metrics_table.add_row("Peak VRAM", f"{self.peak_vram_gb:.2f} GB")
        metrics_table.add_row("Best checkpoint", f"[cyan]{self.best_checkpoint}[/cyan]")
        metrics_table.add_row("Elapsed", elapsed)

        content = Group(
            grid,
            Text("─" * 56, style="dim"),
            checklist_text,
            Text("─" * 56, style="dim"),
            *progress_elements,
            Text("─" * 56, style="dim"),
            metrics_table,
        )

        return Panel(
            content,
            title="[bold green]PruneQuest — Optimization Monitor[/bold green]",
            border_style="green",
            expand=False,
        )


class PlainProgressRenderer:
    """
    Concise line-oriented renderer for non-TTY, redirected, or --no-progress executions.
    """

    def __init__(self, stream=sys.stderr):
        self.stream = stream
        self.start_time = time.time()

    def handle_event(self, event: OptimizationEvent):
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
        payload = event.payload or {}
        cat = event.event_type

        msg = None
        if cat == EventCategory.RUN_STARTED:
            msg = f"Started optimization: model={payload.get('model_id')} benchmark={payload.get('benchmark')} strategy={payload.get('strategy')}"
        elif cat == EventCategory.BASELINE_COMPLETED:
            msg = f"Baseline established: accuracy={payload.get('accuracy', 0.0)*100:.2f}% params={payload.get('total_params', 0):,}"
        elif cat == EventCategory.CANDIDATE_PROPOSED:
            msg = f"Candidate {payload.get('candidate_index')}/{payload.get('max_candidates')}: {payload.get('operation_description')}"
        elif cat == EventCategory.CANDIDATE_ACCEPTED:
            msg = f"Candidate {payload.get('candidate_index')} ACCEPTED: compression={payload.get('compression_pct', 0.0):.2f}% retention={payload.get('accuracy_retention_pct', 100.0):.2f}%"
        elif cat == EventCategory.CANDIDATE_REJECTED:
            msg = f"Candidate {payload.get('candidate_index')} REJECTED: {payload.get('reason')} (rolled back)"
        elif cat == EventCategory.CHECKPOINT_VERIFIED:
            msg = f"Subprocess reload verification PASSED: {payload.get('checkpoint_path')}"
        elif cat == EventCategory.RUN_COMPLETED:
            msg = f"Optimization COMPLETED in {elapsed}"
        elif cat == EventCategory.RUN_INTERRUPTED:
            msg = f"Optimization INTERRUPTED. Preserved checkpoint: {payload.get('best_checkpoint')}"

        if msg:
            print(f"[{elapsed}] [{event.phase}] {msg}", file=self.stream, flush=True)


class JsonStreamRenderer:
    """
    Streams structured JSON event lines to a stream.
    """

    def __init__(self, stream=sys.stdout):
        self.stream = stream

    def handle_event(self, event: OptimizationEvent):
        print(event.to_json(), file=self.stream, flush=True)
