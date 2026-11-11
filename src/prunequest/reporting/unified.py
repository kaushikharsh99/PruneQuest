"""
Unified Reporting Engine for PruneQuest UX Release B.
Generates beginner-friendly Markdown summaries, detailed experiment audit reports,
CSV candidate histories, and genuine Pareto frontier plots from persisted data.
Supports offline regeneration without loading models or running benchmarks.
"""

import os
import json
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prunequest.runs.models import RunDetails


class UnifiedReportGenerator:
    """
    Produces consistent reports and visualizations across all PruneQuest pruning strategies.
    """

    @classmethod
    def generate_all_reports(
        cls,
        run_details: RunDetails,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generates summary.md, detailed_report.md, history.csv, and plots.
        Returns dictionary of generated file paths.
        """
        reports_dir = output_dir or run_details.reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        plots_dir = os.path.join(reports_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Extract candidate records from SQLite, events, or manifest
        records = cls._extract_candidate_records(run_details)

        # 1. Generate summary.md
        summary_path = os.path.join(reports_dir, "summary.md")
        cls._write_summary_report(summary_path, run_details)

        # 2. Generate detailed_report.md
        detailed_path = os.path.join(reports_dir, "detailed_report.md")
        cls._write_detailed_report(detailed_path, run_details, records)

        # 3. Generate history.csv
        csv_path = os.path.join(reports_dir, "history.csv")
        cls._write_history_csv(csv_path, records)

        # 4. Generate Plots
        plot_paths = cls._generate_plots(plots_dir, run_details, records)

        return {
            "summary": summary_path,
            "detailed": detailed_path,
            "history_csv": csv_path,
            **plot_paths,
        }

    @classmethod
    def _extract_candidate_records(cls, run_details: RunDetails) -> List[Dict[str, Any]]:
        db_path = os.path.join(run_details.run_dir, "experiments.db")
        records: List[Dict[str, Any]] = []

        if os.path.exists(db_path):
            try:
                with sqlite3.connect(db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM iterations ORDER BY timestamp ASC")
                    for r in cursor.fetchall():
                        records.append(dict(r))
            except Exception:
                pass

        # If no SQLite records, parse from events.jsonl
        if not records:
            events_path = os.path.join(run_details.run_dir, "events.jsonl")
            if os.path.exists(events_path):
                try:
                    with open(events_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            evt = json.loads(line)
                            if evt.get("event_type") in ("candidate_accepted", "candidate_rejected"):
                                p = evt.get("payload", {})
                                records.append({
                                    "iteration_id": p.get("trial_id", f"cand_{p.get('candidate_index')}"),
                                    "params_after": p.get("params_after", run_details.final_parameters),
                                    "accuracy": p.get("accuracy", run_details.final_accuracy),
                                    "accuracy_retention_pct": p.get("accuracy_retention_pct", 100.0),
                                    "status": "accepted" if evt["event_type"] == "candidate_accepted" else "rejected",
                                    "operation_description": p.get("action_type", ""),
                                    "rejection_reason": p.get("reason", ""),
                                    "checkpoint_path": p.get("checkpoint_path", ""),
                                })
                except Exception:
                    pass

        # Fallback single baseline & best if no detailed records
        if not records and run_details.original_parameters > 0:
            records = [
                {
                    "iteration_id": "baseline",
                    "params_after": run_details.original_parameters,
                    "accuracy": run_details.baseline_accuracy,
                    "accuracy_retention_pct": 100.0,
                    "status": "baseline",
                    "operation_description": "Unpruned Reference Baseline",
                },
                {
                    "iteration_id": "best",
                    "params_after": run_details.final_parameters,
                    "accuracy": run_details.final_accuracy,
                    "accuracy_retention_pct": run_details.accuracy_retention_pct,
                    "status": "accepted",
                    "operation_description": f"Optimized via {run_details.strategy.upper()}",
                },
            ]

        return records

    @classmethod
    def _write_summary_report(cls, path: str, run: RunDetails):
        best_ckpt = run.best_checkpoint or "runs/.../best"
        loading_code = f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "{best_ckpt}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
```"""

        content = f"""# PruneQuest Optimization Report
## PruneQuest — Optimization Summary

### Executive Overview
- **Model:** `{run.model_id}`
- **Benchmark:** `{run.benchmark.upper()}`
- **Strategy:** `{run.strategy.upper()}`
- **Preset:** `{run.preset.upper()}`
- **Status:** `{run.status.upper()}`
- **Checkpoint Reload Verification:** `{ 'PASSED' if run.reload_verified else 'NOT VERIFIED' }`

---

## Key Performance Results
| Metric | Original Baseline | Optimized Model | Reduction / Retention |
| :--- | :--- | :--- | :--- |
| **Parameters** | {run.original_parameters:,} | {run.final_parameters:,} | **-{run.compression_pct:.2f}%** ({run.parameters_pruned:,} removed) |
| **Accuracy** | {run.baseline_accuracy*100:.2f}% | {run.final_accuracy*100:.2f}% | **{run.accuracy_retention_pct:.2f}% retention** (Target: {run.retention_target_pct:.1f}%) |
| **Elapsed Time** | — | {run.elapsed_time_sec:.1f}s | — |

---

## Best Validated Checkpoint
- **Path:** `{best_ckpt}`

### How to Load
{loading_code}
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @classmethod
    def _write_detailed_report(cls, path: str, run: RunDetails, records: List[Dict[str, Any]]):
        table_rows = []
        for r in records:
            acc = r.get("accuracy", 0.0) * 100
            ret = r.get("accuracy_retention_pct", 100.0)
            status = r.get("status", "unknown")
            params = r.get("params_after", 0)
            desc = r.get("operation_description", "")
            table_rows.append(
                f"| `{r.get('iteration_id', '')}` | {desc} | {params:,} | {acc:.2f}% | {ret:.2f}% | `{status}` |"
            )

        rows_str = "\n".join(table_rows) if table_rows else "| None | — | — | — | — | — |"

        content = f"""# PruneQuest — Detailed Experiment Report

## 1. Experiment Metadata
- **Run ID:** `{run.run_id}`
- **Model ID:** `{run.model_id}`
- **Benchmark:** `{run.benchmark.upper()}`
- **Strategy:** `{run.strategy.upper()}`
- **Preset:** `{run.preset.upper()}`
- **Target Accuracy Retention:** {run.retention_target_pct:.1f}%
- **Device:** `{run.device}`
- **Created At:** {run.created_at}
- **Elapsed Duration:** {run.elapsed_time_sec:.1f}s
- **Stopping Reason:** `{run.stopping_reason or 'Completed search budget'}`

---

## 2. Optimizer Efficiency Accounting
- **Actual Full Evaluations Executed:** {len([r for r in records if r.get('status') in ('accepted', 'rejected')])}
- **Accepted Pruning Operations:** {len([r for r in records if r.get('status') == 'accepted'])}
- **Rejected Candidates (Rolled Back):** {len([r for r in records if r.get('status') == 'rejected'])}
- **Parameters Pruned:** {run.parameters_pruned:,} ({run.compression_pct:.2f}%)

---

## 3. Iteration & Candidate Decisions
| Candidate ID | Operation | Parameters After | Accuracy | Retention | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
{rows_str}
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @classmethod
    def _write_history_csv(cls, path: str, records: List[Dict[str, Any]]):
        if records:
            df = pd.DataFrame(records)
            df.to_csv(path, index=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write("iteration_id,params_after,accuracy,accuracy_retention_pct,status,operation_description\n")

    @classmethod
    def _compute_pareto_frontier(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Computes the true non-dominated Pareto frontier points from candidate records.
        Minimizes parameter count and maximizes accuracy.
        """
        valid = [r for r in records if r.get("params_after", 0) > 0 and r.get("accuracy", 0.0) >= 0.0]
        if not valid:
            return []

        # Sort primarily by params ascending, then accuracy descending
        sorted_records = sorted(valid, key=lambda r: (r["params_after"], -r["accuracy"]))

        pareto: List[Dict[str, Any]] = []
        max_acc = -1.0

        for r in sorted_records:
            acc = r["accuracy"]
            if acc > max_acc:
                pareto.append(r)
                max_acc = acc

        # Re-sort descending by params for progressive presentation
        pareto.sort(key=lambda r: r["params_after"], reverse=True)
        return pareto

    @classmethod
    def _generate_plots(
        cls,
        plots_dir: str,
        run: RunDetails,
        records: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        plot_paths: Dict[str, str] = {}
        if not records or len(records) < 2:
            return plot_paths

        df = pd.DataFrame(records)
        if "params_after" not in df.columns or "accuracy" not in df.columns:
            return plot_paths

        # 1. Accuracy vs Parameters
        plt.figure(figsize=(8, 4.8))
        acc_pts = df[df["status"].isin(["accepted", "baseline"])].sort_values("params_after", ascending=False)
        if not acc_pts.empty:
            plt.plot(acc_pts["params_after"] / 1e6, acc_pts["accuracy"] * 100, "o-", color="#2e7d32", linewidth=2, label="Accepted Checkpoints")
        plt.title(f"Accuracy vs. Parameters — {run.model_id.split('/')[-1]}", fontsize=12, fontweight="bold")
        plt.xlabel("Parameters (Millions)", fontsize=11)
        plt.ylabel(f"{run.benchmark.upper()} Accuracy (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        p1 = os.path.join(plots_dir, "accuracy_vs_parameters.png")
        plt.savefig(p1, dpi=180)
        plt.close()
        plot_paths["accuracy_plot"] = p1

        # 2. True Pareto Frontier
        pareto_points = cls._compute_pareto_frontier(records)
        if pareto_points:
            plt.figure(figsize=(8, 4.8))
            all_p = [r["params_after"] / 1e6 for r in records]
            all_a = [r["accuracy"] * 100 for r in records]
            plt.scatter(all_p, all_a, color="#9e9e9e", alpha=0.6, label="Explored Candidates")

            par_p = [r["params_after"] / 1e6 for r in pareto_points]
            par_a = [r["accuracy"] * 100 for r in pareto_points]
            plt.plot(par_p, par_a, "s-", color="#d32f2f", linewidth=2.5, markersize=7, label="Non-Dominated Pareto Frontier")

            plt.title("Pareto Optimal Trade-off Frontier", fontsize=12, fontweight="bold")
            plt.xlabel("Parameters (Millions)", fontsize=11)
            plt.ylabel(f"{run.benchmark.upper()} Accuracy (%)", fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            p2 = os.path.join(plots_dir, "pareto_frontier.png")
            plt.savefig(p2, dpi=180)
            plt.close()
            plot_paths["pareto_plot"] = p2

        return plot_paths
