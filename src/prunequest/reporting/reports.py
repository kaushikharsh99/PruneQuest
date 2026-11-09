"""
Report generation, visualization plotting, and experiment summary export.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any


class ReportGenerator:
    """
    Generates research plots, summary JSONs, and Markdown reports.
    """

    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self.plots_dir = os.path.join(reports_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def generate_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        if df.empty or "accuracy" not in df.columns or "params_after" not in df.columns:
            return {}

        plot_paths = {}
        eval_df = df[df["status"].isin(["accepted", "baseline"])].copy()
        if eval_df.empty:
            eval_df = df.copy()

        eval_df = eval_df.sort_values(by="params_after", ascending=False)
        params_m = eval_df["params_after"] / 1e6

        # 1. Accuracy vs Parameters
        plt.figure(figsize=(8, 5))
        plt.plot(params_m, eval_df["accuracy"] * 100, "o-", color="#1f77b4", linewidth=2, markersize=7)
        plt.title("Benchmark Accuracy vs Parameter Count", fontsize=14, fontweight="bold")
        plt.xlabel("Parameters (Millions)", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        p1 = os.path.join(self.plots_dir, "accuracy_vs_parameters.png")
        plt.tight_layout()
        plt.savefig(p1, dpi=200)
        plt.close()
        plot_paths["accuracy_vs_parameters"] = p1

        # 2. Accuracy vs Pruning Percentage
        prune_pct = (1.0 - eval_df["param_retention_pct"] / 100.0) * 100
        plt.figure(figsize=(8, 5))
        plt.plot(prune_pct, eval_df["accuracy"] * 100, "s-", color="#2ca02c", linewidth=2, markersize=7)
        plt.title("Benchmark Accuracy vs Parameters Pruned", fontsize=14, fontweight="bold")
        plt.xlabel("Parameters Pruned (%)", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        p2 = os.path.join(self.plots_dir, "accuracy_vs_pruning_pct.png")
        plt.tight_layout()
        plt.savefig(p2, dpi=200)
        plt.close()
        plot_paths["accuracy_vs_pruning_pct"] = p2

        # 3. Throughput vs Parameters
        if "tokens_per_sec" in eval_df.columns and (eval_df["tokens_per_sec"] > 0).any():
            plt.figure(figsize=(8, 5))
            plt.plot(params_m, eval_df["tokens_per_sec"], "d-", color="#ff7f0e", linewidth=2, markersize=7)
            plt.title("Inference Throughput vs Parameter Count", fontsize=14, fontweight="bold")
            plt.xlabel("Parameters (Millions)", fontsize=12)
            plt.ylabel("Tokens / sec", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.6)
            p3 = os.path.join(self.plots_dir, "throughput_vs_parameters.png")
            plt.tight_layout()
            plt.savefig(p3, dpi=200)
            plt.close()
            plot_paths["throughput_vs_parameters"] = p3

        # 4. Peak VRAM vs Parameters
        if "peak_vram_gb" in eval_df.columns and (eval_df["peak_vram_gb"] > 0).any():
            plt.figure(figsize=(8, 5))
            plt.plot(params_m, eval_df["peak_vram_gb"], "^-", color="#d62728", linewidth=2, markersize=7)
            plt.title("Peak VRAM Footprint vs Parameter Count", fontsize=14, fontweight="bold")
            plt.xlabel("Parameters (Millions)", fontsize=12)
            plt.ylabel("Peak VRAM (GB)", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.6)
            p4 = os.path.join(self.plots_dir, "vram_vs_parameters.png")
            plt.tight_layout()
            plt.savefig(p4, dpi=200)
            plt.close()
            plot_paths["vram_vs_parameters"] = p4

        # 5. Pareto Frontier Plot
        plt.figure(figsize=(9, 6))
        all_params_m = df["params_after"] / 1e6
        plt.scatter(all_params_m, df["accuracy"] * 100, color="gray", alpha=0.5, label="Explored Candidates")
        plt.plot(params_m, eval_df["accuracy"] * 100, "r-o", linewidth=2.5, label="Accepted Pareto Frontier")
        plt.title("Pareto Frontier: Parameter Count vs Reasoning Accuracy", fontsize=14, fontweight="bold")
        plt.xlabel("Parameters (Millions)", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        p5 = os.path.join(self.plots_dir, "pareto_frontier.png")
        plt.tight_layout()
        plt.savefig(p5, dpi=200)
        plt.close()
        plot_paths["pareto_frontier"] = p5

        return plot_paths

    def generate_report(
        self,
        df: pd.DataFrame,
        summary: Dict[str, Any],
        plot_paths: Dict[str, str],
    ) -> str:
        report_file = os.path.join(self.reports_dir, "EXPERIMENT_REPORT.md")
        lines = [
            "# PruneQuest Experiment Report",
            "",
            "## Summary",
            "",
            f"- **Status:** {summary.get('status', 'completed')}",
            f"- **Baseline Accuracy:** {summary.get('baseline_accuracy', 0.0)*100:.2f}%",
            f"- **Final Accuracy:** {summary.get('final_accuracy', 0.0)*100:.2f}%",
            f"- **Accuracy Retention:** {summary.get('accuracy_retention_pct', 0.0):.2f}%",
            f"- **Original Parameters:** {summary.get('original_params', 0):,}",
            f"- **Final Parameters:** {summary.get('final_params', 0):,}",
            f"- **Best Checkpoint:** `{summary.get('best_checkpoint', '')}`",
            "",
            "## Benchmark & Iteration History",
            "",
        ]

        if not df.empty:
            cols = ["iteration_id", "params_after", "param_retention_pct", "accuracy", "accuracy_retention_pct", "tokens_per_sec", "peak_vram_gb", "status"]
            valid = [c for c in cols if c in df.columns]
            table_df = df[valid].copy()
            if "params_after" in table_df.columns:
                table_df["params_m"] = (table_df["params_after"] / 1e6).round(2)
                table_df.drop(columns=["params_after"], inplace=True)
            if "accuracy" in table_df.columns:
                table_df["acc_%"] = (table_df["accuracy"] * 100).round(2)
                table_df.drop(columns=["accuracy"], inplace=True)
            lines.append(table_df.to_markdown(index=False))
            lines.append("")

        lines.extend([
            "## Visualizations",
            "",
            "![Pareto Frontier](plots/pareto_frontier.png)",
            "![Accuracy vs Parameters](plots/accuracy_vs_parameters.png)",
            "![Accuracy vs Pruning Pct](plots/accuracy_vs_pruning_pct.png)",
            "![Throughput vs Parameters](plots/throughput_vs_parameters.png)",
            "![Peak VRAM vs Parameters](plots/vram_vs_parameters.png)",
            "",
        ])

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Also write summary JSON
        summary_json = os.path.join(self.reports_dir, "summary.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return report_file
