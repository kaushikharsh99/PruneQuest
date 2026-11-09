"""
Transactional experiment tracking using SQLite and JSONL.
"""

import os
import json
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class ExperimentTracker:
    """
    Manages structured iteration logs, candidate records, and Pareto frontiers.
    """

    def __init__(self, run_dir: str, experiment_name: str = "prunequest_experiment"):
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        os.makedirs(run_dir, exist_ok=True)

        self.db_path = os.path.join(run_dir, "experiments.db")
        self.jsonl_dir = os.path.join(run_dir, "generations")
        os.makedirs(self.jsonl_dir, exist_ok=True)

        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iterations (
                    iteration_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    parent_id TEXT,
                    pruning_type TEXT,
                    operation_description TEXT,
                    params_before INTEGER,
                    params_after INTEGER,
                    params_removed INTEGER,
                    param_retention_pct REAL,
                    stage TEXT,
                    sample_size INTEGER,
                    num_correct INTEGER,
                    accuracy REAL,
                    accuracy_ci_lower REAL,
                    accuracy_ci_upper REAL,
                    accuracy_retention_pct REAL,
                    tokens_per_sec REAL,
                    peak_vram_gb REAL,
                    elapsed_time_sec REAL,
                    status TEXT,
                    rejection_reason TEXT,
                    checkpoint_path TEXT,
                    timestamp TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pareto_frontier (
                    iteration_id TEXT PRIMARY KEY,
                    params INTEGER,
                    param_retention_pct REAL,
                    accuracy REAL,
                    accuracy_retention_pct REAL,
                    checkpoint_path TEXT,
                    timestamp TEXT
                )
            """)
            conn.commit()

    def log_iteration(self, record: Dict[str, Any]):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO iterations (
                    iteration_id, experiment_name, parent_id, pruning_type,
                    operation_description, params_before, params_after,
                    params_removed, param_retention_pct, stage, sample_size,
                    num_correct, accuracy, accuracy_ci_lower, accuracy_ci_upper,
                    accuracy_retention_pct, tokens_per_sec, peak_vram_gb,
                    elapsed_time_sec, status, rejection_reason, checkpoint_path,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("iteration_id"),
                self.experiment_name,
                record.get("parent_id", "baseline"),
                record.get("pruning_type", "layer"),
                record.get("operation_description", ""),
                record.get("params_before", 0),
                record.get("params_after", 0),
                record.get("params_removed", 0),
                record.get("param_retention_pct", 100.0),
                record.get("stage", "screening"),
                record.get("sample_size", 0),
                record.get("num_correct", 0),
                record.get("accuracy", 0.0),
                record.get("accuracy_ci_lower", 0.0),
                record.get("accuracy_ci_upper", 0.0),
                record.get("accuracy_retention_pct", 0.0),
                record.get("tokens_per_sec", 0.0),
                record.get("peak_vram_gb", 0.0),
                record.get("elapsed_time_sec", 0.0),
                record.get("status", "pending"),
                record.get("rejection_reason", ""),
                record.get("checkpoint_path", ""),
                now,
            ))
            conn.commit()

    def log_generations(self, iteration_id: str, stage: str, records: List[Dict[str, Any]]):
        filename = f"{iteration_id}_{stage}_generations.jsonl"
        filepath = os.path.join(self.jsonl_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def update_pareto_frontier(self, checkpoint_records: List[Dict[str, Any]]):
        df = pd.DataFrame(checkpoint_records)
        if df.empty or "accuracy" not in df.columns or "params_after" not in df.columns:
            return

        sorted_df = df.sort_values(by=["params_after", "accuracy"], ascending=[True, False])
        pareto_points = []
        max_acc_seen = -1.0

        for _, row in sorted_df.iterrows():
            if row["accuracy"] > max_acc_seen:
                pareto_points.append(row)
                max_acc_seen = row["accuracy"]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pareto_frontier")
            for p in pareto_points:
                cursor.execute("""
                    INSERT INTO pareto_frontier (
                        iteration_id, params, param_retention_pct,
                        accuracy, accuracy_retention_pct, checkpoint_path, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    p["iteration_id"],
                    p["params_after"],
                    p.get("param_retention_pct", 100.0),
                    p["accuracy"],
                    p.get("accuracy_retention_pct", 100.0),
                    p.get("checkpoint_path", ""),
                    datetime.now(timezone.utc).isoformat(),
                ))
            conn.commit()

    def export_csv(self) -> str:
        csv_path = os.path.join(self.run_dir, "iterations_summary.csv")
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM iterations ORDER BY timestamp ASC", conn)
            df.to_csv(csv_path, index=False)
        return csv_path

    def get_history_dataframe(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM iterations", conn)

    def get_latest_accepted(self) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            res = cursor.execute(
                "SELECT * FROM iterations WHERE status IN ('baseline', 'accepted') ORDER BY params_after ASC LIMIT 1"
            ).fetchone()
            if res:
                col_names = [d[0] for d in cursor.description]
                return dict(zip(col_names, res))
        return None

    def get_baseline(self) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            res = cursor.execute(
                "SELECT * FROM iterations WHERE iteration_id = 'baseline' LIMIT 1"
            ).fetchone()
            if res:
                col_names = [d[0] for d in cursor.description]
                return dict(zip(col_names, res))
        return None

    def get_all_accepted(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            res = cursor.execute(
                "SELECT * FROM iterations WHERE status IN ('baseline', 'accepted') ORDER BY timestamp ASC"
            ).fetchall()
            col_names = [d[0] for d in cursor.description]
            return [dict(zip(col_names, row)) for row in res]
