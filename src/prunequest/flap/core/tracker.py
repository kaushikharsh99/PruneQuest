"""
SQLite and JSONL experiment tracker recording candidate evaluations and accepted checkpoints.
"""

from typing import Dict, Any, List, Optional
import os
import sqlite3
import json
import csv
from datetime import datetime, timezone


class ExperimentTracker:
    """
    Records pruning iterations, candidate evaluations, metrics, and checkpoints.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.db_path = os.path.join(output_dir, "experiments.db")
        self.jsonl_path = os.path.join(output_dir, "history.jsonl")
        self.csv_path = os.path.join(output_dir, "candidates.csv")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS iterations (
                candidate_id TEXT PRIMARY KEY,
                step_index INTEGER,
                prune_ratio REAL,
                params_before INTEGER,
                params_after INTEGER,
                params_removed INTEGER,
                screening_loss REAL,
                screening_ppl REAL,
                accuracy REAL,
                num_correct INTEGER,
                sample_size INTEGER,
                retention_pct REAL,
                tokens_per_sec REAL,
                peak_vram_gb REAL,
                elapsed_sec REAL,
                status TEXT,
                rejection_reason TEXT,
                checkpoint_path TEXT,
                created_at TEXT
            )
            """)
            conn.commit()

    def log_candidate(self, record: Dict[str, Any]):
        timestamp = datetime.now(timezone.utc).isoformat()
        rec = dict(record)
        rec.setdefault("created_at", timestamp)

        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO iterations (
                candidate_id, step_index, prune_ratio, params_before, params_after, params_removed,
                screening_loss, screening_ppl, accuracy, num_correct, sample_size, retention_pct,
                tokens_per_sec, peak_vram_gb, elapsed_sec, status, rejection_reason, checkpoint_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.get("candidate_id"),
                rec.get("step_index", 0),
                rec.get("prune_ratio", 0.0),
                rec.get("params_before", 0),
                rec.get("params_after", 0),
                rec.get("params_removed", 0),
                rec.get("screening_loss"),
                rec.get("screening_ppl"),
                rec.get("accuracy", 0.0),
                rec.get("num_correct", 0),
                rec.get("sample_size", 0),
                rec.get("retention_pct", 0.0),
                rec.get("tokens_per_sec", 0.0),
                rec.get("peak_vram_gb", 0.0),
                rec.get("elapsed_sec", 0.0),
                rec.get("status", "unknown"),
                rec.get("rejection_reason", ""),
                rec.get("checkpoint_path", ""),
                rec.get("created_at", timestamp),
            ))
            conn.commit()

        # Append to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

        # Append to CSV
        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        fieldnames = [
            "candidate_id", "step_index", "prune_ratio", "params_before", "params_after", "params_removed",
            "screening_loss", "screening_ppl", "accuracy", "num_correct", "sample_size", "retention_pct",
            "tokens_per_sec", "peak_vram_gb", "elapsed_sec", "status", "rejection_reason", "checkpoint_path", "created_at"
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(rec)

    def get_baseline(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM iterations WHERE candidate_id = 'baseline'")
            row = cur.fetchone()
            return dict(row) if row else None

    def get_latest_accepted(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM iterations WHERE status = 'accepted' ORDER BY step_index DESC LIMIT 1")
            row = cur.fetchone()
            return dict(row) if row else None

    def get_best_accepted(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            # Best is the smallest parameter count among accepted checkpoints
            cur = conn.execute("SELECT * FROM iterations WHERE status = 'accepted' ORDER BY params_after ASC LIMIT 1")
            row = cur.fetchone()
            return dict(row) if row else None

    def get_all_records(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM iterations ORDER BY step_index ASC")
            return [dict(r) for r in cur.fetchall()]
