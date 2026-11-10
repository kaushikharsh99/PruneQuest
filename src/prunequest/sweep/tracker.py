"""
Experiment tracker and Pareto frontier manager for Full Sweep Joint Optimizer.
Tracks trial outcomes, execution metrics, and quantifies saved evaluation compute.
"""

from typing import Dict, Any, List, Optional
import os
import sqlite3
import time


class JointExperimentTracker:
    """
    Persists joint optimization trials into SQLite, tracks Pareto frontiers,
    and quantifies evaluation work saved by joint candidate grouping and cheap screening.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.db_path = os.path.join(run_dir, "experiments.db")

        # In-memory tracking
        self.records: List[Dict[str, Any]] = []
        self.pareto_frontier: List[Dict[str, Any]] = []
        self.evaluations_saved_by_joint: int = 0
        self.evaluations_saved_by_screening: int = 0
        self.total_full_evaluations: int = 0

        self._init_db()
        self._load_existing_records()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    iteration INTEGER,
                    action_type TEXT,
                    parent_id TEXT,
                    params_before INTEGER,
                    params_after INTEGER,
                    params_removed INTEGER,
                    param_retention_pct REAL,
                    screening_loss REAL,
                    accuracy REAL,
                    accuracy_retention_pct REAL,
                    status TEXT,
                    rejection_reason TEXT,
                    checkpoint_path TEXT,
                    elapsed_time_sec REAL,
                    created_at REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()

    def _load_existing_records(self):
        """Loads historical trials from SQLite to populate in-memory state on resume."""
        if not os.path.exists(self.db_path):
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trials ORDER BY created_at ASC")
            for row in cursor.fetchall():
                rec = dict(row)
                self.records.append(rec)
                if rec.get("status") == "accepted":
                    self.update_pareto_frontier(rec)

    def log_trial(self, record: Dict[str, Any]):
        """Records a trial into memory and SQLite."""
        self.records.append(record)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trials (
                    trial_id, iteration, action_type, parent_id,
                    params_before, params_after, params_removed, param_retention_pct,
                    screening_loss, accuracy, accuracy_retention_pct,
                    status, rejection_reason, checkpoint_path,
                    elapsed_time_sec, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("trial_id", f"trial_{len(self.records)}"),
                record.get("iteration", 0),
                record.get("action_type", "joint"),
                record.get("parent_id", "baseline"),
                record.get("params_before", 0),
                record.get("params_after", 0),
                record.get("params_removed", 0),
                record.get("param_retention_pct", 100.0),
                record.get("screening_loss", 0.0),
                record.get("accuracy", 0.0),
                record.get("accuracy_retention_pct", 100.0),
                record.get("status", "pending"),
                str(record.get("rejection_reason", "")),
                str(record.get("checkpoint_path", "")),
                float(record.get("elapsed_time_sec", 0.0)),
                time.time()
            ))
            conn.commit()

        if record.get("status") == "accepted":
            self.update_pareto_frontier(record)

    def record_joint_evaluation_savings(self):
        """Called when a joint candidate passes: saved 1 full evaluation."""
        self.evaluations_saved_by_joint += 1

    def record_screening_rejection_savings(self):
        """Called when cheap screening rejects a poor candidate early without full benchmark validation."""
        self.evaluations_saved_by_screening += 1

    def record_full_evaluation(self):
        """Tracks executed full benchmark generation evaluations."""
        self.total_full_evaluations += 1

    def update_pareto_frontier(self, record: Dict[str, Any]):
        """
        Maintains non-dominated set of checkpoints.
        Candidate A dominates B if A has fewer params and >= accuracy.
        """
        params = record.get("params_after", 0)
        acc = record.get("accuracy", 0.0)

        # Check if dominated
        is_dominated = False
        for p in self.pareto_frontier:
            if p["params_after"] <= params and p["accuracy"] >= acc:
                if p["params_after"] < params or p["accuracy"] > acc:
                    is_dominated = True
                    break

        if not is_dominated:
            # Remove any members dominated by this new record
            self.pareto_frontier = [
                p for p in self.pareto_frontier
                if not (params <= p["params_after"] and acc >= p["accuracy"] and (params < p["params_after"] or acc > p["accuracy"]))
            ]
            self.pareto_frontier.append(record)

    def get_baseline(self) -> Optional[Dict[str, Any]]:
        """Returns baseline record if already established in SQLite DB."""
        if not os.path.exists(self.db_path):
            return None
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trials WHERE trial_id = 'baseline' AND status = 'accepted'")
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_latest_accepted(self) -> Optional[Dict[str, Any]]:
        """Returns the latest accepted checkpoint record that exists on disk."""
        if not os.path.exists(self.db_path):
            return None
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM trials WHERE status = 'accepted' AND trial_id != 'baseline' AND checkpoint_path != '' "
                "ORDER BY iteration DESC, created_at DESC"
            )
            rows = cursor.fetchall()
            for row in rows:
                rec = dict(row)
                ckpt_path = rec.get("checkpoint_path", "")
                if ckpt_path:
                    candidate_paths = [
                        ckpt_path,
                        os.path.join(self.run_dir, ckpt_path),
                        os.path.join(self.run_dir, "checkpoints", os.path.basename(ckpt_path)),
                    ]
                    resolved_path = None
                    for cp in candidate_paths:
                        if os.path.exists(cp):
                            resolved_path = os.path.abspath(cp)
                            break
                    if resolved_path:
                        weights_safetensors = os.path.join(resolved_path, "model.safetensors")
                        weights_bin = os.path.join(resolved_path, "pytorch_model.bin")
                        weights_index = os.path.join(resolved_path, "model.safetensors.index.json")
                        if os.path.exists(weights_safetensors) or os.path.exists(weights_bin) or os.path.exists(weights_index):
                            rec["checkpoint_path"] = resolved_path
                            return rec
        return None

    def get_best_pareto_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Returns the accepted checkpoint with greatest parameter reduction that satisfied retention."""
        accepted = [r for r in self.records if r.get("status") == "accepted" and r.get("checkpoint_path")]
        pruned_accepted = [r for r in accepted if r.get("checkpoint_path") != "unpruned_baseline"]
        if pruned_accepted:
            pruned_accepted.sort(key=lambda x: x["params_after"])
            return pruned_accepted[0]
        if accepted:
            accepted.sort(key=lambda x: x["params_after"])
            return accepted[0]
        return None

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Returns aggregated efficiency and search metrics."""
        return {
            "total_trials": len(self.records),
            "accepted_trials": sum(1 for r in self.records if r.get("status") == "accepted"),
            "full_evaluations_executed": self.total_full_evaluations,
            "evaluations_saved_by_joint": self.evaluations_saved_by_joint,
            "evaluations_saved_by_screening": self.evaluations_saved_by_screening,
            "total_evaluations_saved": self.evaluations_saved_by_joint + self.evaluations_saved_by_screening,
            "pareto_points": len(self.pareto_frontier),
        }
