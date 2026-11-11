"""
Run Management Engine for PruneQuest.
Provides fast directory scanning, manifest parsing, run comparison, log filtering,
and concurrent-run locking without heavy model dependencies.
"""

import os
import json
import time
from typing import List, Optional, Dict, Any

from prunequest.runs.models import RunSummary, RunDetails, RunComparison
from prunequest.events.models import OptimizationEvent


class RunLockError(Exception):
    """Raised when an active lock cannot be acquired on a run directory."""
    pass


class RunLock:
    """
    File-based mutual exclusion lock to prevent concurrent modifications on the same run directory.
    Includes safe stale-lock recovery without deleting data.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.lock_file = os.path.join(run_dir, ".run.lock")
        self._acquired = False

    def acquire(self):
        os.makedirs(self.run_dir, exist_ok=True)
        pid = os.getpid()

        if os.path.exists(self.lock_file):
            holder_pid = None
            try:
                with open(self.lock_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                holder_pid = data.get("pid")
            except Exception:
                pass

            if holder_pid and holder_pid != pid:
                is_alive = False
                try:
                    os.kill(holder_pid, 0)
                    is_alive = True
                except PermissionError:
                    # Process exists, owned by another user (e.g. root/PID 1)
                    is_alive = True
                except ProcessLookupError:
                    # Process is dead
                    is_alive = False
                except OSError:
                    is_alive = False

                if is_alive:
                    raise RunLockError(
                        f"Run directory '{self.run_dir}' is locked by active process PID {holder_pid}. "
                        "Concurrent optimization on the same directory is prevented."
                    )

        # Write current lock
        lock_data = {
            "pid": pid,
            "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.lock_file, "w", encoding="utf-8") as f:
            json.dump(lock_data, f)
        self._acquired = True

    def release(self):
        if self._acquired and os.path.exists(self.lock_file):
            try:
                os.remove(self.lock_file)
            except OSError:
                pass
            self._acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class RunManager:
    """
    High-level manager for discovering, inspecting, comparing, and managing optimization runs.
    """

    @classmethod
    def resolve_run_dir(cls, run_id_or_path: str, runs_root: str = "runs") -> str:
        """
        Resolves a run ID or relative/absolute path to an existing directory path.
        """
        if os.path.isdir(run_id_or_path):
            return os.path.abspath(run_id_or_path)

        # Check under runs_root
        candidate = os.path.join(runs_root, run_id_or_path)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

        # Search for matching run_id in runs_root subdirectories
        if os.path.isdir(runs_root):
            for entry in os.listdir(runs_root):
                sub = os.path.join(runs_root, entry)
                if os.path.isdir(sub):
                    mf = os.path.join(sub, "manifest.json")
                    if os.path.exists(mf):
                        try:
                            with open(mf, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            if data.get("run_id") == run_id_or_path:
                                return os.path.abspath(sub)
                        except Exception:
                            pass

        raise FileNotFoundError(f"Could not locate run directory for: '{run_id_or_path}'")

    @classmethod
    def list_runs(
        cls,
        runs_root: str = "runs",
        limit: Optional[int] = None,
        status: Optional[str] = None,
        model: Optional[str] = None,
        sort: str = "newest",
    ) -> List[RunSummary]:
        """
        Scans runs_root and discovers runs via manifest.json / results.json without loading weights.
        """
        if not os.path.isdir(runs_root):
            return []

        runs: List[RunSummary] = []
        for entry in os.listdir(runs_root):
            dir_path = os.path.join(runs_root, entry)
            if not os.path.isdir(dir_path):
                continue

            summary = cls._parse_run_summary(dir_path)
            if not summary:
                continue

            # Filtering
            if status and summary.status.lower() != status.lower():
                continue
            if model and model.lower() not in summary.model_id.lower():
                continue

            runs.append(summary)

        # Sorting
        if sort == "newest":
            runs.sort(key=lambda r: r.created_at, reverse=True)
        elif sort == "oldest":
            runs.sort(key=lambda r: r.created_at, reverse=False)
        elif sort == "compression":
            runs.sort(key=lambda r: r.compression_pct, reverse=True)
        elif sort == "retention":
            runs.sort(key=lambda r: r.accuracy_retention_pct, reverse=True)

        if limit and limit > 0:
            runs = runs[:limit]

        return runs

    @classmethod
    def get_run(cls, run_id_or_path: str, runs_root: str = "runs") -> RunDetails:
        """
        Loads detailed inspection data for a single run without loading model weights.
        """
        run_dir = cls.resolve_run_dir(run_id_or_path, runs_root=runs_root)
        manifest_path = os.path.join(run_dir, "manifest.json")
        results_path = os.path.join(run_dir, "results.json")

        manifest_data: Dict[str, Any] = {}
        results_data: Dict[str, Any] = {}

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                pass

        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    results_data = json.load(f)
            except Exception:
                pass

        # Parse with fallback
        run_id = manifest_data.get("run_id") or os.path.basename(run_dir)
        schema_version = manifest_data.get("schema_version", 1)
        model_id = manifest_data.get("model_id") or results_data.get("model_id") or "unknown"
        strategy = (
            manifest_data.get("strategy")
            or results_data.get("strategy")
            or manifest_data.get("results", {}).get("strategy")
            or "auto"
        )
        benchmark = (
            manifest_data.get("benchmark")
            or results_data.get("benchmark")
            or manifest_data.get("results", {}).get("benchmark")
            or "unknown"
        )
        status = manifest_data.get("status") or results_data.get("status") or "completed"
        resumable = manifest_data.get("resumable", status in ("running", "interrupted"))

        res = results_data or manifest_data.get("results") or {}
        orig_p = res.get("original_parameters", 0)
        final_p = res.get("final_parameters", orig_p)
        pruned_p = res.get("parameters_pruned", max(0, orig_p - final_p))
        comp_pct = res.get("parameters_pruned_pct", round((pruned_p / orig_p * 100), 2) if orig_p > 0 else 0.0)

        best_ckpt = manifest_data.get("checkpoint_directory") or res.get("best_checkpoint")
        if best_ckpt and not os.path.isabs(best_ckpt):
            best_ckpt = os.path.join(run_dir, best_ckpt)

        return RunDetails(
            run_id=run_id,
            schema_version=schema_version,
            model_id=model_id,
            strategy=strategy,
            benchmark=benchmark,
            preset=res.get("preset", "balanced"),
            device=manifest_data.get("device", "auto"),
            status=status,
            resumable=resumable,
            retention_target_pct=res.get("retention_target_pct", 99.0),
            original_parameters=orig_p,
            final_parameters=final_p,
            parameters_pruned=pruned_p,
            compression_pct=comp_pct,
            baseline_accuracy=res.get("baseline_accuracy", 0.0),
            final_accuracy=res.get("final_accuracy", 0.0),
            accuracy_retention_pct=res.get("accuracy_retention_pct", 100.0),
            best_checkpoint=best_ckpt if (best_ckpt and os.path.exists(best_ckpt)) else None,
            reload_verified=res.get("reload_verified", False),
            elapsed_time_sec=res.get("elapsed_time_sec", 0.0),
            created_at=manifest_data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(run_dir))),
            run_dir=run_dir,
            reports_dir=os.path.join(run_dir, "reports"),
            resolved_config_path=os.path.join(run_dir, "resolved_config.yaml") if os.path.exists(os.path.join(run_dir, "resolved_config.yaml")) else None,
            stopping_reason=res.get("stopping_reason"),
            metadata=manifest_data,
        )

    @classmethod
    def compare_runs(cls, run_a_id: str, run_b_id: str, runs_root: str = "runs") -> RunComparison:
        """
        Compares two runs and checks evaluation compatibility.
        """
        a = cls.get_run(run_a_id, runs_root=runs_root)
        b = cls.get_run(run_b_id, runs_root=runs_root)

        is_compatible = True
        notes = []

        if a.benchmark.lower() != b.benchmark.lower():
            is_compatible = False
            notes.append(f"Incompatible benchmarks: '{a.benchmark}' vs '{b.benchmark}'. Scores cannot be directly compared.")

        if a.model_id.lower() != b.model_id.lower():
            notes.append(f"Different base models: '{a.model_id}' vs '{b.model_id}'.")

        param_diff = a.final_parameters - b.final_parameters
        comp_diff = round(b.compression_pct - a.compression_pct, 2)
        acc_diff = round(b.final_accuracy - a.final_accuracy, 4)
        ret_diff = round(b.accuracy_retention_pct - a.accuracy_retention_pct, 2)
        time_diff = round(b.elapsed_time_sec - a.elapsed_time_sec, 2)

        return RunComparison(
            run_a=a,
            run_b=b,
            is_compatible=is_compatible,
            compatibility_notes=notes,
            param_reduction_diff=param_diff,
            compression_diff_pct=comp_diff,
            accuracy_diff=acc_diff,
            retention_diff_pct=ret_diff,
            elapsed_time_diff_sec=time_diff,
        )

    @classmethod
    def get_run_logs(
        cls,
        run_id_or_path: str,
        phase: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
        runs_root: str = "runs",
    ) -> List[OptimizationEvent]:
        """
        Reads events.jsonl from run directory with phase and category filtering.
        """
        run_dir = cls.resolve_run_dir(run_id_or_path, runs_root=runs_root)
        events_file = os.path.join(run_dir, "events.jsonl")

        if not os.path.exists(events_file):
            return []

        events: List[OptimizationEvent] = []
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = OptimizationEvent.from_json(line)
                    if phase and evt.phase.lower() != phase.lower():
                        continue
                    if category and evt.event_type.lower() != category.lower():
                        continue
                    events.append(evt)
                except Exception:
                    continue

        if limit and limit > 0:
            events = events[-limit:]

        return events

    @classmethod
    def _parse_run_summary(cls, dir_path: str) -> Optional[RunSummary]:
        manifest_file = os.path.join(dir_path, "manifest.json")
        results_file = os.path.join(dir_path, "results.json")

        data: Dict[str, Any] = {}
        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        res = data.get("results") or {}
        if not res and os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    res = json.load(f)
            except Exception:
                pass

        if not data and not res:
            return None

        run_id = data.get("run_id") or os.path.basename(dir_path)
        model_id = data.get("model_id") or res.get("model_id") or "unknown"
        strategy = data.get("strategy") or res.get("strategy") or "auto"
        benchmark = data.get("benchmark") or res.get("benchmark") or "unknown"
        status = data.get("status") or res.get("status") or "completed"
        comp_pct = res.get("parameters_pruned_pct", 0.0)
        ret_pct = res.get("accuracy_retention_pct", 100.0)
        best_ckpt = data.get("checkpoint_directory") or res.get("best_checkpoint")
        created_at = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(dir_path)))
        resumable = data.get("resumable", status in ("running", "interrupted"))

        return RunSummary(
            run_id=run_id,
            model_id=model_id,
            strategy=strategy,
            benchmark=benchmark,
            status=status,
            compression_pct=comp_pct,
            accuracy_retention_pct=ret_pct,
            best_checkpoint=best_ckpt,
            created_at=created_at,
            resumable=resumable,
            run_dir=os.path.abspath(dir_path),
        )
