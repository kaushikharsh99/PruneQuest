"""
Shared test fixtures for creating mock optimization runs.
"""

import os
import json
from prunequest.events.models import OptimizationEvent, EventCategory, EventPhase


def create_mock_run(
    parent_dir: str,
    run_id: str,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    benchmark: str = "gsm8k",
    strategy: str = "joint",
    status: str = "completed",
    compression_pct: float = 18.5,
    retention_pct: float = 99.2,
    timestamp: str = "2026-11-11T10:00:00Z",
    schema_version: int = 2,
):
    run_dir = os.path.join(parent_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    manifest_data = {
        "schema_version": schema_version,
        "run_id": run_id,
        "status": status,
        "resumable": status in ("interrupted", "running"),
        "model_id": model_id,
        "strategy": strategy,
        "benchmark": benchmark,
        "preset": "balanced",
        "retention_target": 0.99,
        "timestamp": timestamp,
        "device": "cuda:0",
        "checkpoint_directory": os.path.join(run_dir, "best"),
        "results": {
            "status": status,
            "run_id": run_id,
            "model_id": model_id,
            "benchmark": benchmark,
            "strategy": strategy,
            "original_parameters": 1000000,
            "final_parameters": int(1000000 * (1 - compression_pct / 100.0)),
            "parameters_pruned": int(1000000 * (compression_pct / 100.0)),
            "compression_ratio": round(1 - compression_pct / 100.0, 4),
            "parameters_pruned_pct": compression_pct,
            "baseline_accuracy": 0.90,
            "final_accuracy": round(0.90 * (retention_pct / 100.0), 4),
            "accuracy_retention_pct": retention_pct,
            "best_checkpoint": os.path.join(run_dir, "best"),
            "reload_verified": True,
            "elapsed_time_sec": 120.0,
        },
    }

    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_data["results"], f, indent=2)

    # Write events.jsonl
    events = [
        OptimizationEvent.create(run_id, 1, EventCategory.RUN_STARTED, EventPhase.PREFLIGHT, {"model_id": model_id}),
        OptimizationEvent.create(run_id, 2, EventCategory.CANDIDATE_ACCEPTED, EventPhase.VALIDATION, {"candidate_index": 1}),
        OptimizationEvent.create(run_id, 3, EventCategory.RUN_COMPLETED, EventPhase.COMPLETED),
    ]
    with open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8") as f:
        for evt in events:
            f.write(evt.to_json() + "\n")

    return run_dir
