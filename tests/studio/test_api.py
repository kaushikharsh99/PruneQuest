"""
Tests for PruneQuest Studio API endpoints.
"""

from fastapi.testclient import TestClient
from prunequest.studio.server import create_app
import tempfile
import os
import json


def test_health_and_system():
    app = create_app()
    client = TestClient(app)

    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data

    sys_res = client.get("/api/v1/system")
    assert sys_res.status_code == 200
    sys_data = sys_res.json()
    assert "os_name" in sys_data
    assert "python_version" in sys_data
    assert "torch_version" in sys_data
    assert "cuda_available" in sys_data
    assert "ram_total_gb" in sys_data


def test_models_inspect_and_popular():
    app = create_app()
    client = TestClient(app)

    pop_res = client.get("/api/v1/models/popular")
    assert pop_res.status_code == 200
    popular = pop_res.json()
    assert len(popular) > 0
    assert any("Qwen" in m["name"] for m in popular)

    # Inspect supported architecture
    insp_res = client.post("/api/v1/models/inspect", json={"model_id": "Qwen/Qwen2.5-0.5B-Instruct"})
    assert insp_res.status_code == 200
    insp_data = insp_res.json()
    assert insp_data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert insp_data["supported"] is True
    assert insp_data["architecture"] == "qwen2"
    assert insp_data["tier"] in ("LEVEL_2_EXPERIMENTAL_PRUNING", "LEVEL_3_VERIFIED_PRUNING")


def test_benchmarks_and_dataset_endpoints():
    app = create_app()
    client = TestClient(app)

    bench_res = client.get("/api/v1/benchmarks")
    assert bench_res.status_code == 200
    benchmarks = bench_res.json()
    assert len(benchmarks) >= 2
    assert any(b["name"] == "gsm8k" for b in benchmarks)

    # Test custom dataset inspect, preview, and validate
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "qa.jsonl")
        records = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3*5?", "answer": "15"},
            {"question": "What is 10/2?", "answer": "5"},
            {"question": "What is 7-4?", "answer": "3"},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Inspect
        insp_res = client.post("/api/v1/datasets/inspect", json={"source": jsonl_path})
        assert insp_res.status_code == 200
        insp_data = insp_res.json()
        assert insp_data["schema_info"]["num_records"] == 4
        assert insp_data["field_mapping"]["input_column"] == "question"
        assert insp_data["field_mapping"]["target_column"] == "answer"

        # Preview
        prev_res = client.post("/api/v1/datasets/preview", json={"source": jsonl_path, "limit": 2})
        assert prev_res.status_code == 200
        prev_data = prev_res.json()
        assert len(prev_data["samples"]) == 2

        # Validate
        val_res = client.post("/api/v1/datasets/validate", json={"source": jsonl_path})
        assert val_res.status_code == 200
        val_data = val_res.json()
        assert val_data["valid"] is True


def test_optimization_preflight():
    app = create_app()
    client = TestClient(app)

    req = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "benchmark": "gsm8k",
        "retain": 0.99,
        "preset": "quick",
        "strategy": "auto",
        "device": "auto",
    }
    res = client.post("/api/v1/optimization/preflight", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert data["strategy"] in ("joint", "flap", "block")
    assert data["target_retention"] == 0.99
    assert data["disk_sufficient"] is True
