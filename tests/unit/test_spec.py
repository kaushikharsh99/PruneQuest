"""
Unit tests for declarative BenchmarkSpec schema, YAML compilation, and native scoring.
"""

import tempfile
import yaml
from prunequest.benchmarks.spec import (
    BenchmarkSpec,
    NativeBenchmark,
    BenchmarkSpecCompiler,
)


def test_spec_yaml_validation_and_scoring():
    spec_dict = {
        "version": 1,
        "name": "custom_test_bench",
        "dataset": {
            "source": "json",
            "path": "dummy.json",
            "fields": {"input": "question", "target": "answer"},
        },
        "task": {
            "type": "generation",
            "prompt_template": "Solve: {{ input }}",
        },
        "scoring": {
            "metric": "numeric_exact_match",
            "higher_is_better": True,
        },
        "optimization": {
            "metric": "numeric_exact_match",
            "retention": 0.95,
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(spec_dict, f)
        yaml_path = f.name

    valid, errors = BenchmarkSpecCompiler.validate_spec(yaml_path)
    assert valid is True
    assert len(errors) == 0

    spec = BenchmarkSpec.from_yaml(yaml_path)
    assert spec.name == "custom_test_bench"
    assert spec.scoring.metric == "numeric_exact_match"

    bench = NativeBenchmark(spec)
    # Test numeric scoring logic
    assert bench._score("The answer is 42", "42") is True
    assert bench._score("Result: 42.0", "42") is True
    assert bench._score("Result: 100", "42") is False


def test_spec_disjoint_partitions():
    import json
    data = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(100)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        json_path = f.name

    spec = BenchmarkSpec(
        name="test_partitions",
        dataset={"source": "json", "path": json_path, "fields": {"input": "question", "target": "answer"}},
        task={"type": "generation"},
        scoring={"metric": "exact_match"},
        optimization={"metric": "exact_match", "retention": 0.99},
    )
    bench = NativeBenchmark(spec)
    partitions = bench.load_data(calibration_size=20, screening_size=10, dev_size=30)

    calib_ids = {s.id for s in partitions["calibration"]}
    screen_ids = {s.id for s in partitions["screening"]}
    dev_ids = {s.id for s in partitions["dev"]}
    test_ids = {s.id for s in partitions["test"]}

    assert len(calib_ids) == 20
    assert len(screen_ids) == 10
    assert len(dev_ids) == 30
    assert len(test_ids) == 40  # 100 - 20 - 10 - 30

    # Ensure zero overlap between partitions
    assert calib_ids.isdisjoint(screen_ids)
    assert calib_ids.isdisjoint(dev_ids)
    assert screen_ids.isdisjoint(dev_ids)
    assert test_ids.isdisjoint(calib_ids)
    assert test_ids.isdisjoint(screen_ids)
    assert test_ids.isdisjoint(dev_ids)

