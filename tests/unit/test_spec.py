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


def test_spec_insufficient_samples_raises_leakage_error():
    import json
    import pytest
    from prunequest.exceptions import BenchmarkError

    # Only 30 samples, but we request 20 calib + 10 screen + 30 dev = 60 needed
    data = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(30)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        json_path = f.name

    spec = BenchmarkSpec(
        name="test_insufficient",
        dataset={"source": "json", "path": json_path, "fields": {"input": "question", "target": "answer"}},
        task={"type": "generation"},
        scoring={"metric": "exact_match"},
        optimization={"metric": "exact_match", "retention": 0.99},
    )
    bench = NativeBenchmark(spec)
    with pytest.raises(BenchmarkError, match="Insufficient data in dataset for disjoint partitions"):
        bench.load_data(calibration_size=20, screening_size=10, dev_size=30)


def test_spec_missing_fields_raises_error():
    import json
    import pytest
    from prunequest.exceptions import BenchmarkError

    # Missing 'answer' column
    data = [{"question": "What is 2+2?"}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        json_path = f.name

    spec = BenchmarkSpec(
        name="test_missing_col",
        dataset={"source": "json", "path": json_path, "fields": {"input": "question", "target": "answer"}},
        task={"type": "generation"},
        scoring={"metric": "exact_match"},
        optimization={"metric": "exact_match", "retention": 0.99},
    )
    bench = NativeBenchmark(spec)
    with pytest.raises(BenchmarkError, match="Required target field 'answer' missing"):
        bench.load_data()


def test_spec_unknown_split_raises_error():
    import json
    import pytest
    from unittest.mock import MagicMock
    from prunequest.exceptions import BenchmarkError

    data = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(100)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        json_path = f.name

    spec = BenchmarkSpec(
        name="test_unknown_split",
        dataset={"source": "json", "path": json_path, "fields": {"input": "question", "target": "answer"}},
        task={"type": "generation"},
        scoring={"metric": "exact_match"},
        optimization={"metric": "exact_match", "retention": 0.99},
    )
    bench = NativeBenchmark(spec)
    bench.load_data(calibration_size=20, screening_size=10, dev_size=30)
    mock_model = MagicMock()
    mock_tok = MagicMock()

    with pytest.raises(BenchmarkError, match="Requested evaluation split 'non_existent_split' is not available"):
        bench.evaluate(mock_model, mock_tok, split_name="non_existent_split")


def test_spec_multiple_choice_and_math_verify():
    import pytest
    from unittest.mock import patch
    from prunequest.exceptions import BenchmarkError

    spec = BenchmarkSpec(
        name="test_mc",
        dataset={"source": "json", "path": "dummy.json", "fields": {"input": "question", "target": "answer"}},
        task={"type": "multiple_choice"},
        scoring={"metric": "multiple_choice"},
        optimization={"metric": "multiple_choice", "retention": 0.99},
    )
    bench = NativeBenchmark(spec)
    assert bench._score("A", "A") is True
    assert bench._score("The correct answer is B.", "B") is True
    assert bench._score("Choice C", "C") is True
    assert bench._score("D", "A") is False

    # math_verify error when library unavailable
    spec_mv = BenchmarkSpec(
        name="test_mv",
        dataset={"source": "json", "path": "dummy.json", "fields": {"input": "question", "target": "answer"}},
        task={"type": "generation"},
        scoring={"metric": "math_verify"},
        optimization={"metric": "math_verify", "retention": 0.99},
    )
    bench_mv = NativeBenchmark(spec_mv)
    with patch("prunequest.benchmarks.spec.MATH_VERIFY_AVAILABLE", False):
        with pytest.raises(BenchmarkError, match="Scorer 'math_verify' requires the 'math_verify' library"):
            bench_mv._score("42", "42")


