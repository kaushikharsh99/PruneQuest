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
