"""
Unit tests for MetricSpec and multi-benchmark ConstraintSuite.
"""

from prunequest.benchmarks.constraints import MetricSpec, ConstraintSuite
from prunequest.benchmarks.base import EvaluationResult


def test_metric_spec_higher_is_better():
    spec = MetricSpec(name="accuracy", higher_is_better=True, retention=0.90)

    # 0.85 >= 0.90 * 0.90 (0.81) -> passes
    passed, msg = spec.check(candidate_val=0.85, baseline_val=0.90)
    assert passed is True

    # 0.75 < 0.90 * 0.90 (0.81) -> fails
    passed, msg = spec.check(candidate_val=0.75, baseline_val=0.90)
    assert passed is False
    assert "retention" in msg.lower() or "required" in msg.lower()


def test_metric_spec_lower_is_better():
    # Perplexity: baseline 10.0, allowed degradation 10% -> max 11.0
    spec = MetricSpec(name="perplexity", higher_is_better=False, allowed_degradation=0.10)

    passed, msg = spec.check(candidate_val=10.5, baseline_val=10.0)
    assert passed is True

    passed, msg = spec.check(candidate_val=11.5, baseline_val=10.0)
    assert passed is False
    assert "max allowed" in msg.lower()


def test_multi_benchmark_constraint_suite():
    suite = ConstraintSuite()
    suite.add_constraint("gsm8k", MetricSpec(name="gsm8k", higher_is_better=True, retention=0.95))
    suite.add_constraint("arc", MetricSpec(name="arc", higher_is_better=True, retention=0.90))

    base_results = {
        "gsm8k": EvaluationResult(total=100, correct=80, accuracy=0.80),
        "arc": EvaluationResult(total=100, correct=70, accuracy=0.70),
    }

    # Candidate satisfies both (gsm8k: 0.78 >= 0.76; arc: 0.65 >= 0.63)
    cand_pass = {
        "gsm8k": EvaluationResult(total=100, correct=78, accuracy=0.78),
        "arc": EvaluationResult(total=100, correct=65, accuracy=0.65),
    }
    all_passed, details = suite.check_all(cand_pass, base_results)
    assert all_passed is True

    # Candidate fails arc (0.55 < 0.63)
    cand_fail = {
        "gsm8k": EvaluationResult(total=100, correct=78, accuracy=0.78),
        "arc": EvaluationResult(total=100, correct=55, accuracy=0.55),
    }
    all_passed, details = suite.check_all(cand_fail, base_results)
    assert all_passed is False
    assert details["gsm8k"]["passed"] is True
    assert details["arc"]["passed"] is False
