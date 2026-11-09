"""
Metric specifications and multi-benchmark retention constraints.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from prunequest.benchmarks.base import EvaluationResult


@dataclass
class MetricSpec:
    """
    Directional metric specification.
    Handles higher-is-better (accuracy, exact match) and lower-is-better (perplexity, loss).
    """

    name: str = "accuracy"
    higher_is_better: bool = True
    retention: float = 0.99
    allowed_degradation: Optional[float] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None

    def check(
        self,
        candidate_val: float,
        baseline_val: float,
    ) -> Tuple[bool, str]:
        """
        Evaluate if candidate metric value satisfies retention constraints against baseline.
        """
        if self.higher_is_better:
            # For higher-is-better (e.g. accuracy: candidate >= baseline * retention)
            if self.min_threshold is not None and candidate_val < self.min_threshold:
                return False, f"{self.name} {candidate_val:.4f} < min threshold {self.min_threshold:.4f}"

            required = baseline_val * self.retention
            if candidate_val < required:
                ret_pct = (candidate_val / baseline_val * 100.0) if baseline_val > 0 else 0.0
                return False, (
                    f"{self.name} {candidate_val*100:.2f}% < required {required*100:.2f}% "
                    f"({ret_pct:.2f}% retained, minimum {self.retention*100:.1f}%)"
                )
            return True, f"{self.name} passed retention ({candidate_val*100:.2f}% >= {required*100:.2f}%)"
        else:
            # For lower-is-better (e.g. perplexity: candidate <= baseline * (1 + allowed_degradation))
            if self.max_threshold is not None and candidate_val > self.max_threshold:
                return False, f"{self.name} {candidate_val:.4f} > max threshold {self.max_threshold:.4f}"

            deg = self.allowed_degradation if self.allowed_degradation is not None else (1.0 - self.retention)
            max_allowed = baseline_val * (1.0 + deg)
            if candidate_val > max_allowed:
                return False, (
                    f"{self.name} {candidate_val:.4f} > max allowed {max_allowed:.4f} "
                    f"(allowed degradation +{deg*100:.1f}%)"
                )
            return True, f"{self.name} passed degradation check ({candidate_val:.4f} <= {max_allowed:.4f})"


class ConstraintSuite:
    """
    Multi-benchmark constraint evaluation engine.
    Candidate models must satisfy ALL individual benchmark retention constraints.
    """

    def __init__(self, constraints: Optional[Dict[str, MetricSpec]] = None):
        self.constraints: Dict[str, MetricSpec] = constraints or {}

    def add_constraint(self, benchmark_name: str, metric_spec: MetricSpec):
        self.constraints[benchmark_name] = metric_spec

    def check_all(
        self,
        candidate_results: Dict[str, EvaluationResult],
        baseline_results: Dict[str, EvaluationResult],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify candidate against baseline across all configured benchmarks.
        Returns: (all_passed, detailed_report).
        """
        all_passed = True
        details = {}

        for bench_name, metric_spec in self.constraints.items():
            if bench_name not in candidate_results or bench_name not in baseline_results:
                details[bench_name] = {
                    "passed": False,
                    "reason": f"Missing evaluation results for benchmark '{bench_name}'",
                }
                all_passed = False
                continue

            cand_res = candidate_results[bench_name]
            base_res = baseline_results[bench_name]

            passed, reason = metric_spec.check(cand_res.accuracy, base_res.accuracy)
            details[bench_name] = {
                "passed": passed,
                "reason": reason,
                "candidate_score": cand_res.accuracy,
                "baseline_score": base_res.accuracy,
                "retention_pct": (cand_res.accuracy / base_res.accuracy * 100.0) if base_res.accuracy > 0 else 0.0,
            }

            if not passed:
                all_passed = False

        return all_passed, details
