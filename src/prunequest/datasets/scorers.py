"""
Scorer Registry and Compatibility Rules for PruneQuest Universal Dataset Onboarding.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from prunequest.exceptions import BenchmarkError
from prunequest.datasets.models import TaskType

try:
    import math_verify  # noqa: F401
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


@dataclass
class ScorerMetadata:
    name: str
    task_compatibility: List[TaskType]
    higher_is_better: bool
    is_bounded_0_1: bool
    dependencies: List[str]
    description: str


REGISTRY: Dict[str, ScorerMetadata] = {
    "exact_match": ScorerMetadata(
        name="exact_match",
        task_compatibility=[TaskType.EXACT_ANSWER, TaskType.GENERATION, TaskType.MULTIPLE_CHOICE],
        higher_is_better=True,
        is_bounded_0_1=True,
        dependencies=[],
        description="Strict string equality after stripping leading and trailing whitespace.",
    ),
    "normalized_match": ScorerMetadata(
        name="normalized_match",
        task_compatibility=[TaskType.EXACT_ANSWER, TaskType.GENERATION, TaskType.MULTIPLE_CHOICE],
        higher_is_better=True,
        is_bounded_0_1=True,
        dependencies=[],
        description="Case-insensitive equality after whitespace and punctuation normalization.",
    ),
    "numeric_exact_match": ScorerMetadata(
        name="numeric_exact_match",
        task_compatibility=[TaskType.MATH_EQUIVALENCE, TaskType.EXACT_ANSWER],
        higher_is_better=True,
        is_bounded_0_1=True,
        dependencies=[],
        description="Extracts the final numeric value and compares using relative tolerance (1e-5).",
    ),
    "math_verify": ScorerMetadata(
        name="math_verify",
        task_compatibility=[TaskType.MATH_EQUIVALENCE],
        higher_is_better=True,
        is_bounded_0_1=True,
        dependencies=["math_verify"],
        description="Canonical mathematical equivalence verification for LaTeX equations and numbers.",
    ),
    "multiple_choice": ScorerMetadata(
        name="multiple_choice",
        task_compatibility=[TaskType.MULTIPLE_CHOICE],
        higher_is_better=True,
        is_bounded_0_1=True,
        dependencies=[],
        description="Extracts multiple-choice option letter (A, B, C, D) or index.",
    ),
    "regex": ScorerMetadata(
        name="regex",
        task_compatibility=[TaskType.EXACT_ANSWER, TaskType.GENERATION],
        higher_is_better=True,
        is_bounded_0_1=True,
        dependencies=[],
        description="Extracts answer candidate using a specified regular expression pattern.",
    ),
    "perplexity": ScorerMetadata(
        name="perplexity",
        task_compatibility=[TaskType.PERPLEXITY],
        higher_is_better=False,
        is_bounded_0_1=False,
        dependencies=[],
        description="Exponential of causal cross-entropy loss over evaluation tokens (lower is better).",
    ),
}


class ScorerRegistry:
    """
    Manages verified evaluation scorers and enforces task-metric compatibility.
    """

    @classmethod
    def get_scorer_info(cls, metric_name: str) -> Optional[ScorerMetadata]:
        return REGISTRY.get(metric_name.lower())

    @classmethod
    def list_scorers(cls) -> List[ScorerMetadata]:
        return list(REGISTRY.values())

    @classmethod
    def resolve_scorer(
        cls,
        task_type: TaskType,
        explicit_metric: Optional[str] = None,
    ) -> Tuple[str, bool, List[str]]:
        """
        Selects an appropriate scorer and validates compatibility.
        Returns: (metric_name, higher_is_better, warnings)
        """
        warnings = []

        if explicit_metric:
            norm_metric = explicit_metric.lower().strip()
            if norm_metric not in REGISTRY:
                valid = list(REGISTRY.keys())
                raise BenchmarkError(
                    f"Unsupported evaluation metric '{explicit_metric}'. "
                    f"Supported metrics: {', '.join(valid)}"
                )

            meta = REGISTRY[norm_metric]
            # Check dependencies
            if "math_verify" in meta.dependencies and not MATH_VERIFY_AVAILABLE:
                raise BenchmarkError(
                    f"Metric '{norm_metric}' requires the 'math_verify' library. "
                    f"Install it via 'pip install math-verify' or specify an alternative metric with --metric "
                    f"(e.g., --metric numeric_exact_match, --metric exact_match)."
                )

            # Check task compatibility
            if task_type not in meta.task_compatibility and task_type != TaskType.UNSUPPORTED:
                warnings.append(
                    f"Metric '{norm_metric}' is typically not recommended for task type '{task_type.value}'. "
                    f"Supported task types for {norm_metric}: {[t.value for t in meta.task_compatibility]}."
                )

            return meta.name, meta.higher_is_better, warnings

        # Automatic scorer selection based on task type
        if task_type == TaskType.MATH_EQUIVALENCE:
            if MATH_VERIFY_AVAILABLE:
                return "math_verify", True, []
            else:
                warnings.append(
                    "Detected mathematical task but 'math_verify' is not installed. "
                    "Selected 'numeric_exact_match' as verified fallback. "
                    "Install 'math_verify' via 'pip install math-verify' for full LaTeX equivalence."
                )
                return "numeric_exact_match", True, warnings
        elif task_type == TaskType.MULTIPLE_CHOICE:
            return "multiple_choice", True, []
        elif task_type == TaskType.EXACT_ANSWER:
            return "normalized_match", True, []
        elif task_type == TaskType.PERPLEXITY:
            return "perplexity", False, []
        else:  # GENERATION
            return "normalized_match", True, []

    @classmethod
    def compute_retention_pct(
        cls,
        baseline: float,
        final: float,
        higher_is_better: bool = True,
    ) -> float:
        """
        Computes accurate retention percentage respecting metric direction.
        """
        if baseline == 0.0:
            return 100.0 if final == 0.0 else 0.0

        if higher_is_better:
            return (final / baseline) * 100.0
        else:
            # Lower is better (e.g. perplexity: baseline 10.0, final 10.2 -> retention = 10.0 / 10.2 = 98.0%)
            if final == 0.0:
                return 0.0
            return (baseline / final) * 100.0
