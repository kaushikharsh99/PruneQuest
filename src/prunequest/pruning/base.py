"""
Base pruner interfaces and candidate data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from prunequest.models.base import BaseModelAdapter


@dataclass
class PruningCandidate:
    candidate_id: str
    pruning_type: str
    target_info: Dict[str, Any]
    estimated_param_reduction: int
    importance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruningResult:
    candidate: PruningCandidate
    params_before: int
    params_after: int
    params_removed: int
    retention_ratio: float
    validation_passed: bool
    error_message: Optional[str] = None


class BasePruner(ABC):
    """
    Abstract interface for structural pruning engines.
    """

    @abstractmethod
    def supports(self, adapter: BaseModelAdapter) -> bool:
        """Check if this pruning backend supports the given model architecture."""
        pass

    @abstractmethod
    def analyze(self, adapter: BaseModelAdapter, **kwargs) -> Dict[str, Any]:
        """Compute structural redundancy or sensitivity scores."""
        pass

    @abstractmethod
    def propose_candidates(
        self,
        adapter: BaseModelAdapter,
        analysis_results: Dict[str, Any],
        top_k: int = 3,
    ) -> List[PruningCandidate]:
        """Propose ranked candidate operations."""
        pass

    @abstractmethod
    def apply_pruning(
        self,
        adapter: BaseModelAdapter,
        candidate: PruningCandidate,
    ) -> PruningResult:
        """Physically execute the pruning operation on the model."""
        pass

    def estimate_savings(
        self,
        adapter: BaseModelAdapter,
        candidate: PruningCandidate,
    ) -> Dict[str, Any]:
        """Estimate parameter and VRAM savings for a candidate."""
        return {
            "estimated_params_removed": candidate.estimated_param_reduction,
            "estimated_vram_mb": (candidate.estimated_param_reduction * 2) / (1024**2),
        }
