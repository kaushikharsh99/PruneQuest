"""
Search strategy abstraction for candidate selection and optimization.
"""

from abc import ABC, abstractmethod
from typing import List
from prunequest.pruning.base import PruningCandidate


class BaseSearchStrategy(ABC):
    """
    Abstract interface for pruning search strategies (Greedy, Beam, Bayesian, etc.).
    """

    @abstractmethod
    def select_candidates(
        self,
        available_candidates: List[PruningCandidate],
        budget: int = 3,
    ) -> List[PruningCandidate]:
        """Rank and select candidates for benchmarking."""
        pass


class GreedySearchStrategy(BaseSearchStrategy):
    """
    Greedy search: selects top candidates ranked by importance score.
    """

    def select_candidates(
        self,
        available_candidates: List[PruningCandidate],
        budget: int = 3,
    ) -> List[PruningCandidate]:
        # Sort ascending by importance score (lowest BI score = highest redundancy)
        sorted_cands = sorted(available_candidates, key=lambda c: c.importance_score)
        return sorted_cands[:budget]
