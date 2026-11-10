"""
Incremental FLAP-Based Structured Width / Channel Pruning Framework.
"""

__version__ = "0.1.0"

from prunequest.flap.config import FLAPConfig
from prunequest.flap.core.optimizer import FLAPOptimizer
from prunequest.flap.metrics import FLAPMetricCalculator
from prunequest.flap.standardization import ScoreStandardizer
from prunequest.flap.allocation import PruningAllocator, AllocationPlan
from prunequest.flap.compensation import BiasCompensator

__all__ = [
    "FLAPConfig",
    "FLAPOptimizer",
    "FLAPMetricCalculator",
    "ScoreStandardizer",
    "PruningAllocator",
    "AllocationPlan",
    "BiasCompensator",
]
