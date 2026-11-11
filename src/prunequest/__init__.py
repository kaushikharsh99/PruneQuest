"""
PruneQuest: Autonomous, Benchmark-Guided LLM Structural Pruning Framework.
- optimize: Universal one-command model optimization.
- FLAP: Width / Channel Pruning via FLAP (Fluctuation-Based Adaptive Pruning).
- Block: Depth / Layer Pruning (Autonomous Search Controller).
- JIPO: Joint Incremental Pruning Optimizer (Width + Depth).
"""

__version__ = "0.3.0"

from prunequest.api import PruneQuest, PruneQuestResult, optimize
from prunequest.orchestrator.optimizer import OptimizationResult
from prunequest.config import PruneQuestConfig
from prunequest.flap.config import FLAPConfig
from prunequest.flap.core.optimizer import FLAPOptimizer
from prunequest.events import OptimizationEvent, EventBus, EventCategory, EventPhase
from prunequest.runs import RunManager, RunSummary, RunDetails, RunComparison
from prunequest.reporting import UnifiedReportGenerator

__all__ = [
    "optimize",
    "OptimizationResult",
    "PruneQuest",
    "PruneQuestResult",
    "PruneQuestConfig",
    "FLAPConfig",
    "FLAPOptimizer",
    "OptimizationEvent",
    "EventBus",
    "EventCategory",
    "EventPhase",
    "RunManager",
    "RunSummary",
    "RunDetails",
    "RunComparison",
    "UnifiedReportGenerator",
]
