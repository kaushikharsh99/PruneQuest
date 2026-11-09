"""
PruneQuest: Autonomous, Two-Stage Benchmark-Guided LLM Structural Pruning Framework.
Stage 1: Width / Channel Pruning via FLAP (Fluctuation-Based Adaptive Pruning).
Stage 2: Depth / Layer Pruning (Autonomous Search Controller).
"""

__version__ = "0.3.0"

from prunequest.api import PruneQuest, PruneQuestResult
from prunequest.config import PruneQuestConfig
from prunequest.stage1.config import Stage1Config
from prunequest.stage1.core.optimizer import FLAPOptimizer

__all__ = [
    "PruneQuest",
    "PruneQuestResult",
    "PruneQuestConfig",
    "Stage1Config",
    "FLAPOptimizer",
]
