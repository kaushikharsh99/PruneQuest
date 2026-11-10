"""
PruneQuest: Autonomous, Benchmark-Guided LLM Structural Pruning Framework.
- FLAP: Width / Channel Pruning via FLAP (Fluctuation-Based Adaptive Pruning).
- Block: Depth / Layer Pruning (Autonomous Search Controller).
"""

__version__ = "0.3.0"

from prunequest.api import PruneQuest, PruneQuestResult
from prunequest.config import PruneQuestConfig
from prunequest.flap.config import FLAPConfig
from prunequest.flap.core.optimizer import FLAPOptimizer

__all__ = [
    "PruneQuest",
    "PruneQuestResult",
    "PruneQuestConfig",
    "FLAPConfig",
    "FLAPOptimizer",
]
