"""
Data structures for FLAP evaluation results.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    total: int
    correct: int
    accuracy: float
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    records: List[Dict[str, Any]] = field(default_factory=list)
    total_time_sec: float = 0.0
    tokens_per_sec: float = 0.0
    peak_vram_gb: float = 0.0
