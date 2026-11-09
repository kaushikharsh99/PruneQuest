"""
Base benchmark interface and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class BenchmarkSample:
    id: str
    question: str
    gold_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    total: int
    correct: int
    accuracy: float
    records: List[Dict[str, Any]]
    total_tokens: int
    total_time_sec: float
    tokens_per_sec: float
    peak_vram_gb: float


class BaseBenchmark(ABC):
    """
    Abstract interface for benchmark adapters.
    Declares metric semantics, split policies, prompting, and evaluation routines.
    """

    metric_name: str = "accuracy"
    metric_direction: str = "higher"  # 'higher' or 'lower'
    requires_generation: bool = True

    def __init__(self, seed: int = 42, max_new_tokens: int = 512):
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.data_splits: Dict[str, List[BenchmarkSample]] = {}

    @abstractmethod
    def load_data(self) -> Dict[str, List[BenchmarkSample]]:
        """Download or load dataset and construct deterministic partitions."""
        pass

    @abstractmethod
    def format_prompt(self, tokenizer: PreTrainedTokenizer, sample: BenchmarkSample) -> str:
        """Construct model-compatible inference prompt."""
        pass

    @abstractmethod
    def verify_answer(self, prediction_text: str, gold_str: str) -> Tuple[bool, str, str]:
        """
        Verify prediction against gold answer.
        Returns: (is_correct, parsed_pred, cleaned_gold).
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        split_name: str = "smoke",
        max_samples: Optional[int] = None,
        device: str = "cuda:0",
        progress_callback: Optional[Any] = None,
    ) -> EvaluationResult:
        """Run inference evaluation over the selected benchmark split."""
        pass
