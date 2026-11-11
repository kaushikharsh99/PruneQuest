"""
Data models and contracts for PruneQuest Universal Dataset Onboarding.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DatasetSourceType(str, Enum):
    LOCAL_FILE = "local_file"
    HUGGINGFACE = "huggingface"
    BENCHMARK_SPEC = "benchmark_spec"
    REGISTERED_BENCHMARK = "registered_benchmark"
    LM_EVAL = "lm_eval"


class TaskType(str, Enum):
    GENERATION = "generation"
    MULTIPLE_CHOICE = "multiple_choice"
    EXACT_ANSWER = "exact_answer"
    MATH_EQUIVALENCE = "math_equivalence"
    PERPLEXITY = "perplexity"
    UNSUPPORTED = "unsupported"


class FieldMappingConfidence(str, Enum):
    HIGH = "high"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


@dataclass
class ResolvedDatasetSource:
    source_type: DatasetSourceType
    path_or_id: str
    is_local: bool
    file_format: Optional[str] = None
    config_name: Optional[str] = None
    revision: Optional[str] = "main"
    split: Optional[str] = None


@dataclass
class DatasetFingerprint:
    source_hash: str
    source_path_or_id: str
    total_samples: int
    partition_counts: Dict[str, int]
    partition_hashes: Dict[str, str]
    seed: int
    input_column: str
    target_column: str
    task_type: str
    scorer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_hash": self.source_hash,
            "source_path_or_id": self.source_path_or_id,
            "total_samples": self.total_samples,
            "partition_counts": self.partition_counts,
            "partition_hashes": self.partition_hashes,
            "seed": self.seed,
            "input_column": self.input_column,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "scorer": self.scorer,
        }


@dataclass
class DatasetSchemaInfo:
    columns: List[str]
    column_types: Dict[str, str]
    sample_count: Optional[int]
    splits: List[str] = field(default_factory=list)
    input_candidates: List[str] = field(default_factory=list)
    target_candidates: List[str] = field(default_factory=list)
    is_nested_chat: bool = False
    has_choices: bool = False
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)


class DatasetInspectionReport(BaseModel):
    source: str
    source_type: str
    format: str
    total_samples: Optional[int] = None
    detected_splits: List[str] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    input_column: Optional[str] = None
    target_column: Optional[str] = None
    mapping_confidence: str
    task_type: str
    proposed_scorer: str
    higher_is_better: bool = True
    sample_preview: List[Dict[str, Any]] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DatasetValidationReport(BaseModel):
    is_valid: bool
    source: str
    task_type: str
    scorer: str
    input_column: str
    target_column: str
    partition_counts: Dict[str, int] = Field(default_factory=dict)
    fingerprint: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
