"""
Benchmark and Dataset inspection API routes for PruneQuest Studio.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from prunequest.datasets import DatasetInspector
from prunequest.studio.schemas import BenchmarkInfo

router = APIRouter(prefix="/api/v1", tags=["benchmarks"])

DEFAULT_BENCHMARKS: List[BenchmarkInfo] = [
    BenchmarkInfo(
        name="gsm8k",
        description="Grade School Math 8K - multi-step mathematical reasoning with strict exact match",
        category="math",
        recommended=True,
    ),
    BenchmarkInfo(
        name="arc_challenge",
        description="AI2 Reasoning Challenge (Challenge Set) - multi-choice science questions",
        category="reasoning",
        recommended=True,
    ),
    BenchmarkInfo(
        name="mmlu",
        description="Massive Multitask Language Understanding across 57 academic subjects",
        category="general_knowledge",
        recommended=False,
    ),
    BenchmarkInfo(
        name="hellaswag",
        description="Challenging sentence completion testing grounded commonsense reasoning",
        category="commonsense",
        recommended=False,
    ),
    BenchmarkInfo(
        name="winogrande",
        description="Commonsense reasoning pronoun resolution benchmark",
        category="commonsense",
        recommended=False,
    ),
]


class DatasetActionRequest(BaseModel):
    source: str
    input_column: Optional[str] = None
    target_column: Optional[str] = None
    split: Optional[str] = None
    limit: int = 3


@router.get("/benchmarks", response_model=List[BenchmarkInfo])
def get_benchmarks():
    """Returns supported native benchmarks."""
    return DEFAULT_BENCHMARKS


@router.post("/datasets/inspect")
def inspect_dataset(req: DatasetActionRequest):
    """Inspects custom dataset schema, detected columns, and inferred task type."""
    try:
        report = DatasetInspector.inspect(
            source=req.source,
            explicit_input=req.input_column,
            explicit_target=req.target_column,
            split=req.split,
        )
        return {
            "source": report.source,
            "source_type": report.source_type,
            "detected_task": report.task_type,
            "schema_info": {
                "columns": report.columns,
                "num_records": report.total_samples or 0,
                "available_splits": report.detected_splits,
            },
            "field_mapping": {
                "input_column": report.input_column,
                "target_column": report.target_column,
                "confidence": report.mapping_confidence,
            },
            "sample_preview": report.sample_preview,
            "warnings": report.warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/datasets/preview")
def preview_dataset(req: DatasetActionRequest):
    """Previews formatted prompt and target completion examples."""
    try:
        samples = DatasetInspector.preview(
            source=req.source,
            explicit_input=req.input_column,
            explicit_target=req.target_column,
            limit=req.limit,
        )
        return {"samples": samples}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/datasets/validate")
def validate_dataset(req: DatasetActionRequest):
    """Validates split partitioning, schema cleanliness, and data leakage absence."""
    try:
        report = DatasetInspector.validate(
            source=req.source,
            explicit_input=req.input_column,
            explicit_target=req.target_column,
        )
        return {
            "valid": report.is_valid,
            "errors": report.errors,
            "warnings": report.warnings,
            "summary": {
                "partition_counts": report.partition_counts,
                "task_type": report.task_type,
                "recommended_scorer": report.scorer,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
