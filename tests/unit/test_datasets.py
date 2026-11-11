"""
Unit tests for Universal Dataset Onboarding (UX Release C).
"""

import json
import tempfile
import pandas as pd
import pytest

from prunequest.exceptions import BenchmarkError
from prunequest.datasets import (
    DatasetSourceResolver,
    DatasetSourceType,
    FieldDetector,
    TaskTypeDetector,
    TaskType,
    FieldMappingConfidence,
    ScorerRegistry,
    DatasetPartitioner,
    BenchmarkSpecBuilder,
    DatasetInspector,
)


def test_dataset_source_resolver_local_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        # JSONL
        jsonl_path = f"{tmpdir}/data.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"question": "What is 1+1?", "answer": "2"}\n')
            f.write('{"question": "What is 2+2?", "answer": "4"}\n')

        res_jsonl = DatasetSourceResolver.resolve(jsonl_path)
        assert res_jsonl.source_type == DatasetSourceType.LOCAL_FILE
        assert res_jsonl.file_format == "jsonl"
        assert res_jsonl.is_local is True

        records, total, splits = DatasetSourceResolver.read_preview_records(res_jsonl, limit=10)
        assert len(records) == 2
        assert total == 2

        # CSV
        csv_path = f"{tmpdir}/data.csv"
        df = pd.DataFrame([{"prompt": "Hello", "completion": "World"}, {"prompt": "Foo", "completion": "Bar"}])
        df.to_csv(csv_path, index=False)

        res_csv = DatasetSourceResolver.resolve(csv_path)
        assert res_csv.source_type == DatasetSourceType.LOCAL_FILE
        assert res_csv.file_format == "csv"

        records_csv, total_csv, _ = DatasetSourceResolver.read_preview_records(res_csv, limit=10)
        assert len(records_csv) == 2

        # Parquet
        parquet_path = f"{tmpdir}/data.parquet"
        df.to_parquet(parquet_path, index=False)
        res_parquet = DatasetSourceResolver.resolve(parquet_path)
        assert res_parquet.file_format == "parquet"

        # Missing file error
        with pytest.raises(BenchmarkError, match="Local dataset file not found"):
            DatasetSourceResolver.resolve(f"{tmpdir}/non_existent.jsonl")


def test_field_detector_confidence_and_overrides():
    # 1. High confidence: unique question & answer
    rows_high = [{"question": "Q1", "answer": "A1"}]
    schema, inp, tgt, conf, ambs = FieldDetector.analyze_schema(rows_high)
    assert inp == "question"
    assert tgt == "answer"
    assert conf == FieldMappingConfidence.HIGH
    assert len(ambs) == 0

    # 2. Ambiguous: multiple input candidates
    rows_amb = [{"instruction": "Do X", "input": "Input Y", "output": "Z"}]
    schema, inp, tgt, conf, ambs = FieldDetector.analyze_schema(rows_amb)
    assert conf == FieldMappingConfidence.AMBIGUOUS
    assert inp in ("instruction", "input")
    assert tgt == "output"
    assert len(ambs) > 0

    # 3. Explicit override resolves ambiguity
    schema, inp, tgt, conf, ambs = FieldDetector.analyze_schema(
        rows_amb, explicit_input="input", explicit_target="output"
    )
    assert inp == "input"
    assert tgt == "output"
    assert conf == FieldMappingConfidence.HIGH

    # 4. Invalid explicit column
    schema, inp, tgt, conf, ambs = FieldDetector.analyze_schema(
        rows_amb, explicit_input="non_existent", explicit_target="output"
    )
    assert conf == FieldMappingConfidence.UNSUPPORTED


def test_task_type_detection():
    # 1. Multiple choice via choices list
    rows_mc = [{"question": "Pick one", "choices": ["Apple", "Banana"], "answer": "A"}]
    schema, _, _, _, _ = FieldDetector.analyze_schema(rows_mc)
    task, reason = TaskTypeDetector.detect(schema, "question", "answer")
    assert task == TaskType.MULTIPLE_CHOICE

    # 2. Mathematical equivalence via \boxed
    rows_math = [{"question": "Solve for x", "answer": r"\boxed{42}"}]
    schema, _, _, _, _ = FieldDetector.analyze_schema(rows_math)
    task, reason = TaskTypeDetector.detect(schema, "question", "answer")
    assert task == TaskType.MATH_EQUIVALENCE

    # 3. Unsupported preference pair
    rows_pref = [{"prompt": "Write story", "chosen": "Good story", "rejected": "Bad story"}]
    schema, _, _, _, _ = FieldDetector.analyze_schema(rows_pref)
    task, reason = TaskTypeDetector.detect(schema, "prompt", "chosen")
    assert task == TaskType.UNSUPPORTED


def test_scorer_registry_and_direction():
    # Math verify resolution
    scorer, higher, warns = ScorerRegistry.resolve_scorer(TaskType.MATH_EQUIVALENCE)
    assert higher is True
    assert scorer in ("math_verify", "numeric_exact_match")

    # Multiple choice
    scorer_mc, higher_mc, _ = ScorerRegistry.resolve_scorer(TaskType.MULTIPLE_CHOICE)
    assert scorer_mc == "multiple_choice"
    assert higher_mc is True

    # Retention calculation: higher is better
    ret_acc = ScorerRegistry.compute_retention_pct(baseline=0.90, final=0.891, higher_is_better=True)
    assert abs(ret_acc - 99.0) < 1e-4

    # Retention calculation: lower is better (perplexity)
    # Baseline perplexity = 10.0, final = 10.1 -> retention = 10.0 / 10.1 = 99.01%
    ret_ppl = ScorerRegistry.compute_retention_pct(baseline=10.0, final=10.1, higher_is_better=False)
    assert abs(ret_ppl - (10.0 / 10.1 * 100.0)) < 1e-4


def test_dataset_partitioner_disjoint_and_leakage_prevention():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
        for i in range(100):
            f.write(json.dumps({"question": f"Question {i}", "answer": f"Answer {i}"}) + "\n")
        f.flush()
        resolved = DatasetSourceResolver.resolve(f.name)

        records = [{"question": f"Question {i}", "answer": f"Answer {i}"} for i in range(100)]

        # Partition into 20 calib + 10 screen + 30 dev = 60
        partitions, fp = DatasetPartitioner.partition_records(
            resolved_source=resolved,
            raw_records=records,
            input_col="question",
            target_col="answer",
            task_type="generation",
            scorer="normalized_match",
            calibration_size=20,
            screening_size=10,
            dev_size=30,
            seed=42,
        )

    calib_ids = {s.id for s in partitions["calibration"]}
    screen_ids = {s.id for s in partitions["screening"]}
    dev_ids = {s.id for s in partitions["dev"]}
    test_ids = {s.id for s in partitions["final_test"]}

    assert len(calib_ids) == 20
    assert len(screen_ids) == 10
    assert len(dev_ids) == 30
    assert len(test_ids) == 40

    # Strict zero-leakage assertions
    assert calib_ids.isdisjoint(screen_ids)
    assert calib_ids.isdisjoint(dev_ids)
    assert screen_ids.isdisjoint(dev_ids)
    assert test_ids.isdisjoint(calib_ids)
    assert test_ids.isdisjoint(dev_ids)

    # Insufficient data error
    with pytest.raises(BenchmarkError, match="Insufficient valid unique samples"):
        DatasetPartitioner.partition_records(
            resolved_source=resolved,
            raw_records=records[:25],
            input_col="question",
            target_col="answer",
            task_type="generation",
            scorer="normalized_match",
            calibration_size=20,
            screening_size=10,
            dev_size=30,
        )


def test_spec_builder_and_inspector():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = f"{tmpdir}/math_bench.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(50):
                f.write(json.dumps({"problem": f"What is {i}+{i}?", "solution": f"#### {2*i}"}) + "\n")

        # 1. Dataset Inspector inspect
        report = DatasetInspector.inspect(jsonl_path)
        assert report.input_column == "problem"
        assert report.target_column == "solution"
        assert report.task_type == "math_equivalence"
        assert len(report.sample_preview) > 0

        # 2. Dataset Inspector preview
        previews = DatasetInspector.preview(jsonl_path, limit=2)
        assert len(previews) == 2
        assert "What is 0+0?" in previews[0]["rendered_prompt"]

        # 3. Dataset Inspector validate
        val_report = DatasetInspector.validate(
            jsonl_path,
            calibration_size=5,
            screening_size=5,
            dev_size=10,
        )
        assert val_report.is_valid is True
        assert val_report.fingerprint is not None

        # 4. Spec Builder synthesis
        spec = BenchmarkSpecBuilder.build(
            name="math_bench",
            source="jsonl",
            path=jsonl_path,
            input_col=report.input_column,
            target_col=report.target_column,
            task_type=report.task_type,
            scorer=report.proposed_scorer,
            retention=0.98,
        )
        spec_path = f"{tmpdir}/resolved_spec.yaml"
        BenchmarkSpecBuilder.save_spec(spec, spec_path)
        assert spec.name == "math_bench"
        assert spec.scoring.metric == report.proposed_scorer
