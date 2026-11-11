"""
Dataset Inspection, Preview & Validation Engine for PruneQuest.
Powering CLI 'prunequest datasets' and high-level Python API.
"""

from typing import Optional, Dict, Any, List
from prunequest.exceptions import BenchmarkError
from prunequest.datasets.models import (
    DatasetInspectionReport,
    DatasetValidationReport,
)
from prunequest.datasets.resolver import DatasetSourceResolver
from prunequest.datasets.schema import FieldDetector, TaskTypeDetector
from prunequest.datasets.scorers import ScorerRegistry
from prunequest.datasets.preview import DatasetPreviewer
from prunequest.datasets.partitioning import DatasetPartitioner


class DatasetInspector:
    """
    Unified inspector for discovery, field detection, schema preview, and safety validation.
    """

    @classmethod
    def inspect(
        cls,
        source: str,
        config: Optional[str] = None,
        revision: Optional[str] = "main",
        split: Optional[str] = None,
        explicit_input: Optional[str] = None,
        explicit_target: Optional[str] = None,
        explicit_metric: Optional[str] = None,
    ) -> DatasetInspectionReport:
        """
        Lightweight inspection of a dataset source without loading model weights.
        """
        resolved = DatasetSourceResolver.resolve(source, config=config, revision=revision, split=split)
        sample_rows, total_samples, detected_splits = DatasetSourceResolver.read_preview_records(resolved, limit=50)

        schema_info, inp_col, tgt_col, confidence, ambiguities = FieldDetector.analyze_schema(
            sample_rows,
            explicit_input=explicit_input,
            explicit_target=explicit_target,
        )

        task_type, task_reason = TaskTypeDetector.detect(schema_info, inp_col, tgt_col)
        scorer_name, higher_is_better, scorer_warnings = ScorerRegistry.resolve_scorer(task_type, explicit_metric)

        previews = []
        for idx, row in enumerate(sample_rows[:3]):
            val_in = str(row.get(inp_col, "")) if inp_col else ""
            val_tgt = str(row.get(tgt_col, "")) if tgt_col else ""
            choices = row.get("choices") or row.get("options")
            p_item = DatasetPreviewer.format_preview_item(
                sample_id=str(row.get("id", idx)),
                input_text=val_in,
                target_text=val_tgt,
                task_type=task_type.value,
                scorer=scorer_name,
                choices=choices if isinstance(choices, list) else None,
            )
            previews.append(p_item)

        warnings = list(scorer_warnings)
        if ambiguities:
            warnings.extend(ambiguities)

        fmt_name = resolved.file_format or resolved.source_type.value

        return DatasetInspectionReport(
            source=source,
            source_type=resolved.source_type.value,
            format=fmt_name,
            total_samples=total_samples,
            detected_splits=detected_splits,
            columns=schema_info.columns,
            input_column=inp_col,
            target_column=tgt_col,
            mapping_confidence=confidence.value,
            task_type=task_type.value,
            proposed_scorer=scorer_name,
            higher_is_better=higher_is_better,
            sample_preview=previews,
            ambiguities=ambiguities,
            warnings=warnings,
        )

    @classmethod
    def preview(
        cls,
        source: str,
        limit: int = 3,
        explicit_input: Optional[str] = None,
        explicit_target: Optional[str] = None,
        prompt_template: str = "{{ input }}",
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates rendered prompt preview cards for the dataset.
        """
        inspection = cls.inspect(
            source,
            explicit_input=explicit_input,
            explicit_target=explicit_target,
        )
        resolved = DatasetSourceResolver.resolve(source)
        sample_rows, _, _ = DatasetSourceResolver.read_preview_records(resolved, limit=limit)

        inp_col = inspection.input_column
        tgt_col = inspection.target_column

        if not inp_col or not tgt_col:
            raise BenchmarkError(
                f"Cannot preview prompts for '{source}' because input/target columns could not be resolved. "
                f"Please specify --input-column and --target-column."
            )

        previews = []
        for idx, row in enumerate(sample_rows[:limit]):
            val_in = str(row.get(inp_col, ""))
            val_tgt = str(row.get(tgt_col, ""))
            choices = row.get("choices") or row.get("options")
            item = DatasetPreviewer.format_preview_item(
                sample_id=str(row.get("id", idx)),
                input_text=val_in,
                target_text=val_tgt,
                task_type=inspection.task_type,
                scorer=inspection.proposed_scorer,
                template=prompt_template,
                system_prompt=system_prompt,
                choices=choices if isinstance(choices, list) else None,
            )
            previews.append(item)
        return previews

    @classmethod
    def validate(
        cls,
        source: str,
        explicit_input: Optional[str] = None,
        explicit_target: Optional[str] = None,
        explicit_metric: Optional[str] = None,
        calibration_size: int = 128,
        screening_size: int = 64,
        dev_size: int = 512,
        seed: int = 42,
    ) -> DatasetValidationReport:
        """
        Validates schema, fields, scorer, and partition integrity.
        """
        errors = []
        warnings = []
        fingerprint_dict = None
        partition_counts = {}

        try:
            resolved = DatasetSourceResolver.resolve(source)
        except Exception as e:
            return DatasetValidationReport(
                is_valid=False,
                source=source,
                task_type="unknown",
                scorer="unknown",
                input_column=explicit_input or "unknown",
                target_column=explicit_target or "unknown",
                errors=[str(e)],
            )

        # Read samples
        try:
            sample_rows, total_count, detected_splits = DatasetSourceResolver.read_preview_records(resolved, limit=1000)
        except Exception as e:
            return DatasetValidationReport(
                is_valid=False,
                source=source,
                task_type="unknown",
                scorer="unknown",
                input_column=explicit_input or "unknown",
                target_column=explicit_target or "unknown",
                errors=[f"Failed reading dataset records: {e}"],
            )

        schema_info, inp_col, tgt_col, confidence, ambiguities = FieldDetector.analyze_schema(
            sample_rows,
            explicit_input=explicit_input,
            explicit_target=explicit_target,
        )

        if not inp_col:
            errors.append(f"No valid input column could be identified. Available columns: {schema_info.columns}")
        if not tgt_col:
            errors.append(f"No valid target column could be identified. Available columns: {schema_info.columns}")

        if ambiguities:
            warnings.extend(ambiguities)

        task_type, task_reason = TaskTypeDetector.detect(schema_info, inp_col, tgt_col)
        if task_type.value == "unsupported":
            errors.append(task_reason)

        try:
            scorer_name, higher_is_better, scorer_warns = ScorerRegistry.resolve_scorer(task_type, explicit_metric)
            warnings.extend(scorer_warns)
        except Exception as e:
            scorer_name = explicit_metric or "unknown"
            errors.append(str(e))

        if inp_col and tgt_col and not errors:
            # Test partitioning
            try:
                # If local file, read full records for partition check
                partitions, fp = DatasetPartitioner.partition_records(
                    resolved_source=resolved,
                    raw_records=sample_rows,
                    input_col=inp_col,
                    target_col=tgt_col,
                    task_type=task_type.value,
                    scorer=scorer_name,
                    calibration_size=min(calibration_size, max(len(sample_rows) // 4, 1)),
                    screening_size=min(screening_size, max(len(sample_rows) // 8, 1)),
                    dev_size=min(dev_size, max(len(sample_rows) // 2, 1)),
                    seed=seed,
                )
                partition_counts = fp.partition_counts
                fingerprint_dict = fp.to_dict()
            except Exception as e:
                errors.append(f"Partition validation failed: {e}")

        is_valid = len(errors) == 0

        return DatasetValidationReport(
            is_valid=is_valid,
            source=source,
            task_type=task_type.value,
            scorer=scorer_name,
            input_column=inp_col or "unresolved",
            target_column=tgt_col or "unresolved",
            partition_counts=partition_counts,
            fingerprint=fingerprint_dict,
            errors=errors,
            warnings=warnings,
        )
