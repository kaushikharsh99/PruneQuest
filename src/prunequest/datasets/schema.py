"""
Automatic Field & Task Type Detection for PruneQuest Universal Dataset Onboarding.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from prunequest.datasets.models import (
    TaskType,
    FieldMappingConfidence,
    DatasetSchemaInfo,
)


INPUT_CANDIDATE_NAMES = [
    "question",
    "prompt",
    "input",
    "instruction",
    "query",
    "problem",
    "text",
    "context",
]

TARGET_CANDIDATE_NAMES = [
    "answer",
    "target",
    "output",
    "response",
    "solution",
    "label",
    "completion",
    "gold",
    "ground_truth",
]


class FieldDetector:
    """
    Detects input and target column mappings with confidence classifications.
    """

    @classmethod
    def analyze_schema(
        cls,
        sample_rows: List[Dict[str, Any]],
        explicit_input: Optional[str] = None,
        explicit_target: Optional[str] = None,
    ) -> Tuple[DatasetSchemaInfo, Optional[str], Optional[str], FieldMappingConfidence, List[str]]:
        """
        Analyzes sample rows and discovers input and target fields.
        Returns: (schema_info, resolved_input, resolved_target, confidence, ambiguities)
        """
        if not sample_rows:
            return (
                DatasetSchemaInfo(columns=[], column_types={}, sample_count=0),
                None,
                None,
                FieldMappingConfidence.UNSUPPORTED,
                ["Dataset is empty; no fields found."],
            )

        first_row = sample_rows[0]
        columns = list(first_row.keys())
        column_types = {k: type(v).__name__ for k, v in first_row.items()}

        # Check for nested chat structure
        is_nested_chat = False
        if "messages" in columns and isinstance(first_row.get("messages"), list):
            is_nested_chat = True
        elif "conversations" in columns and isinstance(first_row.get("conversations"), list):
            is_nested_chat = True

        # Check for choices
        has_choices = "choices" in columns or "options" in columns

        # Find candidates
        input_cands = [c for c in columns if c.lower() in INPUT_CANDIDATE_NAMES]
        target_cands = [c for c in columns if c.lower() in TARGET_CANDIDATE_NAMES]

        schema_info = DatasetSchemaInfo(
            columns=columns,
            column_types=column_types,
            sample_count=len(sample_rows),
            input_candidates=input_cands,
            target_candidates=target_cands,
            is_nested_chat=is_nested_chat,
            has_choices=has_choices,
            sample_rows=sample_rows,
        )

        ambiguities = []

        # 1. Explicit user overrides take absolute precedence
        if explicit_input and explicit_target:
            if explicit_input not in columns:
                return (
                    schema_info,
                    explicit_input,
                    explicit_target,
                    FieldMappingConfidence.UNSUPPORTED,
                    [f"Specified input column '{explicit_input}' not found in columns: {columns}"],
                )
            if explicit_target not in columns:
                return (
                    schema_info,
                    explicit_input,
                    explicit_target,
                    FieldMappingConfidence.UNSUPPORTED,
                    [f"Specified target column '{explicit_target}' not found in columns: {columns}"],
                )
            return schema_info, explicit_input, explicit_target, FieldMappingConfidence.HIGH, []

        resolved_in = explicit_input
        resolved_tgt = explicit_target

        if not resolved_in:
            if len(input_cands) == 1:
                resolved_in = input_cands[0]
            elif len(input_cands) > 1:
                # Rank candidates by priority
                for cand in INPUT_CANDIDATE_NAMES:
                    matches = [c for c in input_cands if c.lower() == cand]
                    if matches:
                        resolved_in = matches[0]
                        break
                ambiguities.append(
                    f"Multiple input column candidates detected: {input_cands}. Selected '{resolved_in}'. "
                    f"Override with --input-column if desired."
                )

        if not resolved_tgt:
            if len(target_cands) == 1:
                resolved_tgt = target_cands[0]
            elif len(target_cands) > 1:
                # Rank candidates by priority
                for cand in TARGET_CANDIDATE_NAMES:
                    matches = [c for c in target_cands if c.lower() == cand]
                    if matches:
                        resolved_tgt = matches[0]
                        break
                ambiguities.append(
                    f"Multiple target column candidates detected: {target_cands}. Selected '{resolved_tgt}'. "
                    f"Override with --target-column if desired."
                )

        # Classify confidence
        if resolved_in and resolved_tgt:
            if len(input_cands) <= 1 and len(target_cands) <= 1:
                confidence = FieldMappingConfidence.HIGH
            else:
                confidence = FieldMappingConfidence.AMBIGUOUS
        else:
            confidence = FieldMappingConfidence.UNSUPPORTED
            if not resolved_in:
                ambiguities.append(f"No recognizable input column found. Candidates: {INPUT_CANDIDATE_NAMES}. Available columns: {columns}")
            if not resolved_tgt:
                ambiguities.append(f"No recognizable target column found. Candidates: {TARGET_CANDIDATE_NAMES}. Available columns: {columns}")

        return schema_info, resolved_in, resolved_tgt, confidence, ambiguities


class TaskTypeDetector:
    """
    Infers evaluation task category from schema structure, fields, and sample contents.
    """

    @classmethod
    def detect(
        cls,
        schema: DatasetSchemaInfo,
        input_col: Optional[str],
        target_col: Optional[str],
    ) -> Tuple[TaskType, str]:
        """
        Returns: (task_type, explanation)
        """
        cols_lower = [c.lower() for c in schema.columns]

        # 1. Unsupported preference / reward pair datasets without objective evaluation
        if ("chosen" in cols_lower and "rejected" in cols_lower) or ("winner" in cols_lower and "loser" in cols_lower):
            return (
                TaskType.UNSUPPORTED,
                "Detected preference pair dataset (e.g. chosen/rejected). Direct evaluation requires an objective target, "
                "reward model, or pairwise loss protocol which is not supported for standard benchmark pruning."
            )

        # 2. Multiple choice detection
        if schema.has_choices or ("choices" in cols_lower or "options" in cols_lower):
            return (
                TaskType.MULTIPLE_CHOICE,
                "Detected choices/options column. Evaluated using multiple-choice letter or index extraction."
            )

        if not schema.sample_rows or not target_col:
            return TaskType.GENERATION, "Standard generation task."

        sample_targets = [str(r.get(target_col, "")).strip() for r in schema.sample_rows[:20] if r.get(target_col) is not None]
        sample_inputs = [str(r.get(input_col, "")).strip() for r in schema.sample_rows[:20] if r.get(input_col) is not None]

        # 3. Multiple choice via single-letter answers
        if all(len(t) == 1 and t.upper() in "ABCD" for t in sample_targets if t):
            return (
                TaskType.MULTIPLE_CHOICE,
                "Detected single-letter answers (A, B, C, D). Evaluated using multiple-choice accuracy protocol."
            )

        # 4. Mathematical equivalence detection
        math_indicators = [
            r"\\boxed\{",
            r"####\s*-?\d+",
            r"\\frac\{",
            r"\\sqrt\{",
            r"\$\$.*\$\$",
        ]
        has_math_target = any(
            any(re.search(pat, t) for pat in math_indicators)
            for t in sample_targets
        )
        has_math_input = any(
            any(re.search(pat, inp) for pat in math_indicators)
            for inp in sample_inputs
        )
        if has_math_target or (has_math_input and any(t.replace(".", "").replace(",", "").replace("-", "").isdigit() for t in sample_targets)):
            return (
                TaskType.MATH_EQUIVALENCE,
                "Detected mathematical reasoning content and LaTeX formatting. Evaluated using mathematical equivalence verification."
            )

        # 5. Short factual / exact answer task
        avg_target_len = sum(len(t) for t in sample_targets) / max(len(sample_targets), 1)
        if avg_target_len <= 15 and all("\n" not in t for t in sample_targets):
            return (
                TaskType.EXACT_ANSWER,
                "Detected short discrete answers. Evaluated using normalized exact match."
            )

        # 6. General open-ended generation
        return (
            TaskType.GENERATION,
            "Detected open-ended generation task with reference solution."
        )
