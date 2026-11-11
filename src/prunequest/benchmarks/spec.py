"""
Declarative PruneQuest Benchmark Specification schema (BenchmarkSpec) and native execution engine.
Enables custom datasets (Hugging Face, JSON, JSONL, CSV, Parquet) with configurable prompts and scorers.
"""

import os
import re
import json
import math
import pandas as pd
import yaml
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from prunequest.benchmarks.base import BaseBenchmark, BenchmarkSample, EvaluationResult
from prunequest.exceptions import BenchmarkError
from prunequest.reporting.metrics import get_gpu_vram_allocated_gb, clean_gpu_memory

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


class DatasetFields(BaseModel):
    input: str = "question"
    target: str = "answer"


class DatasetConfig(BaseModel):
    source: str = "huggingface"  # huggingface, json, jsonl, csv, parquet
    path: str
    revision: Optional[str] = "main"
    fields: DatasetFields = Field(default_factory=DatasetFields)
    split_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "calibration": "train",
        "screening": "train",
        "dev": "validation",
        "smoke": "validation",
        "final_test": "test",
    })


class TaskConfig(BaseModel):
    type: str = "generation"  # generation, multiple_choice, log_likelihood, perplexity
    prompt_template: str = "{{ input }}"
    use_chat_template: bool = False
    system_prompt: Optional[str] = None


class GenerationConfig(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False


class ScoringConfig(BaseModel):
    metric: str = "exact_match"  # exact_match, normalized_match, numeric_exact_match, math_verify, regex
    higher_is_better: bool = True
    regex_pattern: Optional[str] = None


class OptimizationConfig(BaseModel):
    metric: str = "exact_match"
    retention: float = 0.99


class BenchmarkSpec(BaseModel):
    version: int = 1
    name: str
    dataset: DatasetConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BenchmarkSpec":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class NativeBenchmark(BaseBenchmark):
    """
    Executes a declarative BenchmarkSpec natively.
    """

    def __init__(self, spec: BenchmarkSpec, seed: int = 42):
        super().__init__(name=spec.name, seed=seed)
        self.spec = spec
        self._loaded_samples: Optional[Dict[str, List[BenchmarkSample]]] = None

    def _extract_row_record(self, row: Dict[str, Any], idx: int, input_col: str, target_col: str) -> Dict[str, str]:
        if input_col not in row:
            raise BenchmarkError(
                f"Required input field '{input_col}' missing in dataset row {idx}. Available fields: {list(row.keys())}"
            )
        if target_col not in row:
            raise BenchmarkError(
                f"Required target field '{target_col}' missing in dataset row {idx}. Available fields: {list(row.keys())}"
            )
        val_in = str(row[input_col]).strip() if row[input_col] is not None else ""
        val_tgt = str(row[target_col]).strip() if row[target_col] is not None else ""
        if not val_in:
            raise BenchmarkError(f"Dataset row {idx} has empty input field '{input_col}'. Malformed samples cannot be used.")
        if not val_tgt:
            raise BenchmarkError(f"Dataset row {idx} has empty target field '{target_col}'. Malformed samples cannot be used.")
        return {"id": str(idx), "input": val_in, "target": val_tgt}

    def load_data(
        self,
        calibration_size: int = 256,
        dev_size: int = 512,
        smoke_size: int = 16,
        screening_size: int = 64,
    ) -> Dict[str, List[BenchmarkSample]]:
        if self._loaded_samples is not None:
            return self._loaded_samples

        source = self.spec.dataset.source.lower()
        path = self.spec.dataset.path
        input_col = self.spec.dataset.fields.input
        target_col = self.spec.dataset.fields.target

        if source == "huggingface":
            from datasets import load_dataset
            try:
                ds = load_dataset(path, revision=self.spec.dataset.revision)
            except Exception as e:
                raise BenchmarkError(f"Failed to load Hugging Face dataset '{path}': {e}") from e

            # Multi-split dataset handling
            if hasattr(ds, "keys") and callable(ds.keys):
                available_splits = list(ds.keys())
                mapping = self.spec.dataset.split_mapping

                def resolve_split(role: str, candidates: List[str]) -> Optional[str]:
                    target = mapping.get(role)
                    if target and target in available_splits:
                        return target
                    for c in candidates:
                        if c in available_splits:
                            return c
                    return None

                train_sp = resolve_split("calibration", ["train", "training"])
                val_sp = resolve_split("dev", ["validation", "valid", "val", "dev"])
                test_sp = resolve_split("final_test", ["test", "testing", "eval"])

                if train_sp and val_sp:
                    train_records = [self._extract_row_record(r, i, input_col, target_col) for i, r in enumerate(ds[train_sp])]
                    val_records = [self._extract_row_record(r, i, input_col, target_col) for i, r in enumerate(ds[val_sp])]
                    test_records = [self._extract_row_record(r, i, input_col, target_col) for i, r in enumerate(ds[test_sp])] if test_sp else []

                    total_train = len(train_records)
                    min_train_needed = calibration_size + screening_size
                    if total_train < min_train_needed:
                        raise BenchmarkError(
                            f"Insufficient samples in training split '{train_sp}' ({total_train}) for disjoint "
                            f"calibration ({calibration_size}) + screening ({screening_size}) partitions."
                        )

                    calib_samples = [BenchmarkSample(id=f"train_{r['id']}", question=r["input"], gold_answer=r["target"]) for r in train_records[:calibration_size]]
                    screen_samples = [BenchmarkSample(id=f"train_{r['id']}", question=r["input"], gold_answer=r["target"]) for r in train_records[calibration_size:min_train_needed]]

                    dev_len = min(len(val_records), dev_size)
                    if dev_len == 0:
                        raise BenchmarkError(f"Validation split '{val_sp}' is empty.")
                    dev_samples = [BenchmarkSample(id=f"val_{r['id']}", question=r["input"], gold_answer=r["target"]) for r in val_records[:dev_len]]
                    smoke_samples = dev_samples[:min(dev_len, smoke_size)]
                    test_samples = [BenchmarkSample(id=f"test_{r['id']}", question=r["input"], gold_answer=r["target"]) for r in test_records]

                    self._loaded_samples = {
                        "calibration": calib_samples,
                        "screening": screen_samples,
                        "dev": dev_samples,
                        "smoke": smoke_samples,
                        "test": test_samples,
                        "full": calib_samples + screen_samples + dev_samples,
                    }
                    return self._loaded_samples

                # Single split in DatasetDict
                single_sp = train_sp or val_sp or available_splits[0]
                raw_records = [self._extract_row_record(r, i, input_col, target_col) for i, r in enumerate(ds[single_sp])]
            else:
                raw_records = [self._extract_row_record(r, i, input_col, target_col) for i, r in enumerate(ds)]
        elif source in ("json", "jsonl"):
            raw_records = []
            with open(path, "r", encoding="utf-8") as f:
                if source == "jsonl" or path.endswith(".jsonl"):
                    for idx, line in enumerate(f):
                        if line.strip():
                            r = json.loads(line)
                            raw_records.append(self._extract_row_record(r, idx, input_col, target_col))
                else:
                    data = json.load(f)
                    for idx, r in enumerate(data):
                        raw_records.append(self._extract_row_record(r, idx, input_col, target_col))
        elif source in ("csv", "parquet"):
            df = pd.read_csv(path) if source == "csv" else pd.read_parquet(path)
            raw_records = []
            for idx, r in df.iterrows():
                raw_records.append(self._extract_row_record(r.to_dict(), idx, input_col, target_col))
        else:
            raise BenchmarkError(f"Unsupported dataset source '{source}'")

        samples = [
            BenchmarkSample(id=r["id"], question=r["input"], gold_answer=r["target"])
            for r in raw_records
        ]

        # Disjoint deterministic slicing to strictly prevent data leakage between calibration and evaluation
        total_len = len(samples)
        needed = calibration_size + screening_size + dev_size
        if total_len < needed:
            raise BenchmarkError(
                f"Insufficient data in dataset for disjoint partitions: available={total_len}, "
                f"required={needed} (calibration={calibration_size} + screening={screening_size} + dev={dev_size}). "
                f"Data leakage between calibration and development is strictly prevented."
            )

        calib_end = calibration_size
        screen_end = calib_end + screening_size
        dev_end = screen_end + dev_size

        dev_samples = samples[screen_end:dev_end]
        smoke_samples = dev_samples[:min(len(dev_samples), smoke_size)]

        self._loaded_samples = {
            "calibration": samples[:calib_end],
            "screening": samples[calib_end:screen_end],
            "dev": dev_samples,
            "smoke": smoke_samples,
            "test": samples[dev_end:],
            "full": samples,
        }
        return self._loaded_samples

    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        split_name: str = "smoke",
        max_samples: Optional[int] = 16,
        device: str = "cuda:0",
    ) -> EvaluationResult:
        data = self.load_data()
        if split_name not in data or not data[split_name]:
            available = [k for k, v in data.items() if v]
            raise BenchmarkError(
                f"Requested evaluation split '{split_name}' is not available or empty in dataset '{self.spec.name}'. "
                f"Available populated splits: {available}. Please specify a valid split."
            )
        samples = data[split_name]
        if max_samples:
            samples = samples[:max_samples]

        clean_gpu_memory(device)
        model.eval()

        correct = 0
        records = []
        import time
        start_time = time.time()

        for s in samples:
            # Build prompt
            prompt_text = self.spec.task.prompt_template.replace("{{ input }}", s.question)
            if self.spec.task.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                messages = []
                if self.spec.task.system_prompt:
                    messages.append({"role": "system", "content": self.spec.task.system_prompt})
                messages.append({"role": "user", "content": prompt_text})
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = prompt_text

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.spec.generation.max_new_tokens,
                    do_sample=self.spec.generation.do_sample,
                    temperature=self.spec.generation.temperature if self.spec.generation.do_sample else None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            is_corr = self._score(gen_text, s.gold_answer)
            if is_corr:
                correct += 1

            records.append({
                "sample_id": s.id,
                "prompt": prompt,
                "gold_answer": s.gold_answer,
                "generated_text": gen_text,
                "is_correct": is_corr,
            })

        total = len(samples)
        acc = (correct / total) if total > 0 else 0.0
        elapsed = time.time() - start_time
        peak_vram = get_gpu_vram_allocated_gb(device)

        return EvaluationResult(
            total=total,
            correct=correct,
            accuracy=acc,
            records=records,
            total_tokens=0,
            total_time_sec=elapsed,
            tokens_per_sec=(total / elapsed) if elapsed > 0 else 0.0,
            peak_vram_gb=peak_vram,
        )

    def _score(self, prediction: str, gold: str) -> bool:
        metric = self.spec.scoring.metric.lower()
        if metric == "exact_match":
            return prediction.strip() == gold.strip()
        elif metric == "normalized_match":
            return prediction.strip().lower() == gold.strip().lower()
        elif metric == "numeric_exact_match":
            nums_pred = re.findall(r"-?\d+(?:\.\d+)?", prediction.replace(",", ""))
            nums_gold = re.findall(r"-?\d+(?:\.\d+)?", gold.replace(",", ""))
            if nums_pred and nums_gold:
                try:
                    return math.isclose(float(nums_pred[-1]), float(nums_gold[-1]), rel_tol=1e-5)
                except ValueError:
                    return False
            return False
        elif metric == "math_verify":
            if not MATH_VERIFY_AVAILABLE:
                raise BenchmarkError(
                    "Scorer 'math_verify' requires the 'math_verify' library, which is not installed. "
                    "Install it via 'pip install math-verify' or specify an alternative metric with --metric (e.g., numeric_exact_match, exact_match)."
                )
            try:
                pred_p = parse(prediction)
                gold_p = parse(gold)
                return bool(verify(gold_p, pred_p))
            except Exception:
                return False
        elif metric == "multiple_choice":
            pred_c = prediction.strip().upper()
            gold_c = gold.strip().upper()
            if pred_c == gold_c:
                return True
            if len(gold_c) == 1 and gold_c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                matches = re.findall(r"\b([A-Z])\b", pred_c)
                if matches and matches[-1] == gold_c:
                    return True
            return False
        elif metric == "regex" and self.spec.scoring.regex_pattern:
            m = re.search(self.spec.scoring.regex_pattern, prediction)
            if m:
                extracted = m.group(1) if m.groups() else m.group(0)
                return extracted.strip() == gold.strip()
            return False
        return prediction.strip() == gold.strip()


class BenchmarkSpecCompiler:
    """
    Validates BenchmarkSpec files and provides compilation helpers.
    """

    @classmethod
    def validate_spec(cls, spec_path: str) -> Tuple[bool, List[str]]:
        errors = []
        if not os.path.exists(spec_path):
            return False, [f"Spec file not found: {spec_path}"]
        try:
            spec = BenchmarkSpec.from_yaml(spec_path)
            if not spec.name:
                errors.append("Spec 'name' is required.")
            if not spec.dataset.path:
                errors.append("Dataset 'path' is required.")
        except Exception as e:
            errors.append(f"YAML validation error: {str(e)}")

        return len(errors) == 0, errors

    @classmethod
    def preview_spec(cls, spec_path: str, count: int = 3) -> List[Dict[str, Any]]:
        spec = BenchmarkSpec.from_yaml(spec_path)
        bench = NativeBenchmark(spec)
        data = bench.load_data(smoke_size=count)
        samples = data.get("smoke", [])[:count]
        previews = []
        for s in samples:
            prompt = spec.task.prompt_template.replace("{{ input }}", s.question)
            previews.append({
                "id": s.id,
                "question": s.question[:100] + ("..." if len(s.question) > 100 else ""),
                "rendered_prompt": prompt[:150] + ("..." if len(prompt) > 150 else ""),
                "gold_answer": s.gold_answer,
            })
        return previews
