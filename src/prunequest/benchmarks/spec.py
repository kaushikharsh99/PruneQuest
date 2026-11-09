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
        "smoke": "test",
        "screening": "test",
        "dev": "test",
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

        raw_records = []
        if source == "huggingface":
            from datasets import load_dataset
            try:
                ds = load_dataset(path)
                # Take first split or test
                split_key = "test" if "test" in ds else list(ds.keys())[0]
                for idx, row in enumerate(ds[split_key]):
                    raw_records.append({
                        "id": str(idx),
                        "input": str(row.get(input_col, "")),
                        "target": str(row.get(target_col, "")),
                    })
            except Exception as e:
                raise BenchmarkError(f"Failed to load Hugging Face dataset '{path}': {e}") from e
        elif source in ("json", "jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                if source == "jsonl" or path.endswith(".jsonl"):
                    for idx, line in enumerate(f):
                        if line.strip():
                            r = json.loads(line)
                            raw_records.append({
                                "id": str(idx),
                                "input": str(r.get(input_col, "")),
                                "target": str(r.get(target_col, "")),
                            })
                else:
                    data = json.load(f)
                    for idx, r in enumerate(data):
                        raw_records.append({
                            "id": str(idx),
                            "input": str(r.get(input_col, "")),
                            "target": str(r.get(target_col, "")),
                        })
        elif source in ("csv", "parquet"):
            df = pd.read_csv(path) if source == "csv" else pd.read_parquet(path)
            for idx, r in df.iterrows():
                raw_records.append({
                    "id": str(idx),
                    "input": str(r.get(input_col, "")),
                    "target": str(r.get(target_col, "")),
                })
        else:
            raise BenchmarkError(f"Unsupported dataset source '{source}'")

        samples = [
            BenchmarkSample(id=r["id"], question=r["input"], gold_answer=r["target"])
            for r in raw_records
        ]

        # Deterministic slicing
        self._loaded_samples = {
            "calibration": samples[:calibration_size],
            "smoke": samples[:smoke_size],
            "screening": samples[:screening_size],
            "dev": samples[:dev_size],
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
        samples = data.get(split_name, data["smoke"])
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
        elif metric == "math_verify" and MATH_VERIFY_AVAILABLE:
            try:
                pred_p = parse(prediction)
                gold_p = parse(gold)
                return verify(gold_p, pred_p)
            except Exception:
                return self._score(prediction, gold)
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
