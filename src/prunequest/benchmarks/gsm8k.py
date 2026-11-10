"""
GSM8K benchmark adapter with Qwen2.5-Math zero-shot CoT and math-verify normalization.
"""

import time
import random
from typing import List, Dict, Any, Optional, Tuple
import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from prunequest.benchmarks.base import BaseBenchmark, BenchmarkSample, EvaluationResult
from prunequest.exceptions import BenchmarkError
from prunequest.benchmarks.scoring import (
    extract_boxed_content,
    clean_math_answer,
    extract_gsm8k_gold_answer,
    parse_numeric_value,
    verify_gsm8k_answer,
)

__all__ = [
    "GSM8KBenchmark",
    "extract_boxed_content",
    "clean_math_answer",
    "extract_gsm8k_gold_answer",
    "parse_numeric_value",
    "verify_gsm8k_answer",
]


class GSM8KBenchmark(BaseBenchmark):
    """
    GSM8K Benchmark adapter supporting deterministic disjoint partitions
    and official Qwen mathematical prompt formatting.
    """

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    def __init__(self, seed: int = 42, max_new_tokens: int = 512):
        super().__init__(name="gsm8k", seed=seed, max_new_tokens=max_new_tokens)

    def load_data(
        self,
        calibration_size: int = 256,
        dev_size: int = 512,
        smoke_size: int = 16,
        screening_size: int = 64,
    ) -> Dict[str, List[BenchmarkSample]]:
        try:
            ds = load_dataset("openai/gsm8k", "main")
        except Exception as e:
            raise BenchmarkError(f"Failed to load dataset 'openai/gsm8k': {e}") from e

        train_data = list(ds["train"])
        test_data = list(ds["test"])

        rng = random.Random(self.seed)
        indices = list(range(len(train_data)))
        rng.shuffle(indices)

        total_needed = calibration_size + screening_size + dev_size
        if len(train_data) < total_needed:
            raise BenchmarkError(
                f"Dataset 'openai/gsm8k' train split contains {len(train_data)} samples, "
                f"which is fewer than the requested {total_needed} samples "
                f"(calibration={calibration_size}, screening={screening_size}, dev={dev_size})."
            )

        calib_idx = indices[:calibration_size]
        screen_idx = indices[calibration_size : calibration_size + screening_size]
        dev_idx = indices[calibration_size + screening_size : calibration_size + screening_size + dev_size]
        recovery_idx = indices[calibration_size + screening_size + dev_size :]

        def make_samples(items, split_name):
            samples = []
            for i, it in enumerate(items):
                samples.append(BenchmarkSample(
                    id=f"{split_name}_{i}",
                    question=it["question"].strip(),
                    gold_answer=it["answer"].strip(),
                    metadata={"split": split_name, "original_index": i}
                ))
            return samples

        calib_samples = make_samples([train_data[i] for i in calib_idx], "calibration")
        screen_samples = make_samples([train_data[i] for i in screen_idx], "screening")
        dev_samples = make_samples([train_data[i] for i in dev_idx], "dev")
        recovery_samples = make_samples([train_data[i] for i in recovery_idx], "recovery_reserved")
        test_samples = make_samples(test_data, "official_test")

        self.data_splits = {
            "calibration": calib_samples,
            "screening": screen_samples,
            "dev": dev_samples,
            "acceptance": dev_samples,
            "smoke": dev_samples[:smoke_size],  # Diagnostic subset of dev for quick healthchecks
            "recovery_reserved": recovery_samples,
            "test": test_samples,
            "full": test_samples,
        }
        return self.data_splits

    def format_prompt(self, tokenizer: PreTrainedTokenizer, sample: BenchmarkSample) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": sample.question}
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{sample.question}<|im_end|>\n<|im_start|>assistant\n"

    def verify_answer(self, prediction_text: str, gold_str: str) -> Tuple[bool, str, str]:
        return verify_gsm8k_answer(prediction_text, gold_str)

    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        split_name: str = "smoke",
        max_samples: Optional[int] = None,
        device: str = "cuda:0",
        progress_callback: Optional[Any] = None,
    ) -> EvaluationResult:
        if not self.data_splits:
            self.load_data()

        if split_name not in self.data_splits and split_name not in ("full", "test"):
            valid = ", ".join(sorted(self.data_splits.keys()))
            raise BenchmarkError(f"Unknown split '{split_name}'. Available splits: {valid}")

        if split_name in ("full", "test"):
            samples = self.data_splits.get("full", self.data_splits.get("test", []))
        else:
            samples = self.data_splits[split_name]
        if max_samples:
            samples = samples[:max_samples]

        model.eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)

        total = len(samples)
        correct_count = 0
        records = []
        total_tokens = 0
        total_time = 0.0

        for idx, sample in enumerate(samples):
            prompt = self.format_prompt(tokenizer, sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            t1 = time.perf_counter()

            gen_time = t1 - t0
            output_ids = outputs[0][input_len:]
            num_tokens = len(output_ids)
            response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            total_tokens += num_tokens
            total_time += gen_time

            is_corr, parsed_pred, gold_cleaned = self.verify_answer(response_text, sample.gold_answer)
            if is_corr:
                correct_count += 1

            records.append({
                "sample_id": sample.id,
                "question": sample.question,
                "gold_answer": sample.gold_answer,
                "extracted_gold": gold_cleaned,
                "prediction": response_text,
                "parsed_prediction": parsed_pred,
                "is_correct": is_corr,
                "num_tokens": num_tokens,
                "gen_time_sec": gen_time,
                "tokens_per_sec": num_tokens / gen_time if gen_time > 0 else 0.0,
            })

            if progress_callback:
                progress_callback(idx + 1, total, correct_count)

        peak_vram = torch.cuda.max_memory_allocated(device=device) / (1024**3) if torch.cuda.is_available() else 0.0
        avg_tok_s = total_tokens / total_time if total_time > 0 else 0.0

        return EvaluationResult(
            total=total,
            correct=correct_count,
            accuracy=correct_count / total if total > 0 else 0.0,
            records=records,
            total_tokens=total_tokens,
            total_time_sec=total_time,
            tokens_per_sec=avg_tok_s,
            peak_vram_gb=peak_vram,
        )
