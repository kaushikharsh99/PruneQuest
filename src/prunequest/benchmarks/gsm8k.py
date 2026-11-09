"""
GSM8K benchmark adapter with Qwen2.5-Math zero-shot CoT and math-verify normalization.
"""

import re
import time
import math
import random
from typing import List, Dict, Any, Optional, Tuple
import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from prunequest.benchmarks.base import BaseBenchmark, BenchmarkSample, EvaluationResult
from prunequest.reporting.metrics import clean_gpu_memory
from prunequest.exceptions import BenchmarkError

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content from LaTeX \\boxed{...}, properly handling nested braces.
    Returns the last boxed expression if multiple exist.
    """
    results = []
    idx = 0
    while True:
        pos = text.find(r"\boxed{", idx)
        if pos == -1:
            break
        brace_count = 1
        i = pos + len(r"\boxed{")
        start_content = i
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            content = text[start_content:i-1].strip()
            results.append(content)
            idx = i
        else:
            idx = pos + len(r"\boxed{")

    return results[-1] if results else None


def clean_math_answer(raw_str: str) -> str:
    if not raw_str:
        return ""
    s = raw_str.strip().strip("$")
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("%", "").replace(",", "")
    s = s.rstrip(".")
    s = re.sub(r"(?:kg|km|miles|meters|hours|mins|minutes|seconds|days|weeks|dollars|cents|points|degrees)$", "", s, flags=re.IGNORECASE).strip()
    return s


def extract_gsm8k_gold_answer(answer_str: str) -> str:
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_str.replace(",", ""))
    return nums[-1] if nums else answer_str.strip()


def parse_numeric_value(s: str) -> Optional[float]:
    s = s.strip()
    frac_match = re.match(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$", s)
    if frac_match:
        num, denom = float(frac_match.group(1)), float(frac_match.group(2))
        return num / denom if denom != 0 else None
    slash_frac = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", s)
    if slash_frac:
        num, denom = float(slash_frac.group(1)), float(slash_frac.group(2))
        return num / denom if denom != 0 else None
    try:
        return float(s)
    except ValueError:
        return None


class GSM8KBenchmark(BaseBenchmark):
    """
    GSM8K Benchmark adapter supporting deterministic disjoint partitions
    and official Qwen mathematical prompt formatting.
    """

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

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

        calib_idx = indices[:calibration_size]
        dev_idx = indices[calibration_size : calibration_size + dev_size]
        recovery_idx = indices[calibration_size + dev_size :]

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
        dev_samples = make_samples([train_data[i] for i in dev_idx], "dev")
        recovery_samples = make_samples([train_data[i] for i in recovery_idx], "recovery_reserved")
        test_samples = make_samples(test_data, "official_test")

        self.data_splits = {
            "calibration": calib_samples,
            "dev": dev_samples,
            "smoke": dev_samples[:smoke_size],
            "screening": dev_samples[:screening_size],
            "recovery_reserved": recovery_samples,
            "test": test_samples,
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
        gold_cleaned = extract_gsm8k_gold_answer(gold_str)

        if HAS_MATH_VERIFY:
            try:
                p_pred = mv_parse(prediction_text)
                p_gold = mv_parse(gold_cleaned)
                if p_pred and p_gold and mv_verify(p_gold, p_pred):
                    extracted = str(p_pred[-1]) if p_pred else ""
                    return True, extracted, gold_cleaned
            except Exception:
                pass

        boxed = extract_boxed_content(prediction_text)
        if boxed is not None:
            cand = clean_math_answer(boxed)
            if HAS_MATH_VERIFY:
                try:
                    p_cand = mv_parse(boxed)
                    p_gold = mv_parse(gold_cleaned)
                    if p_cand and p_gold and mv_verify(p_gold, p_cand):
                        return True, boxed, gold_cleaned
                except Exception:
                    pass

            cand_val = parse_numeric_value(cand)
            gold_val = parse_numeric_value(gold_cleaned)
            if cand_val is not None and gold_val is not None:
                if math.isclose(cand_val, gold_val, rel_tol=1e-5, abs_tol=1e-5):
                    return True, boxed, gold_cleaned
            if cand.strip().lower() == gold_cleaned.strip().lower():
                return True, boxed, gold_cleaned
            return False, boxed, gold_cleaned

        if "####" in prediction_text:
            pred_cand = prediction_text.split("####")[-1].strip()
            cand = clean_math_answer(pred_cand)
            cand_val = parse_numeric_value(cand)
            gold_val = parse_numeric_value(gold_cleaned)
            if cand_val is not None and gold_val is not None:
                if math.isclose(cand_val, gold_val, rel_tol=1e-5, abs_tol=1e-5):
                    return True, pred_cand, gold_cleaned

        lines = [l.strip() for l in prediction_text.split("\n") if l.strip()]
        if lines:
            last_line = lines[-1]
            nums = re.findall(r"-?\d+(?:\.\d+)?", last_line.replace(",", ""))
            if nums:
                last_num = nums[-1]
                last_val = parse_numeric_value(last_num)
                gold_val = parse_numeric_value(gold_cleaned)
                if last_val is not None and gold_val is not None:
                    if math.isclose(last_val, gold_val, rel_tol=1e-5, abs_tol=1e-5):
                        return True, last_num, gold_cleaned
                return False, last_num, gold_cleaned

        return False, "", gold_cleaned

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

        samples = self.data_splits.get(split_name, self.data_splits.get("smoke", []))
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
