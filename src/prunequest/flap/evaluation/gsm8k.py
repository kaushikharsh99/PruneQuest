"""
Mathematical reasoning evaluation for GSM8K with exact answer verification.
"""

from typing import List, Tuple, Optional
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from prunequest.flap.calibration.dataset import DatasetSample
from prunequest.flap.evaluation.base import EvaluationResult
from prunequest.benchmarks.scoring import (
    extract_boxed_content,
    clean_math_answer,
    parse_numeric_value,
    extract_gsm8k_gold_answer,
    verify_gsm8k_answer,
)

# Backward compatibility aliases
clean_math_str = clean_math_answer
parse_numeric = parse_numeric_value

__all__ = [
    "GSM8KEvaluator",
    "verify_math_prediction",
    "extract_boxed_content",
    "clean_math_str",
    "parse_numeric",
    "clean_math_answer",
    "parse_numeric_value",
    "extract_gsm8k_gold_answer",
    "verify_gsm8k_answer",
]


def verify_math_prediction(pred_text: str, gold_str: str) -> Tuple[bool, str, str]:
    return verify_gsm8k_answer(pred_text, gold_str)


class GSM8KEvaluator:
    """
    Evaluates mathematical reasoning accuracy on GSM8K samples with greedy decoding.
    """

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    @classmethod
    def format_prompt(cls, tokenizer: PreTrainedTokenizer, question: str) -> str:
        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"<|im_start|>system\n{cls.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    @classmethod
    def auto_detect_batch_size(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda:0",
        max_batch_size: int = 64,
    ) -> int:
        """
        Dynamically detects the largest safe batch size that fits in current GPU VRAM.
        """
        if not torch.cuda.is_available() or "cuda" not in device:
            return 4

        try:
            total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            allocated_vram = torch.cuda.memory_allocated(device) / (1024**3)
            free_vram = total_vram - allocated_vram

            # Aggressive VRAM utilization:
            if free_vram >= 2.0 and max_batch_size >= 64:
                return 64
            elif free_vram >= 1.5 and max_batch_size >= 32:
                return 32
            elif free_vram >= 1.0 and max_batch_size >= 16:
                return 16
            elif free_vram >= 0.6 and max_batch_size >= 8:
                return 8
            elif free_vram >= 0.4 and max_batch_size >= 4:
                return 4
            elif free_vram >= 0.2 and max_batch_size >= 2:
                return 2
            return 1
        except Exception:
            return min(32, max_batch_size)

    @classmethod
    def evaluate(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        samples: List[DatasetSample],
        device: str = "cuda:0",
        batch_size: Optional[int] = None,
        max_batch_size: int = 64,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> EvaluationResult:
        model.eval()
        dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        if torch.cuda.is_available() and "cuda" in dev:
            torch.cuda.empty_cache()

        orig_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        curr_batch_size = batch_size or cls.auto_detect_batch_size(model, tokenizer, device=dev, max_batch_size=max_batch_size)

        correct = 0
        records = []
        total_tokens = 0
        start_time = time.time()
        from tqdm import tqdm

        i = 0
        pbar = tqdm(total=len(samples), desc=f"GSM8K Evaluation (Batch={curr_batch_size})", dynamic_ncols=True, leave=False)

        while i < len(samples):
            batch_samples = samples[i : i + curr_batch_size]
            prompts = [cls.format_prompt(tokenizer, s.question) for s in batch_samples]

            try:
                enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                if hasattr(enc, "to"):
                    enc = enc.to(dev)
                else:
                    enc = {k: v.to(dev) if hasattr(v, "to") else v for k, v in enc.items()}

                in_lens = enc["input_ids"].shape[1]

                with torch.inference_mode():
                    outputs = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                for b_idx, sample in enumerate(batch_samples):
                    gen_tokens = outputs[b_idx, in_lens:]
                    total_tokens += len(gen_tokens)
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                    is_correct, pred_cand, gold_cand = verify_math_prediction(gen_text, sample.gold_answer)
                    if is_correct:
                        correct += 1

                    records.append({
                        "sample_id": sample.id,
                        "question": sample.question,
                        "gold_answer": sample.gold_answer,
                        "extracted_gold": gold_cand,
                        "extracted_pred": pred_cand,
                        "generated_text": gen_text,
                        "is_correct": is_correct,
                    })

                pbar.update(len(batch_samples))
                i += len(batch_samples)
                pbar.set_postfix({
                    "score": f"{correct}/{i}",
                    "acc": f"{(correct/i)*100:.2f}%",
                })

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if curr_batch_size > 1:
                    curr_batch_size = max(1, curr_batch_size // 2)
                    pbar.set_description(f"GSM8K Evaluation (OOM Fallback Batch={curr_batch_size})")
                    continue
                else:
                    pbar.close()
                    tokenizer.padding_side = orig_padding_side
                    raise
            except Exception as e:
                pbar.close()
                tokenizer.padding_side = orig_padding_side
                raise e

        pbar.close()
        tokenizer.padding_side = orig_padding_side

        accuracy = (correct / len(samples)) if samples else 0.0
        print(f"\n✔ Evaluation Completed: {correct}/{len(samples)} ({accuracy*100:.2f}%)\n")

        total_time = time.time() - start_time
        peak_vram = 0.0
        if torch.cuda.is_available() and "cuda" in dev:
            peak_vram = torch.cuda.memory_allocated(dev) / (1024**3)

        accuracy = (correct / len(samples)) if samples else 0.0
        tokens_per_sec = (total_tokens / total_time) if total_time > 0 else 0.0

        return EvaluationResult(
            total=len(samples),
            correct=correct,
            accuracy=accuracy,
            records=records,
            total_time_sec=total_time,
            tokens_per_sec=tokens_per_sec,
            peak_vram_gb=peak_vram,
        )
