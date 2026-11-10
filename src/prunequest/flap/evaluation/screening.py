"""
Fast teacher-forced loss proxy screening for early candidate rejection.
"""

from typing import List
import math
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from prunequest.flap.calibration.dataset import DatasetSample
from prunequest.flap.evaluation.base import EvaluationResult


class ProxyScreeningEvaluator:
    """
    Evaluates teacher-forced cross-entropy loss on screening samples without
    expensive autoregressive text generation.
    """

    @classmethod
    def evaluate_loss(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        samples: List[DatasetSample],
        device: str = "cuda:0",
        max_length: int = 512,
    ) -> EvaluationResult:
        model.eval()
        total_loss = 0.0
        sample_count = 0
        start_time = time.time()

        dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"

        with torch.inference_mode():
            for sample in samples:
                text = sample.full_text or f"{sample.question}\n{sample.gold_answer}"
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                if hasattr(enc, "to"):
                    enc = enc.to(dev)
                else:
                    enc = {k: v.to(dev) if hasattr(v, "to") else v for k, v in enc.items()}

                input_ids = enc["input_ids"]
                # In standard HF CausalLM, passing labels=input_ids computes CE loss automatically
                outputs = model(input_ids=input_ids, attention_mask=enc.get("attention_mask"), labels=input_ids)
                loss_val = outputs.loss.item()
                if not math.isnan(loss_val) and not math.isinf(loss_val):
                    total_loss += loss_val
                    sample_count += 1

        avg_loss = (total_loss / max(sample_count, 1)) if sample_count > 0 else float("inf")
        ppl = math.exp(min(avg_loss, 20.0))  # Cap to prevent overflow
        total_time = time.time() - start_time

        peak_vram = 0.0
        if torch.cuda.is_available() and "cuda" in dev:
            peak_vram = torch.cuda.memory_allocated(dev) / (1024**3)

        return EvaluationResult(
            total=len(samples),
            correct=0,
            accuracy=0.0,
            loss=avg_loss,
            perplexity=ppl,
            total_time_sec=total_time,
            peak_vram_gb=peak_vram,
        )
