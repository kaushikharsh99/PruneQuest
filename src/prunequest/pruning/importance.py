"""
ShortGPT Block Influence and layer importance scoring.
"""

import math
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from prunequest.benchmarks.base import BenchmarkSample


def compute_block_influence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    calibration_samples: List[BenchmarkSample],
    metric: str = "cosine",
    max_seq_len: int = 512,
    device: str = "cuda:0",
    max_samples: int = 32,
) -> List[Dict[str, Any]]:
    """
    Compute ShortGPT Block Influence (BI) for every transformer layer using calibration inputs.

    Formula:
        Cosine distance: BI_l = 1 - mean(cosine_similarity(H^{(l-1)}, H^{(l)}))
        Angular distance: BI_l = (1 / pi) * arccos(clamp(mean(cos_sim), -1, 1))

    Lower BI indicates high representation similarity across the block (high redundancy).
    """
    model.eval()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return []

    num_layers = len(model.model.layers)
    subset = calibration_samples[:max_samples]
    layer_cosine_sims = [0.0] * num_layers
    sample_count = 0

    with torch.no_grad():
        for sample in subset:
            prompt = sample.question
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=True,
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            for l in range(num_layers):
                h_in = hidden_states[l]
                h_out = hidden_states[l+1]
                cos_sim = F.cosine_similarity(h_in, h_out, dim=-1).mean().item()
                layer_cosine_sims[l] += cos_sim

            sample_count += 1

    if sample_count == 0:
        return [{"layer_idx": i, "score": 0.0, "cosine_sim": 0.0} for i in range(num_layers)]

    results = []
    for l in range(num_layers):
        avg_sim = layer_cosine_sims[l] / sample_count
        if metric == "angular":
            clamped = max(-1.0, min(1.0, avg_sim))
            bi_score = math.acos(clamped) / math.pi
        else:
            bi_score = 1.0 - avg_sim

        results.append({
            "layer_idx": l,
            "block_influence": bi_score,
            "avg_cosine_sim": avg_sim,
            "score": bi_score,
        })

    # Sort ascending: lowest BI = highest redundancy first
    results.sort(key=lambda x: x["score"])
    return results
