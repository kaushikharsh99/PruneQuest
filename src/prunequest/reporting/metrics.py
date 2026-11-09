"""
Statistical metrics, confidence intervals, and hardware memory helpers.
"""

import math
from typing import Dict, Any, Tuple
import gc
import torch


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a binomial proportion.
    """
    if total <= 0:
        return 0.0, 0.0

    z_dict = {
        0.90: 1.644853,
        0.95: 1.959964,
        0.99: 2.575829,
    }
    z = z_dict.get(confidence, 1.959964)

    p_hat = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = p_hat + z2 / (2 * total)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * total)) / total)

    lower = max(0.0, (centre - spread) / denominator)
    upper = min(1.0, (centre + spread) / denominator)
    return lower, upper


def compute_accuracy_metrics(num_correct: int, total: int, baseline_acc: float = None) -> Dict[str, Any]:
    """
    Compute comprehensive accuracy metrics with Wilson score confidence intervals
    and relative retention percentage.
    """
    if total == 0:
        acc = 0.0
        ci_lower, ci_upper = 0.0, 0.0
    else:
        acc = num_correct / total
        ci_lower, ci_upper = wilson_score_interval(num_correct, total, confidence=0.95)

    metrics = {
        "num_correct": num_correct,
        "total": total,
        "accuracy": acc,
        "accuracy_ci_lower": ci_lower,
        "accuracy_ci_upper": ci_upper,
        "ci_half_width": (ci_upper - ci_lower) / 2.0,
    }

    if baseline_acc is not None and baseline_acc > 0:
        retention = (acc / baseline_acc) * 100.0
        abs_loss = (baseline_acc - acc) * 100.0
        metrics["accuracy_retention_pct"] = retention
        metrics["accuracy_loss_pct"] = abs_loss
    else:
        metrics["accuracy_retention_pct"] = 100.0 if acc > 0 else 0.0
        metrics["accuracy_loss_pct"] = 0.0

    return metrics


def clean_gpu_memory(device: str = "cuda:0"):
    """
    Aggressively collect Python garbage and flush CUDA allocator caches.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_gpu_vram_allocated_gb(device: str = "cuda:0") -> float:
    """
    Return currently allocated GPU VRAM in GB.
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.memory_allocated(device=device) / (1024**3)
    except Exception:
        return 0.0


def get_gpu_memory_info(device: str = "cuda:0") -> Dict[str, float]:
    """
    Return currently allocated, reserved, and peak VRAM in GB.
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
    return {
        "allocated_gb": torch.cuda.memory_allocated(device=device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device=device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device=device) / (1024**3),
    }
