"""
Streaming forward-hook activation statistics collector for FLAP importance profiling.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from prunequest.stage1.calibration.dataset import DatasetSample


@dataclass
class LayerActivationStats:
    layer_idx: int
    intermediate_size: int
    token_count: int
    mean: torch.Tensor          # [intermediate_size]
    variance: torch.Tensor      # [intermediate_size]
    mean_abs: torch.Tensor      # [intermediate_size]
    down_proj_weight_norm2: torch.Tensor  # [intermediate_size]


class FLAPActivationCollector:
    """
    Collects streaming token-level activation statistics (mean, variance, abs mean)
    for each transformer layer's MLP intermediate representation via PyTorch forward hooks.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.hooks: List[Any] = []
        self.stats: Dict[int, Dict[str, Any]] = {}

    def collect(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        samples: List[DatasetSample],
        max_length: int = 512,
        batch_size: int = 1,
    ) -> Dict[int, LayerActivationStats]:
        """
        Executes teacher-forced forward passes over calibration samples to compute streaming stats.
        """
        self.stats.clear()
        self.hooks.clear()

        # Locate layers and down_proj modules
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError("Unsupported model architecture: layers not found.")

        num_layers = len(layers)

        # Initialize accumulators on CPU or device
        for l_idx in range(num_layers):
            mlp = layers[l_idx].mlp
            down_proj = mlp.down_proj
            inter_dim = down_proj.in_features

            self.stats[l_idx] = {
                "token_count": 0,
                "sum_h": torch.zeros(inter_dim, dtype=torch.float64),
                "sum_h2": torch.zeros(inter_dim, dtype=torch.float64),
                "sum_abs": torch.zeros(inter_dim, dtype=torch.float64),
                "inter_dim": inter_dim,
            }

            def make_hook(layer_idx):
                def hook_fn(module, args, output):
                    # args[0] is the intermediate activation h entering down_proj: [B, S, D]
                    if not args:
                        return
                    act = args[0].detach()
                    # Flatten to [N_tokens, D]
                    act_flat = act.view(-1, act.shape[-1]).to(torch.float64).cpu()
                    n_tokens = act_flat.shape[0]

                    st = self.stats[layer_idx]
                    st["token_count"] += n_tokens
                    st["sum_h"] += act_flat.sum(dim=0)
                    st["sum_h2"] += (act_flat ** 2).sum(dim=0)
                    st["sum_abs"] += act_flat.abs().sum(dim=0)
                return hook_fn

            handle = down_proj.register_forward_hook(make_hook(l_idx))
            self.hooks.append(handle)

        model.eval()

        try:
            with torch.inference_mode():
                for sample in tqdm(samples, desc="FLAP Calibration Profiling"):
                    text = sample.full_text or f"{sample.question}\n{sample.gold_answer}"
                    enc = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    ).to(self.device if torch.cuda.is_available() and "cuda" in self.device else "cpu")

                    model(**enc)
        finally:
            # Remove all hooks immediately
            for h in self.hooks:
                h.remove()
            self.hooks.clear()

        # Compute final statistics
        results: Dict[int, LayerActivationStats] = {}
        for l_idx in range(num_layers):
            st = self.stats[l_idx]
            n = max(st["token_count"], 1)
            mean = (st["sum_h"] / n).to(torch.float32)
            # Var(X) = E[X^2] - (E[X])^2
            mean_sq = (st["sum_h2"] / n).to(torch.float32)
            var = torch.clamp(mean_sq - (mean ** 2), min=0.0)
            mean_abs = (st["sum_abs"] / n).to(torch.float32)

            # Compute column L2 norms of down_proj weight
            down_proj = layers[l_idx].mlp.down_proj
            # down_proj.weight: [out_features, in_features]
            w_down = down_proj.weight.detach().cpu().to(torch.float32)
            # squared L2 norm for each input column j: sum over out_features
            w_norm2 = (w_down ** 2).sum(dim=0)  # [in_features]

            results[l_idx] = LayerActivationStats(
                layer_idx=l_idx,
                intermediate_size=st["inter_dim"],
                token_count=n,
                mean=mean,
                variance=var,
                mean_abs=mean_abs,
                down_proj_weight_norm2=w_norm2,
            )

        return results
