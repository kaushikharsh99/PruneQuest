"""
Transformer Layer Pruning engine.
"""

from typing import Dict, Any, List, Optional
from prunequest.pruning.base import BasePruner, PruningCandidate, PruningResult
from prunequest.pruning.importance import compute_block_influence
from prunequest.models.base import BaseModelAdapter
from prunequest.benchmarks.base import BenchmarkSample


class LayerPruner(BasePruner):
    """
    Physical transformer layer pruner.
    Identifies redundant layers via Block Influence and executes ModuleList removal.
    """

    def __init__(self, metric: str = "block_influence"):
        self.metric = metric

    def supports(self, adapter: BaseModelAdapter) -> bool:
        try:
            layers = adapter.get_layers()
            return len(layers) > 2
        except Exception:
            return False

    def analyze(
        self,
        adapter: BaseModelAdapter,
        calibration_samples: Optional[List[BenchmarkSample]] = None,
        max_samples: int = 32,
        **kwargs,
    ) -> Dict[str, Any]:
        if adapter.model is None:
            adapter.load()

        if calibration_samples:
            if adapter.tokenizer is None:
                adapter.load()
            if adapter.tokenizer is not None:
                scores = compute_block_influence(
                    model=adapter.model,
                    tokenizer=adapter.tokenizer,
                    calibration_samples=calibration_samples,
                    metric="cosine",
                    device=adapter.device,
                    max_samples=max_samples,
                )
            else:
                num_layers = len(adapter.get_layers())
                scores = [{"layer_idx": i, "score": float(abs(i - num_layers // 2))} for i in range(num_layers)]
                scores.sort(key=lambda x: x["score"])
        else:
            # Fallback heuristic: middle layers prioritized if no calibration samples
            num_layers = len(adapter.get_layers())
            scores = [{"layer_idx": i, "score": float(abs(i - num_layers // 2))} for i in range(num_layers)]
            scores.sort(key=lambda x: x["score"])

        return {"layer_rankings": scores}

    def propose_candidates(
        self,
        adapter: BaseModelAdapter,
        analysis_results: Dict[str, Any],
        top_k: int = 3,
    ) -> List[PruningCandidate]:
        rankings = analysis_results.get("layer_rankings", [])
        candidates = []

        layers = adapter.get_layers()
        if not layers:
            return []

        # Calculate estimated parameter reduction per block
        first_layer_params = sum(p.numel() for p in layers[0].parameters())

        for item in rankings[:top_k]:
            layer_idx = item["layer_idx"]
            cand = PruningCandidate(
                candidate_id=f"prune_layer_{layer_idx}",
                pruning_type="layer",
                target_info={"layer_idx": layer_idx},
                estimated_param_reduction=first_layer_params,
                importance_score=item["score"],
                metadata={"analysis_item": item},
            )
            candidates.append(cand)

        return candidates

    def apply_pruning(
        self,
        adapter: BaseModelAdapter,
        candidate: PruningCandidate,
    ) -> PruningResult:
        layer_idx = candidate.target_info["layer_idx"]
        params_before, params_after = adapter.prune_layer(layer_idx)
        params_removed = params_before - params_after
        retention = params_after / params_before if params_before > 0 else 1.0

        valid, err = adapter.validate_architecture()

        return PruningResult(
            candidate=candidate,
            params_before=params_before,
            params_after=params_after,
            params_removed=params_removed,
            retention_ratio=retention,
            validation_passed=valid,
            error_message=err,
        )
