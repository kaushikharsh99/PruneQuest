"""
Real-model bounded integration tests using cached Qwen/Qwen2.5-Math-1.5B-Instruct.
Separately executable test suite validating genuine model adapters, physical pruning,
and out-of-process checkpoint reload. Run with:
    PRUNEQUEST_TEST_REAL_MODEL=1 pytest tests/integration/test_real_model.py -v
"""

import os
import shutil
import tempfile
import pytest
import torch

from prunequest.models.registry import ModelRegistry
from prunequest.export.validation import SubprocessValidator
from prunequest.flap.allocation import PruningAllocator
from prunequest.flap.pruning.slicer import MLPSlicer
from prunequest.flap.export.validator import CheckpointValidator


MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"

pytestmark = pytest.mark.skipif(
    not os.environ.get("PRUNEQUEST_TEST_REAL_MODEL"),
    reason="Real model tests run only when PRUNEQUEST_TEST_REAL_MODEL=1",
)


@pytest.mark.real_model
def test_real_model_adapter_and_block_prune():
    """Verify loading, single-layer physical excision, and reload validation on real model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    adapter = ModelRegistry.get_adapter(model_id_or_path=MODEL_ID, device=device)
    model, tokenizer = adapter.load()

    assert model is not None
    assert tokenizer is not None
    original_layers = len(adapter.get_layers())
    assert original_layers == 28

    meta = adapter.get_metadata()
    assert meta.num_layers == 28
    assert meta.total_params > 1_500_000_000

    # Execute one single layer physical excision (e.g. prune layer 14)
    params_before, params_after = adapter.prune_layer(14)
    assert params_after < params_before
    assert len(adapter.get_layers()) == 27
    assert model.config.num_hidden_layers == 27

    # Export to temp directory and validate in isolated subprocess
    tmpdir = tempfile.mkdtemp(prefix="pq_real_block_")
    try:
        saved_dir = adapter.save_checkpoint(tmpdir)
        # Unload model from parent process before spawning subprocess
        # to ensure zero RAM/VRAM contention on memory-constrained systems
        adapter.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ok, msg = SubprocessValidator.validate_checkpoint(saved_dir, device="cpu")
        assert ok, f"Subprocess validation failed for pruned model: {msg}"
    finally:
        adapter.unload()
        shutil.rmtree(tmpdir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@pytest.mark.real_model
def test_real_model_flap_slicing_and_reload():
    """Verify SwiGLU MLP channel slicing on real model and independent reload."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    adapter = ModelRegistry.get_adapter(model_id_or_path=MODEL_ID, device=device)
    model, tokenizer = adapter.load()

    # Slice 5% channels from layer 0
    orig_inter = model.model.layers[0].mlp.gate_proj.out_features
    target_channels = int(orig_inter * 0.95)

    # Create synthetic ranking to slice least important channels
    scores = {0: torch.linspace(0.1, 1.0, orig_inter)}
    plan = PruningAllocator.allocate_uniform(scores, ratio=0.05)
    keep_indices = {0: list(plan.keep_indices[0])}

    # Apply physical slicing
    MLPSlicer.apply_plan(model, plan, active_channels={0: list(range(orig_inter))})

    new_inter = model.model.layers[0].mlp.gate_proj.out_features
    assert new_inter == target_channels
    assert new_inter == len(keep_indices[0])
    assert new_inter < orig_inter

    # Export and reload validate
    tmpdir = tempfile.mkdtemp(prefix="pq_real_flap_")
    try:
        saved_dir = adapter.save_checkpoint(tmpdir)
        # Unload model from parent process before spawning subprocess
        adapter.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ok, msg = CheckpointValidator.validate(saved_dir)
        assert ok, f"CheckpointValidator failed for FLAP sliced model: {msg}"
    finally:
        adapter.unload()
        shutil.rmtree(tmpdir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
