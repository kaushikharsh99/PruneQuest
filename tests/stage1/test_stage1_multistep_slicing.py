import torch
import torch.nn as nn
from prunequest.stage1.flap.allocation import PruningAllocator
from prunequest.stage1.pruning.slicer import MLPSlicer
from prunequest.stage1.pruning.manager import PruningStateManager


class DummyMLP(nn.Module):
    def __init__(self, hidden=16, intermediate=64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden=16, intermediate=64):
        super().__init__()
        self.mlp = DummyMLP(hidden, intermediate)


class DummyModel(nn.Module):
    def __init__(self, num_layers=2, hidden=16, intermediate=64):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(hidden, intermediate) for _ in range(num_layers)])
        self.config = type("Config", (), {"intermediate_size": intermediate})()


def test_multistep_incremental_slicing():
    """Verify that successive pruning steps (e.g. 10% then 25%) do not trigger index out of bounds."""
    model = DummyModel(num_layers=2, hidden=16, intermediate=64)
    active_channels = {
        0: list(range(64)),
        1: list(range(64)),
    }

    scores = {
        0: torch.rand(64),
        1: torch.rand(64),
    }

    # Step 1: 10% pruning (keep 58 channels)
    plan_1 = PruningAllocator.allocate_uniform(scores, prune_ratio=0.10)
    MLPSlicer.apply_plan(model, plan_1, active_channels=active_channels)

    assert model.layers[0].mlp.gate_proj.out_features == 58
    assert model.layers[0].mlp.down_proj.in_features == 58
    assert len(active_channels[0]) == 58
    assert max(active_channels[0]) < 64

    # Step 2: 25% pruning (keep 48 channels out of original 64)
    # The keep_indices in plan_2 are relative to the original 64 channels.
    # MLPSlicer must map these to relative indices within the 58 currently active channels.
    plan_2 = PruningAllocator.allocate_uniform(scores, prune_ratio=0.25)
    backup_2 = PruningStateManager.backup_mlp_weights(model, active_channels=active_channels)
    MLPSlicer.apply_plan(model, plan_2, active_channels=active_channels)

    assert model.layers[0].mlp.gate_proj.out_features == 48
    assert model.layers[0].mlp.down_proj.in_features == 48
    assert len(active_channels[0]) == 48

    # Step 3: Rollback step 2
    PruningStateManager.restore_mlp_weights(model, backup_2, active_channels=active_channels)
    assert model.layers[0].mlp.gate_proj.out_features == 58
    assert model.layers[0].mlp.down_proj.in_features == 58
    assert len(active_channels[0]) == 58

    # Step 4: Try a step-back ratio (15% pruning -> 54 channels)
    plan_3 = PruningAllocator.allocate_uniform(scores, prune_ratio=0.15)
    MLPSlicer.apply_plan(model, plan_3, active_channels=active_channels)
    assert model.layers[0].mlp.gate_proj.out_features == 54
    assert model.layers[0].mlp.down_proj.in_features == 54
    assert len(active_channels[0]) == 54
