import torch
import torch.nn as nn
from prunequest.flap.pruning.slicer import MLPSlicer
from prunequest.flap.pruning.manager import PruningStateManager


class MockLayer(nn.Module):
    def __init__(self, hidden_size=8, intermediate_size=16):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = nn.Module()
        self.config.intermediate_size = 16
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockLayer(), MockLayer()])


def test_transactional_rollback():
    model = MockModel()
    orig_gate_w0 = model.model.layers[0].mlp.gate_proj.weight.clone()
    orig_down_w0 = model.model.layers[0].mlp.down_proj.weight.clone()

    # Take backup
    backup = PruningStateManager.backup_mlp_weights(model)

    # Slice layer 0 to 10 channels
    MLPSlicer.slice_mlp(model.model.layers[0].mlp, list(range(10)))
    assert model.model.layers[0].mlp.gate_proj.out_features == 10
    assert model.model.layers[0].mlp.down_proj.in_features == 10

    # Execute rollback
    PruningStateManager.restore_mlp_weights(model, backup)

    assert model.model.layers[0].mlp.gate_proj.out_features == 16
    assert model.model.layers[0].mlp.down_proj.in_features == 16
    assert torch.allclose(model.model.layers[0].mlp.gate_proj.weight, orig_gate_w0)
    assert torch.allclose(model.model.layers[0].mlp.down_proj.weight, orig_down_w0)
