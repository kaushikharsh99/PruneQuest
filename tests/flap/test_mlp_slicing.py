import torch
import torch.nn as nn
from prunequest.flap.pruning.slicer import MLPSlicer


class MockSwiGLU(nn.Module):
    def __init__(self, hidden_size=16, intermediate_size=32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_slice_swiglu_mlp():
    hidden_size = 8
    intermediate_size = 16
    mlp = MockSwiGLU(hidden_size, intermediate_size)

    # Initial parameter count
    orig_params = sum(p.numel() for p in mlp.parameters())
    assert orig_params == (8 * 16) + (8 * 16) + (16 * 8)  # 128 + 128 + 128 = 384

    # Keep 12 out of 16 channels (prune 4 channels)
    keep_indices = list(range(0, 12))
    res = MLPSlicer.slice_mlp(mlp, keep_indices)

    assert res["new_intermediate_size"] == 12
    assert mlp.gate_proj.out_features == 12
    assert mlp.up_proj.out_features == 12
    assert mlp.down_proj.in_features == 12

    # New parameter count
    new_params = sum(p.numel() for p in mlp.parameters())
    assert new_params == (8 * 12) + (8 * 12) + (12 * 8)  # 96 + 96 + 96 = 288
    assert res["params_removed"] == 384 - 288 == 96

    # Verify forward pass
    x = torch.randn(2, 4, hidden_size)
    out = mlp(x)
    assert out.shape == (2, 4, hidden_size)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
