import torch
import torch.nn as nn
from prunequest.stage1.flap.compensation import BiasCompensator


def test_compute_layer_compensation():
    hidden_size = 4
    intermediate_size = 6
    w_down = torch.randn(hidden_size, intermediate_size)
    mean = torch.randn(intermediate_size)
    prune_indices = [1, 3]

    delta_b = BiasCompensator.compute_layer_compensation(w_down, mean, prune_indices)
    assert delta_b.shape == (hidden_size,)

    # Expected: w_down[:, 1]*mean[1] + w_down[:, 3]*mean[3]
    expected = w_down[:, 1] * mean[1] + w_down[:, 3] * mean[3]
    assert torch.allclose(delta_b, expected, atol=1e-5)


def test_apply_compensation_with_bias():
    linear = nn.Linear(6, 4, bias=True)
    orig_bias = linear.bias.clone()
    delta_b = torch.tensor([1.0, 2.0, 3.0, 4.0])

    ok = BiasCompensator.apply_compensation_to_layer(linear, delta_b)
    assert ok is True
    assert torch.allclose(linear.bias, orig_bias + delta_b, atol=1e-5)


def test_apply_compensation_without_bias():
    linear = nn.Linear(6, 4, bias=False)
    assert linear.bias is None
    delta_b = torch.tensor([1.0, 2.0, 3.0, 4.0])

    ok = BiasCompensator.apply_compensation_to_layer(linear, delta_b)
    assert ok is True
    assert linear.bias is not None
    assert torch.allclose(linear.bias, delta_b, atol=1e-5)
