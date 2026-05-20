import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from neynam import NAM, MultiInputDataset


def _mlp(in_dim=1, hidden=32):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def _simulate(n, seed):
    g = torch.Generator().manual_seed(seed)
    x1 = torch.rand(n, 1, generator=g)
    x2 = -1 + 2 * torch.rand(n, 1, generator=g)
    f1 = torch.sin(2 * math.pi * x1).squeeze(-1)
    f2 = (x2 ** 2).squeeze(-1)
    eps = 0.1 * torch.randn(n, generator=g)
    y = f1 + f2 + eps
    return [x1, x2], y, [f1, f2]


def test_nam_recovers_additive_components():
    torch.manual_seed(0)

    inputs_tr, y_tr, _ = _simulate(n=2000, seed=0)
    inputs_te, y_te, f_true_te = _simulate(n=2000, seed=1)

    loader = DataLoader(
        MultiInputDataset(inputs_tr, y_tr),
        batch_size=128, shuffle=True,
    )

    model = NAM([_mlp(), _mlp()])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    for _ in range(50):
        for xs, y in loader:
            opt.zero_grad()
            loss = ((model(xs) - y) ** 2).mean()
            loss.backward()
            opt.step()

    model.center(inputs_tr)
    model.eval()

    with torch.no_grad():
        mse_y = ((model(inputs_te) - y_te) ** 2).mean().item()
        f1_hat = model.component(0, inputs_te[0])
        f2_hat = model.component(1, inputs_te[1])

    # Compare zero-mean shapes to factor out the unidentifiable level.
    def _center(t):
        return t - t.mean()

    mspe_f1 = ((_center(f1_hat) - _center(f_true_te[0])) ** 2).mean().item()
    mspe_f2 = ((_center(f2_hat) - _center(f_true_te[1])) ** 2).mean().item()

    # Noise variance is 0.01 → y-MSE floor ≈ 0.01.
    assert mse_y < 0.05, f"y-MSE too high: {mse_y:.4f}"
    assert mspe_f1 < 0.05, f"f1 shape MSPE too high: {mspe_f1:.4f}"
    assert mspe_f2 < 0.05, f"f2 shape MSPE too high: {mspe_f2:.4f}"


def test_center_is_idempotent_and_prediction_preserving():
    torch.manual_seed(0)
    inputs, y, _ = _simulate(n=256, seed=0)

    model = NAM([_mlp(), _mlp()])
    with torch.no_grad():
        pred_before = model(inputs).clone()

    model.center(inputs)
    means_after_first = model.means.clone()
    with torch.no_grad():
        pred_after = model(inputs)
    assert torch.allclose(pred_before, pred_after, atol=1e-5)

    model.center(inputs)  # second call: components are zero-mean → no shift
    assert torch.allclose(model.means, means_after_first, atol=1e-6)
