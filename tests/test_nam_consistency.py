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
    """Canonical identifiable decomposition  y = μ + f1(x1) + f2(x2) + ε.

    Each f_k is *population-centered*, i.e. E[f_k(X_k)] = 0, and μ absorbs
    all means so that μ = E[y]. Returns the truth in this parametrization
    so tests compare against population-level quantities directly.

        x1 ~ U[0, 1]   → raw sin(2π x1)   has mean 0   → μ_1 = 0
        x2 ~ U[-1, 1]  → raw x2^2          has mean 1/3 → μ_2 = 1/3
        ε  ~ N(0, 0.1)
    """
    g = torch.Generator().manual_seed(seed)
    x1 = torch.rand(n, 1, generator=g)
    x2 = -1 + 2 * torch.rand(n, 1, generator=g)
    f1 = torch.sin(2 * math.pi * x1).squeeze(-1)          # already 0-mean
    f2 = (x2 ** 2).squeeze(-1) - 1.0 / 3.0                # centered
    mu = torch.tensor(1.0 / 3.0)                          # E[y]
    eps = 0.1 * torch.randn(n, generator=g)
    y = mu + f1 + f2 + eps
    return [x1, x2], y, [f1, f2], mu


def test_nam_recovers_additive_components():
    torch.manual_seed(0)

    inputs_tr, y_tr, _, _ = _simulate(n=2000, seed=0)
    inputs_te, y_te, f_true_te, mu_true = _simulate(n=2000, seed=1)

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
        intercept_hat = model.intercept.detach()

    mspe_f1 = ((f1_hat - f_true_te[0]) ** 2).mean().item()
    mspe_f2 = ((f2_hat - f_true_te[1]) ** 2).mean().item()
    intercept_err = (intercept_hat - mu_true).abs().item()

    # Noise variance is 0.01 → y-MSE floor ≈ 0.01.
    assert mse_y < 0.05, f"y-MSE too high: {mse_y:.4f}"
    assert mspe_f1 < 0.05, f"f1 MSPE too high: {mspe_f1:.4f}"
    assert mspe_f2 < 0.05, f"f2 MSPE too high: {mspe_f2:.4f}"
    assert intercept_err < 0.05, f"μ̂ off by {intercept_err:.4f}"


def test_center_is_idempotent_and_prediction_preserving():
    torch.manual_seed(0)
    inputs, y, _, _ = _simulate(n=256, seed=0)

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
