"""Read-only eval harness for the NAM optimizer benchmark.

DO NOT EDIT during the autoresearch loop. Defines the fixed pieces of the
benchmark: data generating process, training-size grid, evaluation
protocol, scoring, and HP defaults loaded from `hp_defaults.json`.

Convergence-style evaluation: we sweep `n` over a geometric grid, train a
fresh model per (n, seed), and report mean log10 MSPE across all
(n, component, seed) triples. Lower is better.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

N_GRID = [200, 400, 800, 1600, 3200, 6400]
SEEDS = [0, 1, 2, 3, 4]
EVAL_N = 2000
EVAL_SEED_OFFSET = 10_000
EPS_LOG = 1e-12
MU_TRUE = 1.0 / 3.0  # E[y] for the DGP in `simulate`

HERE = Path(__file__).parent
HP_PATH = HERE / "hp_defaults.json"


def simulate(n: int, seed: int):
    """Canonical identifiable 2-component DGP.

        x1 ~ U[0, 1],   f1(x1) = sin(2π x1)                 (mean 0)
        x2 ~ U[-1, 1],  f2(x2) = x2^2 - 1/3                 (mean 0)
        μ  = 1/3,       y = μ + f1 + f2 + ε,  ε ~ N(0, 0.1)

    Returns (inputs, y, f_true, mu) with each component already
    population-centered, so MSPE of component k against `f_true[k]`
    measures shape recovery directly.
    """
    g = torch.Generator().manual_seed(seed)
    x1 = torch.rand(n, 1, generator=g)
    x2 = -1 + 2 * torch.rand(n, 1, generator=g)
    f1 = torch.sin(2 * math.pi * x1).squeeze(-1)
    f2 = (x2 ** 2).squeeze(-1) - 1.0 / 3.0
    mu = torch.tensor(1.0 / 3.0)
    eps = 0.1 * torch.randn(n, generator=g)
    y = mu + f1 + f2 + eps
    return [x1, x2], y, [f1, f2], mu


def evaluate(model, train_seed: int) -> dict:
    """Evaluate on a fresh held-out draw of size EVAL_N.

    Returns:
        mspe          — list of per-component MSPE against the population truth.
        intercept_se  — squared error of model.intercept against MU_TRUE.
    """
    inputs_te, _, f_true_te, _ = simulate(EVAL_N, train_seed + EVAL_SEED_OFFSET)
    model.eval()
    with torch.no_grad():
        mspe = [
            ((model.component(k, inputs_te[k]) - f_true_te[k]) ** 2).mean().item()
            for k in range(len(f_true_te))
        ]
        intercept_se = (model.intercept.item() - MU_TRUE) ** 2
    return {"mspe": mspe, "intercept_se": intercept_se}


def load_hp_defaults() -> dict:
    return json.loads(HP_PATH.read_text()) if HP_PATH.exists() else {}


def hp_for(n: int) -> dict:
    """HP defaults for a given training-set size.

    Falls back to the nearest-n key in `hp_defaults.json`. If no defaults
    file exists, returns a conservative built-in.
    """
    defaults = load_hp_defaults()
    if defaults:
        if str(n) in defaults:
            return dict(defaults[str(n)])
        keys = sorted(int(k) for k in defaults)
        nearest = min(keys, key=lambda k: abs(k - n))
        return dict(defaults[str(nearest)])
    return {"lr": 5e-3, "wd": 0.0, "epochs": 100, "batch_size": 128}


def aggregate(results: list[dict]):
    """Reduce a list of {n, seed, mspe: [...], intercept_se: ...} dicts.

    Returns (headline, mspe_by_n_k, intercept_by_n) where:
      headline        = mean over all (n, k, seed) of log10(MSPE)
      mspe_by_n_k     = {(n, k): [mspe over seeds]}
      intercept_by_n  = {n: [intercept_se over seeds]}
    """
    logs = []
    mspe_by_n_k: dict[tuple[int, int], list[float]] = {}
    intercept_by_n: dict[int, list[float]] = {}
    for r in results:
        for k, m in enumerate(r["mspe"]):
            logs.append(math.log10(max(m, EPS_LOG)))
            mspe_by_n_k.setdefault((r["n"], k), []).append(m)
        intercept_by_n.setdefault(r["n"], []).append(r["intercept_se"])
    headline = sum(logs) / len(logs)
    return headline, mspe_by_n_k, intercept_by_n


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return m, math.sqrt(v)


def format_table(
    mspe_by_n_k: dict[tuple[int, int], list[float]],
    intercept_by_n: dict[int, list[float]],
) -> str:
    """Per-(n, k) MSPE + per-n intercept μ-MSE, mean ± std across seeds."""
    n_vals = sorted({n for (n, _) in mspe_by_n_k})
    k_vals = sorted({k for (_, k) in mspe_by_n_k})
    col_w = 24
    header = f"{'n':>6}  " + "".join(
        f"{'f' + str(k) + ' MSPE (mean ± std)':>{col_w}}" for k in k_vals
    ) + f"{'μ MSE (mean ± std)':>{col_w}}"
    lines = [header]
    for n in n_vals:
        cells = []
        for k in k_vals:
            m, s = _mean_std(mspe_by_n_k[(n, k)])
            cells.append(f"{m:9.4f} ± {s:8.4f}")
        m, s = _mean_std(intercept_by_n[n])
        cells.append(f"{m:9.4f} ± {s:8.4f}")
        lines.append(f"{n:>6d}  " + "".join(f"{c:>{col_w}}" for c in cells))
    return "\n".join(lines)
