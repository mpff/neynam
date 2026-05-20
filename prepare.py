"""Read-only eval harness for the NAM optimizer benchmark.

DO NOT EDIT during the autoresearch loop. Defines the fixed pieces of the
benchmark: data generating process, training-size grid, evaluation
protocol, scoring, and HP defaults loaded from `hp_defaults.json`.

Convergence-style evaluation: we sweep `n` over a geometric grid, train a
fresh model per (n, seed), and report mean log10 MSPE across all
(n, component, seed) triples. Lower is better.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

N_GRID = [200, 400, 800, 1600, 3200, 6400]
SEEDS = [0, 1, 2, 3, 4]
EVAL_N = 2000
# Held-out draws use disjoint seed offsets so HP-search (val) cannot leak
# into the final headline (test). Two independent fresh draws per seed.
VAL_SEED_OFFSET = 10_000
TEST_SEED_OFFSET = 20_000
EPS_LOG = 1e-12
MU_TRUE = 1.0 / 3.0  # E[y] for the DGP in `simulate`
RHO = 0.7            # Gaussian-copula correlation between x1 and x2 (concurvity)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline HPs — classic SGD + cosine LR + early stopping on val y-MSE.
# A wider sweep ({3e-2, 1e-1, 3e-1, 1.0} × {wd 0, 1e-4} across N_GRID)
# found lr=0.1 (no weight decay) dominant at every n; constants below.
HP_DEFAULTS = {"lr": 1e-1, "max_epochs": 500, "patience": 20, "batch_size": 128}


def simulate(n: int, seed: int):
    """Concurvity DGP — *asymmetric difficulty + correlated inputs*.

        (z1, z2) ~ N(0, [[1, ρ], [ρ, 1]]),   ρ = RHO
        x1 = Φ(z1)            ∈ [0, 1]      (uniform marginal)
        x2 = 2 Φ(z2) - 1      ∈ [-1, 1]     (uniform marginal)

        f1(x1) = sin(8π x1)                 (mean 0, 4 cycles)
        f2(x2) = x2^2 - 1/3                 (mean 0, smooth)
        μ  = 1/3,   y = μ + f1 + f2 + ε,    ε ~ N(0, 0.1)

    Marginals are unchanged from the uncorrelated DGP — Var(f_k) and E[f_k]
    are the same — but x1 and x2 are now coupled via a Gaussian copula
    (concurvity). The population-best *additive* decomposition is no
    longer just (f1, f2): naive joint MSE absorbs some of f2's signal
    into f̂1 and vice versa. MSPE here is computed against the
    generative (f1, f2) — so the metric rewards routines that
    *un-confound* the components, e.g. backfitting or Neyman-orthogonal
    score updates.

    f1 remains the optimization bottleneck (high-frequency, sample-bound);
    f2 is smooth.
    """
    g = torch.Generator().manual_seed(seed)
    L = math.sqrt(1.0 - RHO ** 2)
    z = torch.randn(n, 2, generator=g)
    z1 = z[:, 0]
    z2 = RHO * z[:, 0] + L * z[:, 1]
    u1 = 0.5 * (1.0 + torch.erf(z1 / math.sqrt(2.0)))
    u2 = 0.5 * (1.0 + torch.erf(z2 / math.sqrt(2.0)))
    x1 = u1.unsqueeze(-1)
    x2 = (2.0 * u2 - 1.0).unsqueeze(-1)
    f1 = torch.sin(8 * math.pi * x1).squeeze(-1)
    f2 = (x2 ** 2).squeeze(-1) - 1.0 / 3.0
    mu = torch.tensor(1.0 / 3.0)
    eps = 0.1 * torch.randn(n, generator=g)
    y = mu + f1 + f2 + eps
    inputs = [x1.to(DEVICE), x2.to(DEVICE)]
    return inputs, y.to(DEVICE), [f1.to(DEVICE), f2.to(DEVICE)], mu.to(DEVICE)


def evaluate(model, train_seed: int, *, split: str = "test") -> dict:
    """Evaluate on a fresh held-out draw of size EVAL_N.

    Pass `split="val"` for HP search and any tuning use; `split="test"`
    for the final headline (default). The two splits use disjoint seed
    offsets, so HPs cannot leak into the reported score.

    Returns:
        mspe          — list of per-component MSPE against the population truth.
        intercept_se  — squared error of model.intercept against MU_TRUE.
    """
    offset = VAL_SEED_OFFSET if split == "val" else TEST_SEED_OFFSET
    inputs_te, _, f_true_te, _ = simulate(EVAL_N, train_seed + offset)
    model.eval()
    with torch.no_grad():
        mspe = [
            ((model.component(k, inputs_te[k]) - f_true_te[k]) ** 2).mean().item()
            for k in range(len(f_true_te))
        ]
        intercept_se = (model.intercept.item() - MU_TRUE) ** 2
    return {"mspe": mspe, "intercept_se": intercept_se}


def evaluate_curve(model, inputs_tr, train_seed: int) -> dict:
    """Mid-training shape-recovery probe on the VAL split — does NOT
    modify model state.

    Centering during training is normally a one-shot post-hoc step
    (`model.center(...)` rewrites `means` and `intercept`). For per-epoch
    curves we instead compute the training-set mean shift on the fly,
    apply it only for evaluation, and leave the parameters alone. Uses
    VAL_SEED_OFFSET so the test split stays untouched until the final
    `evaluate(..., split='test')` call.
    """
    inputs_te, _, f_true_te, _ = simulate(EVAL_N, train_seed + VAL_SEED_OFFSET)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        shifts = [model.component(k, inputs_tr[k]).mean().item()
                  for k in range(len(f_true_te))]
        mspe = [
            ((model.component(k, inputs_te[k]) - shifts[k] - f_true_te[k]) ** 2).mean().item()
            for k in range(len(f_true_te))
        ]
        intercept_eff = model.intercept.item() + sum(shifts)
        intercept_se = (intercept_eff - MU_TRUE) ** 2
    if was_training:
        model.train()
    return {"mspe": mspe, "intercept_se": intercept_se}


class CurveLogger:
    """Append-only TSV of per-epoch shape recovery curves.

    One row per (n, seed, epoch). Truncates the file on construction so
    each invocation of `train.py` produces a fresh log; the autoresearch
    loop relies on this to keep the file bounded.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")
        self._fh.write("n\tseed\tepoch\ttrain_mse\tmspe_f0\tmspe_f1\tintercept_se\n")
        self._fh.flush()

    def log(self, n, seed, epoch, train_mse, model, inputs_tr):
        ev = evaluate_curve(model, inputs_tr, seed)
        row = [str(n), str(seed), str(epoch), f"{train_mse:.6g}",
               *[f"{m:.6g}" for m in ev["mspe"]],
               f"{ev['intercept_se']:.6g}"]
        self._fh.write("\t".join(row) + "\n")
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def hp_for(n: int) -> dict:
    """HP defaults for a given training-set size — currently constant."""
    return dict(HP_DEFAULTS)


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
