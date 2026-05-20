"""NAM training routine — EDIT THIS FILE during the autoresearch loop.

For each (n, seed) in `prepare.N_GRID × prepare.SEEDS`:
  1. draw a fresh training set + a fresh validation draw (VAL split),
  2. train a NAM with classic SGD + cosine LR; early-stop on val y-MSE
     with patience; restore best-val weights,
  3. center the components,
  4. evaluate per-component MSPE on a *test* draw (disjoint seed offset).

Prints a per-(n, component) table and a single headline scalar:

    score: <mean log10 MSPE across all (n, k, seed)>

Lower is better. The autoresearch loop reads the `score:` line.

Val and test draws use disjoint seed offsets so HPs cannot leak into the
reported score.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from neynam import NAM, MultiInputDataset
from prepare import (
    CurveLogger,
    DEVICE,
    EVAL_N,
    N_GRID,
    SEEDS,
    VAL_SEED_OFFSET,
    aggregate,
    evaluate,
    format_table,
    hp_for,
    simulate,
)

CURVE_LOG_PATH = Path(__file__).parent / "runs" / "curves.tsv"


def make_backbone():
    return nn.Sequential(
        nn.Linear(1, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )


def train_one(n: int, seed: int, logger: CurveLogger | None = None) -> NAM:
    hp = hp_for(n)
    torch.manual_seed(seed)
    inputs_tr, y_tr, _, _ = simulate(n, seed)
    inputs_val, y_val, _, _ = simulate(EVAL_N, seed + VAL_SEED_OFFSET)
    loader = DataLoader(
        MultiInputDataset(inputs_tr, y_tr),
        batch_size=min(hp["batch_size"], n),
        shuffle=True,
    )
    model = NAM([make_backbone(), make_backbone()]).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["max_epochs"])

    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(hp["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xs, y in loader:
            opt.zero_grad()
            loss = ((model(xs) - y) ** 2).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = ((model(inputs_val) - y_val) ** 2).mean().item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if logger is not None:
            logger.log(n, seed, epoch, epoch_loss / n_batches, model, inputs_tr)

        if bad >= hp["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.center(inputs_tr)
    return model


def main():
    t0 = time.time()
    results = []
    with CurveLogger(CURVE_LOG_PATH) as logger:
        for n in N_GRID:
            for seed in SEEDS:
                model = train_one(n, seed, logger=logger)
                results.append({"n": n, "seed": seed,
                                **evaluate(model, seed, split="test")})
    headline, mspe_by_n_k, intercept_by_n = aggregate(results)
    print(format_table(mspe_by_n_k, intercept_by_n))
    print()
    print(f"score: {headline:.6f}")
    print(f"# device: {DEVICE}")
    print(f"# wall: {time.time() - t0:.1f}s   "
          f"({len(results)} trainings over {len(N_GRID)} n × {len(SEEDS)} seeds)")
    print(f"# curves: {CURVE_LOG_PATH}")
    return headline


if __name__ == "__main__":
    main()
