"""NAM training routine — EDIT THIS FILE during the autoresearch loop.

For each (n, seed) in `prepare.N_GRID × prepare.SEEDS`:
  1. draw a fresh training set,
  2. train a NAM with the current routine,
  3. center the components,
  4. evaluate per-component MSPE on a held-out draw of size `prepare.EVAL_N`.

Prints a per-(n, component) table and a single headline scalar:

    score: <mean log10 MSPE across all (n, k, seed)>

Lower is better. The autoresearch loop reads the `score:` line.
"""
from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from neynam import NAM, MultiInputDataset
from prepare import (
    N_GRID,
    SEEDS,
    aggregate,
    evaluate,
    format_table,
    hp_for,
    simulate,
)


def make_backbone():
    return nn.Sequential(
        nn.Linear(1, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )


def train_one(n: int, seed: int) -> NAM:
    hp = hp_for(n)
    torch.manual_seed(seed)
    inputs_tr, y_tr, _, _ = simulate(n, seed)
    loader = DataLoader(
        MultiInputDataset(inputs_tr, y_tr),
        batch_size=min(hp["batch_size"], n),
        shuffle=True,
    )
    model = NAM([make_backbone(), make_backbone()])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    for _ in range(hp["epochs"]):
        for xs, y in loader:
            opt.zero_grad()
            ((model(xs) - y) ** 2).mean().backward()
            opt.step()
    model.center(inputs_tr)
    return model


def main():
    t0 = time.time()
    results = []
    for n in N_GRID:
        for seed in SEEDS:
            model = train_one(n, seed)
            results.append({"n": n, "seed": seed, "mspe": evaluate(model, seed)})
    headline, by_n_k = aggregate(results)
    print(format_table(by_n_k))
    print()
    print(f"score: {headline:.6f}")
    print(f"# wall: {time.time() - t0:.1f}s   "
          f"({len(results)} trainings over {len(N_GRID)} n × {len(SEEDS)} seeds)")
    return headline


if __name__ == "__main__":
    main()
