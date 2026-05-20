"""One-time HP search for the BASELINE training routine.

Searches (lr, wd) on a small grid at each n in `prepare.N_GRID`, picks the
combo with the lowest mean MSPE across components and seeds, and writes
the per-n winners to `hp_defaults.json`.

This is *not* part of the autoresearch loop — it runs once before the loop
starts so the baseline has a fair starting point. The agent is free to
override these HPs in `train.py` for any experiment.
"""
from __future__ import annotations

import itertools
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from neynam import NAM, MultiInputDataset
from prepare import HP_PATH, N_GRID, SEEDS, simulate, evaluate

LR_GRID = [1e-3, 3e-3, 1e-2]
WD_GRID = [0.0, 1e-4]
EPOCHS = 100
BATCH_SIZE = 128
SEARCH_SEEDS = SEEDS[:3]


def _backbone():
    return nn.Sequential(
        nn.Linear(1, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )


def _train(n, seed, lr, wd):
    torch.manual_seed(seed)
    inputs_tr, y_tr, _, _ = simulate(n, seed)
    loader = DataLoader(
        MultiInputDataset(inputs_tr, y_tr),
        batch_size=min(BATCH_SIZE, n),
        shuffle=True,
    )
    model = NAM([_backbone(), _backbone()])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(EPOCHS):
        for xs, y in loader:
            opt.zero_grad()
            ((model(xs) - y) ** 2).mean().backward()
            opt.step()
    model.center(inputs_tr)
    return model


def main():
    t0 = time.time()
    best: dict[str, dict] = {}
    for n in N_GRID:
        rows = []
        for lr, wd in itertools.product(LR_GRID, WD_GRID):
            seed_means = []
            for seed in SEARCH_SEEDS:
                model = _train(n, seed, lr, wd)
                mspe = evaluate(model, seed)["mspe"]
                seed_means.append(sum(mspe) / len(mspe))
            score = sum(seed_means) / len(seed_means)
            rows.append(((lr, wd), score))
            print(f"  n={n:5d}  lr={lr:.0e}  wd={wd:.0e}  mspe={score:.4f}")
        (best_lr, best_wd), best_score = min(rows, key=lambda r: r[1])
        best[str(n)] = {
            "lr": best_lr,
            "wd": best_wd,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }
        print(f"  -> n={n}: lr={best_lr}, wd={best_wd}  (mspe={best_score:.4f})")
        print()
    HP_PATH.write_text(json.dumps(best, indent=2) + "\n")
    print(f"wrote {HP_PATH}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
