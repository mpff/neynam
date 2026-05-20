"""One-time HP search for the BASELINE training routine.

Searches (lr, wd) on a small grid at each n in `prepare.N_GRID` using the
same recipe as `train.py` (classic SGD + cosine LR + early stopping on
val y-MSE), picks the combo with the lowest mean per-component MSPE on
the VAL split across seeds, and writes per-n winners to
`hp_defaults.json`.

This is *not* part of the autoresearch loop — it runs once before the
loop starts so the baseline has a fair starting point. The agent is free
to override these HPs in `train.py` for any experiment.
"""
from __future__ import annotations

import copy
import itertools
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from neynam import NAM, MultiInputDataset
from prepare import (
    DEVICE,
    EVAL_N,
    HP_PATH,
    N_GRID,
    SEEDS,
    VAL_SEED_OFFSET,
    evaluate,
    simulate,
)

# Classic SGD wants 10–100× higher lr than Adam — sweep an order of magnitude.
LR_GRID = [3e-2, 1e-1, 3e-1, 1.0]
WD_GRID = [0.0, 1e-4]
MAX_EPOCHS = 500
PATIENCE = 20
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
    inputs_val, y_val, _, _ = simulate(EVAL_N, seed + VAL_SEED_OFFSET)
    loader = DataLoader(
        MultiInputDataset(inputs_tr, y_tr),
        batch_size=min(BATCH_SIZE, n),
        shuffle=True,
    )
    model = NAM([_backbone(), _backbone()]).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_val = float("inf")
    best_state = None
    bad = 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xs, y in loader:
            opt.zero_grad()
            loss = ((model(xs) - y) ** 2).mean()
            loss.backward()
            opt.step()
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
        if bad >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
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
                mspe = evaluate(model, seed, split="val")["mspe"]
                seed_means.append(sum(mspe) / len(mspe))
            score = sum(seed_means) / len(seed_means)
            rows.append(((lr, wd), score))
            print(f"  n={n:5d}  lr={lr:.0e}  wd={wd:.0e}  mspe={score:.4f}")
        (best_lr, best_wd), best_score = min(rows, key=lambda r: r[1])
        best[str(n)] = {
            "lr": best_lr,
            "wd": best_wd,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
        }
        print(f"  -> n={n}: lr={best_lr}, wd={best_wd}  (mspe={best_score:.4f})")
        print()
    HP_PATH.write_text(json.dumps(best, indent=2) + "\n")
    print(f"wrote {HP_PATH}  ({time.time() - t0:.1f}s)   device={DEVICE}")


if __name__ == "__main__":
    main()
