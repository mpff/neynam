# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project: neynam

Minimal Neural Additive Model — a clean testbed for developing a new
optimization routine for NAMs via autoresearch. Cloned from `cocodeel` but
stripped to the bare essentials.

## Scope

- `src/neynam/model.py` — `NAM`: k generic backbones over k inputs, centering.
- `src/neynam/dataset.py` — `MultiInputDataset`: yields `([x1, ..., xk], y)`.
- `prepare.py` — concurvity DGP, val/test split, `evaluate`, `CurveLogger`,
  `aggregate`, `format_table`, `HP_DEFAULTS`. **Read-only during the
  autoresearch loop.**
- `train.py` — the file the loop edits: backbone, optimizer, training loop,
  early stopping. Prints the `score:` line the loop reads.
- `tests/test_nam_consistency.py` — sanity test for the NAM mechanics
  (independent uncorrelated DGP); not a benchmark.
- `METHOD.md` — DGP one-liner, theory stub, baseline numbers.

## Commands

```bash
pytest tests/
python train.py        # baseline / loop run; writes runs/curves.tsv
```

## Conventions

- PyTorch only; no R/rpy2.
- Inputs to `NAM.forward` are a list/tuple of k tensors (one per backbone).
- Centering is a buffer-update op, not a parameter — it travels with
  `state_dict` and does not change predictions.
