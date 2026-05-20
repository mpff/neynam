# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project: neynam

Minimal Neural Additive Model — a clean testbed for developing a new
optimization routine for NAMs via autoresearch. Cloned from `cocodeel` but
stripped to the bare essentials.

## Scope

- `src/neynam/model.py` — `NAM`: k generic backbones over k inputs, centering.
- `src/neynam/dataset.py` — `MultiInputDataset`: yields `([x1, ..., xk], y)`.
- `tests/test_nam_consistency.py` — synthetic 2-component DGP, recovers
  `f_k` shape under mild thresholds; checks centering invariants.

Out of scope (for now): control variables, post-hoc backfitting, GLM links,
schedulers, AMP, hyperparameter search, simulation grids, R figures.

## Commands

```bash
pytest tests/
```

## Conventions

- PyTorch only; no R/rpy2.
- Inputs to `NAM.forward` are a list/tuple of k tensors (one per backbone).
- Centering is a buffer-update op, not a parameter — it travels with
  `state_dict` and does not change predictions.
