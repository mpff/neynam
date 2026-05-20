# METHOD

## DGP (one line)

Two-component additive model with **concurvity**: $x_1, x_2$ correlated
via a Gaussian copula ($\rho = 0.7$, uniform marginals), $f_1$ a
high-frequency sinusoid (sample-bound), $f_2$ a smooth quadratic
(saturates fast). See `prepare.simulate` for the canonical form.

## Theory (stub)

**NAM model.** With $K$ inputs $x = (x_1, \dots, x_K)$ and per-component
backbones $f_k(\cdot \,; \theta_k) : \mathbb{R} \to \mathbb{R}$:

$$
\hat y(x; c, \theta) \;=\; c \;+\; \sum_{k=1}^K f_k(x_k; \theta_k).
$$

Centering is identifiability post-processing: we subtract
$\mu_k = \mathbb{E}[f_k(X_k)]$ from each component and add $\sum_k \mu_k$
to $c$. This leaves $\hat y$ unchanged and is idempotent.

**Loss.** Squared error on a batch $\mathcal{B}$:

$$
L(c, \theta; \mathcal{B})
\;=\; \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \big(y_i - \hat y(x_i)\big)^2.
$$

**Classic SGD update.** With learning rate $\eta$ and residual
$r_i = y_i - \hat y(x_i)$:

$$
\theta_k \;\leftarrow\; \theta_k \;+\; \frac{2\eta}{|\mathcal{B}|}
  \sum_{i \in \mathcal{B}} r_i \cdot \nabla_{\theta_k} f_k(x_{i,k}; \theta_k),
\qquad
c \;\leftarrow\; c \;+\; \frac{2\eta}{|\mathcal{B}|}
  \sum_{i \in \mathcal{B}} r_i.
$$

## Directions (keywords only)

- Backfitting / cyclic coordinate updates
- Neyman orthogonalization / DML
- Cross-fitting
- Per-component learning rates
- Functional momentum
- Smoothness / spectral regularization
- Muon
- Shampoo

## Baseline

Classic SGD + cosine LR + early-stop-on-val (HPs from `prepare.HP_DEFAULTS`:
lr=0.1, max_epochs=500, patience=100, batch=128), 30 trainings (5 seeds ×
6 n), test split.

|     n |        f1 MSPE (mean ± std) |        f2 MSPE (mean ± std) |          μ MSE (mean ± std) |
|------:|----------------------------:|----------------------------:|----------------------------:|
|   200 |         0.4755 ± 0.0095     |         0.0090 ± 0.0063     |         0.0029 ± 0.0035     |
|   400 |         0.4571 ± 0.0208     |         0.0060 ± 0.0022     |         0.0016 ± 0.0016     |
|   800 |         0.4114 ± 0.0205     |         0.0027 ± 0.0012     |         0.0007 ± 0.0007     |
|  1600 |         0.2916 ± 0.0398     |         0.0024 ± 0.0006     |         0.0006 ± 0.0008     |
|  3200 |         0.0603 ± 0.0771     |         0.0012 ± 0.0009     |         0.0001 ± 0.0001     |
|  6400 |         0.0075 ± 0.0058     |         0.0003 ± 0.0002     |         0.0000 ± 0.0000     |

**Headline:** `score = -1.791` (mean $\log_{10}$ MSPE across all
$(n, k, \text{seed})$, lower is better). Wall: 594 s on CPU.

(Table uses 1-indexed component labels to match the math; the raw
`train.py` console output prints them 0-indexed as `f0`, `f1`.)
