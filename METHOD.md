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

To be regenerated under the current `HP_DEFAULTS` (classic SGD, cosine LR,
early-stop-on-val with patience). Headline metric: mean $\log_{10}$ MSPE
across all $(n, k, \text{seed})$, lower is better. See `train.py` output
or rerun:

```bash
python train.py
```
