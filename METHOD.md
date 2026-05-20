# METHOD

## DGP (one line)

Two-component additive model with **concurvity**: $x_1, x_2$ correlated
via a Gaussian copula ($\rho = 0.7$, uniform marginals), $f_1$ a
high-frequency sinusoid (sample-bound), $f_2$ a smooth quadratic
(saturates fast). See `prepare.simulate` for the canonical form.

## Theory (stub)

**NAM model.** With k inputs $x = (x_1, \dots, x_k)$ and per-component
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

The cross-component coupling is entirely in $r_i$: every $\theta_k$
update reads a residual that depends on the current (mis-)fit of every
other $f_j$.

## Directions

The optimization frontier lives in $f_1(x_1) = \sin(8\pi x_1)$ (high-frequency,
sample-bound) compounded with concurvity ($x_1 \perp\!\!\!\not\perp x_2$):
naive joint SGD on the residual lets $\hat f_1$ and $\hat f_2$ absorb each
other's signal. $f_2(x_2) = x_2^2 - 1/3$ saturates almost immediately under
any sensible routine. Promising avenues to explore:

1. **Backfitting / cyclic coordinate updates.** Fit $f_k$ against the
   *conditional* residual $y - \sum_{j \neq k} f_j(x_j)$ in a cyclic sweep —
   the exact-fit analogue of orthogonalizing the score with respect to
   the other components.
2. **Neyman / DML-style orthogonalization.** Project
   $\nabla_{\theta_k} L$ onto the subspace orthogonal to the span of the
   other components' score directions at the current iterate. First-order
   version of (1); the property that gives debiased ML its $\sqrt{n}$ rate.
3. **Per-component step sizes / schedules.** $f_2$ saturates in a handful
   of epochs; $f_1$ is sample-complexity-bound. A single global $\eta$
   couples them — try $\eta_k$ that adapt to per-component loss curvature
   or gradient norm.
4. **Component-wise functional momentum.** EMA of $f_k(x_k)$ on the
   training set; train against the EMA-residual instead of the live one.
   Damps the cross-component coupling that plain SGD's residual creates.
5. **Frequency / smoothness regularization on $f_1$.** Spectral-norm
   caps, Fourier-feature input encodings, or output L2 on $f_k''$. Not
   "the answer" but useful baselines to beat.
