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

## Baseline

Classic SGD + cosine LR + early-stop-on-val (HPs from `prepare.HP_DEFAULTS`),
30 trainings (5 seeds × 6 n), test split (`TEST_SEED_OFFSET = 20_000`).

|     n |        f1 MSPE (mean ± std) |        f2 MSPE (mean ± std) |          μ MSE (mean ± std) |
|------:|----------------------------:|----------------------------:|----------------------------:|
|   200 |         0.4826 ± 0.0089     |         0.0098 ± 0.0060     |         0.0019 ± 0.0032     |
|   400 |         0.4820 ± 0.0068     |         0.0084 ± 0.0046     |         0.0014 ± 0.0021     |
|   800 |         0.4512 ± 0.0415     |         0.0032 ± 0.0013     |         0.0022 ± 0.0024     |
|  1600 |         0.2916 ± 0.0398     |         0.0024 ± 0.0006     |         0.0006 ± 0.0008     |
|  3200 |         0.1087 ± 0.1852     |         0.0012 ± 0.0009     |         0.0005 ± 0.0009     |
|  6400 |         0.0075 ± 0.0058     |         0.0003 ± 0.0002     |         0.0000 ± 0.0000     |

**Headline:** `score = -1.759` (mean log10 MSPE across all $(n, k, \text{seed})$,
lower is better). Wall: 447 s on CPU (no GPU in this container).

The hard component $f_1(x_1) = \sin(8\pi x_1)$ shows the expected
sample-complexity ramp: variance ~0.5 essentially un-recovered at $n = 200, 400$
(the model collapses to the constant); a clean phase transition between
$n = 1600$ and $n = 3200$; near-perfect recovery by $n = 6400$. The smooth
$f_2$ is at noise floor everywhere.

The wide std at $n = 3200$ (0.19 on a 0.11 mean) reflects exactly the
transition regime — some seeds converge, some don't. This is where new
routines have the most visible room to improve the headline.

---

## Appendix: optimization geometry of NAMs

### A.1 Population objective and the additive fixed point

Write $X = (X_1, \dots, X_K)$ with joint law $P$ and define the Hilbert
space $\mathcal{H} = L^2(P_X)$. Let
$\mathcal{H}_k = \{ g \in L^2(P_{X_k}) : \mathbb{E}[g(X_k)] = 0 \}$ be the
mean-zero functions of $X_k$ alone, embedded in $\mathcal{H}$. The
**additive subspace** is

$$
\mathcal{H}_{\mathrm{add}} \;=\; \mathbb{R} \,\oplus\, \mathcal{H}_1 \oplus \cdots \oplus \mathcal{H}_K
\;\;\subseteq\;\; \mathcal{H},
$$

which is a closed linear subspace. The population risk
$L(c, f) = \mathbb{E}\!\left[(Y - c - \sum_k f_k(X_k))^2\right]$ is
minimized by the orthogonal projection
$\Pi_{\mathrm{add}} Y = c^\star + \sum_k f_k^\star$, unique whenever
$\mathcal{H}_{\mathrm{add}}$ is closed in $\mathcal{H}$ — equivalently,
whenever no $\mathcal{H}_k$ is a perfect linear function of the others
(Stone, 1985; Buja–Hastie–Tibshirani, 1989).

The Gâteaux derivative of $L$ at $(c, f)$ along $h_k \in \mathcal{H}_k$ is

$$
DL[(c, f)](h_k) \;=\; -\,2 \,\mathbb{E}\big[ r(X) \, h_k(X_k) \big],
\qquad
r(X) \;=\; Y - c - \sum_j f_j(X_j).
$$

Vanishing for all $h_k$ is equivalent to the **normal equations**

$$
\mathbb{E}[\,r(X) \mid X_k\,] \;=\; 0 \quad \forall k. \tag{$\star$}
$$

Equation $(\star)$ is the fixed point shared by all three algorithms
below.

### A.2 Three algorithms for $(\star)$

Let $P_k : \mathcal{H} \to \mathcal{H}_k$ denote the conditional-expectation
projector $P_k g = \mathbb{E}[g \mid X_k] - \mathbb{E}[g]$.

**(i) Backfitting.** Cyclic exact coordinate descent on $\mathcal{H}_{\mathrm{add}}$:

$$
f_k \;\leftarrow\; P_k \!\left( Y - c - \sum_{j \neq k} f_j(X_j) \right).
$$

Each sweep is an alternating-projection step; convergence to $(\star)$
is geometric with rate
$\rho_{\mathrm{back}} = \prod_{k} \|P_k P_{k-1} \cdots P_1\| < 1$
whenever the $\mathcal{H}_k$ are not collinear (Hastie–Tibshirani, 1990).

**(ii) Classic SGD on a parametric NAM.** With
$f_k = f_k(\cdot; \theta_k)$:

$$
\theta_k \;\leftarrow\; \theta_k \;+\; \frac{2\eta}{|\mathcal{B}|}
\sum_{i \in \mathcal{B}} r_i \, \nabla_{\theta_k} f_k(X_{i,k}; \theta_k).
$$

The gradient direction depends on the *current* residual $r_i$, which
mixes the errors of all other components. Define the nuisance error
$\delta_j = f_j - f_j^\star$ — then for $j \neq k$,

$$
\mathbb{E}\big[r \nabla_{\theta_k} f_k \mid \delta\big]
\;=\; \underbrace{\mathbb{E}\!\left[(Y - c^\star - \textstyle\sum_l f_l^\star) \nabla_{\theta_k} f_k\right]}_{=\,0}
\;-\; \sum_{j \neq k} \mathbb{E}\big[ \delta_j(X_j) \, \nabla_{\theta_k} f_k(X_k) \big].
$$

The second term is the **plug-in bias**: order $O(\|\delta\|)$ whenever
$\mathbb{E}[\delta_j(X_j) \, \nabla_{\theta_k} f_k(X_k)] \neq 0$, which
happens exactly under concurvity ($X_j \not\!\perp X_k$). Plain SGD does
not "see" $(\star)$ as the unique attractor.

**(iii) Neyman-orthogonal score.** Replace the per-component score
direction by its projection onto the orthocomplement of the nuisance
tangent space. For NAMs the nuisance tangent is spanned by gradients of
the other components; the orthogonalized residual is

$$
\tilde r \;=\; r \;-\; \Pi_{\,\mathrm{span}\{ \nabla_{\theta_j} f_j : j \neq k\}} \, r,
$$

and the update uses $\tilde r$ in place of $r$. This yields a score whose
first-order bias in $\delta$ *vanishes*:

$$
\mathbb{E}\big[\tilde r \, \nabla_{\theta_k} f_k \mid \delta\big]
\;=\; O\big(\|\delta\|^2\big),
$$

the **Neyman near-orthogonality** that gives debiased machine learning
its $\sqrt{n}$ rate even with $n^{-1/4}$ nuisance estimation
(Chernozhukov et al., 2018).

### A.3 At the fixed point all three coincide

At $(\star)$ the residual $r^\star$ is orthogonal to every
$\mathcal{H}_j$, hence in particular to every $\nabla_{\theta_j} f_j$ in
its tangent. Then $\tilde r^\star = r^\star$, the conditional projection
$P_k r^\star = 0$, and the SGD gradient equals the backfitting update
equals the Neyman score — they differ only in the **transient**.

### A.4 Concurvity and the transient

Concurvity controls how aggressively the three transients diverge.
Quantitatively: if
$\rho := \max_{k \neq j} \|P_k P_j\| \in [0, 1)$,
then backfitting's per-sweep contraction is at least $1 - (1 - \rho)^K$,
and the SGD plug-in bias scales like $\rho \, \|\delta\|$. For our DGP
($\rho_{\text{Pearson}}(X_1, X_2) \approx 0.68$ under a Gaussian copula,
$f_1$ high-frequency so $\|\delta_1\|$ stays large for many steps), the
transient is long enough that the three algorithms produce visibly
different per-epoch curves at small $n$ — which is exactly the regime
where the autoresearch loop has room to find improvements.

### A.5 What the loop is searching for

A per-step routine $R_k(\nabla_{\theta_k} L_n, \text{state})$ such that

1. **At the fixed point**: $R_k$ reduces to SGD (no overhead once the
   nuisance is well-fit).
2. **Far from the fixed point**: $R_k$'s expected update has bias
   $O(\|\delta\|^2)$ instead of $O(\|\delta\|)$ — the Neyman property.
3. **Per-step cost**: $O(1)$ extra over SGD; no inner backfitting loop
   per step.

The directions in §"Directions" each correspond to a specific way to
approximate (iii) cheaply: cyclic mini-backfitting trades (3) for (2);
gradient orthogonalization is (iii) literally but needs cheap basis
estimation; functional momentum approximates the conditional-expectation
projector by an EMA of $f_k$ outputs.

### A.6 References

- Buja, A., Hastie, T., Tibshirani, R. (1989). *Linear smoothers and
  additive models.* Ann. Statist. 17(2): 453–510.
- Chernozhukov, V. et al. (2018). *Double/debiased machine learning for
  treatment and structural parameters.* Econom. J. 21(1): C1–C68.
- Hastie, T., Tibshirani, R. (1990). *Generalized Additive Models.*
  Chapman & Hall.
- Stone, C. J. (1985). *Additive regression and other nonparametric
  models.* Ann. Statist. 13(2): 689–705.
