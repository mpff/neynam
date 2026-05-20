import torch
import torch.nn as nn


class NAM(nn.Module):
    """Neural Additive Model with k generic backbones over k inputs.

    Prediction:
        y_hat = intercept + Σ_k (f_k(x_k) - μ_k)

    Each backbone f_k is an arbitrary nn.Module mapping its own input x_k
    to a per-sample scalar. μ_k is a centering offset (buffer; updated by
    `center()` and idempotent).
    """

    def __init__(self, backbones):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.intercept = nn.Parameter(torch.zeros(()))
        self.register_buffer("means", torch.zeros(len(backbones)))

    @property
    def k(self):
        return len(self.backbones)

    def component(self, idx, x):
        raw = self.backbones[idx](x)
        if raw.dim() > 1:
            raw = raw.squeeze(-1)
        return raw - self.means[idx]

    def forward(self, inputs):
        if len(inputs) != self.k:
            raise ValueError(f"expected {self.k} inputs, got {len(inputs)}")
        eta = self.intercept.expand(inputs[0].shape[0])
        for idx, x in enumerate(inputs):
            eta = eta + self.component(idx, x)
        return eta

    @torch.no_grad()
    def center(self, inputs):
        """Absorb each component's empirical mean into the intercept.

        Predictions are unchanged. Idempotent.
        """
        for idx, x in enumerate(inputs):
            shift = self.component(idx, x).mean()
            self.means[idx] += shift
            self.intercept.data += shift
