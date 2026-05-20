# neynam

Minimal Neural Additive Model (NAM) as a starting point for developing a new
optimization routine.

A NAM here is `y_hat = intercept + Σ_k (f_k(x_k) - μ_k)`, where each `f_k` is
an arbitrary `nn.Module` backbone supplied at init time. Centering is a
post-hoc, prediction-preserving, idempotent op.

```python
from torch import nn
from neynam import NAM, MultiInputDataset
from torch.utils.data import DataLoader

backbones = [nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1)),
             nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))]
model = NAM(backbones)

loader = DataLoader(MultiInputDataset([x1, x2], y), batch_size=128, shuffle=True)
# train ... then:
model.center([x1, x2])
```

## Tests

```bash
pytest tests/
```
