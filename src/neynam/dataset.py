from torch.utils.data import Dataset


class MultiInputDataset(Dataset):
    """Dataset yielding ([x1_i, ..., xk_i], y_i) per sample.

    Default DataLoader collation turns this into ([X1, ..., Xk], y) per batch,
    which is exactly what `NAM.forward` consumes.
    """

    def __init__(self, inputs, y):
        n = y.shape[0]
        for i, x in enumerate(inputs):
            if x.shape[0] != n:
                raise ValueError(f"inputs[{i}] has {x.shape[0]} rows, y has {n}")
        self.inputs = list(inputs)
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return [x[idx] for x in self.inputs], self.y[idx]
