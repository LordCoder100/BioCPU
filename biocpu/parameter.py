from __future__ import annotations

import numpy as np


class Parameter:
    def __init__(self, data, requires_learning=True):
        self.accumulator = np.asarray(data, dtype=np.float64).copy()
        self.requires_learning = requires_learning

    @property
    def value(self):
        return self.accumulator

    def accumulate(self, delta, lr):
        if self.requires_learning:
            self.accumulator += lr * delta

    @property
    def shape(self):
        return self.accumulator.shape

    def __array__(self, dtype=None):
        v = self.value
        return v if dtype is None else v.astype(dtype)

    def __repr__(self):
        frozen = "" if self.requires_learning else ", frozen"
        return f"Parameter(shape={self.shape}, dtype=float64{frozen})"
