"""Structured context objects for BioCPU's two-phase learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .nn.modules.linear import SettleLinear


@dataclass
class LayerTrace:
    """Snapshot of one layer's free-phase equilibrium."""

    module: SettleLinear  # the SettleLinear that produced this trace
    input: np.ndarray  # (B, in_dim) — input received by the layer
    equilibrium: np.ndarray  # (B, out_dim) — settled output state


@dataclass
class PhaseContext:
    """Collects per-layer traces during ``model.settle()``."""

    traces: list[LayerTrace] = field(default_factory=list)

    def record(self, module: SettleLinear, inp: np.ndarray, eq: np.ndarray) -> None:
        self.traces.append(LayerTrace(module, np.atleast_2d(inp), eq))

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> LayerTrace:
        return self.traces[idx]
