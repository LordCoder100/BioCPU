"""nn.Linear — core BioCPU layer.

Computes the equilibrium of linear relaxation (Neumann series).
Parameters: W (input weights), optional M (symmetric lateral coupling).
"""
from __future__ import annotations

import numpy as np

from ...parameter import Parameter
from ..functional._functions import equilibrium, random_symmetric
from .module import Module


class SettleLinear(Module):
    """A layer whose output is the equilibrium of linear relaxation.

    Parameters
    ----------
    in_dim : input dimensionality.
    out_dim : output dimensionality.
    gamma : decay factor in (0, 1).
    k : number of Neumann series terms (ignored when coupling is off).
    coupling : lateral coupling scale (``None`` = no coupling, i.e. M = 0).
               Numerical value sets the spectral-norm scale of M.
    seed : RNG seed for weight initialization.
    """

    M: Parameter | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        gamma: float = 0.9,
        k: int = 3,
        coupling: float | None = None,
        seed: int = 0,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        W = rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(out_dim, in_dim))
        self.W = Parameter(W)

        if coupling is not None and coupling > 0:
            M = random_symmetric(out_dim, scale=coupling, gamma=gamma,
                                 seed=seed + 100)
            self.M = Parameter(M, requires_learning=False)
        else:
            self.M = None

        self.gamma = gamma
        self.k = k

    def settle(self, x, ctx=None):
        M = self.M.value if self.M is not None else None
        x_eq = equilibrium(self.W.value, x, M=M, gamma=self.gamma, k=self.k)
        if ctx is not None:
            ctx.record(self, np.atleast_2d(x), x_eq)
        return x_eq
