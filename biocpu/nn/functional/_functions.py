"""Low-level functional operations for BioCPU."""

from __future__ import annotations

import numpy as np


def equilibrium(
    W: np.ndarray,
    u: np.ndarray,
    M: np.ndarray | None = None,
    gamma: float = 0.9,
    k: int = 5,
) -> np.ndarray:
    """Compute the equilibrium of linear relaxation via truncated Neumann series.

        x* = (I − γM)⁻¹ (W u)  ≈  Σ_{j=0}^{k} γʲ Mʲ (W u)

    Evaluated with Horner's scheme (no stored powers of M).

    Parameters
    ----------
    W : (out, in) — input weights.
    u : (in,) | (B, in) — input activations.
    M : (out, out) | None — symmetric lateral coupling; None ≡ zero.
    gamma : decay in (0, 1).
    k : number of series terms (ignored when M is None).

    Returns
    -------
    (B, out) — equilibrium state.
    """
    u = np.atleast_2d(u)
    base = u @ W.T
    if M is None:
        return base
    acc = base.copy()
    y = base
    for _ in range(k):
        y = gamma * (y @ M.T)
        acc = acc + y
    return acc


def equilibrium_exact(W: np.ndarray, u: np.ndarray, M: np.ndarray, gamma: float) -> np.ndarray:
    """Exact equilibrium via matrix inverse (reference for tests)."""
    u = np.atleast_2d(u)
    inv = np.linalg.inv(np.eye(M.shape[0]) - gamma * M)
    return (u @ W.T) @ inv.T


def k_wta(x: np.ndarray, k: int) -> np.ndarray:
    """k-Winners-Take-All (per row): keep *k* largest activations, zero the rest."""
    if k >= x.shape[1]:
        return np.maximum(x, 0.0)
    thresh = np.partition(x, -k, axis=1)[:, -k][:, None]
    return np.where(x >= thresh, x, 0.0)


def k_wta_mask(x: np.ndarray, k: int) -> np.ndarray:
    """Binary mask of k-WTA winners (for projecting targets onto active neurons)."""
    if k >= x.shape[1]:
        return x > 0.0
    thresh = np.partition(x, -k, axis=1)[:, -k][:, None]
    return x >= thresh


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot encode integer labels."""
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def random_symmetric(n: int, scale: float, gamma: float, seed: int = 0) -> np.ndarray:
    """Generate symmetric M with guaranteed ``γ‖M‖₂ < 1``."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, size=(n, n))
    M = (A + A.T) / 2.0
    spec = np.max(np.abs(np.linalg.eigvalsh(M)))
    M = M * (scale / spec)
    if gamma * scale >= 1.0:
        raise ValueError(f"gamma*||M|| = {gamma * scale} must be < 1")
    return M
