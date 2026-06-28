"""optim.Local — local learning rule for BioCPU.

Instead of ``loss.backward() + optimizer.step()``:
  1) Free phase — settle the full stack, collect per-layer traces;
  2) Target signal at the output: δ_top = β(target − logits);
  3) Project the target downward through transposed weights,
     masked by k-WTA (NOT a gradient chain — k-WTA is non-differentiable);
  4) Local weight update: ΔW ∝ δ ⊗ input, accumulated in float.

The model only knows ``settle()``.  All learning logic lives here so
that alternative rules (feedback alignment, predictive coding) can be
swapped in by replacing the learner.
"""
from __future__ import annotations

import numpy as np

from ..context import PhaseContext
from ..nn.functional._functions import k_wta_mask, one_hot


class Local:
    """Local projected-target learning rule.

    Parameters
    ----------
    model : ``nn.Sequential`` (or any Module with ``settle(x, ctx)``).
    n_classes : number of output classes.
    beta : target nudging strength.
    lr : learning rate.
    """

    def __init__(self, model, n_classes: int, beta: float = 0.5,
                 lr: float = 0.05, feedback: str = "symmetric"):
        self.model = model
        self.n_classes = n_classes
        self.beta = beta
        self.lr = lr
        self.feedback = feedback
        self.B_matrices: dict = {}

    # ── core step ────────────────────────────────────────────────

    def step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """One training step on a batch.  Returns monitoring MSE."""
        t = one_hot(y_batch, self.n_classes)

        # 1) free phase — collect per-layer traces
        ctx = PhaseContext()
        logits = self.model.settle(X_batch, ctx=ctx)
        traces = ctx.traces
        last = len(traces) - 1

        # Initialize B_matrices on the fly if needed for Feedback Alignment (FA)
        if self.feedback == "fa" and not self.B_matrices:
            rng = np.random.default_rng(42)
            for trace in traces:
                out_dim, in_dim = trace.module.W.value.shape
                B = rng.normal(0.0, np.sqrt(1.0 / in_dim), size=(out_dim, in_dim))
                self.B_matrices[trace.module] = B

        # 2) target signal at output
        deltas: list[np.ndarray | None] = [None] * len(traces)
        deltas[last] = self.beta * (t - logits)

        # 3) project target downward (variant a)
        kfrac = getattr(self.model, "kwta_frac", 1.0)
        for li in range(last - 1, -1, -1):
            d_next = deltas[li + 1]
            assert d_next is not None, f"delta missing for layer {li + 1}"

            if self.feedback == "fa":
                W_next = self.B_matrices[traces[li + 1].module]
            else:
                W_next = traces[li + 1].module.W.value

            d = d_next @ W_next
            x_li = traces[li].equilibrium
            k = max(1, int(kfrac * x_li.shape[1]))
            mask = k_wta_mask(x_li, k)
            deltas[li] = np.where(mask, d, 0.0)

        # 4) local weight update → float accumulator
        for li, trace in enumerate(traces):
            u_in = trace.input
            delta = deltas[li]
            assert delta is not None, f"delta missing for layer {li}"
            dW = (delta.T @ u_in) / u_in.shape[0]
            trace.module.W.accumulate(dW, self.lr)

        return float(np.mean((logits - t) ** 2))

    # ── convenience ──────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 15,
        batch: int = 128,
        seed: int = 0,
        verbose: bool = True,
    ) -> list[tuple[float, float | None]]:
        """Train for *epochs* epochs.  Returns per-epoch ``(train_acc, val_acc)``."""
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history: list[tuple[float, float | None]] = []

        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                b = perm[s : s + batch]
                self.step(X[b], y[b])

            if verbose:
                tr = self.accuracy(X[:5000], y[:5000])
                msg = f"  epoch {ep + 1:2d}/{epochs}  train(5k)={tr:.4f}"
                if X_val is not None and y_val is not None:
                    va = self.accuracy(X_val, y_val)
                    msg += f"  val={va:.4f}"
                    history.append((tr, va))
                else:
                    history.append((tr, None))
                print(msg)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.model.settle(X, ctx=None)
        return np.argmax(logits, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
