"""MNIST integration test — verifies the full BioCPU training pipeline.

Downloads MNIST via torchvision, trains a 3-layer model with optim.Local,
and checks that accuracy exceeds a baseline threshold.

Run:  pytest tests/test_mnist.py -v -s
"""

from __future__ import annotations

import numpy as np
from torchvision import datasets  # pyrefly: ignore

import biocpu
import biocpu.nn as nn

# ── helpers ──────────────────────────────────────────────────────


def load_mnist():
    """Load MNIST via torchvision → numpy float64 arrays."""

    data_dir = "./data"
    train = datasets.MNIST(data_dir, train=True, download=True)
    test = datasets.MNIST(data_dir, train=False, download=True)

    X_train = train.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y_train = train.targets.numpy()
    X_test = test.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
    y_test = test.targets.numpy()

    return X_train, y_train, X_test, y_test


# ── tests ────────────────────────────────────────────────────────


def test_mnist_m0_baseline():
    """M=0 (no coupling), 3 layers, 10 epochs → accuracy > 90%."""
    X_train, y_train, X_test, y_test = load_mnist()

    model = nn.Sequential(
        nn.SettleLinear(784, 256, gamma=0.9, k=3, seed=42),
        nn.SettleLinear(256, 128, gamma=0.9, k=3, seed=43),
        nn.SettleLinear(128, 10, seed=44),
        kwta_frac=0.1,
    )

    learner = biocpu.optim.Local(model, n_classes=10, lr=0.05, beta=0.5)
    learner.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=15,
        batch=128,
        seed=0,
        verbose=True,
    )

    final_val = learner.accuracy(X_test, y_test)
    print(f"\n  -> Final test accuracy: {final_val:.4f}")

    assert final_val > 0.94, f"Expected > 94% accuracy on MNIST, got {final_val:.2%}"


def test_mnist_with_coupling():
    """Symmetric M coupling (scale=0.5), 3 layers, 10 epochs → accuracy > 90%."""
    X_train, y_train, X_test, y_test = load_mnist()

    model = nn.Sequential(
        nn.SettleLinear(784, 256, gamma=0.9, k=5, coupling=0.5, seed=42),
        nn.SettleLinear(256, 128, gamma=0.9, k=5, coupling=0.5, seed=43),
        nn.SettleLinear(128, 10, seed=44),
        kwta_frac=0.1,
    )

    learner = biocpu.optim.Local(model, n_classes=10, lr=0.05, beta=0.5)
    learner.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=15,
        batch=128,
        seed=0,
        verbose=True,
    )

    final_val = learner.accuracy(X_test, y_test)
    print(f"\n  -> Final test accuracy (coupled): {final_val:.4f}")

    assert final_val > 0.94, f"Expected > 94% accuracy with coupling, got {final_val:.2%}"


def test_phase_context_populated():
    """PhaseContext collects one trace per SettleLinear layer."""
    X_train, _, _, _ = load_mnist()

    model = nn.Sequential(
        nn.SettleLinear(784, 128, seed=42),
        nn.SettleLinear(128, 10, seed=43),
        kwta_frac=0.1,
    )

    ctx = biocpu.PhaseContext()
    logits = model(X_train[:8], ctx=ctx)

    assert len(ctx) == 2, f"Expected 2 traces, got {len(ctx)}"
    assert ctx[0].equilibrium.shape == (8, 128)
    assert ctx[1].equilibrium.shape == (8, 10)
    assert logits.shape == (8, 10)


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING FULL MNIST INTEGRATION TESTING AND TRAINING")
    print("=" * 60)

    print("\n[1/3] Running baseline M=0 training (no lateral coupling)...")
    test_mnist_m0_baseline()

    print("\n[2/3] Running training with lateral coupling (M scale = 0.5)...")
    test_mnist_with_coupling()

    print("\n[3/3] Checking PhaseContext traces...")
    test_phase_context_populated()
    print("PhaseContext OK.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
