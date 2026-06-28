"""Integration test for logical operators (AND, OR, XOR).

Verifies that BioCPU models can learn both linearly separable (AND, OR)
and non-linearly separable (XOR) boolean functions.

Run:  pytest tests/test_logic.py -v
"""
from __future__ import annotations

import numpy as np

import biocpu
import biocpu.nn as nn


def train_logic_gate(
    gate_name: str,
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 8,
    epochs: int = 100,
    lr: float = 0.1,
    beta: float = 0.5,
    seed: int = 42,
) -> float:
    """Helper to train a network on a specific logic gate dataset."""
    # We repeat the 4 items to create a larger training batch size of 32
    # to stabilize local target updates.
    X_train = np.tile(X, (32, 1))
    y_train = np.tile(y, (32,))

    # For XOR we need a hidden layer; for AND/OR we could use a single layer,
    # but we use a general 2-layer architecture to test both cases.
    model = nn.Sequential(
        nn.SettleLinear(2, hidden_dim, gamma=0.9, k=3, seed=seed),
        nn.SettleLinear(hidden_dim, 2, seed=seed + 1),
        kwta_frac=0.5,  # k = 4 active hidden units
    )

    learner = biocpu.optim.Local(model, n_classes=2, lr=lr, beta=beta)

    # Train
    learner.fit(
        X_train, y_train,
        epochs=epochs,
        batch=16,
        seed=seed,
        verbose=False,
    )

    # Evaluate on the original 4 combinations
    predictions = learner.predict(X)
    acc = float(np.mean(predictions == y))
    print(f"Gate {gate_name} predictions: {predictions} (Target: {y}) | Acc: {acc:.2%}")
    return acc


def test_logic_gates():
    # Input data: 4 combinations of two boolean variables
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float64)

    # Gate targets
    gates = {
        "AND": np.array([0, 0, 0, 1], dtype=np.int64),
        "OR":  np.array([0, 1, 1, 1], dtype=np.int64),
        "XOR": np.array([0, 1, 1, 0], dtype=np.int64),
    }

    for name, y in gates.items():
        # XOR is harder, so we give it slightly more epochs to converge
        epochs = 150 if name == "XOR" else 80
        acc = train_logic_gate(name, X, y, epochs=epochs, seed=42)
        assert acc == 1.0, f"Failed to learn {name} gate (Accuracy: {acc:.2%})"


def test_nested_noisy_logic():
    """Tests a complex nested logical expression with input noise:
    f(A, B, C) = (A XOR B) AND (B OR C).
    """
    # 8 combinations of 3 variables
    X_clean = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float64)

    # Truth table for f(A, B, C) = (A XOR B) AND (B OR C)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.int64)

    # Train dataset with Gaussian noise added to inputs
    rng = np.random.default_rng(42)
    X_train = np.tile(X_clean, (64, 1))
    y_train = np.tile(y, (64,))
    X_train += rng.normal(0, 0.15, size=X_train.shape)

    # Build model (3 inputs -> 16 hidden with k-WTA -> 2 outputs)
    model = nn.Sequential(
        nn.SettleLinear(3, 16, gamma=0.9, k=3, seed=42),
        nn.SettleLinear(16, 2, seed=43),
        kwta_frac=0.5,  # k = 8 active hidden units
    )

    learner = biocpu.optim.Local(model, n_classes=2, lr=0.08, beta=0.5)

    # Train with noise
    learner.fit(X_train, y_train, epochs=200, batch=16, seed=0, verbose=False)

    # Generalization test: Evaluate on clean (unseen noise-free) inputs
    predictions = learner.predict(X_clean)
    acc = float(np.mean(predictions == y))
    print(f"\nNested Noisy Logic - Predictions: {predictions} (Target: {y})")
    print(f"Generalization Accuracy on clean inputs: {acc:.2%}")

    assert acc == 1.0, f"Failed to generalize on nested noisy logic (Accuracy: {acc:.2%})"


def test_high_dimensional_logic_generalization():
    """Generates 2000 synthetic samples with 10 boolean features and evaluates
    generalization on a complex nested logic function:
    y = ((x0 != x1) & (x2 | x3)) ^ (x4 == 1)
    """
    rng = np.random.default_rng(42)
    # Generate 2000 samples of 10-dimensional binary variables
    X_bits = rng.integers(0, 2, size=(2000, 10))

    y = (
        ((X_bits[:, 0] != X_bits[:, 1]) & (X_bits[:, 2] | X_bits[:, 3]))
        ^ (X_bits[:, 4] == 1)
    ).astype(np.int64)

    # Inputs converted to float64
    X_clean = X_bits.astype(np.float64)

    # Split: 1600 training, 400 test
    X_train, X_test = X_clean[:1600], X_clean[1600:]
    y_train, y_test = y[:1600], y[1600:]

    # Add input noise during training to evaluate robustness
    X_train_noisy = X_train + rng.normal(0, 0.15, size=X_train.shape)

    # Model: 10 inputs -> 64 hidden -> 2 output classes
    model = nn.Sequential(
        nn.SettleLinear(10, 64, gamma=0.9, k=3, seed=42),
        nn.SettleLinear(64, 2, seed=43),
        kwta_frac=0.3,  # k = 19 active hidden units
    )

    learner = biocpu.optim.Local(model, n_classes=2, lr=0.05, beta=0.5)

    # Train
    learner.fit(X_train_noisy, y_train, epochs=150, batch=32, seed=0, verbose=False)

    # Generalization test on clean test data
    test_acc = learner.accuracy(X_test, y_test)
    print("\nHigh-Dimensional Logic Generalization:")
    print(f"Test Accuracy on clean unseen test data: {test_acc:.2%}")

    assert test_acc > 0.93, f"Expected generalization accuracy > 93%, got {test_acc:.2%}"


def test_feedback_alignment_xor():
    """Tests Feedback Alignment (FA) by training a model on XOR
    using a fixed random feedback matrix B (no weight transport).
    """
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)

    X_train = np.tile(X, (32, 1))
    y_train = np.tile(y, (32,))

    model = nn.Sequential(
        nn.SettleLinear(2, 8, gamma=0.9, k=3, seed=42),
        nn.SettleLinear(8, 2, seed=43),
        kwta_frac=0.5,
    )

    # Use feedback='fa' to enable Feedback Alignment
    learner = biocpu.optim.Local(model, n_classes=2, lr=0.15, beta=0.5, feedback="fa")

    learner.fit(X_train, y_train, epochs=250, batch=16, seed=0, verbose=False)

    predictions = learner.predict(X)
    acc = float(np.mean(predictions == y))
    print(f"\nFeedback Alignment XOR - Predictions: {predictions} (Target: {y}) | Acc: {acc:.2%}")
    assert acc == 1.0, f"Feedback Alignment failed to learn XOR (Accuracy: {acc:.2%})"


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING LOGIC TESTS")
    print("=" * 60)

    print("\n[1/4] Testing standard gates (AND, OR, XOR)...")
    test_logic_gates()

    print("\n[2/4] Testing nested noisy logic generalization (3D input)...")
    test_nested_noisy_logic()

    print("\n[3/4] Testing high-dimensional logic generalization (10D input, 2000 samples)...")
    test_high_dimensional_logic_generalization()

    print("\n[4/4] Testing Feedback Alignment (FA) on XOR...")
    test_feedback_alignment_xor()

    print("\n" + "=" * 60)
    print("ALL LOGIC TESTS PASSED!")
    print("=" * 60)



