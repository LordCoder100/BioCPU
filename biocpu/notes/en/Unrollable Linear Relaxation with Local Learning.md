# Unrollable Linear Relaxation with Local Learning

> Technical specification. Status: concept and core mathematics defined;
> no implementation yet.

---

## 0. Summary

A learning method in which a linear relaxation dynamic with multiplicative decay
admits two equivalent forms: a recurrent form and a form unrolled into a single
parallel integer pass. Learning is local, computed from the difference between two
equilibria, without backpropagation and without a forward pass in the usual sense.

General principle: the trade-off between recurrence, parallelism, and performance
is removed not by heuristics but by restricting the operation to a subclass
(linearity with decay) that possesses two equivalent computational forms.

---

## 1. Target Constraints

The goal is local learning that is efficient on a CPU: integer arithmetic
(int8/int16), vector instructions (e.g. AVX-512), and scaling across many-core
systems and clusters.

Properties the method deliberately avoids, with reasons:

| Approach | Reason for exclusion |
|---|---|
| Backpropagation | biologically implausible (weight-transport problem) |
| Spike-timing-dependent plasticity (STDP) | inefficient on CPU: per-synapse temporal traces, memory-bound; efficient only on neuromorphic hardware |
| Spiking neurons (integrate-and-fire) | require T time steps per example; the temporal axis does not parallelize |
| Forward-Forward | contains a forward pass |
| Equilibrium Propagation (full) | computationally expensive (nonlinear relaxation) and unstable |

General observation: approaches with dynamics in time are expensive on a CPU;
single-pass approaches contain a forward pass. The resolution is the equivalence of
two forms of one operation.

---

## 2. Design Decisions

**Hard invariants:**
1. Computation and weights are integer (weights int8, state int16); multiplication
   is replaced by addition/shifts (ternary weights {−1, 0, +1} or powers of two;
   integer MAC via vector instructions).
2. Learning is local: a synapse update depends only on its two endpoints.
3. No temporal axis — a fixed small number of integer steps per example.
4. Sparsity is activation-based (k-WTA between layers) — a source of speed,
   nonlinearity, and rapid series truncation.
5. Small updates accumulate in floating point (sparingly); weights remain integer.
   High precision is spent only where rounding would otherwise destroy information.
6. Scalability: batch parallelism without temporal dependencies.

**Deliberate trade-off:** target accuracy ~95–96% (feasibility), not 99%. This
permits a linear core with a closed-form equilibrium instead of expensive nonlinear
relaxation.

**Nonlinearity is placed between layers (k-WTA), not inside the relaxation.**
Within a layer the dynamic is linear (unrollable, stable); between layers, k-WTA
provides nonlinearity and sparsity. This keeps the relaxation linear and guarantees
a closed form.

---

## 3. Core Mathematics (single layer)

Notation: state `x`, input from the previous layer `u`, input weights `W`, latent
coupling among the layer's neurons `M`, decay `γ ∈ (0, 1)`.

### 3.1 Recurrent form
```
x[t+1] = γ · M · x[t] + W · u
```
The state evolves toward equilibrium; γ is decay (forgetting); `W·u` is the input
perturbation; `M` is latent interaction. At `M = 0` this is a pure chain.

### 3.2 Equilibrium
```
x* = γ M x* + W u   ⟹   x* = (I − γM)⁻¹ W u
```

### 3.3 Unrolled form (Neumann series)
```
(I − γM)⁻¹ = Σ_{j≥0} (γM)^j = I + γM + γ²M² + …
```
Truncated at k terms:
```
x* ≈ Σ_{j=0}^{k} γ^j M^j (W u)
```
Computed by Horner's scheme (without storing powers of M):
```
y ← W u
acc ← y
for j in 1..k:
    y ← γ · M · y          # one integer matvec (or additions/shifts)
    acc ← acc + y
x* ← acc
```
k terms = k repeated integer matvecs, parallel across the batch. Decay γ < 1 and
k-WTA sparsity truncate the series quickly: typically k ≈ 3–5.

### 3.4 Equivalence of forms
The recurrent form (3.1) and the truncated series (3.3) converge to the same `x*`.
This is one operation in two forms: conceptually the network settles in time;
computationally the equilibrium is reached in k integer steps.

---

## 4. Choice of M and Guarantees

The universality requirement splits into two independent properties:

- **Universal approximation** (covering an arbitrary task/modality) — provided by
  depth and the k-WTA nonlinearity between layers; holds for any M.
- **Convergence** (guaranteed absence of divergence) — provided by symmetry of M.

| M | Equilibrium | Guarantee |
|---|---|---|
| M = 0 | x* = Wu | trivial, always |
| M = Mᵀ (symmetric) | (I−γM)⁻¹Wu | proven unique equilibrium and convergence |
| M arbitrary | may not exist | none (oscillations possible) |

**Symmetric case.** When `M = Mᵀ`, an energy function exists:
```
E(x) = ½ xᵀ (I − γM) x − xᵀ W u
```
If `γ‖M‖₂ < 1`, the matrix `(I − γM)` is positive definite, E is convex, and there
is a unique global minimum to which relaxation always converges.

**Decision:** the core uses a symmetric M with the constraint `γ‖M‖₂ < 1`. The case
M = 0 is its special case (a possible starting point). Universality comes from depth
and k-WTA; stability comes from symmetry.

**Modality-specific structure lives before the core, not in M.** Input of any
modality enters the layer as a vector `u`. The structure of `W·u` defines the
modality: convolutional structure (local receptive fields) for images; positional
encoding for sequences. One universal core, different input front-ends.

---

## 5. Local Learning

A two-phase principle (in the spirit of equilibrium propagation), made cheap by the
fact that under linear relaxation both equilibria have closed forms.

**Two equilibria:**
- free: `x*_free` — equation (3.3) with input only;
- clamped: `x*_clamped` — output pulled toward the target with strength β.

**Local update rule:**
```
ΔW ∝ (x*_clamped ⊗ u) − (x*_free ⊗ u)
```
Each synapse uses only its two endpoints (post-synaptic state and pre-synaptic
input). There is no backward pass through the network, no chain rule, and no
explicit loss function.

ΔW accumulates in floating point; weights are quantized to integer when a
quantization step is crossed.

Compared with full equilibrium propagation: both phases are computed in closed
form, so the second phase requires no expensive nonlinear relaxation — removing both
cost and instability.

---

## 6. k-WTA (k-Winners-Take-All)

An activation-sparsity mechanism. For a layer of N neurons, only the k neurons with
the largest activations remain active; the rest are zeroed.

```
input:  N activations
output: k largest kept, (N − k) zeroed
```

The biological analogue is lateral inhibition: the most active neurons suppress
their neighbors through inhibitory connections, so only a small fraction stays
active.

In this method k-WTA serves three roles:
1. **Sparsity** — only k of N active; inactive units are skipped in computation.
2. **Between-layer nonlinearity** — top-k selection is nonlinear and sits between
   layers, so the within-layer dynamic stays linear (and unrollable).
3. **Series-truncation acceleration** — a sparse vector after each series term (3.3)
   speeds the decay of subsequent terms, so k ≈ 3–5 suffices.

The non-differentiability of top-k selection is not a problem, since learning is
local (Section 5) and uses no gradient.

The parameter k (typically 2–10% of a layer's neurons) is a hyperparameter: too
small reduces capacity, too large reduces sparsity.

---

## 7. Open Questions

1. The actual number of terms k for the target accuracy (empirical).
2. k-WTA: fixed k or adaptive threshold.
3. Quantizing the update accumulator to integer: per-layer or per-channel scale.
4. Maintaining symmetry of M: explicit (M ← ½(M+Mᵀ)) or parameterization M = LLᵀ.
5. Clamping strength β: fixed or scheduled.
6. Input front-ends: integer convolution for images; positional encoding for sequences.
7. Possible advantage of the recurrent form at inference (constant memory).

---

## 8. Next Step

A minimal feasibility prototype: core (3.3) with M = 0, then symmetric M, k-WTA
between layers, the two-phase rule (Section 5). First goal — exceeding a baseline
accuracy and determining the actual k.
