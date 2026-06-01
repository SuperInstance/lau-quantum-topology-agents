# lau-quantum-topology-agents

**Topological Quantum Field Theory (TQFT) applied to agent systems — Frobenius algebras, cobordisms, anyons, knot invariants, and topologically protected communication.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

142 tests · 2,838 lines of Rust · 9 modules

---

## What This Does

This crate implements the mathematical framework of **Topological Quantum Field Theory** (TQFT) and applies it to multi-agent systems. In TQFT:

- **Boundaries** (agent interfaces) are assigned **vector spaces**
- **Cobordisms** (manifold-shaped interactions) are assigned **linear maps**
- **The partition function** Z(M) computes amplitudes over agent histories
- **Topological invariants** (Jones polynomial, anyon braiding) capture properties that survive any continuous deformation

This gives you:
- **Frobenius algebras** — the algebraic heart of 2D TQFT, with multiplication, comultiplication, and the Frobenius condition
- **Cobordism category** — compose agent interactions as manifold gluings with functorial verification
- **Ising anyons** — non-abelian braiding statistics where swapping agents A↔B ≠ swapping B↔A
- **Jones polynomial** — knot invariants computed from agent interaction patterns via the Kauffman bracket
- **Partition function** — weighted sums over agent histories with temperature control
- **Surgery** — Dehn twists, S-matrices, and cutting/gluging for mapping class group operations
- **Topological protection** — agent states encoded in topological degrees of freedom, robust to local perturbations

---

## Key Idea

The central analogy:

| TQFT Concept | Agent Counterpart |
|---|---|
| (d−1)-manifold Σ | Agent boundary / interface |
| Vector space Z(Σ) | Agent state space |
| Cobordism M: Σ₁ → Σ₂ | Interaction history connecting two agent states |
| Linear map Z(M) | State transition operator |
| Partition function Z(M) | Amplitude / weight of interaction history |
| Frobenius algebra A | Internal structure of agent operations |
| Anyon braiding | Non-commutative agent interactions |
| Jones polynomial | Topological invariant of interaction pattern |
| Surgery | Restructuring agent interaction topology |
| Topological protection | Error-robust agent state encoding |

The TQFT functor Z: **Cob** → **Vect** is a symmetric monoidal functor. For agents, this means:
- Composition of interactions = composition of linear maps
- Parallel agents = tensor product of state spaces
- No interaction = identity map
- Closed manifold = scalar (amplitude)

---

## Install

```toml
[dependencies]
lau-quantum-topology-agents = "0.1"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `serde`, `serde_json`.

---

## Quick Start

```rust
use lau_quantum_topology_agents::*;

// 1. Create a TQFT from a Frobenius algebra
let tqft = TQFT::z2(); // The Z/2Z TQFT

// 2. Define agent boundaries
let mut agent_a = AgentBoundary::new("agent-a", 2);
let mut agent_b = AgentBoundary::new("agent-b", 2);
agent_a.connect_to(&mut agent_b);

// 3. The TQFT assigns vector spaces to boundaries
let dim = tqft.assign_vector_space(&agent_a);
// dim = agent.state_dimension × algebra.dimension = 2 × 2 = 4

// 4. Compose cobordisms (interactions)
let cob1 = Cobordism::identity(vec![
    BoundaryComponent { label: "a".into(), dimension: 2, orientation: Orientation::Ingoing },
]);
let cob2 = cob1.clone();
let composed = cob1.compose(&cob2)?; // Z(M₂ ∘ M₁) = Z(M₂) · Z(M₁)

// 5. Compute partition function (amplitude)
let pf = PartitionFunction::new(FrobeniusAlgebra::z2()).with_temperature(0.5);
let amplitude = pf.compute(&composed);

// 6. Compute Jones polynomial of an interaction braid
let mut jones = JonesPolynomial::new();
let mut braid = BraidWord::new(3);
braid.sigma(0);      // σ₁
braid.sigma(1);       // σ₂
braid.sigma_inv(0);   // σ₁⁻¹
let poly = jones.evaluate_braid(&braid);

// 7. Anyon braiding (Ising model)
let anyons = AnyonSystem::ising();
let fusion = anyons.fuse("σ", "σ"); // → ["1", "ψ"]
let braid_matrix = anyons.braid("σ", "σ"); // R-matrix

// 8. Topological protection
let protection = TopologicalProtection::new(anyons, 3);
let logical = DVector::from_vec(vec![1.0, 0.0]);
let encoded = protection.encode(&logical);
let is_valid = protection.verify(&encoded);
```

---

## API Reference

### `FrobeniusAlgebra`
The algebraic structure underlying 2D TQFT. Contains multiplication μ, unit η, comultiplication Δ, and counit ε satisfying the Frobenius condition.

```rust
let alg = FrobeniusAlgebra::z2();
let product = alg.multiply(&a, &b);      // μ: A⊗A → A
let unit = alg.unit(1.0);                // η: ℝ → A
let comult = alg.comultiply(&x);         // Δ: A → A⊗A
let frobenius_ok = alg.verify_frobenius_condition(1e-10);
let frob_matrix = alg.frobenius_matrix(); // β_ij = ε(μ(e_i, e_j))
```

Pre-built algebras: `trivial()` (1D), `z2()` (ℤ/2ℤ group algebra).

### `Cobordism`
A manifold connecting boundaries, represented as a linear map.

```rust
let cob = Cobordism::new(incoming, outgoing, genus, linear_map);
let identity = Cobordism::identity(boundaries);  // Cylinder Σ × [0,1]
let pair = Cobordism::pair(dim);                  // Cap: ∅ → Σ ⊔ Σ
let copair = Cobordism::copair(dim);              // Cup: Σ ⊔ Σ → ∅
let composed = cob1.compose(&cob2)?;             // M₂ ∘ M₁
let tensor = cob1.tensor(&cob2);                  // M₁ ⊔ M₂
```

### `TQFT`
The full TQFT functor. Orchestrates all components.

```rust
let tqft = TQFT::new(frobenius_algebra);
let functorial = tqft.verify_functoriality(1e-10);  // Z(M₂∘M₁) = Z(M₂)·Z(M₁)
let monoidal = tqft.verify_monoidal(1e-10);          // Z(M₁⊔M₂) = Z(M₁)⊗Z(M₂)
```

### `AnyonSystem`
Non-abelian anyon model (Ising). Supports fusion rules, braiding, and F-matrix recoupling.

```rust
let anyons = AnyonSystem::ising();
let fusion_outputs = anyons.fuse("σ", "σ");  // ["1", "ψ"]
let braid_R = anyons.braid("σ", "ψ");        // R-matrix
let total_dim = anyons.total_quantum_dimension(); // √2 for Ising
```

Fusion rules: 1×1→1, 1×ψ→ψ, 1×σ→σ, ψ×ψ→1, ψ×σ→σ, σ×σ→1+ψ.

### `JonesPolynomial`
Knot invariant computed via the Kauffman bracket.

```rust
let jones = JonesPolynomial::new();
let unlink = jones.bracket_unlink(2);     // (-A² - A⁻²)²
let trefoil = jones.trefoil();             // V(t) for trefoil knot

let mut braid = BraidWord::new(3);
braid.sigma(0); braid.sigma(1); braid.sigma(0);
let eval = jones.evaluate_braid(&braid);
```

### `PartitionFunction`
Computes amplitudes for closed cobordisms and agent histories.

```rust
let pf = PartitionFunction::new(algebra).with_temperature(0.5);
let z = pf.compute(&cobordism);                                   // Z(M)
let amp = pf.amplitude(&cob, &input_state, &output_state);       // ⟨out|Z(M)|in⟩
let surface_z = pf.closed_surface_amplitude(genus);               // Z(Σ_g)
```

### `Surgery`
Cutting and gluing operations. S-matrix and Dehn twists.

```rust
let surgery = Surgery::new(algebra);
let s_matrix = surgery.s_matrix();                // Modular S-matrix
let twist = surgery.dehn_twist(genus);            // Mapping class group generator
let (left, right) = surgery.cut(&cobordism, dim); // Cut along submanifold
let glued = surgery.glue(&left, &right);          // Reglue
```

### `TopologicalProtection`
Encodes agent states in topological degrees of freedom for error robustness.

```rust
let tp = TopologicalProtection::new(anyons, code_distance);
let encoded = tp.encode(&logical_state);      // Encode into protected subspace
let valid = tp.verify(&encoded);              // Check error syndrome
let decoded = tp.decode(&encoded);            // Recover logical state
let level = tp.protection_level();            // None, Level1, Level2, Maximum
```

Protection levels scale with code distance: d≥1 → Level1, d≥3 → Level2, d≥5 → Maximum.

### `AgentBoundary`
An agent's interface as a vector space, with connection channels.

```rust
let mut a = AgentBoundary::new("researcher", 4);
let mut b = AgentBoundary::new("coder", 4);
a.connect_to(&mut b);
a.normalize();
let overlap = a.inner_product(&b);
```

---

## How It Works

```
FrobeniusAlgebra ──────────────────────────────────────┐
    │ μ, η, Δ, ε, β                                     │
    ▼                                                     │
AgentBoundary ──→ Cobordism ──→ TQFT (functor)           │
    │ Σ            │ M: Σ₁→Σ₂   │ Z                     │
    │              │             │                        │
    ▼              ▼             ▼                        │
PartitionFunction  Surgery     AnyonSystem               │
    │ Z(M)         │ S, T      │ R, F matrices           │
    ▼              ▼             ▼                        │
JonesPolynomial  ──→ Protection  ←───────────────────────┘
    │ V(L)           │ encode/decode
    ▼                ▼
Amplitudes     Topologically protected states
```

1. **Frobenius algebra** defines the internal algebraic structure: multiplication (combining states), comultiplication (splitting states), and the Frobenius form β.
2. **Agent boundaries** are (d−1)-manifolds assigned vector spaces by the TQFT.
3. **Cobordisms** are d-manifolds connecting boundaries, assigned linear maps by the TQFT functor.
4. **TQFT** verifies functoriality (Z respects composition) and monoidality (Z respects tensor products).
5. **Partition function** computes scalar amplitudes Z(M) for closed cobordisms.
6. **Anyons** provide non-abelian braiding: fusing and braiding particles whose statistics depend on topology, not geometry.
7. **Jones polynomial** computes knot invariants from braided interaction patterns via the Kauffman bracket.
8. **Surgery** implements the mapping class group: S-matrix (modular transformation), Dehn twists, and cut/reglue.
9. **Topological protection** encodes logical states in topological degrees of freedom, achieving error robustness that scales with code distance.

---

## The Math

### Frobenius Algebras and 2D TQFT

A **commutative Frobenius algebra** (A, μ, η, Δ, ε) consists of:
- **Multiplication** μ: A ⊗ A → A with unit η
- **Comultiplication** Δ: A → A ⊗ A with counit ε
- **Frobenius condition**: (μ ⊗ id)∘(id ⊗ Δ) = Δ∘μ

The **classification theorem** (Atiyah, Dijkgraaf): 2D TQFTs are in bijection with commutative Frobenius algebras. The TQFT functor is completely determined by the algebra structure.

For a closed surface of genus *g*:
```
Z(Σ_g) = ε(μ^g(Δ^g(η(1))))
```

### Cobordism Category

The **cobordism category** Cob_d has:
- **Objects**: closed (d−1)-manifolds
- **Morphisms**: d-dimensional cobordisms M: Σ_in → Σ_out (where ∂M = Σ_in ⊔ Σ̄_out)

A **TQFT** is a symmetric monoidal functor Z: Cob_d → Vect_ℝ satisfying:
- **Functoriality**: Z(M₂ ∘ M₁) = Z(M₂) ∘ Z(M₁)
- **Monoidality**: Z(Σ₁ ⊔ Σ₂) = Z(Σ₁) ⊗ Z(Σ₂)
- **Normalization**: Z(∅) = ℝ

### Anyons and Braiding

In 2+1 dimensions, particles can have **anyonic statistics**: exchanging two particles applies a unitary R-matrix that is neither +1 (bosons) nor −1 (fermions).

The **Ising anyon model** has three anyon types: {1, ψ, σ} with:
- Quantum dimensions: d₁ = 1, d_ψ = 1, d_σ = √2
- Total quantum dimension: D = √(1 + 1 + 2) = 2
- Fusion: σ × σ → 1 + ψ (non-abelian: two possible outcomes)
- Braiding: R_σσ = e^{-iπ/8} (non-trivial phase)

### Jones Polynomial

The **Jones polynomial** V_L(t) is a knot invariant computed via the Kauffman bracket:
```
⟨◯⟩ = 1
⟨L ⊔ ◯⟩ = (-A² - A⁻²) ⟨L⟩
⟨crossing⟩ = A ⟨0-smoothing⟩ + A⁻¹ ⟨1-smoothing⟩
```

The braid representation: every knot/link is the closure of a braid. The Jones polynomial is computed from the braid word using the Burau representation (or equivalently, from the TQFT applied to the braid).

### Surgery and the S-Matrix

**Surgery** on a 3-manifold: cut out a solid torus S¹ × D² and reglue via a diffeomorphism of the boundary torus. The **S-matrix** encodes the modular transformation of the TQFT under this operation:

```
S_ij = (1/D) Σ_k d_k · N_{ik}^j · e^{2πi s_k/c}
```

where D is the total quantum dimension, d_k are quantum dimensions, N_{ik}^j are fusion multiplicities, s_k are conformal spins, and c is the central charge.

### Topological Protection

Agent states encoded in the **fusion space** of anyons are topologically protected: local perturbations cannot distinguish different fusion outcomes without performing a global measurement. The **code distance** d determines how many local errors are needed to cause a logical error:

- d = 1: No protection
- d = 3: Surface code level (corrects 1 error)
- d = 5+: Full topological protection (corrects ⌊(d−1)/2⌋ errors)

---

## Module Overview

| Module | Tests | Key Types |
|--------|-------|-----------|
| `frobenius` | 22 | `FrobeniusAlgebra` |
| `cobordism` | 14 | `Cobordism`, `BoundaryComponent` |
| `tqft` | 15 | `TQFT` |
| `anyon` | 19 | `AnyonSystem`, `AnyonType`, `FusionRule` |
| `jones` | 18 | `JonesPolynomial`, `BraidWord` |
| `amplitude` | 12 | `PartitionFunction` |
| `surgery` | 10 | `Surgery` |
| `protection` | 17 | `TopologicalProtection`, `ProtectedState` |
| `agent` | 15 | `AgentBoundary` |

---

## License

MIT
