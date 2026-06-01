# lau-quantum-topology-agents

**Topological Quantum Field Theory (TQFT) for agent systems** — Frobenius algebras, cobordisms, anyons, knot invariants, and topologically protected communication, all mapped to agent interaction semantics.

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-142-green.svg)](#testing)

---

## Why TQFT for Agents?

A TQFT is a rule that assigns vector spaces to boundaries and linear maps to the manifolds connecting them. Translated to agent systems:

- Each agent's interface is a **vector space** (its possible states)
- An interaction history between agents is a **cobordism** (manifold connecting boundaries)
- The **partition function** Z(M) computes amplitudes — the weight of each interaction history
- **Anyons** model interactions where order matters (swapping A then B ≠ B then A)
- **Topological protection** makes agent communication robust to local noise
- **Knot invariants** (Jones polynomial) classify the topology of interaction patterns

This crate implements these structures from the ground up, with full axiom verification and agent-oriented APIs.

---

## Quick Start

```toml
# Cargo.toml
[dependencies]
lau-quantum-topology-agents = "0.1"
```

```rust
use lau_quantum_topology_agents::*;

// Start with the Z/2Z TQFT
let tqft = TQFT::z2();

// Verify all axioms hold
let report = tqft.verify_axioms(1e-10);
assert!(report.all_pass());

// Create agents
let mut agent_a = AgentBoundary::new("alice", 2);
let mut agent_b = AgentBoundary::new("bob", 2);
agent_a.connect_to(&mut agent_b);

// Compute interaction amplitude via Frobenius multiplication
let alg = FrobeniusAlgebra::z2();
let network = AgentNetwork::new(2);
let result = alg.multiply(&agent_a.state_vector(), &agent_b.state_vector());

// Compute Jones polynomial for an interaction braid
let mut braid = JonesPolynomial::new();
let mut trefoil = BraidWord::new(2);
trefoil.sigma(0);
trefoil.sigma(0);
trefoil.sigma(0);
let jones_val = braid.evaluate_braid(&trefoil);
```

---

## Architecture

### Module Map

```
┌─────────────────────────────────────────────────┐
│  TQFT (top-level functor)                       │
│  ┌──────────────┐  ┌──────────────┐             │
│  │ FrobeniusAlg  │  │ Cobordism    │             │
│  │ (algebraic    │  │ (geometric   │             │
│  │  structure)   │  │  structure)  │             │
│  └──────────────┘  └──────────────┘             │
│  ┌──────────────┐  ┌──────────────┐             │
│  │ AgentBoundry  │  │ PartitionFn  │             │
│  │ (agent state  │  │ (amplitudes  │             │
│  │  spaces)      │  │  & weights)  │             │
│  └──────────────┘  └──────────────┘             │
│  ┌──────────────┐  ┌──────────────┐             │
│  │ Surgery       │  │ JonesPoly    │             │
│  │ (S-matrix,    │  │ (knot        │             │
│  │  Dehn twists)  │  │  invariants) │             │
│  └──────────────┘  └──────────────┘             │
│  ┌──────────────┐  ┌──────────────┐             │
│  │ AnyonSystem   │  │ TopoProtect  │             │
│  │ (braiding,    │  │ (error       │             │
│  │  fusion)      │  │  correction) │             │
│  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────┘
```

---

## Modules in Detail

### Frobenius Algebra (`frobenius`)

The algebraic heart of 2D TQFT. A commutative Frobenius algebra over ℝ with four operations:

| Operation | Symbol | Map | Meaning |
|---|---|---|---|
| Multiplication | μ | A ⊗ A → A | Agent state combination |
| Unit | η | ℝ → A | Default agent state |
| Comultiplication | Δ | A → A ⊗ A | State splitting |
| Counit | ε | A → ℝ | State evaluation |

```rust
let alg = FrobeniusAlgebra::z2(); // ℤ/2ℤ group algebra
let e0 = DVector::from_vec(vec![1.0, 0.0]);
let e1 = DVector::from_vec(vec![0.0, 1.0]);

// e1 * e1 = e0 (self-inverse)
let result = alg.multiply(&e1, &e1);
assert!((&result - &e0).norm() < 1e-10);

// Verify all axioms
assert!(alg.is_valid_tqft_algebra(1e-10));
```

**Built-in algebras:**
- `FrobeniusAlgebra::trivial()` — 1D (ℝ with standard structure)
- `FrobeniusAlgebra::z2()` — 2D (ℤ/2ℤ group algebra)

**Axiom checks:** commutativity, associativity, unit axiom, counit axiom, Frobenius condition, Frobenius form non-degeneracy.

### Cobordism (`cobordism`)

Manifolds connecting agent boundaries, represented as linear maps.

```rust
// Identity cobordism (cylinder: no interaction)
let id = Cobordism::identity(vec![boundary]);

// Compose: sequential interaction
let composed = cobordism_a.compose(&cobordism_b)?;

// Tensor: parallel (independent) interaction
let parallel = cobordism_a.tensor(&cobordism_b);

// Evaluate on a state
let new_state = cobordism.evaluate(&state);
```

**Operations:**
- `compose()` — sequential composition (matrix multiplication)
- `tensor()` — disjoint union (block diagonal)
- `euler_characteristic()` — χ = 2 - 2g - b
- `is_isomorphism()` — invertibility check

### Agent Boundaries (`agent`)

Vector spaces assigned to agent interfaces, with connection topology.

```rust
let mut alice = AgentBoundary::new("alice", 2);
let mut bob = AgentBoundary::new("bob", 2);
alice.connect_to(&mut bob);

// Networks of agents
let mut net = AgentNetwork::new(2);
net.add_agent(alice);
net.add_agent(bob);
let joint = net.joint_state(); // tensor product of all states
let interaction = net.interact("alice", "bob", &alg); // Frobenius multiply
```

**`AgentNetwork`** supports:
- Agent addition and lookup by ID
- Joint state computation (tensor product)
- Pairwise interaction via Frobenius multiplication
- Total dimension computation

### Partition Function (`amplitude`)

Computes amplitudes for interaction histories — the weighted sum over agent trajectories.

```rust
let pf = PartitionFunction::new(alg).with_temperature(0.5);

// Amplitude for specific input/output states
let amp = pf.amplitude(&cobordism, &input_state, &output_state);

// Partition function Z(M)
let z = pf.compute(&cobordism);

// Closed surface amplitude Z(Σ_g)
let z_sphere = pf.closed_surface_amplitude(0); // genus 0

// Probability distribution over agent states
let probs = pf.state_probabilities(&cobordism, &basis_states);

// Expectation value of an observable
let exp = pf.expectation(&cobordism, &basis_states, |state| state[0]);
```

### Surgery (`surgery`)

Cutting and gluing operations — the TQFT analog of modifying interaction topology.

```rust
let surgery = Surgery::new(algebra);

// S-matrix (modular transformation)
let s = surgery.s_matrix();

// Dehn twist (mapping class group generator)
let twist = surgery.dehn_twist(1);

// Cut a cobordism along a submanifold
let (left, right) = surgery.cut(&cobordism, split_dim);

// Glue back together
let glued = surgery.glue(&left, &right)?;

// Modular S-transformation (space ↔ time exchange)
let s_transform = surgery.modular_s_transform();

// Verify braid relation STS = TST
assert!(surgery.verify_braid_relation(1e-6));
```

### Jones Polynomial (`jones`)

Knot invariants computed via the Kauffman bracket and Burau representation.

```rust
let jones = JonesPolynomial::new();

// Define a braid (trefoil knot)
let mut braid = BraidWord::new(2);
braid.sigma(0); // positive crossing
braid.sigma(0);
braid.sigma(0);

// Evaluate the Jones polynomial
let value = jones.evaluate_braid(&braid);

// Compute writhe (sum of crossing signs)
let w = jones.writhe(&braid); // = 3

// Check if two braids are topologically equivalent
let equiv = jones.are_equivalent(&braid1, &braid2, 1e-6);

// Knot determinant |V(-1)|
let det = jones.knot_determinant(&braid);
```

**Key concepts:**
- **Braid generators**: σᵢ (positive crossing), σᵢ⁻¹ (negative crossing)
- **Writhe**: signed crossing count
- **Kauffman bracket**: recursive relation for knot evaluation
- **Burau representation**: matrix representation of the braid group

### Anyon System (`anyon`)

Non-abelian braiding statistics — interactions where swap order matters.

```rust
let ising = AnyonSystem::ising();

// Fusion: σ × σ → 1 + ψ (two possible outcomes)
let channels = ising.fuse("σ", "σ");

// Total quantum dimension
let d = ising.total_quantum_dimension();

// Braid (swap) anyons
let braided = ising.braid(&state_matrix, 0, 1);

// Non-abelian check
assert!(ising.is_non_abelian());
```

**Built-in models:**

| Model | Types | Key Fusion Rule | Non-Abelian? |
|---|---|---|---|
| **Ising** | 1, ψ, σ | σ × σ → 1 + ψ | ✅ |
| **Fibonacci** | 1, τ | τ × τ → 1 + τ | ✅ |

**Operations:**
- `fuse()` — compute fusion channels
- `braid()` / `braid_inverse()` — apply R-matrix braiding
- `topological_spins()` — θₐ = Rₐₐ for each type
- `is_non_abelian()` — check if braiding is non-commutative

### Topological Protection (`protection`)

Error-resistant agent communication using topological codes.

```rust
let tp = TopologicalProtection::new(AnyonSystem::ising(), 5);

// Encode a logical state
let logical = DVector::from_vec(vec![1.0, 0.0]);
let encoded = tp.encode(&logical);

// Decode back
let decoded = tp.decode(&encoded);

// Detect and correct errors
let errors = tp.detect_errors(&noisy_state);
let num_corrected = tp.correct(&mut noisy_state);

// Protected channel with noise
let received = tp.protected_channel(&sender_state, &receiver_state, noise_level);
```

**Protection levels:**

| Code Distance | Level | Correctable Errors |
|---|---|---|
| 0 | None | 0 |
| 1 | Level1 | 0 |
| 3 | Level2 | 1 |
| 5+ | Maximum | ⌊(d-1)/2⌋ |

### TQFT Functor (`tqft`)

The top-level object — a symmetric monoidal functor from the cobordism category to vector spaces.

```rust
let tqft = TQFT::z2();

// Assign vector spaces to agent boundaries
let dim = tqft.assign_vector_space(&agent);

// Assign linear maps to cobordisms
let map = tqft.assign_linear_map(&cobordism);

// Full axiom verification
let report = tqft.verify_axioms(1e-10);
assert!(report.all_pass());
```

**`TQFTAxiomReport`** checks:
- Algebra validity (commutative Frobenius)
- Functoriality (Z(id) = id, Z(g∘f) = Z(g)∘Z(f))
- Monoidality (Z(Σ₁ ⊔ Σ₂) = Z(Σ₁) ⊗ Z(Σ₂))
- Frobenius condition
- Associativity and commutativity

---

## Testing

142 tests covering every module and axiom:

```bash
cargo test
```

Test categories:
- **Frobenius algebra axioms** — commutativity, associativity, unit/counit, Frobenius condition, non-degeneracy
- **Cobordism operations** — composition, tensor product, evaluation, isomorphism
- **Agent networks** — state vectors, connections, joint states, interactions
- **Partition functions** — amplitudes, closed surfaces, probabilities, expectations
- **Surgery** — S-matrix, Dehn twists, cutting/gluing, modular transformations
- **Jones polynomial** — writhe, bracket, braid evaluation, knot determinant
- **Anyon systems** — fusion rules, braiding, non-abelian detection, quantum dimensions
- **Topological protection** — encode/decode, error detection/correction, protected channels
- **TQFT functor** — functoriality, monoidality, full axiom verification
- **Serialization** — round-trip tests for all types

---

## Mathematical Background

### 2D TQFT (Atiyah's Axioms)

A 2D TQFT is equivalent to a commutative Frobenius algebra. The correspondence:

- **Pair of pants** (multiplication) → μ: A ⊗ A → A
- **Cap** (unit) → η: ℝ → A
- **Copants** (comultiplication) → Δ: A → A ⊗ A
- **Cup** (counit) → ε: A → ℝ

The Frobenius condition `β(μ(a,b), c) = β(a, μ(b,c))` (where β = ε∘μ) is the algebraic encoding of topological invariance.

### Anyons and Braiding

In 2+1 dimensions, particle exchange can produce unitary transformations (not just ±1 phase). The **Ising anyon model** has three topological charges:

```
σ × σ → 1 + ψ    (non-abelian: two fusion channels)
σ × ψ → σ         (abelian)
ψ × ψ → 1         (abelian, fermionic)
```

The **Fibonacci model** is the simplest non-abelian theory:

```
τ × τ → 1 + τ    (quantum dimension d = (1+√5)/2)
```

### Jones Polynomial

The Jones polynomial V(L) is a knot invariant computable from the Kauffman bracket:

```
⟨◯⟩ = 1
⟨L ⊔ ◯⟩ = (-A² - A⁻²)⟨L⟩
```

Each crossing splits into two smoothings, weighted by A and A⁻¹. The normalized bracket `(-A³)^{-w(L)}⟨L⟩` gives the Jones polynomial.

---

## Dependencies

| Crate | Purpose |
|---|---|
| `nalgebra` | Linear algebra (matrices, vectors) with serde support |
| `serde` / `serde_json` | Serialization of all types |

---

## License

MIT
