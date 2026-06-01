# lau-quantum-topology-agents

**Quantum topology (TQFT) applied to agent systems.**

A Rust crate implementing topological quantum field theory concepts for modeling agent interactions. In TQFT, vector spaces are assigned to boundaries and linear maps to cobordisms (manifolds connecting boundaries). For agents:

- Each agent's **boundary** (interface with the world) is a vector space
- The **cobordism** (interaction history) is a linear map between boundaries
- The **partition function** Z(M) computes amplitudes over agent histories
- **Topological protection** makes agent states robust to perturbation

## Core Structures

| Module | Description |
|--------|-------------|
| `frobenius` | Frobenius algebra (μ, η, Δ, ε) — the algebraic heart of 2D TQFT |
| `cobordism` | Cobordism category — manifolds connecting agent boundaries |
| `agent` | Agent boundaries as vector spaces with input/output channels |
| `amplitude` | Partition function Z(M) — weighted sum over agent histories |
| `surgery` | Surgery operations: cutting, gluing, S-matrix, Dehn twists |
| `jones` | Jones polynomial — knot invariants from agent interaction patterns |
| `anyon` | Anyon systems — non-abelian braiding statistics (Ising, Fibonacci) |
| `protection` | Topologically protected agent communication channels |
| `tqft` | Full TQFT functor: Cob → Vect_ℝ with axiom verification |

## Quick Start

```rust
use lau_quantum_topology_agents::*;

// Create a TQFT from the Z/2Z Frobenius algebra
let tqft = TQFT::z2();

// Verify all TQFT axioms hold
let report = tqft.verify_axioms(1e-10);
assert!(report.all_pass());

// Create agents with boundary vector spaces
let mut agent_a = AgentBoundary::new("alice", 2);
let mut agent_b = AgentBoundary::new("bob", 2);
agent_a.connect_to(&mut agent_b);

// Anyon braiding (non-abelian: order matters!)
let anyons = AnyonSystem::ising();
let channels = anyons.fuse("σ", "σ"); // → {1, ψ}
assert!(anyons.is_non_abelian());

// Topologically protected communication
let protection = TopologicalProtection::new(AnyonSystem::ising(), 5);
let logical = nalgebra::DVector::from_vec(vec![1.0, 0.0]);
let encoded = protection.encode(&logical);
let decoded = protection.decode(&encoded);
```

## Mathematical Background

### 2D TQFT ≅ Frobenius Algebra

A 2D TQFT is equivalent to a commutative Frobenius algebra over the ground field ℝ:

- **Multiplication** μ: A ⊗ A → A (merging two boundaries)
- **Unit** η: ℝ → A (creating from vacuum)
- **Comultiplication** Δ: A → A ⊗ A (splitting a boundary)
- **Counit** ε: A → ℝ (annihilating to vacuum)

These satisfy the Frobenius condition: (μ ⊗ id) ∘ (id ⊗ Δ) = Δ ∘ μ.

### Anyons

In 2+1D, particles can have **anyonic** statistics. The Ising anyon model has:

- σ × σ → 1 + ψ (non-abelian fusion)
- σ × ψ → σ
- ψ × ψ → 1

The order of braiding matters, enabling richer agent interaction semantics.

### Jones Polynomial

Knot and link invariants computed via the Kauffman bracket:

⟨L⟩ = Σ smoothings A^a(A⁻¹)^b(-A² - A⁻²)^c

## License

MIT
