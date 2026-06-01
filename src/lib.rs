//! # lau-quantum-topology-agents
//!
//! Topological Quantum Field Theory (TQFT) applied to agent systems.
//!
//! A TQFT assigns vector spaces to boundaries and linear maps to cobordisms
//! (manifolds connecting boundaries). For agents:
//! - Each agent's boundary (interface) is a vector space
//! - Interaction history (manifold) is a linear map
//! - The partition function Z(M) computes amplitudes over agent histories
//!
//! Core structures:
//! - **Frobenius algebra**: the algebraic heart of 2D TQFT
//! - **Cobordism**: manifolds connecting agent boundaries
//! - **Anyons**: non-abelian braiding statistics
//! - **Jones polynomial**: knot invariants from agent interactions
//! - **Surgery**: cutting and gluing for S-matrices

pub mod frobenius;
pub mod cobordism;
pub mod agent;
pub mod amplitude;
pub mod surgery;
pub mod jones;
pub mod anyon;
pub mod protection;
pub mod tqft;

pub use frobenius::FrobeniusAlgebra;
pub use cobordism::Cobordism;
pub use agent::AgentBoundary;
pub use amplitude::PartitionFunction;
pub use surgery::Surgery;
pub use jones::JonesPolynomial;
pub use anyon::AnyonSystem;
pub use protection::TopologicalProtection;
pub use tqft::TQFT;
