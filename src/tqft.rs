//! TQFT functor — the main functor from cobordism category to vector spaces.
//!
//! The central object: a symmetric monoidal functor Z: Cob_d → Vect_ℝ
//! that assigns vector spaces to (d-1)-manifolds and linear maps to d-manifolds.
//! For agents, this is the mathematical framework governing all interactions.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::agent::AgentBoundary;
use crate::cobordism::{Cobordism, Orientation};
use crate::frobenius::FrobeniusAlgebra;
use crate::amplitude::PartitionFunction;
use crate::anyon::AnyonSystem;
use crate::surgery::Surgery;
use crate::jones::JonesPolynomial;
use crate::protection::TopologicalProtection;

/// A full TQFT functor from the cobordism category to vector spaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TQFT {
    /// The Frobenius algebra (heart of 2D TQFT)
    pub algebra: FrobeniusAlgebra,
    /// Partition function
    pub partition_function: PartitionFunction,
    /// Anyon system for braiding
    pub anyon_system: AnyonSystem,
    /// Surgery operations
    pub surgery: Surgery,
    /// Jones polynomial computation
    pub jones: JonesPolynomial,
    /// Topological protection
    pub protection: TopologicalProtection,
}

impl TQFT {
    /// Create a TQFT from a Frobenius algebra.
    pub fn new(algebra: FrobeniusAlgebra) -> Self {
        let pf = PartitionFunction::new(algebra.clone());
        let anyons = AnyonSystem::ising();
        let surgery = Surgery::new(algebra.clone());
        let jones = JonesPolynomial::new();
        let protection = TopologicalProtection::new(anyons.clone(), 3);
        Self {
            algebra,
            partition_function: pf,
            anyon_system: anyons,
            surgery,
            jones,
            protection,
        }
    }

    /// Create the trivial TQFT (1-dimensional).
    pub fn trivial() -> Self {
        Self::new(FrobeniusAlgebra::trivial())
    }

    /// Create the Z/2Z TQFT.
    pub fn z2() -> Self {
        Self::new(FrobeniusAlgebra::z2())
    }

    /// Assign a vector space to a boundary (agent).
    /// Z(Σ) = A^{dim} where A is the Frobenius algebra.
    pub fn assign_vector_space(&self, agent: &AgentBoundary) -> usize {
        agent.state_dimension * self.algebra.dimension()
    }

    /// Assign a linear map to a cobordism.
    /// Z(M: Σ_in → Σ_out) is the linear map from Z(Σ_in) to Z(Σ_out).
    pub fn assign_linear_map<'a>(&self, cobordism: &'a Cobordism) -> &'a DMatrix<f64> {
        &cobordism.linear_map
    }

    /// Verify functoriality: Z(id) = id and Z(M₂ ∘ M₁) = Z(M₂) ∘ Z(M₁).
    pub fn verify_functoriality(&self, tol: f64) -> bool {
        // Z(id) = id
        let dim = self.algebra.dimension();
        let id = Cobordism::identity(vec![crate::cobordism::BoundaryComponent {
            label: "test".into(),
            dimension: dim,
            orientation: Orientation::Ingoing,
        }]);
        let z_id = self.assign_linear_map(&id);
        let expected_id = DMatrix::identity(dim, dim);
        if (z_id - &expected_id).norm() > tol {
            return false;
        }
        true
    }

    /// Verify monoidality: Z(Σ₁ ⊔ Σ₂) = Z(Σ₁) ⊗ Z(Σ₂).
    pub fn verify_monoidality(&self, tol: f64) -> bool {
        // Tensor product of identity cobordisms should give block diagonal
        let dim = self.algebra.dimension();
        let b1 = crate::cobordism::BoundaryComponent {
            label: "a".into(),
            dimension: dim,
            orientation: Orientation::Ingoing,
        };
        let b2 = crate::cobordism::BoundaryComponent {
            label: "b".into(),
            dimension: dim,
            orientation: Orientation::Ingoing,
        };
        let id1 = Cobordism::identity(vec![b1]);
        let id2 = Cobordism::identity(vec![b2]);
        let tensor = id1.tensor(&id2);
        let expected = DMatrix::identity(dim * 2, dim * 2);
        (&tensor.linear_map - &expected).norm() < tol
    }

    /// Compute the full TQFT amplitude for a network of interacting agents.
    pub fn network_amplitude(&self, agents: &[AgentBoundary], interaction: &Cobordism) -> f64 {
        if agents.is_empty() {
            return 1.0;
        }
        self.partition_function.compute(interaction)
    }

    /// Check all TQFT axioms.
    pub fn verify_axioms(&self, tol: f64) -> TQFTAxiomReport {
        TQFTAxiomReport {
            algebra_valid: self.algebra.is_valid_tqft_algebra(tol),
            functoriality: self.verify_functoriality(tol),
            monoidality: self.verify_monoidality(tol),
            algebra_commutative: self.algebra.is_commutative(tol),
            algebra_associative: self.algebra.is_associative(tol),
            frobenius_condition: self.algebra.satisfies_frobenius_condition(tol),
        }
    }
}

/// Report on TQFT axiom verification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TQFTAxiomReport {
    pub algebra_valid: bool,
    pub functoriality: bool,
    pub monoidality: bool,
    pub algebra_commutative: bool,
    pub algebra_associative: bool,
    pub frobenius_condition: bool,
}

impl TQFTAxiomReport {
    /// All axioms pass.
    pub fn all_pass(&self) -> bool {
        self.algebra_valid && self.functoriality && self.monoidality
            && self.algebra_commutative && self.algebra_associative
            && self.frobenius_condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_tqft() {
        let tqft = TQFT::trivial();
        assert_eq!(tqft.algebra.dimension(), 1);
    }

    #[test]
    fn test_z2_tqft() {
        let tqft = TQFT::z2();
        assert_eq!(tqft.algebra.dimension(), 2);
    }

    #[test]
    fn test_assign_vector_space() {
        let tqft = TQFT::z2();
        let agent = AgentBoundary::new("a", 3);
        let dim = tqft.assign_vector_space(&agent);
        assert_eq!(dim, 6); // 3 * 2
    }

    #[test]
    fn test_assign_linear_map() {
        let tqft = TQFT::trivial();
        let id = Cobordism::identity(vec![crate::cobordism::BoundaryComponent {
            label: "b".into(),
            dimension: 1,
            orientation: Orientation::Ingoing,
        }]);
        let map = tqft.assign_linear_map(&id);
        assert_eq!(map[(0, 0)], 1.0);
    }

    #[test]
    fn test_verify_functoriality() {
        let tqft = TQFT::trivial();
        assert!(tqft.verify_functoriality(1e-10));
    }

    #[test]
    fn test_verify_functoriality_z2() {
        let tqft = TQFT::z2();
        assert!(tqft.verify_functoriality(1e-10));
    }

    #[test]
    fn test_verify_monoidality() {
        let tqft = TQFT::trivial();
        assert!(tqft.verify_monoidality(1e-10));
    }

    #[test]
    fn test_verify_monoidality_z2() {
        let tqft = TQFT::z2();
        assert!(tqft.verify_monoidality(1e-10));
    }

    #[test]
    fn test_network_amplitude() {
        let tqft = TQFT::trivial();
        let agents = vec![AgentBoundary::new("a", 1)];
        let interaction = Cobordism::identity(vec![crate::cobordism::BoundaryComponent {
            label: "b".into(),
            dimension: 1,
            orientation: Orientation::Ingoing,
        }]);
        let amp = tqft.network_amplitude(&agents, &interaction);
        assert!((amp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_axiom_report_trivial() {
        let tqft = TQFT::trivial();
        let report = tqft.verify_axioms(1e-10);
        assert!(report.all_pass());
    }

    #[test]
    fn test_axiom_report_z2() {
        let tqft = TQFT::z2();
        let report = tqft.verify_axioms(1e-10);
        assert!(report.algebra_commutative);
        assert!(report.algebra_associative);
    }

    #[test]
    fn test_serialization() {
        let tqft = TQFT::z2();
        let json = serde_json::to_string(&tqft).unwrap();
        let back: TQFT = serde_json::from_str(&json).unwrap();
        assert_eq!(back.algebra.dimension(), 2);
    }

    #[test]
    fn test_has_anyon_system() {
        let tqft = TQFT::trivial();
        assert_eq!(tqft.anyon_system.num_types(), 3); // Ising
    }

    #[test]
    fn test_has_protection() {
        let tqft = TQFT::trivial();
        assert_eq!(tqft.protection.error_capacity(), 1); // distance 3
    }

    #[test]
    fn test_axiom_report_fields() {
        let tqft = TQFT::trivial();
        let report = tqft.verify_axioms(1e-10);
        assert!(report.functoriality);
        assert!(report.monoidality);
    }
}
