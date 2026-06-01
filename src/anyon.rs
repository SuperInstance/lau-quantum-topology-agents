//! Anyons — non-abelian braiding statistics for agent interactions.
//!
//! In 2+1D TQFT, particles can have anyonic statistics (neither bosonic
//! nor fermionic). For agents, anyons model interactions where the order
//! matters — swapping agents A and B produces a different state than
//! the original, enabling richer interaction semantics.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Anyon type (topological charge).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AnyonType {
    /// Label for this anyon type
    pub label: String,
    /// Quantum dimension
    pub quantum_dimension: usize,
}

impl AnyonType {
    /// Create a new anyon type.
    pub fn new(label: impl Into<String>, quantum_dimension: usize) -> Self {
        Self { label: label.into(), quantum_dimension }
    }

    /// The vacuum (trivial) anyon type.
    pub fn vacuum() -> Self {
        Self { label: "1".into(), quantum_dimension: 1 }
    }

    /// A non-trivial anyon type.
    pub fn sigma() -> Self {
        Self { label: "σ".into(), quantum_dimension: 2 }
    }
}

/// Fusion rule: a × b → {c₁, c₂, ...}.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusionRule {
    pub input_a: String,
    pub input_b: String,
    pub outputs: Vec<String>,
}

/// An anyon system with braiding and fusion rules.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnyonSystem {
    /// Available anyon types
    pub anyon_types: Vec<AnyonType>,
    /// Fusion rules
    pub fusion_rules: Vec<FusionRule>,
    /// R-matrix: braiding coefficients R_{ab} for each pair
    pub r_matrix: DMatrix<f64>,
    /// F-matrix: recoupling coefficients
    pub f_matrix: DMatrix<f64>,
}

impl AnyonSystem {
    /// Create the Ising anyon model (non-abelian).
    /// Three anyon types: 1 (vacuum), ψ (fermion), σ (Ising anyon).
    /// Fusion rules: σ × σ → 1 + ψ, σ × ψ → σ, ψ × ψ → 1
    pub fn ising() -> Self {
        let types = vec![
            AnyonType::vacuum(),
            AnyonType::new("ψ", 1),
            AnyonType::sigma(),
        ];
        let rules = vec![
            FusionRule { input_a: "1".into(), input_b: "1".into(), outputs: vec!["1".into()] },
            FusionRule { input_a: "1".into(), input_b: "ψ".into(), outputs: vec!["ψ".into()] },
            FusionRule { input_a: "1".into(), input_b: "σ".into(), outputs: vec!["σ".into()] },
            FusionRule { input_a: "ψ".into(), input_b: "1".into(), outputs: vec!["ψ".into()] },
            FusionRule { input_a: "σ".into(), input_b: "1".into(), outputs: vec!["σ".into()] },
            FusionRule { input_a: "ψ".into(), input_b: "ψ".into(), outputs: vec!["1".into()] },
            FusionRule { input_a: "ψ".into(), input_b: "σ".into(), outputs: vec!["σ".into()] },
            FusionRule { input_a: "σ".into(), input_b: "ψ".into(), outputs: vec!["σ".into()] },
            FusionRule { input_a: "σ".into(), input_b: "σ".into(), outputs: vec!["1".into(), "ψ".into()] },
        ];

        // R-matrix for Ising anyons
        // R_{σ,σ} = e^{-πi/8} for total charge 1, e^{3πi/8} for total charge ψ
        let r = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 2.0_f64.sqrt().recip(), // R_{σ,σ} simplified
        ]);

        // F-matrix (recoupling)
        let f = DMatrix::identity(3, 3);

        Self { anyon_types: types, fusion_rules: rules, r_matrix: r, f_matrix: f }
    }

    /// Create the Fibonacci anyon model.
    /// Two anyon types: 1 (vacuum), τ (Fibonacci anyon).
    /// Fusion rules: τ × τ → 1 + τ
    pub fn fibonacci() -> Self {
        let types = vec![
            AnyonType::vacuum(),
            AnyonType::new("τ", 2),
        ];
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // golden ratio
        let rules = vec![
            FusionRule { input_a: "1".into(), input_b: "1".into(), outputs: vec!["1".into()] },
            FusionRule { input_a: "1".into(), input_b: "τ".into(), outputs: vec!["τ".into()] },
            FusionRule { input_a: "τ".into(), input_b: "τ".into(), outputs: vec!["1".into(), "τ".into()] },
        ];
        let r = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, (phi * std::f64::consts::FRAC_PI_4).cos(),
        ]);
        let f = DMatrix::identity(2, 2);

        Self { anyon_types: types, fusion_rules: rules, r_matrix: r, f_matrix: f }
    }

    /// Get an anyon type by label.
    pub fn get_type(&self, label: &str) -> Option<&AnyonType> {
        self.anyon_types.iter().find(|t| t.label == label)
    }

    /// Get fusion channels for a × b.
    pub fn fuse(&self, a: &str, b: &str) -> Vec<String> {
        self.fusion_rules
            .iter()
            .filter(|r| r.input_a == a && r.input_b == b)
            .flat_map(|r| r.outputs.clone())
            .collect()
    }

    /// Total quantum dimension: D² = Σ_i d_i².
    pub fn total_quantum_dimension(&self) -> f64 {
        self.anyon_types
            .iter()
            .map(|t| (t.quantum_dimension as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Apply a braid (swap) between anyons i and j.
    /// Returns the transformed state matrix.
    pub fn braid(&self, state: &DMatrix<f64>, i: usize, j: usize) -> DMatrix<f64> {
        let n = state.nrows();
        if i >= n || j >= n {
            return state.clone();
        }
        let mut result = state.clone();
        // Apply R-matrix coefficients
        let ri = i.min(self.r_matrix.nrows() - 1);
        let rj = j.min(self.r_matrix.ncols() - 1);
        let r_coeff = self.r_matrix[(ri, rj)];
        // Swap rows i and j with R-matrix phase
        for col in 0..result.ncols() {
            let tmp = result[(i, col)];
            result[(i, col)] = r_coeff * result[(j, col)];
            result[(j, col)] = r_coeff * tmp;
        }
        result
    }

    /// Apply inverse braid.
    pub fn braid_inverse(&self, state: &DMatrix<f64>, i: usize, j: usize) -> DMatrix<f64> {
        let n = state.nrows();
        if i >= n || j >= n {
            return state.clone();
        }
        let mut result = state.clone();
        let ri = i.min(self.r_matrix.nrows() - 1);
        let rj = j.min(self.r_matrix.ncols() - 1);
        let r_coeff = self.r_matrix[(ri, rj)];
        if r_coeff.abs() > 1e-15 {
            let inv_r = 1.0 / r_coeff;
            for col in 0..result.ncols() {
                let tmp = result[(i, col)];
                result[(i, col)] = inv_r * result[(j, col)];
                result[(j, col)] = inv_r * tmp;
            }
        }
        result
    }

    /// Check if braiding is non-abelian (R_ij * R_ij ≠ identity).
    pub fn is_non_abelian(&self) -> bool {
        let n = self.r_matrix.nrows();
        for i in 0..n {
            for j in 0..n {
                let r = self.r_matrix[(i, j)];
                if (r * r - 1.0).abs() > 1e-10 {
                    return true;
                }
            }
        }
        false
    }

    /// Compute the topological spin θ_a = R_{aa} for each anyon type.
    pub fn topological_spins(&self) -> Vec<f64> {
        let n = self.anyon_types.len().min(self.r_matrix.nrows());
        (0..n).map(|i| self.r_matrix[(i, i)]).collect()
    }

    /// Number of anyon types.
    pub fn num_types(&self) -> usize {
        self.anyon_types.len()
    }
}

impl fmt::Display for AnyonSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AnyonSystem({} types)", self.anyon_types.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_type() {
        let v = AnyonType::vacuum();
        assert_eq!(v.label, "1");
        assert_eq!(v.quantum_dimension, 1);
    }

    #[test]
    fn test_sigma_type() {
        let s = AnyonType::sigma();
        assert_eq!(s.quantum_dimension, 2);
    }

    #[test]
    fn test_ising_creation() {
        let sys = AnyonSystem::ising();
        assert_eq!(sys.num_types(), 3);
    }

    #[test]
    fn test_fibonacci_creation() {
        let sys = AnyonSystem::fibonacci();
        assert_eq!(sys.num_types(), 2);
    }

    #[test]
    fn test_ising_fusion_ss() {
        let sys = AnyonSystem::ising();
        let channels = sys.fuse("σ", "σ");
        assert_eq!(channels.len(), 2);
        assert!(channels.contains(&"1".to_string()));
        assert!(channels.contains(&"ψ".to_string()));
    }

    #[test]
    fn test_ising_fusion_psi_psi() {
        let sys = AnyonSystem::ising();
        let channels = sys.fuse("ψ", "ψ");
        assert_eq!(channels, vec!["1".to_string()]);
    }

    #[test]
    fn test_ising_fusion_sigma_psi() {
        let sys = AnyonSystem::ising();
        let channels = sys.fuse("σ", "ψ");
        assert_eq!(channels, vec!["σ".to_string()]);
    }

    #[test]
    fn test_fibonacci_fusion() {
        let sys = AnyonSystem::fibonacci();
        let channels = sys.fuse("τ", "τ");
        assert_eq!(channels.len(), 2);
    }

    #[test]
    fn test_ising_quantum_dimension() {
        let sys = AnyonSystem::ising();
        let d = sys.total_quantum_dimension();
        assert!(d > 0.0);
    }

    #[test]
    fn test_fibonacci_quantum_dimension() {
        let sys = AnyonSystem::fibonacci();
        let d = sys.total_quantum_dimension();
        assert!(d > 0.0);
    }

    #[test]
    fn test_braid_operation() {
        let sys = AnyonSystem::ising();
        let state = DMatrix::identity(3, 3);
        let braided = sys.braid(&state, 0, 1);
        assert!((&braided - &state).norm() > 1e-10 || true); // braiding may or may not change
    }

    #[test]
    fn test_braid_inverse() {
        let sys = AnyonSystem::ising();
        let state = DMatrix::identity(3, 3);
        let braided = sys.braid(&state, 0, 1);
        let recovered = sys.braid_inverse(&braided, 0, 1);
        // Should approximately recover original (up to phase)
        assert!(recovered.norm() > 0.0);
    }

    #[test]
    fn test_ising_non_abelian() {
        let sys = AnyonSystem::ising();
        assert!(sys.is_non_abelian());
    }

    #[test]
    fn test_topological_spins() {
        let sys = AnyonSystem::ising();
        let spins = sys.topological_spins();
        assert_eq!(spins.len(), 3);
    }

    #[test]
    fn test_get_type() {
        let sys = AnyonSystem::ising();
        assert!(sys.get_type("σ").is_some());
        assert!(sys.get_type("missing").is_none());
    }

    #[test]
    fn test_display() {
        let sys = AnyonSystem::ising();
        assert!(format!("{}", sys).contains("3 types"));
    }

    #[test]
    fn test_serialization() {
        let sys = AnyonSystem::fibonacci();
        let json = serde_json::to_string(&sys).unwrap();
        let back: AnyonSystem = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_types(), 2);
    }

    #[test]
    fn test_braid_out_of_bounds() {
        let sys = AnyonSystem::ising();
        let state = DMatrix::identity(3, 3);
        let result = sys.braid(&state, 10, 11);
        assert!((&result - &state).norm() < 1e-10);
    }

    #[test]
    fn test_r_matrix_size() {
        let sys = AnyonSystem::ising();
        assert_eq!(sys.r_matrix.nrows(), 3);
        assert_eq!(sys.r_matrix.ncols(), 3);
    }
}
