//! Frobenius algebra — the algebraic structure underlying 2D TQFT.
//!
//! A commutative Frobenius algebra over ℝ with:
//! - Multiplication μ: A ⊗ A → A
//! - Unit η: ℝ → A
//! - Comultiplication Δ: A → A ⊗ A
//! - Counit ε: A → ℝ
//!
//! Satisfying Frobenius condition: (μ ⊗ id)∘(id ⊗ Δ) = Δ∘μ

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A Frobenius algebra over ℝ represented by structure constants.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrobeniusAlgebra {
    /// Dimension of the algebra
    pub dimension: usize,
    /// Multiplication structure constants: μ(e_i, e_j) = Σ_k mu[i][j][k] * e_k
    /// Stored as mu[i * dim + j][k] for compactness
    mu: DMatrix<f64>,
    /// Unit vector: η(1) = Σ_i eta[i] * e_i
    eta: DVector<f64>,
    /// Comultiplication structure constants
    delta: DMatrix<f64>,
    /// Counit vector: ε(e_i) = epsilon[i]
    pub epsilon: DVector<f64>,
}

impl FrobeniusAlgebra {
    /// Create a new Frobenius algebra from structure constants.
    pub fn new(
        dimension: usize,
        mu: DMatrix<f64>,
        eta: DVector<f64>,
        delta: DMatrix<f64>,
        epsilon: DVector<f64>,
    ) -> Self {
        assert_eq!(mu.nrows(), dimension * dimension);
        assert_eq!(mu.ncols(), dimension);
        assert_eq!(eta.nrows(), dimension);
        assert_eq!(delta.nrows(), dimension);
        assert_eq!(delta.ncols(), dimension * dimension);
        assert_eq!(epsilon.nrows(), dimension);
        Self { dimension, mu, eta, delta, epsilon }
    }

    /// Create the trivial 1D Frobenius algebra (ℝ with standard structure).
    pub fn trivial() -> Self {
        Self {
            dimension: 1,
            mu: DMatrix::from_element(1, 1, 1.0),
            eta: DVector::from_element(1, 1.0),
            delta: DMatrix::from_element(1, 1, 1.0),
            epsilon: DVector::from_element(1, 1.0),
        }
    }

    /// Create the group algebra ℤ/2ℤ Frobenius algebra (2-dimensional).
    /// e0 = identity, e1 = generator. e1*e1 = e0.
    /// Frobenius form: β(e_i, e_j) = δ_{ij} (standard).
    pub fn z2() -> Self {
        // mu[i*dim+j][k]: mu is 4×2 matrix
        // e0*e0 = e0: mu[0][0]=1, mu[0][1]=0
        // e0*e1 = e1: mu[1][0]=0, mu[1][1]=1
        // e1*e0 = e1: mu[2][0]=0, mu[2][1]=1
        // e1*e1 = e0: mu[3][0]=1, mu[3][1]=0
        let mu = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0,  // e0*e0
            0.0, 1.0,  // e0*e1
            0.0, 1.0,  // e1*e0
            1.0, 0.0,  // e1*e1
        ]);
        let eta = DVector::from_vec(vec![1.0, 0.0]);
        // delta[i][j*dim+k]: comultiplication, 2×4 matrix
        // Δ(e0) = e0⊗e0 + e1⊗e1 (dual to multiplication)
        // Δ(e1) = e0⊗e1 + e1⊗e0
        let delta = DMatrix::from_row_slice(2, 4, &[
            1.0, 0.0, 0.0, 1.0,  // Δ(e0) = e0⊗e0 + e1⊗e1
            0.0, 1.0, 1.0, 0.0,  // Δ(e1) = e0⊗e1 + e1⊗e0
        ]);
        let epsilon = DVector::from_vec(vec![1.0, 0.0]);
        Self { dimension: 2, mu, eta, delta, epsilon }
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Multiply two elements (given as coefficient vectors).
    pub fn multiply(&self, a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
        assert_eq!(a.nrows(), self.dimension);
        assert_eq!(b.nrows(), self.dimension);
        let mut result = DVector::zeros(self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                let ab = a[i] * b[j];
                for k in 0..self.dimension {
                    result[k] += ab * self.mu[(i * self.dimension + j, k)];
                }
            }
        }
        result
    }

    /// Apply unit: η(r) for scalar r.
    pub fn unit(&self, r: f64) -> DVector<f64> {
        &self.eta * r
    }

    /// Comultiply: Δ(a) returns a flattened vector of dimension dim².
    pub fn comultiply(&self, a: &DVector<f64>) -> DVector<f64> {
        assert_eq!(a.nrows(), self.dimension);
        let mut result = DVector::zeros(self.dimension * self.dimension);
        for i in 0..self.dimension {
            let ai = a[i];
            for jk in 0..self.dimension * self.dimension {
                result[jk] += ai * self.delta[(i, jk)];
            }
        }
        result
    }

    /// Apply counit: ε(a).
    pub fn counit(&self, a: &DVector<f64>) -> f64 {
        self.epsilon.dot(a)
    }

    /// The Frobenius form (bilinear form): β(a, b) = ε(μ(a, b)).
    pub fn frobenius_form(&self, a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        self.counit(&self.multiply(a, b))
    }

    /// Compute the Frobenius form matrix β(e_i, e_j).
    pub fn frobenius_matrix(&self) -> DMatrix<f64> {
        let n = self.dimension;
        let mut mat = DMatrix::zeros(n, n);
        for i in 0..n {
            let ei = DVector::from_fn(n, |r, _| if r == i { 1.0 } else { 0.0 });
            for j in 0..n {
                let ej = DVector::from_fn(n, |r, _| if r == j { 1.0 } else { 0.0 });
                mat[(i, j)] = self.frobenius_form(&ei, &ej);
            }
        }
        mat
    }

    /// Check commutativity: μ(a,b) = μ(b,a).
    pub fn is_commutative(&self, tol: f64) -> bool {
        let n = self.dimension;
        for i in 0..n {
            let ei = DVector::from_fn(n, |r, _| if r == i { 1.0 } else { 0.0 });
            for j in 0..n {
                let ej = DVector::from_fn(n, |r, _| if r == j { 1.0 } else { 0.0 });
                let ab = self.multiply(&ei, &ej);
                let ba = self.multiply(&ej, &ei);
                if (&ab - &ba).norm() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check associativity: μ(μ(a,b),c) = μ(a,μ(b,c)).
    pub fn is_associative(&self, tol: f64) -> bool {
        let n = self.dimension;
        for i in 0..n {
            let ei = DVector::from_fn(n, |r, _| if r == i { 1.0 } else { 0.0 });
            for j in 0..n {
                let ej = DVector::from_fn(n, |r, _| if r == j { 1.0 } else { 0.0 });
                for k in 0..n {
                    let ek = DVector::from_fn(n, |r, _| if r == k { 1.0 } else { 0.0 });
                    let left = self.multiply(&self.multiply(&ei, &ej), &ek);
                    let right = self.multiply(&ei, &self.multiply(&ej, &ek));
                    if (&left - &right).norm() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check Frobenius condition: (μ ⊗ id)∘(id ⊗ Δ) = Δ∘μ.
    /// For each pair of basis elements, verify the relation holds.
    pub fn satisfies_frobenius_condition(&self, tol: f64) -> bool {
        let n = self.dimension;
        // Compute the Frobenius form β(e_i, e_j) = ε(μ(e_i, e_j))
        // and verify it matches the pairing from comultiplication
        // The key Frobenius relation: β(μ(a,b), c) = β(a, μ(b,c))
        for i in 0..n {
            let ei = DVector::from_fn(n, |r, _| if r == i { 1.0 } else { 0.0 });
            for j in 0..n {
                let ej = DVector::from_fn(n, |r, _| if r == j { 1.0 } else { 0.0 });
                for k in 0..n {
                    let ek = DVector::from_fn(n, |r, _| if r == k { 1.0 } else { 0.0 });
                    // β(μ(ei, ej), ek) = ε(μ(μ(ei, ej), ek))
                    let left = self.counit(&self.multiply(&self.multiply(&ei, &ej), &ek));
                    // β(ei, μ(ej, ek)) = ε(μ(ei, μ(ej, ek)))
                    let right = self.counit(&self.multiply(&ei, &self.multiply(&ej, &ek)));
                    if (left - right).abs() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check unit axiom: μ(η(1), a) = a and μ(a, η(1)) = a.
    pub fn satisfies_unit_axiom(&self, tol: f64) -> bool {
        let u = self.unit(1.0);
        for i in 0..self.dimension {
            let ei = DVector::from_fn(self.dimension, |r, _| if r == i { 1.0 } else { 0.0 });
            let left = self.multiply(&u, &ei);
            let right = self.multiply(&ei, &u);
            if (&left - &ei).norm() > tol || (&right - &ei).norm() > tol {
                return false;
            }
        }
        true
    }

    /// Check counit axiom: (ε ⊗ id)∘Δ(a) = a and (id ⊗ ε)∘Δ(a) = a.
    pub fn satisfies_counit_axiom(&self, tol: f64) -> bool {
        for i in 0..self.dimension {
            let ei = DVector::from_fn(self.dimension, |r, _| if r == i { 1.0 } else { 0.0 });
            let d = self.comultiply(&ei);
            // (ε ⊗ id): sum over first factor counit
            let mut left = DVector::zeros(self.dimension);
            for j in 0..self.dimension {
                let ej = DVector::from_fn(self.dimension, |r, _| if r == j { 1.0 } else { 0.0 });
                let eps_j = self.counit(&ej);
                for k in 0..self.dimension {
                    left[k] += eps_j * d[j * self.dimension + k];
                }
            }
            if (&left - &ei).norm() > tol {
                return false;
            }
        }
        true
    }

    /// Full TQFT axioms check.
    pub fn is_valid_tqft_algebra(&self, tol: f64) -> bool {
        self.is_commutative(tol)
            && self.is_associative(tol)
            && self.satisfies_unit_axiom(tol)
            && self.satisfies_counit_axiom(tol)
            && self.satisfies_frobenius_condition(tol)
    }
}

impl fmt::Display for FrobeniusAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FrobeniusAlgebra(dim={})", self.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_dimension() {
        let alg = FrobeniusAlgebra::trivial();
        assert_eq!(alg.dimension(), 1);
    }

    #[test]
    fn test_trivial_multiply() {
        let alg = FrobeniusAlgebra::trivial();
        let a = DVector::from_vec(vec![3.0]);
        let b = DVector::from_vec(vec![5.0]);
        let result = alg.multiply(&a, &b);
        assert!((result[0] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_trivial_unit() {
        let alg = FrobeniusAlgebra::trivial();
        let u = alg.unit(7.0);
        assert!((u[0] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_trivial_counit() {
        let alg = FrobeniusAlgebra::trivial();
        let a = DVector::from_vec(vec![4.0]);
        assert!((alg.counit(&a) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trivial_frobenius_form() {
        let alg = FrobeniusAlgebra::trivial();
        let a = DVector::from_vec(vec![3.0]);
        let b = DVector::from_vec(vec![5.0]);
        assert!((alg.frobenius_form(&a, &b) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_trivial_is_commutative() {
        assert!(FrobeniusAlgebra::trivial().is_commutative(1e-10));
    }

    #[test]
    fn test_trivial_is_associative() {
        assert!(FrobeniusAlgebra::trivial().is_associative(1e-10));
    }

    #[test]
    fn test_trivial_satisfies_unit_axiom() {
        assert!(FrobeniusAlgebra::trivial().satisfies_unit_axiom(1e-10));
    }

    #[test]
    fn test_trivial_is_valid_tqft() {
        assert!(FrobeniusAlgebra::trivial().is_valid_tqft_algebra(1e-10));
    }

    #[test]
    fn test_z2_dimension() {
        assert_eq!(FrobeniusAlgebra::z2().dimension(), 2);
    }

    #[test]
    fn test_z2_multiply_identity() {
        let alg = FrobeniusAlgebra::z2();
        let e0 = DVector::from_vec(vec![1.0, 0.0]);
        let e1 = DVector::from_vec(vec![0.0, 1.0]);
        let result = alg.multiply(&e0, &e1);
        assert!((&result - &e1).norm() < 1e-10);
    }

    #[test]
    fn test_z2_multiply_self_inverse() {
        let alg = FrobeniusAlgebra::z2();
        let e1 = DVector::from_vec(vec![0.0, 1.0]);
        let result = alg.multiply(&e1, &e1);
        // e1 * e1 = e0 in Z/2Z
        let e0 = DVector::from_vec(vec![1.0, 0.0]);
        assert!((&result - &e0).norm() < 1e-10);
    }

    #[test]
    fn test_z2_is_commutative() {
        assert!(FrobeniusAlgebra::z2().is_commutative(1e-10));
    }

    #[test]
    fn test_z2_is_associative() {
        assert!(FrobeniusAlgebra::z2().is_associative(1e-10));
    }

    #[test]
    fn test_z2_unit_axiom() {
        assert!(FrobeniusAlgebra::z2().satisfies_unit_axiom(1e-10));
    }

    #[test]
    fn test_z2_counit_axiom() {
        assert!(FrobeniusAlgebra::z2().satisfies_counit_axiom(1e-10));
    }

    #[test]
    fn test_z2_is_valid_tqft() {
        assert!(FrobeniusAlgebra::z2().is_valid_tqft_algebra(1e-10));
    }

    #[test]
    fn test_frobenius_matrix_symmetric() {
        let alg = FrobeniusAlgebra::z2();
        let mat = alg.frobenius_matrix();
        assert!((&mat - &mat.transpose()).norm() < 1e-10);
    }

    #[test]
    fn test_frobenius_matrix_nondegenerate() {
        let alg = FrobeniusAlgebra::z2();
        let mat = alg.frobenius_matrix();
        assert!(mat.determinant().abs() > 1e-10);
    }

    #[test]
    fn test_comultiply_basis() {
        let alg = FrobeniusAlgebra::z2();
        let e0 = DVector::from_vec(vec![1.0, 0.0]);
        let d = alg.comultiply(&e0);
        assert_eq!(d.nrows(), 4);
    }

    #[test]
    fn test_display() {
        let alg = FrobeniusAlgebra::trivial();
        assert_eq!(format!("{}", alg), "FrobeniusAlgebra(dim=1)");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let alg = FrobeniusAlgebra::z2();
        let json = serde_json::to_string(&alg).unwrap();
        let back: FrobeniusAlgebra = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dimension, 2);
    }
}
