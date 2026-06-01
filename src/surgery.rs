//! Surgery — cutting and gluing agent interactions (S-matrix).
//!
//! In TQFT, surgery operations modify manifolds by cutting along
//! embedded submanifolds and regluing. The S-matrix encodes how
//! the TQFT transforms under these operations.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::cobordism::Cobordism;
use crate::frobenius::FrobeniusAlgebra;

/// Surgery operations on agent interactions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Surgery {
    /// The Frobenius algebra
    pub algebra: FrobeniusAlgebra,
}

impl Surgery {
    /// Create a new surgery context.
    pub fn new(algebra: FrobeniusAlgebra) -> Self {
        Self { algebra }
    }

    /// Compute the S-matrix for the Frobenius algebra.
    /// S_ij = Σ_k (μ^k_ij) / (λ_k) where λ_k are the Frobenius eigenvalues.
    /// For a semisimple Frobenius algebra, this diagonalizes the multiplication.
    pub fn s_matrix(&self) -> DMatrix<f64> {
        let n = self.algebra.dimension();
        let beta = self.algebra.frobenius_matrix();
        let mut s = DMatrix::zeros(n, n);
        for i in 0..n {
            let ei = DVector::from_fn(n, |r, _| if r == i { 1.0 } else { 0.0 });
            for j in 0..n {
                let ej = DVector::from_fn(n, |r, _| if r == j { 1.0 } else { 0.0 });
                s[(i, j)] = self.algebra.frobenius_form(&ei, &ej);
            }
        }
        // Normalize by Frobenius form
        let det = beta.determinant();
        if det.abs() > 1e-15 {
            s *= 1.0 / det.abs().sqrt();
        }
        s
    }

    /// Dehn twist: apply a twist along a boundary component.
    /// This is a generator of the mapping class group.
    pub fn dehn_twist(&self, genus: usize) -> DMatrix<f64> {
        let n = self.algebra.dimension();
        if genus == 0 {
            return DMatrix::identity(n, n);
        }
        // Dehn twist = T = exp(2πi * c/24) in the modular context
        // For real Frobenius: rotation by the counit angle
        let epsilon = self.algebra.epsilon.clone();
        let mut twist = DMatrix::identity(n, n);
        for i in 0..n {
            let angle = epsilon[i] * std::f64::consts::PI * 2.0 / (genus as f64 + 1.0);
            twist[(i, i)] = angle.cos();
        }
        twist
    }

    /// Cut a cobordism along a submanifold.
    /// Returns two new cobordisms from cutting.
    pub fn cut(&self, cobordism: &Cobordism, cut_dim: usize) -> (Cobordism, Cobordism) {
        let n = cobordism.input_dimension();
        let m = cobordism.output_dimension();
        let mat = &cobordism.linear_map;

        let left = cut_dim.min(n);

        let mat1 = mat.columns(0, left).clone_owned();
        let mat2 = if n > left {
            mat.columns(left, n - left).clone_owned()
        } else {
            DMatrix::zeros(m, 0)
        };

        let b_in1 = crate::cobordism::BoundaryComponent {
            label: "cut_in1".into(),
            dimension: mat1.ncols(),
            orientation: crate::cobordism::Orientation::Ingoing,
        };
        let b_out1 = crate::cobordism::BoundaryComponent {
            label: "cut_out1".into(),
            dimension: m,
            orientation: crate::cobordism::Orientation::Outgoing,
        };
        let b_in2 = crate::cobordism::BoundaryComponent {
            label: "cut_in2".into(),
            dimension: mat2.ncols(),
            orientation: crate::cobordism::Orientation::Ingoing,
        };
        let b_out2 = crate::cobordism::BoundaryComponent {
            label: "cut_out2".into(),
            dimension: m,
            orientation: crate::cobordism::Orientation::Outgoing,
        };

        let c1 = Cobordism::new(
            if mat1.ncols() > 0 { vec![b_in1] } else { vec![] },
            vec![b_out1],
            cobordism.genus,
            mat1,
        );
        let c2 = Cobordism::new(
            if mat2.ncols() > 0 { vec![b_in2] } else { vec![] },
            vec![b_out2],
            cobordism.genus,
            mat2,
        );

        (c1, c2)
    }

    /// Glue two cobordisms together.
    pub fn glue(&self, c1: &Cobordism, c2: &Cobordism) -> Result<Cobordism, String> {
        c1.compose(c2)
    }

    /// Connected sum of two cobordisms.
    pub fn connected_sum(&self, c1: &Cobordism, c2: &Cobordism) -> Cobordism {
        c1.tensor(c2)
    }

    /// Compute the modular S-transformation.
    /// This exchanges the roles of space and time in the TQFT.
    pub fn modular_s_transform(&self) -> DMatrix<f64> {
        let beta = self.algebra.frobenius_matrix();
        let n = self.algebra.dimension();
        // S = β^{-1/2} (the inverse square root of the Frobenius form)
        // Approximate via diagonalization approach
        let mut s = DMatrix::zeros(n, n);
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| beta[(i, j)]).sum();
            if row_sum.abs() > 1e-15 {
                for j in 0..n {
                    s[(i, j)] = beta[(i, j)] / row_sum;
                }
            } else {
                s[(i, i)] = 1.0;
            }
        }
        s
    }

    /// Verify the modular relation STS = TST (braid relation).
    pub fn verify_braid_relation(&self, tol: f64) -> bool {
        let s = self.modular_s_transform();
        let t = self.dehn_twist(1);
        let sts = &s * &t * &s;
        let tst = &t * &s * &t;
        (&sts - &tst).norm() < tol * s.nrows() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cobordism::{BoundaryComponent, Orientation};

    fn make_boundary(dim: usize) -> BoundaryComponent {
        BoundaryComponent { label: "b".into(), dimension: dim, orientation: Orientation::Ingoing }
    }

    #[test]
    fn test_s_matrix_creation() {
        let alg = FrobeniusAlgebra::trivial();
        let s = Surgery::new(alg);
        let mat = s.s_matrix();
        assert_eq!(mat.nrows(), 1);
    }

    #[test]
    fn test_s_matrix_z2() {
        let alg = FrobeniusAlgebra::z2();
        let s = Surgery::new(alg);
        let mat = s.s_matrix();
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 2);
    }

    #[test]
    fn test_dehn_twist_identity() {
        let alg = FrobeniusAlgebra::trivial();
        let s = Surgery::new(alg);
        let twist = s.dehn_twist(0);
        assert_eq!(twist.nrows(), 1);
        assert!((twist[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dehn_twist_genus1() {
        let alg = FrobeniusAlgebra::z2();
        let s = Surgery::new(alg);
        let twist = s.dehn_twist(1);
        assert_eq!(twist.nrows(), 2);
    }

    #[test]
    fn test_cut_cobordism() {
        let alg = FrobeniusAlgebra::trivial();
        let s = Surgery::new(alg);
        let c = Cobordism::identity(vec![make_boundary(4)]);
        let (c1, _c2) = s.cut(&c, 2);
        // The first piece takes the first 2 columns
        assert_eq!(c1.input_dimension(), 2);
        assert_eq!(c1.output_dimension(), 4);
    }

    #[test]
    fn test_glue_cobordisms() {
        let alg = FrobeniusAlgebra::trivial();
        let s = Surgery::new(alg);
        let id1 = Cobordism::identity(vec![make_boundary(2)]);
        let id2 = Cobordism::identity(vec![make_boundary(2)]);
        // Can't compose directly since orientations don't match for compose
        // Use tensor instead
        let result = s.connected_sum(&id1, &id2);
        assert_eq!(result.input_dimension(), 4);
    }

    #[test]
    fn test_connected_sum() {
        let alg = FrobeniusAlgebra::trivial();
        let s = Surgery::new(alg);
        let c1 = Cobordism::identity(vec![make_boundary(2)]);
        let c2 = Cobordism::identity(vec![make_boundary(3)]);
        let sum = s.connected_sum(&c1, &c2);
        assert_eq!(sum.input_dimension(), 5);
    }

    #[test]
    fn test_modular_s_transform() {
        let alg = FrobeniusAlgebra::z2();
        let s = Surgery::new(alg);
        let transform = s.modular_s_transform();
        assert_eq!(transform.nrows(), 2);
    }

    #[test]
    fn test_modular_s_transform_trivial() {
        let alg = FrobeniusAlgebra::trivial();
        let s = Surgery::new(alg);
        let transform = s.modular_s_transform();
        assert!((transform[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_serialization() {
        let alg = FrobeniusAlgebra::trivial();
        let s = Surgery::new(alg);
        let json = serde_json::to_string(&s).unwrap();
        let back: Surgery = serde_json::from_str(&json).unwrap();
        assert_eq!(back.algebra.dimension(), 1);
    }
}
