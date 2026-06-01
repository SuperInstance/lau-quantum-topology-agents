//! Cobordism category — manifolds connecting boundaries.
//!
//! In TQFT, a cobordism M from Σ_in to Σ_out is a manifold whose boundary
//! is ∂M = Σ_in ⊔ Σ_out. For agents, cobordisms represent interactions
//! between agent boundaries.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Orientation of a boundary component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Orientation {
    Ingoing,
    Outgoing,
}

/// A boundary component (agent interface channel).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryComponent {
    /// Label for this boundary
    pub label: String,
    /// Dimension of the vector space assigned to this boundary
    pub dimension: usize,
    /// Orientation
    pub orientation: Orientation,
}

/// A cobordism connecting boundaries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cobordism {
    /// Input boundary components
    pub incoming: Vec<BoundaryComponent>,
    /// Output boundary components
    pub outgoing: Vec<BoundaryComponent>,
    /// Genus of the cobordism (number of handles)
    pub genus: usize,
    /// Linear map representing the TQFT assignment
    pub linear_map: DMatrix<f64>,
}

impl Cobordism {
    /// Create a new cobordism.
    pub fn new(
        incoming: Vec<BoundaryComponent>,
        outgoing: Vec<BoundaryComponent>,
        genus: usize,
        linear_map: DMatrix<f64>,
    ) -> Self {
        let in_dim: usize = incoming.iter().map(|b| b.dimension).sum();
        let out_dim: usize = outgoing.iter().map(|b| b.dimension).sum();
        assert_eq!(linear_map.nrows(), out_dim);
        assert_eq!(linear_map.ncols(), in_dim);
        Self { incoming, outgoing, genus, linear_map }
    }

    /// Identity cobordism: cylinder Σ × [0,1].
    pub fn identity(boundaries: Vec<BoundaryComponent>) -> Self {
        let dim: usize = boundaries.iter().map(|b| b.dimension).sum();
        Self {
            incoming: boundaries.clone(),
            outgoing: boundaries,
            genus: 0,
            linear_map: DMatrix::identity(dim, dim),
        }
    }

    /// Compose two cobordisms: M₂ ∘ M₁.
    pub fn compose(&self, other: &Cobordism) -> Result<Cobordism, String> {
        // Check compatibility: self's outputs must match other's inputs
        if self.outgoing.len() != other.incoming.len() {
            return Err("Boundary mismatch: output count != input count".into());
        }
        for (a, b) in self.outgoing.iter().zip(other.incoming.iter()) {
            if a.dimension != b.dimension {
                return Err(format!("Dimension mismatch: {} != {}", a.dimension, b.dimension));
            }
        }
        let composed = &other.linear_map * &self.linear_map;
        Ok(Cobordism {
            incoming: self.incoming.clone(),
            outgoing: other.outgoing.clone(),
            genus: self.genus + other.genus,
            linear_map: composed,
        })
    }

    /// Tensor product: M₁ ⊔ M₂ (disjoint union).
    pub fn tensor(&self, other: &Cobordism) -> Cobordism {
        let in_dim = self.input_dimension() + other.input_dimension();
        let out_dim = self.output_dimension() + other.output_dimension();
        // Block diagonal composition
        let mut map = DMatrix::zeros(out_dim, in_dim);
        let si = self.input_dimension();
        let so = self.output_dimension();
        for i in 0..self.output_dimension() {
            for j in 0..self.input_dimension() {
                map[(i, j)] = self.linear_map[(i, j)];
            }
        }
        for i in 0..other.output_dimension() {
            for j in 0..other.input_dimension() {
                map[(so + i, si + j)] = other.linear_map[(i, j)];
            }
        }
        Cobordism {
            incoming: self.incoming.iter().chain(other.incoming.iter()).cloned().collect(),
            outgoing: self.outgoing.iter().chain(other.outgoing.iter()).cloned().collect(),
            genus: self.genus + other.genus,
            linear_map: map,
        }
    }

    /// Total input dimension.
    pub fn input_dimension(&self) -> usize {
        self.incoming.iter().map(|b| b.dimension).sum()
    }

    /// Total output dimension.
    pub fn output_dimension(&self) -> usize {
        self.outgoing.iter().map(|b| b.dimension).sum()
    }

    /// Evaluate the cobordism on a state vector.
    pub fn evaluate(&self, state: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        assert_eq!(state.nrows(), self.input_dimension());
        &self.linear_map * state
    }

    /// The Euler characteristic: χ = 2 - 2g - b where b = total boundary components.
    pub fn euler_characteristic(&self) -> i32 {
        let b = self.incoming.len() + self.outgoing.len();
        2 - 2 * (self.genus as i32) - (b as i32)
    }

    /// Check if this is a closed cobordism (no boundaries).
    pub fn is_closed(&self) -> bool {
        self.incoming.is_empty() && self.outgoing.is_empty()
    }

    /// Check if this is an isomorphism (invertible map).
    pub fn is_isomorphism(&self, tol: f64) -> bool {
        if self.input_dimension() != self.output_dimension() {
            return false;
        }
        self.linear_map.determinant().abs() > tol
    }
}

impl fmt::Display for Cobordism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cobordism(genus={}, in={}, out={})",
            self.genus,
            self.incoming.len(),
            self.outgoing.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_boundary(dim: usize, orientation: Orientation) -> BoundaryComponent {
        BoundaryComponent {
            label: "b".into(),
            dimension: dim,
            orientation,
        }
    }

    #[test]
    fn test_identity_cobordism() {
        let b = make_boundary(2, Orientation::Ingoing);
        let id = Cobordism::identity(vec![b]);
        assert_eq!(id.input_dimension(), 2);
        assert_eq!(id.output_dimension(), 2);
        assert!(id.is_isomorphism(1e-10));
    }

    #[test]
    fn test_identity_actually_identity() {
        let b = make_boundary(3, Orientation::Ingoing);
        let id = Cobordism::identity(vec![b]);
        let v = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = id.evaluate(&v);
        assert!((&result - &v).norm() < 1e-10);
    }

    #[test]
    fn test_compose_identity() {
        let b = make_boundary(2, Orientation::Ingoing);
        let id1 = Cobordism::identity(vec![b.clone()]);
        let id2 = Cobordism::identity(vec![make_boundary(2, Orientation::Outgoing)]);
        let comp = id1.compose(&id2).unwrap();
        let v = nalgebra::DVector::from_vec(vec![1.0, 0.0]);
        let result = comp.evaluate(&v);
        assert!((&result - &v).norm() < 1e-10);
    }

    #[test]
    fn test_compose_dimension_mismatch() {
        let id1 = Cobordism::identity(vec![make_boundary(2, Orientation::Ingoing)]);
        let id2 = Cobordism::identity(vec![make_boundary(3, Orientation::Outgoing)]);
        assert!(id1.compose(&id2).is_err());
    }

    #[test]
    fn test_tensor_product() {
        let c1 = Cobordism::identity(vec![make_boundary(2, Orientation::Ingoing)]);
        let c2 = Cobordism::identity(vec![make_boundary(3, Orientation::Ingoing)]);
        let t = c1.tensor(&c2);
        assert_eq!(t.input_dimension(), 5);
        assert_eq!(t.output_dimension(), 5);
    }

    #[test]
    fn test_euler_characteristic_cylinder() {
        let id = Cobordism::identity(vec![make_boundary(1, Orientation::Ingoing)]);
        // genus=0, b=2 → χ = 2 - 0 - 2 = 0
        assert_eq!(id.euler_characteristic(), 0);
    }

    #[test]
    fn test_euler_characteristic_genus() {
        let c = Cobordism::new(
            vec![],
            vec![],
            1,
            DMatrix::identity(0, 0),
        );
        // genus=1, b=0 → χ = 2 - 2 - 0 = 0
        assert_eq!(c.euler_characteristic(), 0);
    }

    #[test]
    fn test_is_closed() {
        let c = Cobordism::new(vec![], vec![], 0, DMatrix::identity(0, 0));
        assert!(c.is_closed());
    }

    #[test]
    fn test_not_closed() {
        let c = Cobordism::identity(vec![make_boundary(1, Orientation::Ingoing)]);
        assert!(!c.is_closed());
    }

    #[test]
    fn test_evaluate() {
        let mat = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let c = Cobordism::new(
            vec![make_boundary(2, Orientation::Ingoing)],
            vec![make_boundary(2, Orientation::Outgoing)],
            0,
            mat,
        );
        let v = nalgebra::DVector::from_vec(vec![3.0, 4.0]);
        let result = c.evaluate(&v);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
    }

    #[test]
    fn test_serialization() {
        let c = Cobordism::identity(vec![make_boundary(2, Orientation::Ingoing)]);
        let json = serde_json::to_string(&c).unwrap();
        let back: Cobordism = serde_json::from_str(&json).unwrap();
        assert_eq!(back.incoming.len(), 1);
    }

    #[test]
    fn test_display() {
        let c = Cobordism::identity(vec![make_boundary(2, Orientation::Ingoing)]);
        assert!(format!("{}", c).contains("genus=0"));
    }

    #[test]
    fn test_isomorphism() {
        let mat = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let c = Cobordism::new(
            vec![make_boundary(2, Orientation::Ingoing)],
            vec![make_boundary(2, Orientation::Outgoing)],
            0,
            mat,
        );
        assert!(c.is_isomorphism(1e-10));
    }

    #[test]
    fn test_not_isomorphism() {
        let mat = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let c = Cobordism::new(
            vec![make_boundary(2, Orientation::Ingoing)],
            vec![make_boundary(2, Orientation::Outgoing)],
            0,
            mat,
        );
        assert!(!c.is_isomorphism(1e-10));
    }
}
