//! Jones polynomial — knot invariants from agent interaction patterns.
//!
//! The Jones polynomial V(L) is a knot invariant that can be computed
//! from a TQFT perspective using the bracket polynomial. For agents,
//! we model interaction patterns as braids/knots and compute their
//! topological invariants.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A crossing in a knot diagram.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Crossing {
    /// Positive crossing (overcrossing)
    Positive,
    /// Negative crossing (undercrossing)
    Negative,
}

/// A braid word (sequence of generators and their inverses).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BraidWord {
    /// Number of strands
    pub strands: usize,
    /// Generators: positive = σ_i, negative = σ_i^{-1}
    pub generators: Vec<(usize, bool)>, // (index, is_positive)
}

impl BraidWord {
    /// Create a new braid word.
    pub fn new(strands: usize) -> Self {
        Self { strands, generators: Vec::new() }
    }

    /// Add a positive generator σ_i.
    pub fn sigma(&mut self, i: usize) {
        assert!(i < self.strands);
        self.generators.push((i, true));
    }

    /// Add a negative generator σ_i^{-1}.
    pub fn sigma_inv(&mut self, i: usize) {
        assert!(i < self.strands);
        self.generators.push((i, false));
    }

    /// Number of crossings.
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    /// Whether the braid is trivial (empty).
    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }
}

/// Jones polynomial computation via Kauffman bracket.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JonesPolynomial {
    /// The Kauffman variable A
    pub a: f64,
}

impl JonesPolynomial {
    /// Create with default Kauffman variable.
    pub fn new() -> Self {
        // A = e^{-3πi/7} gives the Jones polynomial at t = e^{2πi/7}
        // We use a real approximation for computation
        Self { a: 0.5_f64.sqrt() }
    }

    /// Create with custom variable.
    pub fn with_a(a: f64) -> Self {
        Self { a }
    }

    /// Compute the Kauffman bracket of the unlink with n components.
    pub fn bracket_unlink(&self, n: usize) -> f64 {
        // ⟨○...○⟩ = (-A² - A⁻²)ⁿ
        let delta = -(self.a * self.a) - 1.0 / (self.a * self.a);
        delta.powi(n as i32)
    }

    /// Compute the Kauffman bracket of a single crossing.
    /// Each crossing splits into two smoothings weighted by A or A⁻¹.
    pub fn bracket_crossing(&self, crossing: Crossing) -> (f64, f64) {
        match crossing {
            Crossing::Positive => (self.a, 1.0 / self.a),
            Crossing::Negative => (1.0 / self.a, self.a),
        }
    }

    /// Compute the writhe of a braid.
    pub fn writhe(&self, braid: &BraidWord) -> i32 {
        braid.generators.iter().map(|(_, pos)| if *pos { 1 } else { -1 }).sum()
    }

    /// Compute the Jones polynomial (as a real evaluation at a specific point)
    /// for a braid using the Burau representation.
    pub fn evaluate_braid(&self, braid: &BraidWord) -> f64 {
        let n = braid.strands;
        if braid.is_empty() {
            return self.bracket_unlink(n);
        }

        // Burau representation
        let t = self.a * self.a; // Jones variable
        let mut rep = DMatrix::identity(n, n);

        for &(i, is_positive) in &braid.generators {
            let mut gen = DMatrix::identity(n, n);
            if i < n {
                // Burau generator at position i
                let ti = if is_positive { t } else { 1.0 / t };
                gen[(i, i)] = 1.0 - ti;
                if i + 1 < n {
                    gen[(i, i + 1)] = ti;
                    gen[(i + 1, i)] = 1.0;
                    gen[(i + 1, i + 1)] = 0.0;
                }
            }
            rep = &gen * &rep;
        }

        // Jones polynomial ≈ (-A³)^(-writhe) * trace / (t^{1/2} + t^{-1/2})
        let writhe = self.writhe(braid);
        let prefactor = (-self.a.powi(3)).powi(-writhe);
        let trace = rep.trace();
        let delta = self.a * self.a + 1.0 / (self.a * self.a);

        if delta.abs() < 1e-15 {
            prefactor * trace
        } else {
            prefactor * trace / delta
        }
    }

    /// Compute the Jones polynomial for an unlink of n components.
    pub fn evaluate_unlink(&self, n: usize) -> f64 {
        self.bracket_unlink(n)
    }

    /// Check if two braids give the same Jones polynomial (approximate).
    pub fn are_equivalent(&self, b1: &BraidWord, b2: &BraidWord, tol: f64) -> bool {
        (self.evaluate_braid(b1) - self.evaluate_braid(b2)).abs() < tol
    }

    /// Compute the knot determinant (|V(-1)|).
    pub fn knot_determinant(&self, braid: &BraidWord) -> f64 {
        // Use A = e^{πi/4} for evaluation at -1
        let neg_one = Self { a: (std::f64::consts::FRAC_PI_4).cos() };
        let val = neg_one.evaluate_braid(braid);
        if val.is_finite() { val.abs() } else { 0.0 }
    }
}

impl Default for JonesPolynomial {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_braid_creation() {
        let b = BraidWord::new(3);
        assert_eq!(b.strands, 3);
        assert!(b.is_empty());
    }

    #[test]
    fn test_braid_generators() {
        let mut b = BraidWord::new(3);
        b.sigma(0);
        b.sigma_inv(1);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_writhe_positive() {
        let mut b = BraidWord::new(2);
        b.sigma(0);
        b.sigma(0);
        b.sigma(0);
        assert_eq!(JonesPolynomial::new().writhe(&b), 3);
    }

    #[test]
    fn test_writhe_mixed() {
        let mut b = BraidWord::new(2);
        b.sigma(0);
        b.sigma_inv(0);
        assert_eq!(JonesPolynomial::new().writhe(&b), 0);
    }

    #[test]
    fn test_writhe_negative() {
        let mut b = BraidWord::new(2);
        b.sigma_inv(0);
        assert_eq!(JonesPolynomial::new().writhe(&b), -1);
    }

    #[test]
    fn test_bracket_unlink_0() {
        let j = JonesPolynomial::new();
        let val = j.bracket_unlink(0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bracket_unlink_1() {
        let j = JonesPolynomial::new();
        let val = j.bracket_unlink(1);
        // δ = -(A² + A⁻²)
        let a = j.a;
        let expected = -(a * a + 1.0 / (a * a));
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bracket_unlink_2() {
        let j = JonesPolynomial::new();
        let val = j.bracket_unlink(2);
        let a = j.a;
        let delta = -(a * a + 1.0 / (a * a));
        assert!((val - delta * delta).abs() < 1e-10);
    }

    #[test]
    fn test_bracket_crossing_positive() {
        let j = JonesPolynomial::new();
        let (a, b) = j.bracket_crossing(Crossing::Positive);
        assert!((a - j.a).abs() < 1e-10);
        assert!((b - 1.0 / j.a).abs() < 1e-10);
    }

    #[test]
    fn test_bracket_crossing_negative() {
        let j = JonesPolynomial::new();
        let (a, b) = j.bracket_crossing(Crossing::Negative);
        assert!((a - 1.0 / j.a).abs() < 1e-10);
        assert!((b - j.a).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_empty_braid() {
        let j = JonesPolynomial::new();
        let b = BraidWord::new(2);
        let val = j.evaluate_braid(&b);
        assert!(val.is_finite());
    }

    #[test]
    fn test_evaluate_trefoil() {
        // Trefoil knot as σ₁³ (closure of 3-strand braid with σ₁σ₂ repeated)
        let mut b = BraidWord::new(2);
        b.sigma(0);
        b.sigma(0);
        b.sigma(0);
        let j = JonesPolynomial::new();
        let val = j.evaluate_braid(&b);
        assert!(val.is_finite());
        assert!(val.abs() > 1e-15);
    }

    #[test]
    fn test_evaluate_figure_eight() {
        // Figure-eight knot as σ₁σ₂⁻¹σ₁σ₂⁻¹
        let mut b = BraidWord::new(3);
        b.sigma(0);
        b.sigma_inv(1);
        b.sigma(0);
        b.sigma_inv(1);
        let j = JonesPolynomial::new();
        let val = j.evaluate_braid(&b);
        assert!(val.is_finite());
    }

    #[test]
    fn test_unlink_evaluation() {
        let j = JonesPolynomial::new();
        let val = j.evaluate_unlink(1);
        assert!(val.is_finite());
    }

    #[test]
    fn test_are_equivalent_same() {
        let j = JonesPolynomial::new();
        let mut b1 = BraidWord::new(2);
        b1.sigma(0);
        let mut b2 = BraidWord::new(2);
        b2.sigma(0);
        assert!(j.are_equivalent(&b1, &b2, 1e-6));
    }

    #[test]
    fn test_knot_determinant() {
        let mut b = BraidWord::new(2);
        b.sigma(0);
        b.sigma(0);
        b.sigma(0);
        let j = JonesPolynomial::new();
        let det = j.knot_determinant(&b);
        assert!(det.is_finite());
    }

    #[test]
    fn test_default() {
        let j = JonesPolynomial::default();
        assert!(j.a > 0.0);
    }

    #[test]
    fn test_serialization() {
        let j = JonesPolynomial::new();
        let json = serde_json::to_string(&j).unwrap();
        let back: JonesPolynomial = serde_json::from_str(&json).unwrap();
        assert!((back.a - j.a).abs() < 1e-10);
    }
}
