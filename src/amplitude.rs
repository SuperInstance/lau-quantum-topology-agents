//! Partition function and amplitudes.
//!
//! The partition function Z(M) assigns a number (amplitude) to a closed
//! cobordism M. For agents, this is the weighted sum over interaction histories.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::cobordism::Cobordism;
use crate::frobenius::FrobeniusAlgebra;

/// Partition function for computing amplitudes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartitionFunction {
    /// The Frobenius algebra
    pub algebra: FrobeniusAlgebra,
    /// Weight parameter for history contributions
    pub temperature: f64,
}

impl PartitionFunction {
    /// Create a new partition function with the given algebra.
    pub fn new(algebra: FrobeniusAlgebra) -> Self {
        Self { algebra, temperature: 1.0 }
    }

    /// Set the temperature parameter.
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t.max(0.001);
        self
    }

    /// Compute Z(M) for a closed cobordism (scalar amplitude).
    /// For a matrix cobordism, Z = trace of the linear map.
    pub fn compute(&self, cobordism: &Cobordism) -> f64 {
        if cobordism.is_closed() && cobordism.input_dimension() == 0 && cobordism.output_dimension() == 0 {
            // Closed manifold: Z = 1 (normalized)
            return 1.0;
        }
        // For a cobordism with equal in/out dims, Z = trace
        if cobordism.input_dimension() == cobordism.output_dimension() {
            return cobordism.linear_map.trace();
        }
        0.0
    }

    /// Compute the amplitude for a specific agent history.
    /// Given input/output states, compute <output|Z(M)|input>.
    pub fn amplitude(
        &self,
        cobordism: &Cobordism,
        input_state: &DVector<f64>,
        output_state: &DVector<f64>,
    ) -> f64 {
        let evolved = cobordism.evaluate(input_state);
        output_state.dot(&evolved)
    }

    /// Weighted sum over all basis agent histories.
    pub fn weighted_history_sum(&self, cobordism: &Cobordism, histories: &[DVector<f64>]) -> f64 {
        let mut sum = 0.0;
        for history in histories {
            let amp = self.amplitude(cobordism, history, history);
            let weight = (-amp / self.temperature).exp();
            sum += weight * amp;
        }
        sum
    }

    /// Compute the Frobenius algebra value on a closed surface of genus g.
    /// Z(Σ_g) = ε(μ^{g}(Δ^{g}(η(1)))) in the Frobenius algebra.
    pub fn closed_surface_amplitude(&self, genus: usize) -> f64 {
        let n = self.algebra.dimension();
        let unit = self.algebra.unit(1.0);
        let mut state = unit;
        // Apply comultiplication genus times
        for _ in 0..genus {
            let comult = self.algebra.comultiply(&state);
            // Pair the outputs back via multiplication
            let mut paired = DVector::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    let coeff = comult[i * n + j];
                    let ei = DVector::from_fn(n, |r, _| if r == i { 1.0 } else { 0.0 });
                    let ej = DVector::from_fn(n, |r, _| if r == j { 1.0 } else { 0.0 });
                    let prod = self.algebra.multiply(&ei, &ej);
                    paired += coeff * prod;
                }
            }
            state = paired;
        }
        self.algebra.counit(&state)
    }

    /// Compute probability distribution over agent states.
    pub fn state_probabilities(&self, cobordism: &Cobordism, basis_states: &[DVector<f64>]) -> Vec<f64> {
        let amps: Vec<f64> = basis_states
            .iter()
            .map(|s| self.amplitude(cobordism, s, s))
            .collect();
        let weights: Vec<f64> = amps.iter().map(|a| (-a / self.temperature).exp()).collect();
        let total: f64 = weights.iter().sum();
        if total.abs() < 1e-15 {
            return vec![0.0; weights.len()];
        }
        weights.iter().map(|w| w / total).collect()
    }

    /// Expectation value of an observable over agent histories.
    pub fn expectation<F>(&self, cobordism: &Cobordism, basis_states: &[DVector<f64>], observable: F) -> f64
    where
        F: Fn(&DVector<f64>) -> f64,
    {
        let probs = self.state_probabilities(cobordism, basis_states);
        probs
            .iter()
            .zip(basis_states.iter())
            .map(|(p, s)| p * observable(s))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cobordism::{BoundaryComponent, Orientation};

    fn make_boundary(dim: usize) -> BoundaryComponent {
        BoundaryComponent {
            label: "b".into(),
            dimension: dim,
            orientation: Orientation::Ingoing,
        }
    }

    #[test]
    fn test_partition_function_creation() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        assert!((pf.temperature - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_temperature() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg).with_temperature(2.0);
        assert!((pf.temperature - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_closed_cobordism_amplitude() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let c = Cobordism::new(vec![], vec![], 0, DMatrix::identity(0, 0));
        assert!((pf.compute(&c) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace_amplitude() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let mat = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let c = Cobordism::new(
            vec![make_boundary(2)],
            vec![make_boundary(2)],
            0,
            mat,
        );
        assert!((pf.compute(&c) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_with_states() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let c = Cobordism::identity(vec![make_boundary(2)]);
        let input = DVector::from_vec(vec![1.0, 0.0]);
        let output = DVector::from_vec(vec![1.0, 0.0]);
        let amp = pf.amplitude(&c, &input, &output);
        assert!((amp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplitude_orthogonal() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let c = Cobordism::identity(vec![make_boundary(2)]);
        let input = DVector::from_vec(vec![1.0, 0.0]);
        let output = DVector::from_vec(vec![0.0, 1.0]);
        assert!(pf.amplitude(&c, &input, &output).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_history_sum() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let c = Cobordism::identity(vec![make_boundary(2)]);
        let histories = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let sum = pf.weighted_history_sum(&c, &histories);
        assert!(sum.is_finite());
    }

    #[test]
    fn test_closed_surface_genus_0() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let amp = pf.closed_surface_amplitude(0);
        // Z(S²) = ε(η(1)) = 1
        assert!((amp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_closed_surface_genus_1() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let amp = pf.closed_surface_amplitude(1);
        // Should give ε(μ(Δ(η(1))))
        assert!(amp.is_finite());
    }

    #[test]
    fn test_state_probabilities() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let c = Cobordism::identity(vec![make_boundary(2)]);
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let probs = pf.state_probabilities(&c, &basis);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expectation_value() {
        let alg = FrobeniusAlgebra::trivial();
        let pf = PartitionFunction::new(alg);
        let c = Cobordism::identity(vec![make_boundary(2)]);
        let basis = vec![
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let exp = pf.expectation(&c, &basis, |s| s[0]);
        assert!(exp.is_finite());
    }

    #[test]
    fn test_z2_genus_0() {
        let alg = FrobeniusAlgebra::z2();
        let pf = PartitionFunction::new(alg);
        let amp = pf.closed_surface_amplitude(0);
        assert!((amp - 1.0).abs() < 1e-10);
    }
}
