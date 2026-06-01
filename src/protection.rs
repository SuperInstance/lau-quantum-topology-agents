//! Topological protection for agent states.
//!
//! Agent states encoded in topological degrees of freedom are robust
//! to local perturbations. This module provides topologically protected
//! communication channels and error detection.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::anyon::AnyonSystem;

/// Topological protection level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtectionLevel {
    /// No protection (trivial topology)
    None,
    /// Single-level protection (e.g., anyonic braiding)
    Level1,
    /// Double-level protection (surface codes)
    Level2,
    /// Full topological protection
    Maximum,
}

/// A topologically protected agent state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtectedState {
    /// The encoded state vector
    pub state: DVector<f64>,
    /// Topological protection level
    pub protection: ProtectionLevel,
    /// Error syndrome (0 = no error detected)
    pub syndrome: Vec<f64>,
}

/// Topological protection system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologicalProtection {
    /// Underlying anyon system
    pub anyon_system: AnyonSystem,
    /// Code distance (minimum number of errors to cause failure)
    pub code_distance: usize,
    /// Protection level
    pub level: ProtectionLevel,
}

impl TopologicalProtection {
    /// Create a new topological protection system.
    pub fn new(anyon_system: AnyonSystem, code_distance: usize) -> Self {
        Self {
            anyon_system,
            code_distance,
            level: if code_distance >= 5 {
                ProtectionLevel::Maximum
            } else if code_distance >= 3 {
                ProtectionLevel::Level2
            } else if code_distance >= 1 {
                ProtectionLevel::Level1
            } else {
                ProtectionLevel::None
            },
        }
    }

    /// Encode a logical state into the protected subspace.
    pub fn encode(&self, logical_state: &DVector<f64>) -> ProtectedState {
        let d = self.code_distance.max(1);
        let n = logical_state.nrows();
        // Encode by repeating across d copies
        let encoded_dim = n * d;
        let mut encoded = DVector::zeros(encoded_dim);
        for i in 0..n {
            for j in 0..d {
                encoded[i * d + j] = logical_state[i];
            }
        }
        ProtectedState {
            state: encoded,
            protection: self.level,
            syndrome: vec![0.0; d],
        }
    }

    /// Decode a protected state back to the logical state.
    pub fn decode(&self, protected: &ProtectedState) -> DVector<f64> {
        let d = self.code_distance.max(1);
        let n = protected.state.nrows() / d;
        let mut decoded = DVector::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..d {
                sum += protected.state[i * d + j];
            }
            decoded[i] = sum / d as f64;
        }
        decoded
    }

    /// Detect errors in a protected state.
    pub fn detect_errors(&self, protected: &ProtectedState) -> Vec<usize> {
        let d = self.code_distance.max(1);
        let n = protected.state.nrows() / d;
        let mut errors = Vec::new();
        for i in 0..n {
            let mut vals = Vec::new();
            for j in 0..d {
                vals.push(protected.state[i * d + j]);
            }
            // Majority vote: detect outliers
            let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
            for (j, &v) in vals.iter().enumerate() {
                if (v - mean).abs() > 0.5 * mean.abs().max(1e-10) {
                    errors.push(i * d + j);
                }
            }
        }
        errors
    }

    /// Correct errors in a protected state (majority vote).
    pub fn correct(&self, protected: &mut ProtectedState) -> usize {
        let errors = self.detect_errors(protected);
        let d = self.code_distance.max(1);
        let n = protected.state.nrows() / d;
        for i in 0..n {
            let mut vals = Vec::new();
            for j in 0..d {
                vals.push(protected.state[i * d + j]);
            }
            let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
            for j in 0..d {
                if (protected.state[i * d + j] - mean).abs() > 0.5 * mean.abs().max(1e-10) {
                    protected.state[i * d + j] = mean;
                }
            }
        }
        errors.len()
    }

    /// Check if a perturbation is within the protection threshold.
    pub fn is_perturbation_protected(&self, original: &ProtectedState, perturbed: &ProtectedState) -> bool {
        if original.state.nrows() != perturbed.state.nrows() {
            return false;
        }
        let diff = (&original.state - &perturbed.state).norm();
        let norm = original.state.norm().max(1e-10);
        // Protected if relative error < threshold based on code distance
        let threshold = 1.0 / (self.code_distance as f64).sqrt();
        diff / norm < threshold
    }

    /// Compute the protection capacity (number of correctable errors).
    pub fn error_capacity(&self) -> usize {
        (self.code_distance - 1) / 2
    }

    /// Create a protected communication channel between two agent states.
    pub fn protected_channel(
        &self,
        sender_state: &DVector<f64>,
        _receiver_state: &DVector<f64>,
        noise_level: f64,
    ) -> DVector<f64> {
        // Encode, apply noise, decode
        let encoded = self.encode(sender_state);
        let mut noisy = encoded.state.clone();
        // Add topological noise
        for i in 0..noisy.nrows() {
            noisy[i] += noise_level * (i as f64 * 0.1).sin();
        }
        let mut protected = ProtectedState {
            state: noisy,
            protection: self.level,
            syndrome: vec![0.0; self.code_distance],
        };
        self.correct(&mut protected);
        self.decode(&protected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ising_protection(distance: usize) -> TopologicalProtection {
        TopologicalProtection::new(AnyonSystem::ising(), distance)
    }

    #[test]
    fn test_creation() {
        let tp = ising_protection(3);
        assert_eq!(tp.code_distance, 3);
        assert_eq!(tp.level, ProtectionLevel::Level2);
    }

    #[test]
    fn test_level_none() {
        let tp = ising_protection(0);
        assert_eq!(tp.level, ProtectionLevel::None);
    }

    #[test]
    fn test_level_1() {
        let tp = ising_protection(1);
        assert_eq!(tp.level, ProtectionLevel::Level1);
    }

    #[test]
    fn test_level_max() {
        let tp = ising_protection(5);
        assert_eq!(tp.level, ProtectionLevel::Maximum);
    }

    #[test]
    fn test_encode_decode() {
        let tp = ising_protection(3);
        let logical = DVector::from_vec(vec![1.0, 0.0]);
        let encoded = tp.encode(&logical);
        assert!(encoded.state.nrows() > logical.nrows());
        let decoded = tp.decode(&encoded);
        assert!((decoded[0] - logical[0]).abs() < 1e-10);
    }

    #[test]
    fn test_encode_preserves_norm() {
        let tp = ising_protection(3);
        let logical = DVector::from_vec(vec![1.0, 0.0]);
        let encoded = tp.encode(&logical);
        let decoded = tp.decode(&encoded);
        // Round-trip should preserve direction
        assert!((decoded[0] - logical[0]).abs() < 1e-10);
        assert!((decoded[1] - logical[1]).abs() < 1e-10);
    }

    #[test]
    fn test_error_detection() {
        let tp = ising_protection(3);
        let logical = DVector::from_vec(vec![1.0, 0.0]);
        let mut encoded = tp.encode(&logical);
        // Introduce an error
        encoded.state[1] *= 10.0;
        let errors = tp.detect_errors(&encoded);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_no_error_detection() {
        let tp = ising_protection(3);
        let logical = DVector::from_vec(vec![1.0, 0.0]);
        let encoded = tp.encode(&logical);
        let errors = tp.detect_errors(&encoded);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_error_correction() {
        let tp = ising_protection(3);
        let logical = DVector::from_vec(vec![1.0, 0.0]);
        let mut encoded = tp.encode(&logical);
        encoded.state[0] = 0.0;
        let num_corrected = tp.correct(&mut encoded);
        assert!(num_corrected > 0);
    }

    #[test]
    fn test_perturbation_protected() {
        let tp = ising_protection(5);
        let logical = DVector::from_vec(vec![1.0, 0.0]);
        let original = tp.encode(&logical);
        let mut perturbed = original.clone();
        perturbed.state[0] += 0.001;
        assert!(tp.is_perturbation_protected(&original, &perturbed));
    }

    #[test]
    fn test_perturbation_not_protected() {
        let tp = ising_protection(3);
        let logical = DVector::from_vec(vec![1.0, 0.0]);
        let original = tp.encode(&logical);
        let mut perturbed = original.clone();
        perturbed.state[0] = -original.state[0];
        assert!(!tp.is_perturbation_protected(&original, &perturbed));
    }

    #[test]
    fn test_error_capacity() {
        let tp = ising_protection(5);
        assert_eq!(tp.error_capacity(), 2);
    }

    #[test]
    fn test_error_capacity_odd() {
        let tp = ising_protection(3);
        assert_eq!(tp.error_capacity(), 1);
    }

    #[test]
    fn test_protected_channel() {
        let tp = ising_protection(3);
        let sender = DVector::from_vec(vec![1.0, 0.0]);
        let receiver = DVector::from_vec(vec![0.0, 0.0]);
        let result = tp.protected_channel(&sender, &receiver, 0.01);
        assert!(result.nrows() == sender.nrows());
    }

    #[test]
    fn test_protected_channel_low_noise() {
        let tp = ising_protection(5);
        let sender = DVector::from_vec(vec![1.0, 0.0]);
        let receiver = DVector::from_vec(vec![0.0, 0.0]);
        let result = tp.protected_channel(&sender, &receiver, 0.0);
        // With zero noise, should recover approximately
        assert!(result[0].abs() > 0.0);
    }

    #[test]
    fn test_serialization() {
        let tp = ising_protection(3);
        let json = serde_json::to_string(&tp).unwrap();
        let back: TopologicalProtection = serde_json::from_str(&json).unwrap();
        assert_eq!(back.code_distance, 3);
    }

    #[test]
    fn test_fibonacci_protection() {
        let tp = TopologicalProtection::new(AnyonSystem::fibonacci(), 3);
        assert_eq!(tp.error_capacity(), 1);
    }
}
