//! Agent boundaries — vector spaces assigned to agent interfaces.
//!
//! Each agent has a boundary (interface with the world). In TQFT,
//! this boundary is assigned a vector space. The dimension of this
//! space corresponds to the number of distinct agent states.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::frobenius::FrobeniusAlgebra;

/// An agent with a boundary that is a vector space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentBoundary {
    /// Unique agent identifier
    pub id: String,
    /// Dimension of the agent's state space
    pub state_dimension: usize,
    /// Current state vector
    pub state: Vec<f64>,
    /// Input channels (other agents this one receives from)
    pub input_channels: Vec<String>,
    /// Output channels (other agents this one sends to)
    pub output_channels: Vec<String>,
}

impl AgentBoundary {
    /// Create a new agent boundary.
    pub fn new(id: impl Into<String>, dimension: usize) -> Self {
        Self {
            id: id.into(),
            state_dimension: dimension,
            state: vec![1.0; dimension], // default to unit vector sum
            input_channels: Vec::new(),
            output_channels: Vec::new(),
        }
    }

    /// Get the state as a nalgebra vector.
    pub fn state_vector(&self) -> DVector<f64> {
        DVector::from_vec(self.state.clone())
    }

    /// Set the state from a vector.
    pub fn set_state(&mut self, v: &DVector<f64>) {
        assert_eq!(v.nrows(), self.state_dimension);
        self.state = v.iter().cloned().collect();
    }

    /// Connect this agent's output to another's input.
    pub fn connect_to(&mut self, other: &mut Self) {
        if !self.output_channels.contains(&other.id) {
            self.output_channels.push(other.id.clone());
        }
        if !other.input_channels.contains(&self.id) {
            other.input_channels.push(self.id.clone());
        }
    }

    /// The boundary vector space dimension.
    pub fn dimension(&self) -> usize {
        self.state_dimension
    }

    /// Normalize the agent's state.
    pub fn normalize(&mut self) {
        let v = self.state_vector();
        let norm = v.norm();
        if norm > 1e-15 {
            let normalized = v / norm;
            self.set_state(&normalized);
        }
    }

    /// Compute inner product with another agent's state.
    pub fn inner_product(&self, other: &AgentBoundary) -> f64 {
        assert_eq!(self.state_dimension, other.state_dimension);
        self.state_vector().dot(&other.state_vector())
    }
}

/// A network of agents connected via cobordisms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentNetwork {
    /// Agents indexed by ID
    pub agents: Vec<AgentBoundary>,
    /// Frobenius algebra governing interactions
    pub algebra_dimension: usize,
}

impl AgentNetwork {
    /// Create an empty network.
    pub fn new(algebra_dimension: usize) -> Self {
        Self {
            agents: Vec::new(),
            algebra_dimension,
        }
    }

    /// Add an agent to the network.
    pub fn add_agent(&mut self, agent: AgentBoundary) {
        self.agents.push(agent);
    }

    /// Get agent by ID.
    pub fn get_agent(&self, id: &str) -> Option<&AgentBoundary> {
        self.agents.iter().find(|a| a.id == id)
    }

    /// Get mutable agent by ID.
    pub fn get_agent_mut(&mut self, id: &str) -> Option<&mut AgentBoundary> {
        self.agents.iter_mut().find(|a| a.id == id)
    }

    /// Total state space dimension (tensor product of all agents).
    pub fn total_dimension(&self) -> usize {
        if self.agents.is_empty() {
            0
        } else {
            self.agents.iter().map(|a| a.state_dimension).product()
        }
    }

    /// Compute the joint state as tensor product.
    pub fn joint_state(&self) -> DVector<f64> {
        if self.agents.is_empty() {
            return DVector::zeros(0);
        }
        let mut result = self.agents[0].state_vector();
        for agent in &self.agents[1..] {
            let s = agent.state_vector();
            let mut tensor = DVector::zeros(result.nrows() * s.nrows());
            for i in 0..result.nrows() {
                for j in 0..s.nrows() {
                    tensor[i * s.nrows() + j] = result[i] * s[j];
                }
            }
            result = tensor;
        }
        result
    }

    /// Interact two agents using the Frobenius algebra multiplication.
    pub fn interact(&self, id_a: &str, id_b: &str, algebra: &FrobeniusAlgebra) -> Option<DVector<f64>> {
        let a = self.get_agent(id_a)?;
        let b = self.get_agent(id_b)?;
        Some(algebra.multiply(&a.state_vector(), &b.state_vector()))
    }

    /// Number of agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether network is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

impl fmt::Display for AgentBoundary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agent({}, dim={})", self.id, self.state_dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let a = AgentBoundary::new("a1", 3);
        assert_eq!(a.id, "a1");
        assert_eq!(a.dimension(), 3);
    }

    #[test]
    fn test_state_vector() {
        let mut a = AgentBoundary::new("a1", 2);
        a.state = vec![1.0, 2.0];
        let v = a.state_vector();
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_state() {
        let mut a = AgentBoundary::new("a1", 2);
        a.set_state(&DVector::from_vec(vec![3.0, 4.0]));
        assert!((a.state[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_connect() {
        let mut a = AgentBoundary::new("a", 2);
        let mut b = AgentBoundary::new("b", 2);
        a.connect_to(&mut b);
        assert!(a.output_channels.contains(&"b".to_string()));
        assert!(b.input_channels.contains(&"a".to_string()));
    }

    #[test]
    fn test_normalize() {
        let mut a = AgentBoundary::new("a", 2);
        a.state = vec![3.0, 4.0];
        a.normalize();
        let v = a.state_vector();
        assert!((v.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let mut a = AgentBoundary::new("a", 2);
        let mut b = AgentBoundary::new("b", 2);
        a.state = vec![1.0, 0.0];
        b.state = vec![0.0, 1.0];
        assert!(a.inner_product(&b).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product_parallel() {
        let mut a = AgentBoundary::new("a", 2);
        let mut b = AgentBoundary::new("b", 2);
        a.state = vec![1.0, 2.0];
        b.state = vec![2.0, 4.0];
        assert!(a.inner_product(&b) > 0.0);
    }

    #[test]
    fn test_network_creation() {
        let net = AgentNetwork::new(2);
        assert!(net.is_empty());
    }

    #[test]
    fn test_network_add_agent() {
        let mut net = AgentNetwork::new(2);
        net.add_agent(AgentBoundary::new("a1", 2));
        assert_eq!(net.len(), 1);
    }

    #[test]
    fn test_network_get_agent() {
        let mut net = AgentNetwork::new(2);
        net.add_agent(AgentBoundary::new("a1", 2));
        assert!(net.get_agent("a1").is_some());
        assert!(net.get_agent("missing").is_none());
    }

    #[test]
    fn test_network_total_dimension() {
        let mut net = AgentNetwork::new(2);
        net.add_agent(AgentBoundary::new("a", 2));
        net.add_agent(AgentBoundary::new("b", 3));
        assert_eq!(net.total_dimension(), 6);
    }

    #[test]
    fn test_network_joint_state() {
        let mut net = AgentNetwork::new(2);
        let mut a = AgentBoundary::new("a", 2);
        a.state = vec![1.0, 0.0];
        let mut b = AgentBoundary::new("b", 2);
        b.state = vec![0.0, 1.0];
        net.add_agent(a);
        net.add_agent(b);
        let joint = net.joint_state();
        assert_eq!(joint.nrows(), 4);
    }

    #[test]
    fn test_network_interact() {
        let mut net = AgentNetwork::new(2);
        net.add_agent(AgentBoundary::new("a", 2));
        net.add_agent(AgentBoundary::new("b", 2));
        let alg = FrobeniusAlgebra::z2();
        let result = net.interact("a", "b", &alg);
        assert!(result.is_some());
    }

    #[test]
    fn test_display() {
        let a = AgentBoundary::new("x", 4);
        assert!(format!("{}", a).contains("x"));
    }

    #[test]
    fn test_serialization() {
        let a = AgentBoundary::new("a", 3);
        let json = serde_json::to_string(&a).unwrap();
        let back: AgentBoundary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "a");
    }
}
