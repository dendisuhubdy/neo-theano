use std::collections::HashMap;

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_types::Device;

/// Trait for neural network modules, mirroring PyTorch's `nn.Module`.
pub trait Module: Send + Sync {
    /// Forward pass.
    fn forward(&self, input: &Variable) -> Variable;

    /// Get all trainable parameters.
    fn parameters(&self) -> Vec<Variable>;

    /// Get named parameters.
    fn named_parameters(&self) -> Vec<(String, Variable)> {
        self.parameters()
            .into_iter()
            .enumerate()
            .map(|(i, p)| (format!("param_{i}"), p))
            .collect()
    }

    /// Number of trainable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.tensor().numel()).sum()
    }
}
