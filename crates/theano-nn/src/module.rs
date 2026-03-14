use std::collections::HashMap;

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_types::{Device, Result};

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

    /// Move all parameters to a different device, returning a new module.
    ///
    /// Like `model.to(device)` in PyTorch. Since Rust modules own their
    /// parameters as `Variable`, this creates a new module with parameters
    /// transferred to the target device.
    ///
    /// Individual layer types implement this by transferring their weight/bias
    /// Variables. The default implementation is not available on the trait
    /// directly because Module is object-safe and `to()` returns `Self`.
    /// Use the layer-specific `.to()` methods or `module_to_device()` helper.
    fn to_device_params(&self, device: &Device) -> Result<Vec<Variable>> {
        self.parameters()
            .iter()
            .map(|p| p.to(device))
            .collect()
    }

    /// Return the state dict: a map from parameter names to their tensors.
    ///
    /// Like `model.state_dict()` in PyTorch. Useful for saving/loading models.
    fn state_dict(&self) -> HashMap<String, Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(name, var)| (name, var.into_tensor()))
            .collect()
    }
}
