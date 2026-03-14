//! Container modules.

use theano_autograd::Variable;
use theano_types::{Device, Result};

use crate::module::Module;

/// Sequential container. Like `torch.nn.Sequential`.
///
/// Chains modules in order: output of one is input to the next.
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(modules: Vec<Box<dyn Module>>) -> Self {
        Self { modules }
    }

    /// Add a module to the end of the sequence.
    pub fn add(mut self, module: impl Module + 'static) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Number of modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Transfer all parameters in the sequential to a different device.
    ///
    /// Since Sequential uses `Box<dyn Module>`, we cannot reconstruct a new
    /// Sequential with moved layers. Instead, this transfers each parameter
    /// Variable to the target device and returns the new parameter list.
    ///
    /// Like calling `model.to(device)` in PyTorch for the parameter transfer.
    pub fn to_device_params(&self, device: &Device) -> Result<Vec<Variable>> {
        self.modules
            .iter()
            .flat_map(|m| m.parameters())
            .map(|p| p.to(device))
            .collect()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Variable) -> Variable {
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Variable> {
        self.modules
            .iter()
            .flat_map(|m| m.parameters())
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        self.modules
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                m.named_parameters()
                    .into_iter()
                    .map(move |(name, var)| (format!("{i}.{name}"), var))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, ReLU};
    use theano_core::Tensor;

    #[test]
    fn test_sequential() {
        let model = Sequential::new(vec![])
            .add(Linear::new(10, 5))
            .add(ReLU)
            .add(Linear::new(5, 2));

        let input = Variable::new(Tensor::ones(&[1, 10]));
        let output = model.forward(&input);
        assert_eq!(output.tensor().shape(), &[1, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let model = Sequential::new(vec![])
            .add(Linear::new(10, 5))
            .add(ReLU)
            .add(Linear::new(5, 2));

        let params = model.parameters();
        // Linear(10,5): weight + bias = 2
        // ReLU: 0
        // Linear(5,2): weight + bias = 2
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_sequential_backward() {
        let model = Sequential::new(vec![])
            .add(Linear::new(4, 3))
            .add(ReLU)
            .add(Linear::new(3, 1));

        let input = Variable::requires_grad(Tensor::ones(&[2, 4]));
        let output = model.forward(&input);
        let loss = output.sum().unwrap();
        loss.backward();

        // All linear weight params should have gradients
        for param in model.parameters() {
            assert!(param.grad().is_some(), "parameter should have gradient");
        }
    }

    #[test]
    fn test_sequential_to_device_params() {
        use theano_types::Device;

        let model = Sequential::new(vec![])
            .add(Linear::new(10, 5))
            .add(ReLU)
            .add(Linear::new(5, 2));

        let gpu_params = model.to_device_params(&Device::Cuda(0)).unwrap();
        assert_eq!(gpu_params.len(), 4); // 2 weights + 2 biases
        for param in &gpu_params {
            assert_eq!(param.device(), &Device::Cuda(0));
        }
    }

    #[test]
    fn test_sequential_named_parameters() {
        let model = Sequential::new(vec![])
            .add(Linear::new(10, 5))
            .add(ReLU)
            .add(Linear::new(5, 2));

        let named = model.named_parameters();
        assert_eq!(named.len(), 4);
        // First Linear's weight and bias get prefix "0."
        assert_eq!(named[0].0, "0.weight");
        assert_eq!(named[1].0, "0.bias");
        // ReLU has no params, so next Linear gets prefix "2."
        assert_eq!(named[2].0, "2.weight");
        assert_eq!(named[3].0, "2.bias");
    }

    #[test]
    fn test_sequential_state_dict() {
        let model = Sequential::new(vec![])
            .add(Linear::new(4, 3))
            .add(ReLU)
            .add(Linear::new(3, 1));

        let sd = model.state_dict();
        assert_eq!(sd.len(), 4);
        assert!(sd.contains_key("0.weight"));
        assert!(sd.contains_key("0.bias"));
        assert!(sd.contains_key("2.weight"));
        assert!(sd.contains_key("2.bias"));
    }
}
