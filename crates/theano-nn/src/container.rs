//! Container modules.

use theano_autograd::Variable;

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
}
