//! Linear (fully connected) layer — `nn.Linear`.

use theano_autograd::Variable;
use theano_core::Tensor;

use crate::init;
use crate::module::Module;

/// Applies a linear transformation: y = xW^T + b.
///
/// Like `torch.nn.Linear`.
pub struct Linear {
    weight: Variable,
    bias: Option<Variable>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Create a new Linear layer.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = init::kaiming_uniform(&[out_features, in_features], in_features);
        let bias = Some(init::uniform_init(
            &[out_features],
            -(1.0 / (in_features as f64).sqrt()),
            1.0 / (in_features as f64).sqrt(),
        ));
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Create a Linear layer without bias.
    pub fn no_bias(in_features: usize, out_features: usize) -> Self {
        let weight = init::kaiming_uniform(&[out_features, in_features], in_features);
        Self {
            weight,
            bias: None,
            in_features,
            out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn weight(&self) -> &Variable {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Variable> {
        self.bias.as_ref()
    }

    /// Reconstruct a Linear layer from pre-trained tensors.
    pub fn from_tensors(weight: Tensor, bias: Option<Tensor>) -> Self {
        let shape = weight.shape().to_vec();
        let out_features = shape[0];
        let in_features = shape[1];
        Self {
            weight: Variable::requires_grad(weight),
            bias: bias.map(Variable::requires_grad),
            in_features,
            out_features,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Variable) -> Variable {
        // y = x @ W^T + b
        let w_t = self.weight.t().unwrap();
        let out = input.matmul(&w_t).unwrap();
        if let Some(ref bias) = self.bias {
            out.add(bias).unwrap()
        } else {
            out
        }
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref bias) = self.bias {
            params.push(("bias".to_string(), bias.clone()));
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_shape() {
        let layer = Linear::new(10, 5);
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert_eq!(layer.weight().tensor().shape(), &[5, 10]);
        assert!(layer.bias().is_some());
    }

    #[test]
    fn test_linear_forward() {
        let layer = Linear::new(3, 2);
        let input = Variable::new(Tensor::ones(&[4, 3]));
        let output = layer.forward(&input);
        assert_eq!(output.tensor().shape(), &[4, 2]);
    }

    #[test]
    fn test_linear_no_bias() {
        let layer = Linear::no_bias(3, 2);
        assert!(layer.bias().is_none());
        assert_eq!(layer.parameters().len(), 1);
    }

    #[test]
    fn test_linear_parameters() {
        let layer = Linear::new(3, 2);
        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_linear_backward() {
        let layer = Linear::new(3, 2);
        let input = Variable::requires_grad(Tensor::ones(&[1, 3]));
        let output = layer.forward(&input);
        let loss = output.sum().unwrap();
        loss.backward();

        // Weight should have a gradient
        assert!(layer.weight().grad().is_some());
    }
}
