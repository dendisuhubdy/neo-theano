//! Activation functions as nn modules.

use theano_autograd::Variable;

use crate::module::Module;

/// ReLU activation. Like `torch.nn.ReLU`.
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Variable) -> Variable {
        input.relu().unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

/// Sigmoid activation. Like `torch.nn.Sigmoid`.
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Variable {
        input.sigmoid().unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

/// Tanh activation. Like `torch.nn.Tanh`.
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Variable {
        input.tanh().unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

/// GELU activation. Like `torch.nn.GELU`.
pub struct GELU;

impl Module for GELU {
    fn forward(&self, input: &Variable) -> Variable {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let x2 = input.mul(input).unwrap();
        let x3 = x2.mul(input).unwrap();
        let inner = input.add(&x3.mul_scalar(0.044715).unwrap()).unwrap();
        let scaled = inner.mul_scalar((2.0_f64 / std::f64::consts::PI).sqrt()).unwrap();
        let tanh_val = scaled.tanh().unwrap();
        let one_plus = tanh_val.add_scalar(1.0).unwrap();
        input.mul(&one_plus).unwrap().mul_scalar(0.5).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

/// SiLU (Swish) activation. Like `torch.nn.SiLU`.
pub struct SiLU;

impl Module for SiLU {
    fn forward(&self, input: &Variable) -> Variable {
        // SiLU(x) = x * sigmoid(x)
        let sig = input.sigmoid().unwrap();
        input.mul(&sig).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

/// Softmax activation. Like `torch.nn.Softmax`.
pub struct Softmax {
    dim: i64,
}

impl Softmax {
    pub fn new(dim: i64) -> Self {
        Self { dim }
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Variable) -> Variable {
        input.softmax(self.dim).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

/// LogSoftmax activation. Like `torch.nn.LogSoftmax`.
pub struct LogSoftmax {
    dim: i64,
}

impl LogSoftmax {
    pub fn new(dim: i64) -> Self {
        Self { dim }
    }
}

impl Module for LogSoftmax {
    fn forward(&self, input: &Variable) -> Variable {
        let sm = input.softmax(self.dim).unwrap();
        sm.log().unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theano_core::Tensor;

    #[test]
    fn test_relu() {
        let relu = ReLU;
        let input = Variable::new(Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]));
        let output = relu.forward(&input);
        assert_eq!(output.tensor().to_vec_f64().unwrap(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let sig = Sigmoid;
        let input = Variable::new(Tensor::scalar(0.0));
        let output = sig.forward(&input);
        let v = output.tensor().item().unwrap();
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let sm = Softmax::new(0);
        let input = Variable::new(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let output = sm.forward(&input);
        let data = output.tensor().to_vec_f64().unwrap();
        let sum: f64 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
