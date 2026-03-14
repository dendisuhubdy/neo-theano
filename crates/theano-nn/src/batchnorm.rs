//! Batch normalization layers.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_types::{Device, Result};

use crate::init;
use crate::module::Module;

/// 1D Batch Normalization. Like `torch.nn.BatchNorm1d`.
///
/// Normalizes over the batch dimension for inputs of shape [N, C] or [N, C, L].
pub struct BatchNorm1d {
    num_features: usize,
    eps: f64,
    momentum: f64,
    affine: bool,
    weight: Option<Variable>, // gamma
    bias: Option<Variable>,   // beta
    running_mean: Vec<f64>,
    running_var: Vec<f64>,
    training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            weight: Some(Variable::requires_grad(Tensor::ones(&[num_features]))),
            bias: Some(init::zeros(&[num_features])),
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            training: true,
        }
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Move this layer to a different device, returning a new BatchNorm1d.
    pub fn to(&self, device: &Device) -> Result<Self> {
        let weight = match &self.weight {
            Some(w) => Some(w.to(device)?),
            None => None,
        };
        let bias = match &self.bias {
            Some(b) => Some(b.to(device)?),
            None => None,
        };
        Ok(Self {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            affine: self.affine,
            weight,
            bias,
            running_mean: self.running_mean.clone(),
            running_var: self.running_var.clone(),
            training: self.training,
        })
    }

    /// Move to CPU.
    pub fn cpu(&self) -> Result<Self> {
        self.to(&Device::Cpu)
    }

    /// Move to CUDA device 0.
    pub fn cuda(&self) -> Result<Self> {
        self.to(&Device::Cuda(0))
    }

    /// Set training mode.
    pub fn set_training(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Check if in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape();
        assert!(
            shape.len() == 2 || shape.len() == 3,
            "BatchNorm1d expects 2D or 3D input, got {}D",
            shape.len()
        );

        let n = shape[0];
        let c = shape[1];
        assert_eq!(c, self.num_features, "expected {} features, got {}", self.num_features, c);

        if self.training {
            // Compute batch mean and variance
            let mean = input.mean_dim(0, false).unwrap();
            let diff = input.sub(&mean.reshape(&[1, c]).unwrap()).unwrap();
            let var = diff.mul(&diff).unwrap().mean_dim(0, false).unwrap();

            // Normalize: (x - mean) / sqrt(var + eps)
            let eps_tensor = Variable::new(Tensor::full(&[c], self.eps));
            let std = var.add(&eps_tensor).unwrap().sqrt().unwrap();
            let normalized = diff.div(&std.reshape(&[1, c]).unwrap()).unwrap();

            // Affine transform: gamma * normalized + beta
            if self.affine {
                let weight = self.weight.as_ref().unwrap();
                let bias = self.bias.as_ref().unwrap();
                let scaled = normalized.mul(&weight.reshape(&[1, c]).unwrap()).unwrap();
                scaled.add(&bias.reshape(&[1, c]).unwrap()).unwrap()
            } else {
                normalized
            }
        } else {
            // Use running stats
            let mean = Variable::new(Tensor::from_slice(&self.running_mean, &[c]));
            let var = Variable::new(Tensor::from_slice(&self.running_var, &[c]));
            let eps_tensor = Variable::new(Tensor::full(&[c], self.eps));

            let diff = input.sub(&mean.reshape(&[1, c]).unwrap()).unwrap();
            let std = var.add(&eps_tensor).unwrap().sqrt().unwrap();
            let normalized = diff.div(&std.reshape(&[1, c]).unwrap()).unwrap();

            if self.affine {
                let weight = self.weight.as_ref().unwrap();
                let bias = self.bias.as_ref().unwrap();
                let scaled = normalized.mul(&weight.reshape(&[1, c]).unwrap()).unwrap();
                scaled.add(&bias.reshape(&[1, c]).unwrap()).unwrap()
            } else {
                normalized
            }
        }
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = vec![];
        if let Some(ref w) = self.weight {
            params.push(w.clone());
        }
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        let mut params = vec![];
        if let Some(ref w) = self.weight {
            params.push(("weight".to_string(), w.clone()));
        }
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batchnorm1d_shape() {
        let bn = BatchNorm1d::new(4);
        let input = Variable::new(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        ));
        let output = bn.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 4]);
    }

    #[test]
    fn test_batchnorm1d_parameters() {
        let bn = BatchNorm1d::new(4);
        assert_eq!(bn.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn test_batchnorm1d_to_device() {
        let bn = BatchNorm1d::new(4);
        let bn_gpu = bn.to(&Device::Cuda(0)).unwrap();
        for param in bn_gpu.parameters() {
            assert_eq!(param.device(), &Device::Cuda(0));
        }

        let bn_cpu = bn_gpu.cpu().unwrap();
        for param in bn_cpu.parameters() {
            assert_eq!(param.device(), &Device::Cpu);
        }
    }

    #[test]
    fn test_batchnorm1d_train_eval() {
        let mut bn = BatchNorm1d::new(4);
        assert!(bn.is_training());

        bn.eval();
        assert!(!bn.is_training());

        bn.train();
        assert!(bn.is_training());

        bn.set_training(false);
        assert!(!bn.is_training());
    }

    #[test]
    fn test_batchnorm1d_eval_mode() {
        let mut bn = BatchNorm1d::new(4);
        bn.eval();

        // In eval mode, uses running stats (default: mean=0, var=1)
        let input = Variable::new(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        ));
        let output = bn.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 4]);
    }

    #[test]
    fn test_batchnorm1d_named_parameters() {
        let bn = BatchNorm1d::new(4);
        let named = bn.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_batchnorm1d_state_dict() {
        let bn = BatchNorm1d::new(4);
        let sd = bn.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape(), &[4]);
    }
}
