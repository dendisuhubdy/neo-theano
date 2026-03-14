//! Normalization layers.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_types::{Device, Result};
use crate::init;
use crate::module::Module;

/// Layer Normalization. Like `torch.nn.LayerNorm`.
/// Normalizes over the last D dimensions.
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f64,
    weight: Variable,
    bias: Variable,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let weight = Variable::requires_grad(Tensor::ones(&normalized_shape));
        let bias = init::zeros(&normalized_shape);
        Self {
            normalized_shape,
            eps: 1e-5,
            weight,
            bias,
        }
    }

    /// Move this layer to a different device, returning a new LayerNorm.
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            normalized_shape: self.normalized_shape.clone(),
            eps: self.eps,
            weight: self.weight.to(device)?,
            bias: self.bias.to(device)?,
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

    /// Reconstruct a LayerNorm from pre-trained weight and bias tensors.
    pub fn from_tensors(weight: Tensor, bias: Tensor) -> Self {
        let normalized_shape = weight.shape().to_vec();
        Self {
            normalized_shape,
            eps: 1e-5,
            weight: Variable::requires_grad(weight),
            bias: Variable::requires_grad(bias),
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape().to_vec();
        let ndim = shape.len();
        let norm_dims = self.normalized_shape.len();
        let reduce_start = ndim - norm_dims;

        let outer_size: usize = shape[..reduce_start].iter().product();
        let inner_size: usize = shape[reduce_start..].iter().product();

        // Reshape to [outer, inner] for mean/var computation
        let flat = input.reshape(&[outer_size, inner_size]).unwrap();
        let mean = flat.mean_dim(1, true).unwrap();
        let diff = flat.sub(&mean).unwrap();
        let var = diff.mul(&diff).unwrap().mean_dim(1, true).unwrap();
        let eps = Variable::new(Tensor::full(&[1, 1], self.eps));
        let std = var.add(&eps).unwrap().sqrt().unwrap();
        let normalized = diff.div(&std).unwrap();

        // Reshape back to original shape
        let normalized = normalized.reshape(&shape).unwrap();

        // Apply affine: weight and bias have shape normalized_shape
        // Need to broadcast: prepend 1s for outer dims
        let mut broadcast_shape = vec![1usize; reduce_start];
        broadcast_shape.extend_from_slice(&self.normalized_shape);
        let weight = self.weight.reshape(&broadcast_shape).unwrap();
        let bias = self.bias.reshape(&broadcast_shape).unwrap();

        normalized.mul(&weight).unwrap().add(&bias).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }
}

/// Group Normalization. Like `torch.nn.GroupNorm`.
/// Divides channels into groups and normalizes within each group.
pub struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    weight: Variable,
    bias: Variable,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        assert_eq!(num_channels % num_groups, 0, "num_channels must be divisible by num_groups");
        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            weight: Variable::requires_grad(Tensor::ones(&[num_channels])),
            bias: init::zeros(&[num_channels]),
        }
    }
}

impl GroupNorm {
    /// Move this layer to a different device, returning a new GroupNorm.
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            num_groups: self.num_groups,
            num_channels: self.num_channels,
            eps: self.eps,
            weight: self.weight.to(device)?,
            bias: self.bias.to(device)?,
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
}

impl Module for GroupNorm {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape().to_vec();
        assert!(shape.len() >= 2);
        let n = shape[0];
        let c = shape[1];
        assert_eq!(c, self.num_channels);

        let spatial: usize = shape[2..].iter().product();
        let cpg = c / self.num_groups;
        let group_size = cpg * spatial;

        // Reshape to [N, G, cpg*spatial] for per-group normalization
        let reshaped = input.reshape(&[n, self.num_groups, group_size]).unwrap();
        let mean = reshaped.mean_dim(2, true).unwrap();
        let diff = reshaped.sub(&mean).unwrap();
        let var = diff.mul(&diff).unwrap().mean_dim(2, true).unwrap();
        let eps = Variable::new(Tensor::full(&[1, 1, 1], self.eps));
        let std = var.add(&eps).unwrap().sqrt().unwrap();
        let normalized = diff.div(&std).unwrap();

        // Reshape back to original shape
        let normalized = normalized.reshape(&shape).unwrap();

        // Apply per-channel affine: weight/bias are [C], broadcast to [1, C, 1, ...]
        let mut weight_shape = vec![1usize; shape.len()];
        weight_shape[1] = c;
        let weight = self.weight.reshape(&weight_shape).unwrap();
        let bias = self.bias.reshape(&weight_shape).unwrap();

        normalized.mul(&weight).unwrap().add(&bias).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_shape() {
        let ln = LayerNorm::new(vec![4]);
        let input = Variable::new(Tensor::ones(&[2, 4]));
        let output = ln.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 4]);
    }

    #[test]
    fn test_layernorm_params() {
        let ln = LayerNorm::new(vec![8]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_groupnorm_shape() {
        let gn = GroupNorm::new(2, 4);
        let input = Variable::new(Tensor::ones(&[2, 4, 3, 3]));
        let output = gn.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 4, 3, 3]);
    }

    #[test]
    fn test_layernorm_to_device() {
        let ln = LayerNorm::new(vec![8]);
        let ln_gpu = ln.to(&Device::Cuda(0)).unwrap();
        for param in ln_gpu.parameters() {
            assert_eq!(param.device(), &Device::Cuda(0));
        }

        let ln_cpu = ln_gpu.cpu().unwrap();
        for param in ln_cpu.parameters() {
            assert_eq!(param.device(), &Device::Cpu);
        }
    }

    #[test]
    fn test_groupnorm_to_device() {
        let gn = GroupNorm::new(2, 4);
        let gn_gpu = gn.to(&Device::Cuda(0)).unwrap();
        for param in gn_gpu.parameters() {
            assert_eq!(param.device(), &Device::Cuda(0));
        }

        let gn_cpu = gn_gpu.cpu().unwrap();
        for param in gn_cpu.parameters() {
            assert_eq!(param.device(), &Device::Cpu);
        }
    }

    #[test]
    fn test_layernorm_named_parameters() {
        let ln = LayerNorm::new(vec![8]);
        let named = ln.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_groupnorm_named_parameters() {
        let gn = GroupNorm::new(2, 4);
        let named = gn.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_layernorm_state_dict() {
        let ln = LayerNorm::new(vec![8]);
        let sd = ln.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape(), &[8]);
    }
}
