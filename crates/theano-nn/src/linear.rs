//! Linear (fully connected) layer — `nn.Linear`.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_types::{Device, Result};

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

    /// Move this layer to a different device, returning a new Linear layer.
    ///
    /// Like `layer.to(device)` in PyTorch.
    ///
    /// # Examples
    /// ```ignore
    /// let layer = Linear::new(784, 256);
    /// let layer_gpu = layer.to(&Device::Cuda(0))?;
    /// ```
    pub fn to(&self, device: &Device) -> Result<Self> {
        let weight = self.weight.to(device)?;
        let bias = match &self.bias {
            Some(b) => Some(b.to(device)?),
            None => None,
        };
        Ok(Self {
            weight,
            bias,
            in_features: self.in_features,
            out_features: self.out_features,
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

    #[test]
    fn test_linear_to_device() {
        use theano_types::Device;

        let layer = Linear::new(10, 5);
        let layer_gpu = layer.to(&Device::Cuda(0)).unwrap();

        // Verify shapes preserved
        assert_eq!(layer_gpu.in_features(), 10);
        assert_eq!(layer_gpu.out_features(), 5);

        // Verify weights transferred
        assert_eq!(layer_gpu.weight().device(), &Device::Cuda(0));
        assert!(layer_gpu.bias().is_some());
        assert_eq!(layer_gpu.bias().unwrap().device(), &Device::Cuda(0));

        // Verify data preserved
        let orig_w = layer.weight().tensor().to_vec_f64().unwrap();
        let gpu_w = layer_gpu.weight().tensor().to_vec_f64().unwrap();
        assert_eq!(orig_w, gpu_w);
    }

    #[test]
    fn test_linear_to_cpu() {
        let layer = Linear::new(4, 3);
        let layer_cpu = layer.cpu().unwrap();
        assert_eq!(layer_cpu.weight().device(), &Device::Cpu);
    }

    #[test]
    fn test_linear_roundtrip() {
        use theano_types::Device;

        let layer = Linear::new(8, 4);
        let orig_data = layer.weight().tensor().to_vec_f64().unwrap();

        let layer_gpu = layer.to(&Device::Cuda(0)).unwrap();
        let layer_back = layer_gpu.to(&Device::Cpu).unwrap();

        let back_data = layer_back.weight().tensor().to_vec_f64().unwrap();
        assert_eq!(orig_data, back_data);
    }
}
