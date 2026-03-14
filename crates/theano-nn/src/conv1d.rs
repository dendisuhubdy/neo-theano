//! 1D Convolution layer.

use std::sync::Arc;

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_core::tensor::GradFn;
use theano_types::{Device, Result};

use crate::init;
use crate::module::Module;

/// 1D Convolution layer. Like `torch.nn.Conv1d`.
///
/// Input shape: [N, C_in, L]
/// Output shape: [N, C_out, L_out]
pub struct Conv1d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weight: Variable,
    bias: Option<Variable>,
}

impl Conv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_options(in_channels, out_channels, kernel_size, 1, 0, true)
    }

    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let weight = init::kaiming_uniform(&[out_channels, in_channels, kernel_size], fan_in);
        let bias = if use_bias {
            let bound = 1.0 / (fan_in as f64).sqrt();
            Some(init::uniform_init(&[out_channels], -bound, bound))
        } else {
            None
        };

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias,
        }
    }

    pub fn to(&self, device: &Device) -> Result<Self> {
        let weight = self.weight.to(device)?;
        let bias = match &self.bias {
            Some(b) => Some(b.to(device)?),
            None => None,
        };
        Ok(Self {
            weight,
            bias,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
        })
    }

    pub fn cpu(&self) -> Result<Self> {
        self.to(&Device::Cpu)
    }

    pub fn cuda(&self) -> Result<Self> {
        self.to(&Device::Cuda(0))
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape();
        assert_eq!(shape.len(), 3, "Conv1d expects 3D input [N, C, L]");
        let (n, c_in, l_in) = (shape[0], shape[1], shape[2]);
        assert_eq!(c_in, self.in_channels);

        let l_out = (l_in + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let input_data = input.tensor().to_vec_f64().unwrap();
        let weight_data = self.weight.tensor().to_vec_f64().unwrap();
        let mut output_data = vec![0.0f64; n * self.out_channels * l_out];

        for batch in 0..n {
            for co in 0..self.out_channels {
                for ol in 0..l_out {
                    let mut val = 0.0f64;
                    for ci in 0..c_in {
                        for k in 0..self.kernel_size {
                            let il = (ol * self.stride + k) as isize - self.padding as isize;
                            if il >= 0 && (il as usize) < l_in {
                                let in_idx = batch * c_in * l_in + ci * l_in + il as usize;
                                let w_idx = co * c_in * self.kernel_size + ci * self.kernel_size + k;
                                val += input_data[in_idx] * weight_data[w_idx];
                            }
                        }
                    }
                    if let Some(ref bias) = self.bias {
                        val += bias.tensor().to_vec_f64().unwrap()[co];
                    }
                    output_data[batch * self.out_channels * l_out + co * l_out + ol] = val;
                }
            }
        }

        let output_tensor = Tensor::from_slice(&output_data, &[n, self.out_channels, l_out]);
        let grad_fn = Arc::new(Conv1dBackward {
            input_tensor: input.tensor().detach(),
            weight_tensor: self.weight.tensor().detach(),
            has_bias: self.bias.is_some(),
            stride: self.stride,
            padding: self.padding,
            kernel_size: self.kernel_size,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
        });

        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(ref bias) = self.bias {
            inputs.push(bias.clone());
        }
        Variable::from_op_public(output_tensor, grad_fn, inputs)
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

struct Conv1dBackward {
    input_tensor: Tensor,
    weight_tensor: Tensor,
    has_bias: bool,
    stride: usize,
    padding: usize,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
}

impl GradFn for Conv1dBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let grad_data = grad.to_vec_f64().unwrap();
        let input_data = self.input_tensor.to_vec_f64().unwrap();
        let weight_data = self.weight_tensor.to_vec_f64().unwrap();

        let in_shape = self.input_tensor.shape();
        let (n, c_in, l_in) = (in_shape[0], in_shape[1], in_shape[2]);
        let l_out = grad.shape()[2];
        let ks = self.kernel_size;

        let mut grad_weight = vec![0.0f64; self.out_channels * c_in * ks];
        let mut grad_input = vec![0.0f64; n * c_in * l_in];

        for batch in 0..n {
            for co in 0..self.out_channels {
                for ol in 0..l_out {
                    let g = grad_data[batch * self.out_channels * l_out + co * l_out + ol];
                    for ci in 0..c_in {
                        for k in 0..ks {
                            let il = (ol * self.stride + k) as isize - self.padding as isize;
                            if il >= 0 && (il as usize) < l_in {
                                let in_idx = batch * c_in * l_in + ci * l_in + il as usize;
                                let w_idx = co * c_in * ks + ci * ks + k;
                                grad_weight[w_idx] += g * input_data[in_idx];
                                grad_input[in_idx] += weight_data[w_idx] * g;
                            }
                        }
                    }
                }
            }
        }

        let gi = Tensor::from_slice(&grad_input, &[n, c_in, l_in]);
        let gw = Tensor::from_slice(&grad_weight, &[self.out_channels, c_in, ks]);

        if self.has_bias {
            let mut grad_bias = vec![0.0f64; self.out_channels];
            for batch in 0..n {
                for co in 0..self.out_channels {
                    for ol in 0..l_out {
                        grad_bias[co] += grad_data[batch * self.out_channels * l_out + co * l_out + ol];
                    }
                }
            }
            vec![Some(gi), Some(gw), Some(Tensor::from_slice(&grad_bias, &[self.out_channels]))]
        } else {
            vec![Some(gi), Some(gw)]
        }
    }

    fn name(&self) -> &str { "Conv1dBackward" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_shape() {
        let conv = Conv1d::new(1, 8, 3);
        let input = Variable::new(Tensor::ones(&[2, 1, 16]));
        let output = conv.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 8, 14]);
    }

    #[test]
    fn test_conv1d_with_padding() {
        let conv = Conv1d::with_options(3, 16, 3, 1, 1, true);
        let input = Variable::new(Tensor::ones(&[1, 3, 10]));
        let output = conv.forward(&input);
        assert_eq!(output.tensor().shape(), &[1, 16, 10]);
    }

    #[test]
    fn test_conv1d_params() {
        let conv = Conv1d::new(3, 16, 5);
        assert_eq!(conv.parameters().len(), 2);
        assert_eq!(conv.parameters()[0].tensor().shape(), &[16, 3, 5]);
    }

    #[test]
    fn test_conv1d_to_device() {
        let conv = Conv1d::new(3, 16, 3);
        let conv_gpu = conv.to(&Device::Cuda(0)).unwrap();
        for param in conv_gpu.parameters() {
            assert_eq!(param.device(), &Device::Cuda(0));
        }

        let conv_back = conv_gpu.cpu().unwrap();
        for param in conv_back.parameters() {
            assert_eq!(param.device(), &Device::Cpu);
        }
    }

    #[test]
    fn test_conv1d_backward() {
        let conv = Conv1d::with_options(1, 1, 3, 1, 0, false);
        let input = Variable::requires_grad(Tensor::ones(&[1, 1, 5]));
        let output = conv.forward(&input);
        let loss = output.sum().unwrap();
        loss.backward();
        assert!(input.grad().is_some());
        assert!(conv.parameters()[0].grad().is_some());
    }

    #[test]
    fn test_conv1d_named_parameters() {
        let conv = Conv1d::new(3, 16, 3);
        let named = conv.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_conv1d_state_dict() {
        let conv = Conv1d::new(3, 16, 5);
        let sd = conv.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape(), &[16, 3, 5]);
    }
}
