//! Convolutional layers.

use std::sync::Arc;

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_core::tensor::GradFn;
use theano_types::{Device, Result};

use crate::init;
use crate::module::Module;

/// 2D Convolution layer. Like `torch.nn.Conv2d`.
///
/// Input shape: [N, C_in, H, W]
/// Output shape: [N, C_out, H_out, W_out]
pub struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    weight: Variable,
    bias: Option<Variable>,
}

impl Conv2d {
    /// Create a Conv2d layer with square kernel.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_options(in_channels, out_channels, (kernel_size, kernel_size), (1, 1), (0, 0), true)
    }

    /// Create with full options.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let weight = init::kaiming_uniform(
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
            fan_in,
        );
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

    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Reconstruct a Conv2d layer from pre-trained tensors.
    pub fn from_tensors(weight: Tensor, bias: Option<Tensor>, stride: (usize, usize), padding: (usize, usize)) -> Self {
        let shape = weight.shape().to_vec();
        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_size = (shape[2], shape[3]);
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight: Variable::requires_grad(weight),
            bias: bias.map(Variable::requires_grad),
        }
    }

    /// Move this layer to a different device, returning a new Conv2d layer.
    ///
    /// Like `layer.to(device)` in PyTorch.
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

    /// Move to CPU.
    pub fn cpu(&self) -> Result<Self> {
        self.to(&Device::Cpu)
    }

    /// Move to CUDA device 0.
    pub fn cuda(&self) -> Result<Self> {
        self.to(&Device::Cuda(0))
    }

    /// Compute output spatial dimensions.
    fn output_size(&self, h_in: usize, w_in: usize) -> (usize, usize) {
        let h_out = (h_in + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let w_out = (w_in + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (h_out, w_out)
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape();
        assert_eq!(shape.len(), 4, "Conv2d expects 4D input [N, C, H, W], got {}D", shape.len());
        let n = shape[0];
        let c_in = shape[1];
        let h_in = shape[2];
        let w_in = shape[3];
        assert_eq!(c_in, self.in_channels);

        let (h_out, w_out) = self.output_size(h_in, w_in);
        let kh = self.kernel_size.0;
        let kw = self.kernel_size.1;

        let input_data = input.tensor().to_vec_f64().unwrap();
        let weight_data = self.weight.tensor().to_vec_f64().unwrap();

        let mut output_data = vec![0.0f64; n * self.out_channels * h_out * w_out];

        for batch in 0..n {
            for co in 0..self.out_channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut val = 0.0f64;
                        for ci in 0..c_in {
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    let ih = (oh * self.stride.0 + kh_i) as isize - self.padding.0 as isize;
                                    let iw = (ow * self.stride.1 + kw_i) as isize - self.padding.1 as isize;
                                    if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                        let in_idx = batch * c_in * h_in * w_in + ci * h_in * w_in + ih as usize * w_in + iw as usize;
                                        let w_idx = co * c_in * kh * kw + ci * kh * kw + kh_i * kw + kw_i;
                                        val += input_data[in_idx] * weight_data[w_idx];
                                    }
                                }
                            }
                        }
                        if let Some(ref bias) = self.bias {
                            val += bias.tensor().to_vec_f64().unwrap()[co];
                        }
                        output_data[batch * self.out_channels * h_out * w_out + co * h_out * w_out + oh * w_out + ow] = val;
                    }
                }
            }
        }

        let output_tensor = Tensor::from_slice(&output_data, &[n, self.out_channels, h_out, w_out]);
        let grad_fn = Arc::new(Conv2dBackward {
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

struct Conv2dBackward {
    input_tensor: Tensor,
    weight_tensor: Tensor,
    has_bias: bool,
    stride: (usize, usize),
    padding: (usize, usize),
    kernel_size: (usize, usize),
    in_channels: usize,
    out_channels: usize,
}

impl GradFn for Conv2dBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let grad_data = grad.to_vec_f64().unwrap();
        let input_data = self.input_tensor.to_vec_f64().unwrap();
        let weight_data = self.weight_tensor.to_vec_f64().unwrap();

        let in_shape = self.input_tensor.shape();
        let (n, c_in, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let out_shape = grad.shape();
        let (h_out, w_out) = (out_shape[2], out_shape[3]);
        let (kh, kw) = self.kernel_size;

        // grad_weight[co][ci][kh_i][kw_i] = sum_n,oh,ow grad[n][co][oh][ow] * input[n][ci][ih][iw]
        let mut grad_weight = vec![0.0f64; self.out_channels * c_in * kh * kw];
        for batch in 0..n {
            for co in 0..self.out_channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g = grad_data[batch * self.out_channels * h_out * w_out + co * h_out * w_out + oh * w_out + ow];
                        for ci in 0..c_in {
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    let ih = (oh * self.stride.0 + kh_i) as isize - self.padding.0 as isize;
                                    let iw = (ow * self.stride.1 + kw_i) as isize - self.padding.1 as isize;
                                    if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                        let in_idx = batch * c_in * h_in * w_in + ci * h_in * w_in + ih as usize * w_in + iw as usize;
                                        grad_weight[co * c_in * kh * kw + ci * kh * kw + kh_i * kw + kw_i] += g * input_data[in_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // grad_input[n][ci][ih][iw] = sum_co,kh_i,kw_i weight[co][ci][kh_i][kw_i] * grad[n][co][oh][ow]
        let mut grad_input = vec![0.0f64; n * c_in * h_in * w_in];
        for batch in 0..n {
            for co in 0..self.out_channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g = grad_data[batch * self.out_channels * h_out * w_out + co * h_out * w_out + oh * w_out + ow];
                        for ci in 0..c_in {
                            for kh_i in 0..kh {
                                for kw_i in 0..kw {
                                    let ih = (oh * self.stride.0 + kh_i) as isize - self.padding.0 as isize;
                                    let iw = (ow * self.stride.1 + kw_i) as isize - self.padding.1 as isize;
                                    if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                        grad_input[batch * c_in * h_in * w_in + ci * h_in * w_in + ih as usize * w_in + iw as usize]
                                            += weight_data[co * c_in * kh * kw + ci * kh * kw + kh_i * kw + kw_i] * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad_input_tensor = Tensor::from_slice(&grad_input, &[n, c_in, h_in, w_in]);
        let grad_weight_tensor = Tensor::from_slice(&grad_weight, &[self.out_channels, c_in, kh, kw]);

        if self.has_bias {
            // grad_bias[co] = sum_n,oh,ow grad[n][co][oh][ow]
            let mut grad_bias = vec![0.0f64; self.out_channels];
            for batch in 0..n {
                for co in 0..self.out_channels {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            grad_bias[co] += grad_data[batch * self.out_channels * h_out * w_out + co * h_out * w_out + oh * w_out + ow];
                        }
                    }
                }
            }
            vec![Some(grad_input_tensor), Some(grad_weight_tensor), Some(Tensor::from_slice(&grad_bias, &[self.out_channels]))]
        } else {
            vec![Some(grad_input_tensor), Some(grad_weight_tensor)]
        }
    }

    fn name(&self) -> &str { "Conv2dBackward" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_output_shape() {
        let conv = Conv2d::new(1, 16, 3);
        let input = Variable::new(Tensor::ones(&[1, 1, 28, 28]));
        let output = conv.forward(&input);
        // 28 - 3 + 1 = 26
        assert_eq!(output.tensor().shape(), &[1, 16, 26, 26]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        let conv = Conv2d::with_options(1, 8, (3, 3), (1, 1), (1, 1), true);
        let input = Variable::new(Tensor::ones(&[2, 1, 14, 14]));
        let output = conv.forward(&input);
        // Same padding: (14 + 2*1 - 3)/1 + 1 = 14
        assert_eq!(output.tensor().shape(), &[2, 8, 14, 14]);
    }

    #[test]
    fn test_conv2d_parameters() {
        let conv = Conv2d::new(3, 16, 3);
        let params = conv.parameters();
        assert_eq!(params.len(), 2); // weight + bias
        assert_eq!(params[0].tensor().shape(), &[16, 3, 3, 3]);
        assert_eq!(params[1].tensor().shape(), &[16]);
    }

    #[test]
    fn test_conv2d_stride() {
        let conv = Conv2d::with_options(1, 4, (3, 3), (2, 2), (0, 0), false);
        let input = Variable::new(Tensor::ones(&[1, 1, 7, 7]));
        let output = conv.forward(&input);
        // (7 - 3)/2 + 1 = 3
        assert_eq!(output.tensor().shape(), &[1, 4, 3, 3]);
    }

    #[test]
    fn test_conv2d_to_device() {
        let conv = Conv2d::new(3, 16, 3);
        let conv_gpu = conv.to(&Device::Cuda(0)).unwrap();

        // Verify shapes preserved
        assert_eq!(conv_gpu.in_channels(), 3);
        assert_eq!(conv_gpu.out_channels(), 16);

        // Verify parameters transferred
        for param in conv_gpu.parameters() {
            assert_eq!(param.device(), &Device::Cuda(0));
        }

        // Roundtrip: GPU -> CPU
        let conv_back = conv_gpu.cpu().unwrap();
        for param in conv_back.parameters() {
            assert_eq!(param.device(), &Device::Cpu);
        }
    }

    #[test]
    fn test_conv2d_named_parameters() {
        let conv = Conv2d::new(3, 16, 3);
        let named = conv.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_conv2d_state_dict() {
        let conv = Conv2d::new(3, 16, 3);
        let sd = conv.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape(), &[16, 3, 3, 3]);
    }

    #[test]
    fn test_conv2d_forward_correctness() {
        let conv = Conv2d::with_options(1, 1, (1, 1), (1, 1), (0, 0), false);
        let input = Variable::new(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 1, 2, 2],
        ));
        let output = conv.forward(&input);
        assert_eq!(output.tensor().shape(), &[1, 1, 2, 2]);
        assert_eq!(output.tensor().numel(), 4);
    }

    #[test]
    fn test_conv2d_backward() {
        let conv = Conv2d::with_options(1, 1, (3, 3), (1, 1), (0, 0), false);
        let input = Variable::requires_grad(Tensor::ones(&[1, 1, 5, 5]));
        let output = conv.forward(&input);
        let loss = output.sum().unwrap();
        loss.backward();
        // Both input and weight should have gradients
        assert!(input.grad().is_some());
        assert!(conv.parameters()[0].grad().is_some());
    }
}
