//! Convolutional layers.

use theano_autograd::Variable;
use theano_core::Tensor;

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

        // Naive convolution via im2col-like approach
        // For each output position, gather the input patch and dot with kernel
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
                                    let ih = oh * self.stride.0 + kh_i;
                                    let iw = ow * self.stride.1 + kw_i;
                                    let ih = ih as isize - self.padding.0 as isize;
                                    let iw = iw as isize - self.padding.1 as isize;

                                    if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        let in_idx = batch * c_in * h_in * w_in
                                            + ci * h_in * w_in
                                            + ih * w_in
                                            + iw;
                                        let w_idx = co * c_in * kh * kw
                                            + ci * kh * kw
                                            + kh_i * kw
                                            + kw_i;
                                        val += input_data[in_idx] * weight_data[w_idx];
                                    }
                                }
                            }
                        }

                        // Add bias
                        if let Some(ref bias) = self.bias {
                            let bias_data = bias.tensor().to_vec_f64().unwrap();
                            val += bias_data[co];
                        }

                        let out_idx = batch * self.out_channels * h_out * w_out
                            + co * h_out * w_out
                            + oh * w_out
                            + ow;
                        output_data[out_idx] = val;
                    }
                }
            }
        }

        Variable::new(Tensor::from_slice(
            &output_data,
            &[n, self.out_channels, h_out, w_out],
        ))
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
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
}
