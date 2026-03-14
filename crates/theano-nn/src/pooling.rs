//! Pooling layers.

use std::sync::Arc;

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_core::tensor::GradFn;
use crate::module::Module;

/// 2D Max Pooling. Like `torch.nn.MaxPool2d`.
/// Input: [N, C, H, W], Output: [N, C, H_out, W_out]
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size: (kernel_size, kernel_size),
            stride: (kernel_size, kernel_size),
            padding: (0, 0),
        }
    }

    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let w_out = (w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (h_out, w_out)
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape();
        assert_eq!(shape.len(), 4);
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = self.output_size(h, w);
        let data = input.tensor().to_vec_f64().unwrap();

        let mut output = vec![0.0f64; n * c * h_out * w_out];
        let mut max_indices = vec![0usize; n * c * h_out * w_out];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_idx = 0usize;
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = (oh * self.stride.0 + kh) as isize - self.padding.0 as isize;
                                let iw = (ow * self.stride.1 + kw) as isize - self.padding.1 as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let idx = batch * c * h * w + ch * h * w + ih as usize * w + iw as usize;
                                    if data[idx] > max_val {
                                        max_val = data[idx];
                                        max_idx = idx;
                                    }
                                }
                            }
                        }
                        let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = max_val;
                        max_indices[out_idx] = max_idx;
                    }
                }
            }
        }

        let output_tensor = Tensor::from_slice(&output, &[n, c, h_out, w_out]);
        let grad_fn = Arc::new(MaxPool2dBackward {
            max_indices,
            input_shape: vec![n, c, h, w],
        });
        Variable::from_op_public(output_tensor, grad_fn, vec![input.clone()])
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
}

struct MaxPool2dBackward {
    max_indices: Vec<usize>,
    input_shape: Vec<usize>,
}

impl GradFn for MaxPool2dBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_data = grad_output[0].to_vec_f64().unwrap();
        let numel: usize = self.input_shape.iter().product();
        let mut grad_input = vec![0.0f64; numel];

        for (i, &idx) in self.max_indices.iter().enumerate() {
            grad_input[idx] += grad_data[i];
        }

        vec![Some(Tensor::from_slice(&grad_input, &self.input_shape))]
    }

    fn name(&self) -> &str { "MaxPool2dBackward" }
}

/// 2D Average Pooling. Like `torch.nn.AvgPool2d`.
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size: (kernel_size, kernel_size),
            stride: (kernel_size, kernel_size),
            padding: (0, 0),
        }
    }

    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let h_out = (h + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let w_out = (w + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (h_out, w_out)
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape();
        assert_eq!(shape.len(), 4);
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = self.output_size(h, w);
        let data = input.tensor().to_vec_f64().unwrap();

        let mut output = vec![0.0f64; n * c * h_out * w_out];
        let pool_area = (self.kernel_size.0 * self.kernel_size.1) as f64;

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f64;
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = (oh * self.stride.0 + kh) as isize - self.padding.0 as isize;
                                let iw = (ow * self.stride.1 + kw) as isize - self.padding.1 as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    sum += data[batch * c * h * w + ch * h * w + ih as usize * w + iw as usize];
                                }
                            }
                        }
                        output[batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow] = sum / pool_area;
                    }
                }
            }
        }

        let output_tensor = Tensor::from_slice(&output, &[n, c, h_out, w_out]);
        let grad_fn = Arc::new(AvgPool2dBackward {
            input_shape: vec![n, c, h, w],
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
        });
        Variable::from_op_public(output_tensor, grad_fn, vec![input.clone()])
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
}

struct AvgPool2dBackward {
    input_shape: Vec<usize>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl GradFn for AvgPool2dBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_data = grad_output[0].to_vec_f64().unwrap();
        let (n, c, h, w) = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]);
        let out_shape = grad_output[0].shape();
        let (h_out, w_out) = (out_shape[2], out_shape[3]);
        let pool_area = (self.kernel_size.0 * self.kernel_size.1) as f64;

        let mut grad_input = vec![0.0f64; n * c * h * w];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g = grad_data[batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow] / pool_area;
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = (oh * self.stride.0 + kh) as isize - self.padding.0 as isize;
                                let iw = (ow * self.stride.1 + kw) as isize - self.padding.1 as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    grad_input[batch * c * h * w + ch * h * w + ih as usize * w + iw as usize] += g;
                                }
                            }
                        }
                    }
                }
            }
        }

        vec![Some(Tensor::from_slice(&grad_input, &self.input_shape))]
    }

    fn name(&self) -> &str { "AvgPool2dBackward" }
}

/// Adaptive Average Pooling (outputs fixed spatial size). Like `torch.nn.AdaptiveAvgPool2d`.
pub struct AdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape();
        assert_eq!(shape.len(), 4);
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (oh_size, ow_size) = self.output_size;
        let data = input.tensor().to_vec_f64().unwrap();

        let mut output = vec![0.0f64; n * c * oh_size * ow_size];
        // Store the region boundaries for backward
        let mut regions: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(oh_size * ow_size);
        for oh in 0..oh_size {
            for ow in 0..ow_size {
                regions.push(((oh * h) / oh_size, ((oh + 1) * h) / oh_size,
                              (ow * w) / ow_size, ((ow + 1) * w) / ow_size));
            }
        }

        for batch in 0..n {
            for ch in 0..c {
                for (r, &(h_start, h_end, w_start, w_end)) in regions.iter().enumerate() {
                    let oh = r / ow_size;
                    let ow = r % ow_size;
                    let mut sum = 0.0f64;
                    let mut count = 0;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += data[batch * c * h * w + ch * h * w + ih * w + iw];
                            count += 1;
                        }
                    }
                    output[batch * c * oh_size * ow_size + ch * oh_size * ow_size + oh * ow_size + ow] =
                        if count > 0 { sum / count as f64 } else { 0.0 };
                }
            }
        }

        let output_tensor = Tensor::from_slice(&output, &[n, c, oh_size, ow_size]);
        let grad_fn = Arc::new(AdaptiveAvgPool2dBackward {
            input_shape: vec![n, c, h, w],
            output_size: self.output_size,
        });
        Variable::from_op_public(output_tensor, grad_fn, vec![input.clone()])
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
}

struct AdaptiveAvgPool2dBackward {
    input_shape: Vec<usize>,
    output_size: (usize, usize),
}

impl GradFn for AdaptiveAvgPool2dBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_data = grad_output[0].to_vec_f64().unwrap();
        let (n, c, h, w) = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]);
        let (oh_size, ow_size) = self.output_size;

        let mut grad_input = vec![0.0f64; n * c * h * w];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..oh_size {
                    for ow in 0..ow_size {
                        let h_start = (oh * h) / oh_size;
                        let h_end = ((oh + 1) * h) / oh_size;
                        let w_start = (ow * w) / ow_size;
                        let w_end = ((ow + 1) * w) / ow_size;
                        let count = (h_end - h_start) * (w_end - w_start);
                        let g = grad_data[batch * c * oh_size * ow_size + ch * oh_size * ow_size + oh * ow_size + ow]
                            / count as f64;
                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                grad_input[batch * c * h * w + ch * h * w + ih * w + iw] += g;
                            }
                        }
                    }
                }
            }
        }

        vec![Some(Tensor::from_slice(&grad_input, &self.input_shape))]
    }

    fn name(&self) -> &str { "AdaptiveAvgPool2dBackward" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d() {
        let pool = MaxPool2d::new(2);
        let input = Variable::new(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            &[1, 1, 4, 4],
        ));
        let output = pool.forward(&input);
        assert_eq!(output.tensor().shape(), &[1, 1, 2, 2]);
        let data = output.tensor().to_vec_f64().unwrap();
        assert_eq!(data, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_maxpool2d_backward() {
        let pool = MaxPool2d::new(2);
        let input = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            &[1, 1, 4, 4],
        ));
        let output = pool.forward(&input);
        let loss = output.sum().unwrap();
        loss.backward();
        let grad = input.grad().unwrap().to_vec_f64().unwrap();
        // Only max positions get gradient
        assert_eq!(grad[5], 1.0);  // position of 6.0
        assert_eq!(grad[7], 1.0);  // position of 8.0
        assert_eq!(grad[13], 1.0); // position of 14.0
        assert_eq!(grad[15], 1.0); // position of 16.0
        assert_eq!(grad[0], 0.0);  // non-max
    }

    #[test]
    fn test_avgpool2d() {
        let pool = AvgPool2d::new(2);
        let input = Variable::new(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            &[1, 1, 4, 4],
        ));
        let output = pool.forward(&input);
        assert_eq!(output.tensor().shape(), &[1, 1, 2, 2]);
        let data = output.tensor().to_vec_f64().unwrap();
        assert_eq!(data, vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_avgpool2d_backward() {
        let pool = AvgPool2d::new(2);
        let input = Variable::requires_grad(Tensor::ones(&[1, 1, 4, 4]));
        let output = pool.forward(&input);
        let loss = output.sum().unwrap();
        loss.backward();
        let grad = input.grad().unwrap().to_vec_f64().unwrap();
        // Each input contributes to exactly one output, divided by pool_area=4
        assert!((grad[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_avg_pool2d() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Variable::new(Tensor::ones(&[2, 3, 7, 7]));
        let output = pool.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 3, 1, 1]);
    }

    #[test]
    fn test_adaptive_avg_pool2d_backward() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Variable::requires_grad(Tensor::ones(&[1, 1, 4, 4]));
        let output = pool.forward(&input);
        let loss = output.sum().unwrap();
        loss.backward();
        let grad = input.grad().unwrap().to_vec_f64().unwrap();
        // Global avg pool: each element contributes 1/16
        assert!((grad[0] - 1.0/16.0).abs() < 1e-10);
    }
}
