//! Pooling layers.

use theano_autograd::Variable;
use theano_core::Tensor;
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
            stride: (kernel_size, kernel_size), // default stride = kernel_size
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

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f64::NEG_INFINITY;
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = (oh * self.stride.0 + kh) as isize - self.padding.0 as isize;
                                let iw = (ow * self.stride.1 + kw) as isize - self.padding.1 as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let idx = batch * c * h * w + ch * h * w + ih as usize * w + iw as usize;
                                    max_val = max_val.max(data[idx]);
                                }
                            }
                        }
                        let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }

        Variable::new(Tensor::from_slice(&output, &[n, c, h_out, w_out]))
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
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
                                    let idx = batch * c * h * w + ch * h * w + ih as usize * w + iw as usize;
                                    sum += data[idx];
                                }
                            }
                        }
                        let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = sum / pool_area;
                    }
                }
            }
        }

        Variable::new(Tensor::from_slice(&output, &[n, c, h_out, w_out]))
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
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

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..oh_size {
                    for ow in 0..ow_size {
                        let h_start = (oh * h) / oh_size;
                        let h_end = ((oh + 1) * h) / oh_size;
                        let w_start = (ow * w) / ow_size;
                        let w_end = ((ow + 1) * w) / ow_size;

                        let mut sum = 0.0f64;
                        let mut count = 0;
                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                let idx = batch * c * h * w + ch * h * w + ih * w + iw;
                                sum += data[idx];
                                count += 1;
                            }
                        }
                        let out_idx = batch * c * oh_size * ow_size + ch * oh_size * ow_size + oh * ow_size + ow;
                        output[out_idx] = if count > 0 { sum / count as f64 } else { 0.0 };
                    }
                }
            }
        }

        Variable::new(Tensor::from_slice(&output, &[n, c, oh_size, ow_size]))
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
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
    fn test_adaptive_avg_pool2d() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Variable::new(Tensor::ones(&[2, 3, 7, 7]));
        let output = pool.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 3, 1, 1]);
    }
}
