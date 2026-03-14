//! Padding and shape manipulation layers.

use std::sync::Arc;

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_core::tensor::GradFn;
use crate::module::Module;

/// Zero-pad a 2D input. Like `torch.nn.ZeroPad2d`.
pub struct ZeroPad2d {
    padding: (usize, usize, usize, usize), // (left, right, top, bottom)
}

impl ZeroPad2d {
    /// Uniform padding on all sides.
    pub fn new(padding: usize) -> Self {
        Self { padding: (padding, padding, padding, padding) }
    }

    /// Asymmetric padding: (left, right, top, bottom).
    pub fn asymmetric(left: usize, right: usize, top: usize, bottom: usize) -> Self {
        Self { padding: (left, right, top, bottom) }
    }
}

impl Module for ZeroPad2d {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape();
        assert_eq!(shape.len(), 4, "ZeroPad2d expects 4D input");
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (pl, pr, pt, pb) = self.padding;
        let h_out = h + pt + pb;
        let w_out = w + pl + pr;

        let data = input.tensor().to_vec_f64().unwrap();
        let mut output = vec![0.0f64; n * c * h_out * w_out];

        for batch in 0..n {
            for ch in 0..c {
                for ih in 0..h {
                    for iw in 0..w {
                        let oh = ih + pt;
                        let ow = iw + pl;
                        let in_idx = batch * c * h * w + ch * h * w + ih * w + iw;
                        let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = data[in_idx];
                    }
                }
            }
        }

        let output_tensor = Tensor::from_slice(&output, &[n, c, h_out, w_out]);
        let grad_fn = Arc::new(ZeroPad2dBackward {
            input_shape: vec![n, c, h, w],
            padding: self.padding,
        });
        Variable::from_op_public(output_tensor, grad_fn, vec![input.clone()])
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
}

struct ZeroPad2dBackward {
    input_shape: Vec<usize>,
    padding: (usize, usize, usize, usize),
}

impl GradFn for ZeroPad2dBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad_data = grad_output[0].to_vec_f64().unwrap();
        let (n, c, h, w) = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]);
        let (pl, _pr, pt, _pb) = self.padding;
        let h_out = grad_output[0].shape()[2];
        let w_out = grad_output[0].shape()[3];

        let mut grad_input = vec![0.0f64; n * c * h * w];

        for batch in 0..n {
            for ch in 0..c {
                for ih in 0..h {
                    for iw in 0..w {
                        let oh = ih + pt;
                        let ow = iw + pl;
                        grad_input[batch * c * h * w + ch * h * w + ih * w + iw] =
                            grad_data[batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow];
                    }
                }
            }
        }

        vec![Some(Tensor::from_slice(&grad_input, &self.input_shape))]
    }

    fn name(&self) -> &str { "ZeroPad2dBackward" }
}

/// Flatten layer. Like `torch.nn.Flatten`.
pub struct Flatten {
    start_dim: i64,
    end_dim: i64,
}

impl Flatten {
    pub fn new() -> Self {
        Self { start_dim: 1, end_dim: -1 }
    }

    pub fn with_dims(start_dim: i64, end_dim: i64) -> Self {
        Self { start_dim, end_dim }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape();
        let ndim = shape.len() as i64;
        let start = if self.start_dim < 0 { (ndim + self.start_dim) as usize } else { self.start_dim as usize };
        let end = if self.end_dim < 0 { (ndim + self.end_dim) as usize } else { self.end_dim as usize };

        let mut new_shape = Vec::new();
        for i in 0..start {
            new_shape.push(shape[i]);
        }
        let flat_size: usize = shape[start..=end].iter().product();
        new_shape.push(flat_size);
        for i in (end + 1)..shape.len() {
            new_shape.push(shape[i]);
        }

        input.reshape(&new_shape).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeropad2d() {
        let pad = ZeroPad2d::new(1);
        let input = Variable::new(Tensor::ones(&[1, 1, 2, 2]));
        let output = pad.forward(&input);
        assert_eq!(output.tensor().shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_flatten() {
        let flatten = Flatten::new();
        let input = Variable::new(Tensor::ones(&[2, 3, 4, 5]));
        let output = flatten.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 60]);
    }
}
