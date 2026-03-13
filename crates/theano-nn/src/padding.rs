//! Padding and shape manipulation layers.

use theano_autograd::Variable;
use theano_core::Tensor;
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

        Variable::new(Tensor::from_slice(&output, &[n, c, h_out, w_out]))
    }

    fn parameters(&self) -> Vec<Variable> { vec![] }
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
        input.tensor().flatten(self.start_dim, self.end_dim)
            .map(Variable::new)
            .unwrap()
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
