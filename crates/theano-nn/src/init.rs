//! Parameter initialization functions, mirroring PyTorch's `nn.init`.

use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use theano_autograd::Variable;
use theano_core::Tensor;

/// Kaiming uniform initialization (He initialization).
/// Good for ReLU networks.
pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Variable {
    let bound = (6.0 / fan_in as f64).sqrt();
    uniform_init(shape, -bound, bound)
}

/// Kaiming normal initialization.
pub fn kaiming_normal(shape: &[usize], fan_in: usize) -> Variable {
    let std = (2.0 / fan_in as f64).sqrt();
    normal_init(shape, 0.0, std)
}

/// Xavier (Glorot) uniform initialization.
/// Good for sigmoid/tanh networks.
pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Variable {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    uniform_init(shape, -bound, bound)
}

/// Xavier (Glorot) normal initialization.
pub fn xavier_normal(shape: &[usize], fan_in: usize, fan_out: usize) -> Variable {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    normal_init(shape, 0.0, std)
}

/// Uniform initialization in [low, high).
pub fn uniform_init(shape: &[usize], low: f64, high: f64) -> Variable {
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    let data: Vec<f64> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    Variable::requires_grad(Tensor::from_slice(&data, shape))
}

/// Normal initialization with mean and std.
pub fn normal_init(shape: &[usize], mean: f64, std: f64) -> Variable {
    let numel: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Normal::new(mean, std).unwrap();
    let data: Vec<f64> = (0..numel).map(|_| dist.sample(&mut rng)).collect();
    Variable::requires_grad(Tensor::from_slice(&data, shape))
}

/// Zero initialization.
pub fn zeros(shape: &[usize]) -> Variable {
    Variable::requires_grad(Tensor::zeros(shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaiming_uniform() {
        let w = kaiming_uniform(&[64, 32], 32);
        assert_eq!(w.tensor().shape(), &[64, 32]);
        assert!(w.requires_grad_flag());
    }

    #[test]
    fn test_xavier_normal() {
        let w = xavier_normal(&[64, 32], 32, 64);
        assert_eq!(w.tensor().shape(), &[64, 32]);
    }

    #[test]
    fn test_zeros_init() {
        let w = zeros(&[3, 3]);
        let data = w.tensor().to_vec_f64().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }
}
