//! Dropout layers.

use rand::Rng;
use theano_autograd::Variable;
use theano_core::Tensor;

use crate::module::Module;

/// Dropout layer. Like `torch.nn.Dropout`.
///
/// During training, randomly zeroes elements with probability `p`.
/// During inference (eval mode), passes through unchanged.
pub struct Dropout {
    p: f64,
    training: bool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1)");
        Self { p, training: true }
    }

    /// Set training mode.
    pub fn set_training(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Check if in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Variable) -> Variable {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let shape = input.tensor().shape();
        let numel = input.tensor().numel();
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);

        let mask_data: Vec<f64> = (0..numel)
            .map(|_| if rng.gen::<f64>() >= self.p { scale } else { 0.0 })
            .collect();

        let mask = Variable::new(Tensor::from_slice(&mask_data, shape));
        input.mul(&mask).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_eval_passthrough() {
        let mut dropout = Dropout::new(0.5);
        dropout.eval();
        let input = Variable::new(Tensor::ones(&[100]));
        let output = dropout.forward(&input);
        let data = output.tensor().to_vec_f64().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_dropout_train_zeroes_some() {
        let dropout = Dropout::new(0.5);
        let input = Variable::new(Tensor::ones(&[1000]));
        let output = dropout.forward(&input);
        let data = output.tensor().to_vec_f64().unwrap();
        let num_zero = data.iter().filter(|&&x| x == 0.0).count();
        // With p=0.5 and 1000 elements, should zero roughly 500 (+/- some)
        assert!(num_zero > 300 && num_zero < 700, "num_zero = {num_zero}");
    }

    #[test]
    fn test_dropout_zero_prob() {
        let dropout = Dropout::new(0.0);
        let input = Variable::new(Tensor::ones(&[10]));
        let output = dropout.forward(&input);
        let data = output.tensor().to_vec_f64().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_dropout_set_training() {
        let mut dropout = Dropout::new(0.5);
        assert!(dropout.is_training());

        dropout.set_training(false);
        assert!(!dropout.is_training());

        // In eval mode, should pass through
        let input = Variable::new(Tensor::ones(&[100]));
        let output = dropout.forward(&input);
        let data = output.tensor().to_vec_f64().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));

        dropout.set_training(true);
        assert!(dropout.is_training());
    }
}
