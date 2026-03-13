use theano_autograd::Variable;
use theano_core::Tensor;

use crate::optimizer::Optimizer;

/// Adagrad optimizer, matching `torch.optim.Adagrad`.
///
/// Implements the Adagrad algorithm (Duchi et al., 2011) with optional
/// weight decay and learning rate decay.
///
/// # Examples
/// ```ignore
/// use theano_optim::{Adagrad, Optimizer};
/// let mut optim = Adagrad::new(params, 0.01);
/// optim.zero_grad();
/// // ... forward + backward ...
/// optim.step();
/// ```
pub struct Adagrad {
    /// The parameter variables being optimized.
    pub params: Vec<Variable>,
    lr: f64,
    eps: f64,
    weight_decay: f64,
    lr_decay: f64,
    step_count: usize,
    sum_sq: Vec<Option<Vec<f64>>>,
}

impl Adagrad {
    /// Create a new Adagrad optimizer for the given parameters and learning rate.
    pub fn new(params: Vec<Variable>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params, lr, eps: 1e-10, weight_decay: 0.0, lr_decay: 0.0,
            step_count: 0, sum_sq: vec![None; n],
        }
    }

    /// Set the epsilon term for numerical stability (default 1e-10).
    pub fn eps(mut self, eps: f64) -> Self { self.eps = eps; self }
    /// Set the weight decay (L2 penalty, default 0).
    pub fn weight_decay(mut self, wd: f64) -> Self { self.weight_decay = wd; self }
    /// Set the learning rate decay (default 0).
    pub fn lr_decay(mut self, d: f64) -> Self { self.lr_decay = d; self }
}

impl Optimizer for Adagrad {
    fn step(&mut self) {
        self.step_count += 1;
        let clr = self.lr / (1.0 + (self.step_count - 1) as f64 * self.lr_decay);

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = match param.grad() { Some(g) => g, None => continue };
            let param_data = param.tensor().to_vec_f64().unwrap();
            let mut grad_data = grad.to_vec_f64().unwrap();
            let n = param_data.len();

            if self.weight_decay != 0.0 {
                for (g, p) in grad_data.iter_mut().zip(param_data.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            if self.sum_sq[i].is_none() {
                self.sum_sq[i] = Some(vec![0.0; n]);
            }
            let ss = self.sum_sq[i].as_mut().unwrap();
            for j in 0..n {
                ss[j] += grad_data[j] * grad_data[j];
            }

            let new_data: Vec<f64> = (0..n).map(|j| {
                param_data[j] - clr * grad_data[j] / (ss[j].sqrt() + self.eps)
            }).collect();

            param.update_param(Tensor::from_slice(&new_data, param.tensor().shape()));
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.tensor().set_grad(Tensor::zeros(param.tensor().shape()));
        }
    }

    fn params(&self) -> &[Variable] { &self.params }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adagrad_basic() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = Adagrad::new(vec![x], 0.01);
        optim.params[0].tensor().set_grad(Tensor::from_slice(&[1.0], &[1]));
        optim.step();
        let updated = optim.params()[0].tensor().to_vec_f64().unwrap();
        assert!(updated[0] < 1.0);
    }
}
