use theano_autograd::Variable;
use theano_core::Tensor;

use crate::optimizer::Optimizer;

/// RMSprop optimizer, matching `torch.optim.RMSprop`.
///
/// Implements the RMSprop algorithm (Hinton, 2012) with optional momentum
/// and centered variant.
///
/// # Examples
/// ```ignore
/// use theano_optim::{RMSprop, Optimizer};
/// let mut optim = RMSprop::new(params, 0.01).alpha(0.99);
/// optim.zero_grad();
/// // ... forward + backward ...
/// optim.step();
/// ```
pub struct RMSprop {
    /// The parameter variables being optimized.
    pub params: Vec<Variable>,
    lr: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
    centered: bool,
    step_count: usize,
    sq_avg: Vec<Option<Vec<f64>>>,
    grad_avg: Vec<Option<Vec<f64>>>,
    momentum_buf: Vec<Option<Vec<f64>>>,
}

impl RMSprop {
    /// Create a new RMSprop optimizer for the given parameters and learning rate.
    pub fn new(params: Vec<Variable>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params, lr, alpha: 0.99, eps: 1e-8, weight_decay: 0.0,
            momentum: 0.0, centered: false, step_count: 0,
            sq_avg: vec![None; n], grad_avg: vec![None; n], momentum_buf: vec![None; n],
        }
    }

    /// Set the smoothing constant alpha (default 0.99).
    pub fn alpha(mut self, alpha: f64) -> Self { self.alpha = alpha; self }
    /// Set the epsilon term for numerical stability (default 1e-8).
    pub fn eps(mut self, eps: f64) -> Self { self.eps = eps; self }
    /// Set the momentum factor (default 0).
    pub fn momentum(mut self, m: f64) -> Self { self.momentum = m; self }
    /// Set the weight decay (L2 penalty, default 0).
    pub fn weight_decay(mut self, wd: f64) -> Self { self.weight_decay = wd; self }
    /// Enable or disable the centered variant (default false).
    pub fn centered(mut self, c: bool) -> Self { self.centered = c; self }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        self.step_count += 1;
        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };
            let param_data = param.tensor().to_vec_f64().unwrap();
            let mut grad_data = grad.to_vec_f64().unwrap();
            let n = param_data.len();

            if self.weight_decay != 0.0 {
                for (g, p) in grad_data.iter_mut().zip(param_data.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            if self.sq_avg[i].is_none() {
                self.sq_avg[i] = Some(vec![0.0; n]);
                if self.centered { self.grad_avg[i] = Some(vec![0.0; n]); }
                if self.momentum > 0.0 { self.momentum_buf[i] = Some(vec![0.0; n]); }
            }

            let sq = self.sq_avg[i].as_mut().unwrap();
            for j in 0..n {
                sq[j] = self.alpha * sq[j] + (1.0 - self.alpha) * grad_data[j] * grad_data[j];
            }

            let avg = if self.centered {
                let ga = self.grad_avg[i].as_mut().unwrap();
                for j in 0..n {
                    ga[j] = self.alpha * ga[j] + (1.0 - self.alpha) * grad_data[j];
                }
                (0..n).map(|j| sq[j] - ga[j] * ga[j]).collect::<Vec<_>>()
            } else {
                sq.clone()
            };

            let new_data: Vec<f64> = if self.momentum > 0.0 {
                let buf = self.momentum_buf[i].as_mut().unwrap();
                for j in 0..n {
                    buf[j] = self.momentum * buf[j] + grad_data[j] / (avg[j].sqrt() + self.eps);
                }
                (0..n).map(|j| param_data[j] - self.lr * buf[j]).collect()
            } else {
                (0..n).map(|j| param_data[j] - self.lr * grad_data[j] / (avg[j].sqrt() + self.eps)).collect()
            };

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
    fn test_rmsprop_basic() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = RMSprop::new(vec![x], 0.01);
        optim.params[0].tensor().set_grad(Tensor::from_slice(&[1.0], &[1]));
        optim.step();
        let updated = optim.params()[0].tensor().to_vec_f64().unwrap();
        assert!(updated[0] < 1.0);
    }
}
