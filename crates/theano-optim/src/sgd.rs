use theano_autograd::Variable;
use theano_core::Tensor;

use crate::optimizer::Optimizer;

/// Stochastic Gradient Descent optimizer, matching `torch.optim.SGD`.
///
/// Supports momentum, dampening, weight decay, and Nesterov momentum.
///
/// # Examples
/// ```ignore
/// use theano_optim::{SGD, Optimizer};
/// let mut optim = SGD::new(params, 0.01).momentum(0.9);
/// optim.zero_grad();
/// // ... forward + backward ...
/// optim.step();
/// ```
pub struct SGD {
    /// The parameter variables being optimized.
    pub params: Vec<Variable>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    dampening: f64,
    nesterov: bool,
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    /// Create a new SGD optimizer for the given parameters and learning rate.
    pub fn new(params: Vec<Variable>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocities: vec![None; n],
        }
    }

    /// Set the momentum factor (default 0).
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the weight decay (L2 penalty, default 0).
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set the dampening factor for momentum (default 0).
    pub fn dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    /// Enable or disable Nesterov momentum (default false).
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let param_data = param.tensor().to_vec_f64().unwrap();
            let mut grad_data = grad.to_vec_f64().unwrap();

            // Weight decay: grad += weight_decay * param
            if self.weight_decay != 0.0 {
                for (g, p) in grad_data.iter_mut().zip(param_data.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            let update = if self.momentum != 0.0 {
                // Momentum update
                let vel = if let Some(v) = &self.velocities[i] {
                    let v_data = v.to_vec_f64().unwrap();
                    let new_v: Vec<f64> = v_data
                        .iter()
                        .zip(grad_data.iter())
                        .map(|(&vi, &gi)| self.momentum * vi + (1.0 - self.dampening) * gi)
                        .collect();
                    new_v
                } else {
                    grad_data.clone()
                };
                self.velocities[i] = Some(Tensor::from_slice(&vel, param.tensor().shape()));

                if self.nesterov {
                    grad_data
                        .iter()
                        .zip(vel.iter())
                        .map(|(&g, &v)| g + self.momentum * v)
                        .collect::<Vec<f64>>()
                } else {
                    vel
                }
            } else {
                grad_data
            };

            // param = param - lr * update
            let new_data: Vec<f64> = param_data
                .iter()
                .zip(update.iter())
                .map(|(&p, &u)| p - self.lr * u)
                .collect();

            let new_tensor = Tensor::from_slice(&new_data, param.tensor().shape());
            param.update_param(new_tensor);
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.tensor().set_grad(Tensor::zeros(param.tensor().shape()));
        }
    }

    fn params(&self) -> &[Variable] {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_basic() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let mut optim = SGD::new(vec![x], 0.1);

        // Simulate gradient = [2.0, 4.0, 6.0] (grad of x^2)
        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[2.0, 4.0, 6.0], &[3]));
        optim.step();

        let updated = optim.params()[0].tensor().to_vec_f64().unwrap();
        // new = old - lr * grad = [1-0.2, 2-0.4, 3-0.6] = [0.8, 1.6, 2.4]
        assert!((updated[0] - 0.8).abs() < 1e-10);
        assert!((updated[1] - 1.6).abs() < 1e-10);
        assert!((updated[2] - 2.4).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = SGD::new(vec![x], 0.1).momentum(0.9);

        // Step 1: grad = 1.0
        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[1.0], &[1]));
        optim.step();
        let v1 = optim.params()[0].tensor().to_vec_f64().unwrap()[0];
        // velocity = 1.0, param = 1.0 - 0.1*1.0 = 0.9
        assert!((v1 - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_zero_grad() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0], &[2]));
        let mut optim = SGD::new(vec![x], 0.1);

        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[5.0, 5.0], &[2]));
        optim.zero_grad();

        let grad = optim.params[0].grad().unwrap().to_vec_f64().unwrap();
        assert!((grad[0]).abs() < 1e-10);
        assert!((grad[1]).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_weight_decay() {
        let x = Variable::requires_grad(Tensor::from_slice(&[2.0], &[1]));
        let mut optim = SGD::new(vec![x], 0.1).weight_decay(0.1);

        // grad = 1.0, weight_decay adds 0.1 * 2.0 = 0.2, so effective grad = 1.2
        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[1.0], &[1]));
        optim.step();

        let updated = optim.params()[0].tensor().to_vec_f64().unwrap()[0];
        // param = 2.0 - 0.1 * 1.2 = 2.0 - 0.12 = 1.88
        assert!((updated - 1.88).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_no_grad_skips_param() {
        // A param with no gradient set should be skipped
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = SGD::new(vec![x], 0.1);
        // Don't set any gradient
        optim.step();
        let val = optim.params()[0].tensor().to_vec_f64().unwrap()[0];
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_multiple_steps() {
        let x = Variable::requires_grad(Tensor::from_slice(&[10.0], &[1]));
        let mut optim = SGD::new(vec![x], 1.0);

        // Step 1: grad=2.0, param = 10 - 1*2 = 8
        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[2.0], &[1]));
        optim.step();
        let v = optim.params()[0].tensor().to_vec_f64().unwrap()[0];
        assert!((v - 8.0).abs() < 1e-10);

        // Step 2: grad=3.0, param = 8 - 1*3 = 5
        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[3.0], &[1]));
        optim.step();
        let v = optim.params()[0].tensor().to_vec_f64().unwrap()[0];
        assert!((v - 5.0).abs() < 1e-10);
    }
}
