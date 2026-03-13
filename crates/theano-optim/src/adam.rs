use theano_autograd::Variable;
use theano_core::Tensor;

use crate::optimizer::Optimizer;

/// Adam optimizer, matching `torch.optim.Adam`.
///
/// Implements the Adam algorithm (Kingma & Ba, 2014) with optional AMSGrad.
///
/// # Examples
/// ```ignore
/// use theano_optim::{Adam, Optimizer};
/// let mut optim = Adam::new(params, 0.001).betas(0.9, 0.999);
/// optim.zero_grad();
/// // ... forward + backward ...
/// optim.step();
/// ```
pub struct Adam {
    /// The parameter variables being optimized.
    pub params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    amsgrad: bool,
    // State
    step_count: usize,
    m: Vec<Option<Vec<f64>>>,     // first moment estimates
    v: Vec<Option<Vec<f64>>>,     // second moment estimates
    v_max: Vec<Option<Vec<f64>>>, // max second moment (amsgrad)
}

impl Adam {
    /// Create a new Adam optimizer for the given parameters and learning rate.
    pub fn new(params: Vec<Variable>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            m: vec![None; n],
            v: vec![None; n],
            v_max: vec![None; n],
        }
    }

    /// Set the beta coefficients (exponential decay rates for moment estimates).
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set the epsilon term for numerical stability (default 1e-8).
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the weight decay (L2 penalty, default 0).
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Enable or disable AMSGrad variant (default false).
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.step_count += 1;
        let t = self.step_count as f64;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let param_data = param.tensor().to_vec_f64().unwrap();
            let grad_data = grad.to_vec_f64().unwrap();
            let n = param_data.len();

            // Initialize moments if needed
            if self.m[i].is_none() {
                self.m[i] = Some(vec![0.0; n]);
                self.v[i] = Some(vec![0.0; n]);
                if self.amsgrad {
                    self.v_max[i] = Some(vec![0.0; n]);
                }
            }

            let m = self.m[i].as_mut().unwrap();
            let v = self.v[i].as_mut().unwrap();

            // Add weight decay (L2 regularization for vanilla Adam)
            let grad_data: Vec<f64> = if self.weight_decay != 0.0 {
                grad_data
                    .iter()
                    .zip(param_data.iter())
                    .map(|(&g, &p)| g + self.weight_decay * p)
                    .collect()
            } else {
                grad_data
            };

            // Update biased first and second moment estimates
            for j in 0..n {
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * grad_data[j];
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * grad_data[j] * grad_data[j];
            }

            // Bias correction factors
            let bc1 = 1.0 - self.beta1.powf(t);
            let bc2 = 1.0 - self.beta2.powf(t);

            let new_data: Vec<f64> = if self.amsgrad {
                let v_max = self.v_max[i].as_mut().unwrap();
                for j in 0..n {
                    v_max[j] = v_max[j].max(v[j]);
                }
                (0..n)
                    .map(|j| {
                        let m_hat = m[j] / bc1;
                        let v_max_hat = v_max[j] / bc2;
                        param_data[j] - self.lr * m_hat / (v_max_hat.sqrt() + self.eps)
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|j| {
                        let m_hat = m[j] / bc1;
                        let v_hat = v[j] / bc2;
                        param_data[j] - self.lr * m_hat / (v_hat.sqrt() + self.eps)
                    })
                    .collect()
            };

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

/// AdamW optimizer — decoupled weight decay regularization.
///
/// Implements the AdamW algorithm (Loshchilov & Hutter, 2017) where weight
/// decay is decoupled from the gradient-based update, unlike vanilla Adam
/// which applies weight decay through the gradient (L2 regularization).
///
/// # Examples
/// ```ignore
/// use theano_optim::{AdamW, Optimizer};
/// let mut optim = AdamW::new(params, 0.001).weight_decay(0.01);
/// optim.zero_grad();
/// // ... forward + backward ...
/// optim.step();
/// ```
pub struct AdamW {
    /// The parameter variables being optimized.
    pub params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    // State
    step_count: usize,
    m: Vec<Option<Vec<f64>>>,
    v: Vec<Option<Vec<f64>>>,
}

impl AdamW {
    /// Create a new AdamW optimizer for the given parameters and learning rate.
    pub fn new(params: Vec<Variable>, lr: f64) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    /// Set the beta coefficients (exponential decay rates for moment estimates).
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set the epsilon term for numerical stability (default 1e-8).
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the weight decay coefficient (default 0.01).
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.step_count += 1;
        let t = self.step_count as f64;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            let param_data = param.tensor().to_vec_f64().unwrap();
            let grad_data = grad.to_vec_f64().unwrap();
            let n = param_data.len();

            if self.m[i].is_none() {
                self.m[i] = Some(vec![0.0; n]);
                self.v[i] = Some(vec![0.0; n]);
            }

            let m = self.m[i].as_mut().unwrap();
            let v = self.v[i].as_mut().unwrap();

            // Update moments (no weight decay in gradient for AdamW!)
            for j in 0..n {
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * grad_data[j];
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * grad_data[j] * grad_data[j];
            }

            // Bias correction factors
            let bc1 = 1.0 - self.beta1.powf(t);
            let bc2 = 1.0 - self.beta2.powf(t);

            // Decoupled weight decay + Adam update
            let new_data: Vec<f64> = (0..n)
                .map(|j| {
                    let m_hat = m[j] / bc1;
                    let v_hat = v[j] / bc2;
                    let adam_update = m_hat / (v_hat.sqrt() + self.eps);
                    // Decoupled weight decay: applied to param directly, not through gradient
                    param_data[j] * (1.0 - self.lr * self.weight_decay) - self.lr * adam_update
                })
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
    fn test_adam_basic() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = Adam::new(vec![x], 0.1);

        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[1.0], &[1]));
        optim.step();

        let updated = optim.params()[0].tensor().to_vec_f64().unwrap();
        // After one step, param should have decreased
        assert!(updated[0] < 1.0);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let x = Variable::requires_grad(Tensor::from_slice(&[5.0], &[1]));
        let mut optim = Adam::new(vec![x], 0.1);

        for _ in 0..10 {
            optim.params[0]
                .tensor()
                .set_grad(Tensor::from_slice(&[1.0], &[1]));
            optim.step();
        }

        let updated = optim.params()[0].tensor().to_vec_f64().unwrap();
        // After 10 steps with constant gradient, param should have decreased significantly
        assert!(updated[0] < 5.0);
    }

    #[test]
    fn test_adam_no_grad_skips_param() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = Adam::new(vec![x], 0.1);
        optim.step();
        let val = optim.params()[0].tensor().to_vec_f64().unwrap()[0];
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adam_zero_grad() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = Adam::new(vec![x], 0.1);

        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[5.0], &[1]));
        optim.zero_grad();

        let grad = optim.params[0].grad().unwrap().to_vec_f64().unwrap();
        assert!((grad[0]).abs() < 1e-10);
    }

    #[test]
    fn test_adamw_basic() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = AdamW::new(vec![x], 0.1);

        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[1.0], &[1]));
        optim.step();

        let updated = optim.params()[0].tensor().to_vec_f64().unwrap();
        assert!(updated[0] < 1.0);
    }

    #[test]
    fn test_adamw_weight_decay_effect() {
        // With weight_decay, AdamW should decay params more than Adam
        let x1 = Variable::requires_grad(Tensor::from_slice(&[10.0], &[1]));
        let x2 = Variable::requires_grad(Tensor::from_slice(&[10.0], &[1]));
        let mut optim_no_wd = AdamW::new(vec![x1], 0.1).weight_decay(0.0);
        let mut optim_wd = AdamW::new(vec![x2], 0.1).weight_decay(0.1);

        optim_no_wd.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[1.0], &[1]));
        optim_wd.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[1.0], &[1]));

        optim_no_wd.step();
        optim_wd.step();

        let v_no_wd = optim_no_wd.params()[0].tensor().to_vec_f64().unwrap()[0];
        let v_wd = optim_wd.params()[0].tensor().to_vec_f64().unwrap()[0];

        // Weight decay should make the parameter smaller
        assert!(v_wd < v_no_wd);
    }

    #[test]
    fn test_adamw_zero_grad() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0], &[1]));
        let mut optim = AdamW::new(vec![x], 0.1);

        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[5.0], &[1]));
        optim.zero_grad();

        let grad = optim.params[0].grad().unwrap().to_vec_f64().unwrap();
        assert!((grad[0]).abs() < 1e-10);
    }

    #[test]
    fn test_adam_multidim() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let mut optim = Adam::new(vec![x], 0.01);

        optim.params[0]
            .tensor()
            .set_grad(Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], &[2, 2]));
        optim.step();

        let updated = optim.params()[0].tensor().to_vec_f64().unwrap();
        // All elements should have decreased
        assert!(updated[0] < 1.0);
        assert!(updated[1] < 2.0);
        assert!(updated[2] < 3.0);
        assert!(updated[3] < 4.0);
    }
}
