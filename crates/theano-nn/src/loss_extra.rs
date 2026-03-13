//! Additional loss functions.

use theano_autograd::Variable;
use theano_core::Tensor;

/// L1 Loss (Mean Absolute Error). Like `torch.nn.L1Loss`.
pub struct L1Loss;

impl L1Loss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, prediction: &Variable, target: &Variable) -> Variable {
        let diff = prediction.sub(target).unwrap();
        diff.abs().unwrap().mean().unwrap()
    }
}

impl Default for L1Loss {
    fn default() -> Self {
        Self::new()
    }
}

/// Smooth L1 Loss (Huber Loss). Like `torch.nn.SmoothL1Loss`.
pub struct SmoothL1Loss {
    beta: f64,
}

impl SmoothL1Loss {
    pub fn new() -> Self {
        Self { beta: 1.0 }
    }

    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn forward(&self, prediction: &Variable, target: &Variable) -> Variable {
        let diff = prediction.sub(target).unwrap();
        let abs_diff = diff.abs().unwrap();
        let abs_data = abs_diff.tensor().to_vec_f64().unwrap();
        let beta = self.beta;

        let result: Vec<f64> = abs_data
            .iter()
            .map(|&x| {
                if x < beta {
                    0.5 * x * x / beta
                } else {
                    x - 0.5 * beta
                }
            })
            .collect();

        let t = Variable::new(Tensor::from_slice(&result, abs_diff.tensor().shape()));
        t.mean().unwrap()
    }
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary Cross-Entropy Loss. Like `torch.nn.BCELoss`.
/// Input and target should be in [0, 1].
pub struct BCELoss;

impl BCELoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Variable, target: &Variable) -> Variable {
        let eps = 1e-12;
        let input_data = input.tensor().to_vec_f64().unwrap();
        let target_data = target.tensor().to_vec_f64().unwrap();

        let result: Vec<f64> = input_data
            .iter()
            .zip(target_data.iter())
            .map(|(&p, &t)| {
                let p = p.clamp(eps, 1.0 - eps);
                -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
            })
            .collect();

        let t = Variable::new(Tensor::from_slice(&result, input.tensor().shape()));
        t.mean().unwrap()
    }
}

impl Default for BCELoss {
    fn default() -> Self {
        Self::new()
    }
}

/// BCE with logits. Like `torch.nn.BCEWithLogitsLoss`.
/// Combines sigmoid and BCE for numerical stability.
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Variable, target: &Variable) -> Variable {
        let input_data = input.tensor().to_vec_f64().unwrap();
        let target_data = target.tensor().to_vec_f64().unwrap();

        let result: Vec<f64> = input_data
            .iter()
            .zip(target_data.iter())
            .map(|(&x, &t)| {
                // max(x, 0) - x*t + log(1 + exp(-|x|))
                let relu_x = x.max(0.0);
                relu_x - x * t + (1.0 + (-x.abs()).exp()).ln()
            })
            .collect();

        let t = Variable::new(Tensor::from_slice(&result, input.tensor().shape()));
        t.mean().unwrap()
    }
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// KL Divergence Loss. Like `torch.nn.KLDivLoss`.
/// Input: log probabilities, target: probabilities.
pub struct KLDivLoss;

impl KLDivLoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Variable, target: &Variable) -> Variable {
        // KL(P || Q) = sum(P * (log(P) - log(Q)))
        // input is log(Q), target is P
        let input_data = input.tensor().to_vec_f64().unwrap();
        let target_data = target.tensor().to_vec_f64().unwrap();

        let result: Vec<f64> = input_data
            .iter()
            .zip(target_data.iter())
            .map(|(&log_q, &p)| {
                if p > 0.0 {
                    p * (p.ln() - log_q)
                } else {
                    0.0
                }
            })
            .collect();

        let t = Variable::new(Tensor::from_slice(&result, input.tensor().shape()));
        t.mean().unwrap()
    }
}

impl Default for KLDivLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Negative Log Likelihood Loss. Like `torch.nn.NLLLoss`.
/// Input: [N, C] log probabilities, Target: [N] class indices.
pub struct NLLLoss;

impl NLLLoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Variable, target: &Variable) -> Variable {
        let n = input.tensor().shape()[0];
        let c = input.tensor().shape()[1];
        let input_data = input.tensor().to_vec_f64().unwrap();
        let target_data = target.tensor().to_vec_f64().unwrap();

        let mut loss_sum = 0.0f64;
        for i in 0..n {
            let class_idx = target_data[i] as usize;
            assert!(class_idx < c);
            loss_sum -= input_data[i * c + class_idx];
        }

        Variable::new(Tensor::scalar(loss_sum / n as f64))
    }
}

impl Default for NLLLoss {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_loss() {
        let pred = Variable::new(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let target = Variable::new(Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]));
        let loss = L1Loss::new().forward(&pred, &target);
        assert!((loss.tensor().item().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smooth_l1() {
        let pred = Variable::new(Tensor::from_slice(&[0.0], &[1]));
        let target = Variable::new(Tensor::from_slice(&[0.0], &[1]));
        let loss = SmoothL1Loss::new().forward(&pred, &target);
        assert!((loss.tensor().item().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_bce_loss() {
        let pred = Variable::new(Tensor::from_slice(&[0.5, 0.5], &[2]));
        let target = Variable::new(Tensor::from_slice(&[1.0, 0.0], &[2]));
        let loss = BCELoss::new().forward(&pred, &target);
        let v = loss.tensor().item().unwrap();
        assert!((v - 0.6931471805599453).abs() < 1e-6); // -ln(0.5)
    }

    #[test]
    fn test_bce_with_logits() {
        let logits = Variable::new(Tensor::from_slice(&[0.0], &[1]));
        let target = Variable::new(Tensor::from_slice(&[1.0], &[1]));
        let loss = BCEWithLogitsLoss::new().forward(&logits, &target);
        let v = loss.tensor().item().unwrap();
        assert!((v - 0.6931471805599453).abs() < 1e-6);
    }

    #[test]
    fn test_nll_loss() {
        let log_probs = Variable::new(Tensor::from_slice(
            &[-0.5, -1.0, -2.0, -0.3, -1.5, -2.5],
            &[2, 3],
        ));
        let targets = Variable::new(Tensor::from_slice(&[0.0, 2.0], &[2]));
        let loss = NLLLoss::new().forward(&log_probs, &targets);
        let v = loss.tensor().item().unwrap();
        // -((-0.5) + (-2.5)) / 2 = 1.5
        assert!((v - 1.5).abs() < 1e-10);
    }
}
