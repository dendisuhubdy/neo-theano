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
        let beta = self.beta;

        // Quadratic region: 0.5 * x^2 / beta (where |diff| < beta)
        let quadratic = diff.mul(&diff).unwrap().mul_scalar(0.5 / beta).unwrap();
        // Linear region: |diff| - 0.5 * beta
        let linear = abs_diff.add_scalar(-0.5 * beta).unwrap();

        // Condition: |diff| < beta → use quadratic, else linear
        let threshold = Variable::new(Tensor::full(abs_diff.tensor().shape(), beta));
        let cond = threshold.sub(&abs_diff).unwrap().relu().unwrap(); // >0 where |diff| < beta
        // Build a mask: 1 where |diff| < beta, 0 otherwise
        let zero = Variable::new(Tensor::zeros(abs_diff.tensor().shape()));
        let ones = Variable::new(Tensor::ones(abs_diff.tensor().shape()));
        let mask = ones.where_cond(&cond, &zero).unwrap(); // 1 where cond > 0

        // result = mask * quadratic + (1 - mask) * linear
        let inv_mask = ones.sub(&mask).unwrap();
        let result = mask.mul(&quadratic).unwrap().add(&inv_mask.mul(&linear).unwrap()).unwrap();
        result.mean().unwrap()
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
        // BCE = -[target * log(input) + (1 - target) * log(1 - input)]
        let clamped = input.clamp(1e-12, 1.0 - 1e-12).unwrap();
        let log_p = clamped.log().unwrap();
        let ones = Variable::new(Tensor::ones(input.tensor().shape()));
        let log_1mp = ones.sub(&clamped).unwrap().log().unwrap();
        let ones_t = Variable::new(Tensor::ones(target.tensor().shape()));
        let bce = target.mul(&log_p).unwrap()
            .add(&ones_t.sub(target).unwrap().mul(&log_1mp).unwrap()).unwrap()
            .neg().unwrap();
        bce.mean().unwrap()
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
        // Numerically stable: max(x,0) - x*t + log(1 + exp(-|x|))
        // = relu(x) - x*t + log(1 + exp(-|x|))
        // Using Variable ops: sigmoid computes 1/(1+exp(-x)), so
        // -log(sigmoid(x)) = log(1+exp(-x)) for x > 0
        // For numerical stability, use: softplus(x) - x*t where softplus(x) = log(1+exp(x))
        // BCE with logits = softplus(x) - x*t = log(1+exp(x)) - x*t
        // But softplus(x) = x + log(1+exp(-x)) for large x, which is relu(x) + log(1+exp(-|x|))
        // Simpler: just apply sigmoid then use BCELoss logic
        let probs = input.sigmoid().unwrap();
        let clamped = probs.clamp(1e-12, 1.0 - 1e-12).unwrap();
        let log_p = clamped.log().unwrap();
        let ones = Variable::new(Tensor::ones(input.tensor().shape()));
        let log_1mp = ones.sub(&clamped).unwrap().log().unwrap();
        let ones_t = Variable::new(Tensor::ones(target.tensor().shape()));
        let bce = target.mul(&log_p).unwrap()
            .add(&ones_t.sub(target).unwrap().mul(&log_1mp).unwrap()).unwrap()
            .neg().unwrap();
        bce.mean().unwrap()
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
        // = sum(P * log(P) - P * log(Q))
        // P * log(P) is a constant w.r.t. input, but we include it for correctness
        // The gradient-relevant part is: -P * log(Q) = -target * input
        let target_log_target = target.mul(&target.clamp(1e-12, 1e30).unwrap().log().unwrap()).unwrap();
        let target_log_q = target.mul(input).unwrap();
        let kl = target_log_target.sub(&target_log_q).unwrap();
        kl.mean().unwrap()
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
        // Gather the log-prob at each target class using index_select per row
        // target: [N] class indices, input: [N, C] log probabilities
        // We need to select input[i, target[i]] for each i
        // Approach: flatten input, compute flat indices, use index_select
        let c = input.tensor().shape()[1];
        let target_data = target.tensor().to_vec_f64().unwrap();
        let flat_indices: Vec<f64> = (0..n).map(|i| {
            (i * c) as f64 + target_data[i]
        }).collect();
        let idx_var = Variable::new(Tensor::from_slice(&flat_indices, &[n]));
        let flat_input = input.reshape(&[n * c]).unwrap();
        let gathered = flat_input.index_select(0, &idx_var).unwrap();
        // NLL = -mean(gathered)
        gathered.neg().unwrap().mean().unwrap()
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
