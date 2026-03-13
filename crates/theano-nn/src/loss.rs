//! Loss functions.

use theano_autograd::Variable;
use theano_core::Tensor;

/// Mean Squared Error loss. Like `torch.nn.MSELoss`.
///
/// loss = mean((prediction - target)^2)
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }

    /// Compute MSE loss.
    pub fn forward(&self, prediction: &Variable, target: &Variable) -> Variable {
        let diff = prediction.sub(target).unwrap();
        let sq = diff.mul(&diff).unwrap();
        sq.mean().unwrap()
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-entropy loss. Like `torch.nn.CrossEntropyLoss`.
///
/// Combines log_softmax and NLL loss.
/// Input: [N, C] (raw logits), Target: [N] (class indices as f64).
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }

    /// Compute cross-entropy loss.
    ///
    /// `input`: [N, C] raw logits (unnormalized)
    /// `target`: [N] class indices (as f64, e.g., 0.0, 1.0, 2.0...)
    pub fn forward(&self, input: &Variable, target: &Variable) -> Variable {
        let n = input.tensor().shape()[0];
        let c = input.tensor().shape()[1];

        // Compute log_softmax
        let log_probs = input.softmax(1).unwrap().log().unwrap();

        // Gather the log probabilities at target indices
        let log_probs_data = log_probs.tensor().to_vec_f64().unwrap();
        let target_data = target.tensor().to_vec_f64().unwrap();

        let mut nll_data = vec![0.0f64; n];
        for i in 0..n {
            let class_idx = target_data[i] as usize;
            assert!(class_idx < c, "target class {class_idx} >= num classes {c}");
            nll_data[i] = -log_probs_data[i * c + class_idx];
        }

        // Mean over batch
        let nll_sum: f64 = nll_data.iter().sum();
        let loss = nll_sum / n as f64;

        // For gradient flow, we need to express this as Variable operations.
        // We'll compute it directly:
        // loss = -mean(sum over n: log_softmax(input)[n, target[n]])
        //
        // Simplified: compute NLL via Variable ops for autograd support.
        // Create a one-hot encoding of target and use it to gather.
        let mut one_hot = vec![0.0f64; n * c];
        for i in 0..n {
            let class_idx = target_data[i] as usize;
            one_hot[i * c + class_idx] = 1.0;
        }
        let one_hot_var = Variable::new(Tensor::from_slice(&one_hot, &[n, c]));

        // NLL = -sum(one_hot * log_softmax) / N
        let masked = log_probs.mul(&one_hot_var).unwrap();
        let total = masked.sum().unwrap();
        let neg_mean = total.neg().unwrap().mul_scalar(1.0 / n as f64).unwrap();
        neg_mean
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        let pred = Variable::new(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let target = Variable::new(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let loss = MSELoss::new().forward(&pred, &target);
        assert!((loss.tensor().item().unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = Variable::new(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let target = Variable::new(Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]));
        let loss = MSELoss::new().forward(&pred, &target);
        // MSE = mean(1 + 1 + 1) = 1.0
        assert!((loss.tensor().item().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Perfect prediction: logits heavily favor correct class
        let logits = Variable::new(Tensor::from_slice(
            &[10.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            &[2, 3],
        ));
        let targets = Variable::new(Tensor::from_slice(&[0.0, 1.0], &[2]));
        let loss = CrossEntropyLoss::new().forward(&logits, &targets);
        let v = loss.tensor().item().unwrap();
        // Loss should be very small for confident correct predictions
        assert!(v < 0.01, "Expected small loss, got {v}");
    }

    #[test]
    fn test_cross_entropy_loss_gradient() {
        let logits = Variable::requires_grad(Tensor::from_slice(
            &[2.0, 1.0, 0.1],
            &[1, 3],
        ));
        let targets = Variable::new(Tensor::from_slice(&[0.0], &[1]));
        let loss = CrossEntropyLoss::new().forward(&logits, &targets);
        loss.backward();

        assert!(logits.grad().is_some());
    }
}
