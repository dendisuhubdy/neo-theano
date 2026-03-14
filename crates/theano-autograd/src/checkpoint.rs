//! Gradient checkpointing for memory-efficient training.
//!
//! Like `torch.utils.checkpoint.checkpoint` in PyTorch. During the forward pass,
//! intermediate activations are not saved. During the backward pass, the forward
//! function is re-run to recompute them. This trades compute for memory.
//!
//! # Example
//! ```ignore
//! let output = checkpoint(|input| {
//!     let h = layer1.forward(input);
//!     let h = layer2.forward(&h);
//!     layer3.forward(&h)
//! }, &input);
//! ```

use std::sync::Arc;

use theano_core::Tensor;
use theano_core::tensor::GradFn;

use crate::no_grad::{set_grad_enabled, NoGradGuard};
use crate::variable::Variable;

/// Run a function with gradient checkpointing.
///
/// During forward: runs the function but doesn't save intermediates (runs under no_grad).
/// During backward: re-runs the function to recompute intermediates, then backprops through them.
///
/// This trades compute for memory — useful for very deep models where memory is the bottleneck.
///
/// # Arguments
/// * `func` - The function to checkpoint. Must be deterministic.
/// * `input` - The input variable.
///
/// # Returns
/// The output variable, connected to the autograd graph via a checkpoint node.
pub fn checkpoint<F>(func: F, input: &Variable) -> Variable
where
    F: Fn(&Variable) -> Variable + Send + Sync + 'static,
{
    // Run forward under no_grad — don't save intermediates
    let output = {
        let _guard = NoGradGuard::new();
        func(input)
    };

    // If input doesn't require grad, just return the output as-is
    if !input.requires_grad_flag() {
        return output;
    }

    // Create a checkpoint backward node that stores the function and input
    let grad_fn = Arc::new(CheckpointBackward {
        func: Arc::new(func),
        input_tensor: input.tensor().detach(),
        _output_shape: output.tensor().shape().to_vec(),
    });

    Variable::from_op_public(output.into_tensor(), grad_fn, vec![input.clone()])
}

/// GradFn that re-executes the forward function during backward to recompute
/// intermediate activations.
struct CheckpointBackward {
    func: Arc<dyn Fn(&Variable) -> Variable + Send + Sync>,
    input_tensor: Tensor,
    _output_shape: Vec<usize>,
}

impl GradFn for CheckpointBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];

        // Re-run forward with grad enabled to build the computation graph
        let prev = set_grad_enabled(true);
        let input_var = Variable::requires_grad(self.input_tensor.clone());
        let output_var = (self.func)(&input_var);

        // Now backprop through the recomputed graph
        // Create a custom gradient for the output and propagate
        let output_sum = output_var.mul(&Variable::new(grad.clone())).unwrap();
        let loss = output_sum.sum().unwrap();
        loss.backward();

        set_grad_enabled(prev);

        // Retrieve the gradient of the input
        let input_grad = input_var.grad();
        vec![input_grad]
    }

    fn name(&self) -> &str {
        "CheckpointBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_forward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let y = checkpoint(|input| input.mul_scalar(2.0).unwrap(), &x);
        let data = y.tensor().to_vec_f64().unwrap();
        assert_eq!(data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_checkpoint_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let y = checkpoint(
            |input| {
                // f(x) = 2*x
                input.mul_scalar(2.0).unwrap()
            },
            &x,
        );
        let loss = y.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        // d/dx sum(2*x) = 2 for each element
        assert_eq!(g, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_checkpoint_no_grad_input() {
        let x = Variable::new(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let y = checkpoint(|input| input.mul_scalar(3.0).unwrap(), &x);
        let data = y.tensor().to_vec_f64().unwrap();
        assert_eq!(data, vec![3.0, 6.0, 9.0]);
        // No grad_fn since input doesn't require grad
        assert!(y.grad_fn().is_none());
    }

    #[test]
    fn test_checkpoint_chain() {
        let x = Variable::requires_grad(Tensor::from_slice(&[2.0, 3.0], &[2]));
        let y = checkpoint(
            |input| {
                // f(x) = x^2 (via mul with self)
                input.mul(input).unwrap()
            },
            &x,
        );
        let loss = y.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        // d/dx x^2 = 2x => [4.0, 6.0]
        assert!(
            (g[0] - 4.0).abs() < 1e-10 && (g[1] - 6.0).abs() < 1e-10,
            "Expected [4.0, 6.0], got {:?}",
            g
        );
    }
}
