use theano_autograd::Variable;

/// Trait for all optimizers, mirroring PyTorch's `torch.optim.Optimizer` interface.
///
/// Every optimizer holds a set of parameter [`Variable`]s and knows how to
/// update them based on their accumulated gradients.
pub trait Optimizer {
    /// Perform a single optimization step (parameter update).
    fn step(&mut self);
    /// Clear all accumulated gradients.
    fn zero_grad(&mut self);
    /// Get all parameter variables.
    fn params(&self) -> &[Variable];
}
