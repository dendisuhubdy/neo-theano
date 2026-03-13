//! Automatic differentiation engine for Theano.
//!
//! This module implements reverse-mode automatic differentiation (backpropagation),
//! matching PyTorch's autograd semantics: dynamic computational graph rebuilt every
//! forward pass, topological-sort backward traversal, gradient accumulation.

pub mod engine;
pub mod function;
pub mod grad_fns;
pub mod variable;
pub mod no_grad;

pub use engine::backward;
pub use function::Function;
pub use no_grad::{no_grad, is_grad_enabled, set_grad_enabled, NoGradGuard};
pub use variable::Variable;
