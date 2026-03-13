//! Optimizers for the Theano deep learning framework.
//!
//! This crate provides parameter optimization algorithms matching PyTorch's
//! `torch.optim` module: SGD (with momentum, Nesterov), Adam, and AdamW.

pub mod optimizer;
pub mod sgd;
pub mod adam;
pub mod scheduler;
pub mod rmsprop;
pub mod adagrad;

pub use optimizer::Optimizer;
pub use sgd::SGD;
pub use adam::{Adam, AdamW};
pub use scheduler::{LRScheduler, StepLR, ExponentialLR, CosineAnnealingLR, MultiStepLR, LinearLR, ReduceLROnPlateau};
pub use rmsprop::RMSprop;
pub use adagrad::Adagrad;
