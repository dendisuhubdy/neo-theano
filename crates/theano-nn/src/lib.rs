//! Neural network modules for the Theano deep learning framework.
//!
//! Mirrors PyTorch's `torch.nn` API: Module trait, layers, activations, loss functions.

pub mod module;
pub mod linear;
pub mod conv;
pub mod activation;
pub mod dropout;
pub mod batchnorm;
pub mod loss;
pub mod container;
pub mod init;

pub use module::Module;
pub use linear::Linear;
pub use conv::Conv2d;
pub use activation::{ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax, LogSoftmax};
pub use dropout::Dropout;
pub use batchnorm::BatchNorm1d;
pub use loss::{MSELoss, CrossEntropyLoss};
pub use container::Sequential;
pub use init::{kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal};
