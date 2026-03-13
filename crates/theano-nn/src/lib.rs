//! Neural network modules for the Theano deep learning framework.
//!
//! Mirrors PyTorch's `torch.nn` API: Module trait, layers, activations, loss functions.

pub mod module;
pub mod linear;
pub mod conv;
pub mod conv1d;
pub mod activation;
pub mod dropout;
pub mod batchnorm;
pub mod normalization;
pub mod pooling;
pub mod embedding;
pub mod padding;
pub mod rnn;
pub mod transformer;
pub mod loss;
pub mod loss_extra;
pub mod container;
pub mod init;

pub use module::Module;
pub use linear::Linear;
pub use conv::Conv2d;
pub use conv1d::Conv1d;
pub use activation::{ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax, LogSoftmax};
pub use dropout::Dropout;
pub use batchnorm::BatchNorm1d;
pub use normalization::{LayerNorm, GroupNorm};
pub use pooling::{MaxPool2d, AvgPool2d, AdaptiveAvgPool2d};
pub use embedding::Embedding;
pub use padding::{ZeroPad2d, Flatten};
pub use rnn::{RNNCell, LSTMCell};
pub use transformer::MultiheadAttention;
pub use loss::{MSELoss, CrossEntropyLoss};
pub use loss_extra::{L1Loss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss, KLDivLoss, NLLLoss};
pub use container::Sequential;
pub use init::{kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal};
