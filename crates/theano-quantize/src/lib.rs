//! Quantization support for Theano.
//! Post-training quantization (PTQ), quantization-aware training (QAT),
//! INT8/INT4/FP8 support.

pub mod observer;
pub mod quantize;
pub mod qconfig;

pub use observer::{Observer, MinMaxObserver, PerChannelMinMaxObserver};
pub use quantize::{quantize_tensor, dequantize_tensor, quantize_per_tensor};
pub use qconfig::{QConfig, QuantDType};
