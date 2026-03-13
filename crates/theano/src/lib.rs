//! # Theano — A PyTorch-compatible Deep Learning Framework in Rust
//!
//! Neo Theano provides a 100% PyTorch-parity deep learning framework built in Rust,
//! with NVIDIA CUDA and AMD ROCm GPU acceleration, and Python bindings via PyO3.
//!
//! ## Quick Start
//!
//! ```rust
//! use theano::prelude::*;
//!
//! // Create tensors
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
//! let b = Tensor::ones(&[2, 2]);
//! let c = &a + &b;
//!
//! // Matrix multiply
//! let d = a.matmul(&b).unwrap();
//! ```

// Re-export core crates
pub use theano_types as types;
pub use theano_core as core;
pub use theano_backend as backend;

#[cfg(feature = "cpu")]
pub use theano_cpu as cpu;

// Re-export key types at the top level for convenience
pub use theano_types::{DType, Device, DeviceType, Layout, Shape, TheanoError, Result};
pub use theano_core::Tensor;

/// Commonly used imports.
pub mod prelude {
    pub use crate::{DType, Device, Layout, Shape, Tensor, Result};
}
