//! Python bindings for Theano via PyO3.
//!
//! This crate defines the Python API surface for Theano. When built with
//! PyO3 and maturin, it produces a Python module that can be imported as:
//!
//! ```python
//! import theano as torch
//! x = torch.tensor([1.0, 2.0, 3.0])
//! y = torch.zeros(2, 3)
//! z = x + y
//! ```
//!
//! The API mirrors PyTorch for maximum familiarity.

pub mod tensor_api;
pub mod nn_api;
pub mod dtype_api;

pub use tensor_api::TensorAPI;
pub use nn_api::NNAPI;
pub use dtype_api::DTypeAPI;
