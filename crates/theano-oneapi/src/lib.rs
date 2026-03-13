//! Intel oneAPI/Level Zero backend for Theano.
//! Targets Intel Arc GPUs, Gaudi accelerators, and Intel data center GPUs.
//!
//! # Architecture
//!
//! - `OneApiDevice` — manages an Intel GPU device (context, queue, properties)
//! - `OneApiStorage` — device memory buffer with typed data
//! - `OneApiBackend` — implements `theano_backend::Backend`
//!
//! When compiled without real Level Zero / SYCL runtime, operates in mock mode
//! for development and testing.

pub mod error;
pub mod device;
pub mod storage;
pub mod oneapi_backend;

pub use oneapi_backend::OneApiBackend;
pub use device::OneApiDevice;
pub use storage::OneApiStorage;
pub use error::OneApiError;
