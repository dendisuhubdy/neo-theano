//! Apple Metal GPU backend for Theano.
//! Targets Apple M1/M2/M3/M4 and future Apple Silicon GPUs.
//! Uses Metal Performance Shaders (MPS) for optimized ML primitives.

pub mod error;
pub mod device;
pub mod storage;
pub mod metal_backend;
pub mod msl_kernels;

pub use metal_backend::MetalBackend;
pub use device::MetalDevice;
pub use storage::MetalStorage;
pub use error::MetalError;
