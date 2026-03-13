//! ROCm/HIP GPU backend for the Theano deep learning framework.
//!
//! This crate provides GPU acceleration via AMD ROCm using the HIP runtime.
//! The binary compiles without ROCm installed; HIP libraries are loaded at
//! runtime via dlopen when the `hip` feature is enabled.
//!
//! # Architecture
//!
//! - `RocmDevice` -- manages a HIP device (context, stream, rocBLAS handle)
//! - `RocmStorage` -- GPU memory buffer with typed data
//! - `CachingAllocator` -- PyTorch-style memory pool to avoid hipMalloc per tensor
//! - `RocmBackend` -- implements `theano_backend::Backend`

pub mod error;
pub mod allocator;
pub mod device;
pub mod storage;
pub mod rocm_backend;

pub use rocm_backend::RocmBackend;
pub use device::RocmDevice;
pub use storage::RocmStorage;
pub use error::RocmError;
