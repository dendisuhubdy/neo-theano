//! CUDA GPU backend for the Theano deep learning framework.
//!
//! This crate provides GPU acceleration via NVIDIA CUDA using the `cudarc` crate
//! for safe, dynamic-loading FFI. The binary compiles without CUDA installed;
//! CUDA libraries are loaded at runtime via dlopen.
//!
//! # Architecture
//!
//! - `CudaDevice` — manages a CUDA device (context, stream, cuBLAS handle)
//! - `CudaStorage` — GPU memory buffer with typed data
//! - `CachingAllocator` — PyTorch-style memory pool to avoid cudaMalloc per tensor
//! - `CudaBackend` — implements `theano_backend::Backend`

pub mod allocator;
pub mod device;
pub mod storage;
pub mod cuda_backend;
pub mod error;

pub use cuda_backend::CudaBackend;
pub use device::CudaDevice;
pub use storage::CudaStorage;
pub use error::CudaError;
