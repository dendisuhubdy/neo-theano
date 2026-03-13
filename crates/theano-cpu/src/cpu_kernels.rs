//! CPU kernel implementations.
//!
//! This module will eventually house optimized SIMD kernels, BLAS wrappers,
//! and rayon-parallelized operations. For now, the core computation lives
//! in `cpu_storage.rs` using scalar Rust code.

// Future: SIMD-optimized elementwise kernels
// Future: gemm crate integration for matmul
// Future: rayon-parallelized reductions
