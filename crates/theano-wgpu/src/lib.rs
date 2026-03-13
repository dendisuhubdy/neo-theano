//! WebGPU backend via the `wgpu` crate.
//! Cross-platform: Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows), WebGPU (browsers).
//! Primary use: browser inference via WebAssembly, cross-platform portability.

pub mod error;
pub mod device;
pub mod storage;
pub mod wgpu_backend;
pub mod wgsl_kernels;

pub use wgpu_backend::WgpuBackend;
pub use device::WgpuDevice;
pub use storage::WgpuStorage;
pub use error::WgpuError;
