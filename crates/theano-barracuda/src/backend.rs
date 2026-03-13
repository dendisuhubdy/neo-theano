//! BarraCUDA backend marker.
//!
//! BarraCUDA is an experimental backend that compiles CUDA kernels directly
//! to AMD GFX11 machine code. It shares the AMD GPU memory space with ROCm
//! and will eventually use RocmStorage for data management.
//!
//! The full Backend trait implementation will be added when BarraCUDA
//! is integrated with the ROCm memory management layer.

/// Marker type for the BarraCUDA backend.
#[derive(Clone, Debug)]
pub struct BarracudaBackend;

impl BarracudaBackend {
    pub fn name() -> &'static str {
        "barracuda"
    }

    pub fn description() -> &'static str {
        "BarraCUDA: Direct .cu → AMD GFX11 compilation (no HIP translation)"
    }
}
