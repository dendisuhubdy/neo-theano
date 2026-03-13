//! BarraCUDA backend for the Theano deep learning framework.
//!
//! [BarraCUDA](https://github.com/Zaneham/BarraCUDA) is an open-source CUDA compiler
//! (~15K lines of C99) that compiles `.cu` files directly to AMD RDNA (GFX11) GPU
//! machine code — **without NVIDIA's toolchain or HIP translation**.
//!
//! # Architecture
//!
//! - Uses the same CUDA kernel sources as `theano-cuda-kernels`
//! - Build-time: BarraCUDA compiler transforms `.cu` → AMD GFX11 binaries
//! - Runtime: loads compiled AMD binaries via HSA runtime or direct dispatch
//! - Advantage over ROCm/HIP: eliminates the CUDA→HIP translation layer entirely
//!
//! # Status
//!
//! Experimental. BarraCUDA supports atomics, warp intrinsics, shared memory,
//! cooperative groups. This backend integrates it as an alternative AMD backend
//! alongside ROCm.

pub mod compiler;
pub mod device;
pub mod backend;

pub use backend::BarracudaBackend;
pub use device::BarracudaDevice;
pub use compiler::BarracudaCompiler;

/// Check if BarraCUDA compiler is available on this system.
pub fn is_available() -> bool {
    // Check if barracuda binary exists in PATH or standard locations
    which_barracuda().is_some()
}

fn which_barracuda() -> Option<std::path::PathBuf> {
    // Check PATH
    if let Ok(path) = std::env::var("BARRACUDA_PATH") {
        let p = std::path::PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    // Check common locations
    for path in &["/usr/local/bin/barracuda", "/opt/barracuda/bin/barracuda"] {
        let p = std::path::PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_availability_check() {
        // On most systems BarraCUDA won't be installed — that's fine
        let _ = is_available();
    }
}
