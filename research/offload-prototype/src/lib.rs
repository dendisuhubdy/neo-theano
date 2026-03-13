//! # GPU Offload Prototype for Neo Theano
//!
//! This crate demonstrates how Rust's `core::intrinsics::offload` can be used
//! to execute forward+backward passes as single GPU kernels.
//!
//! ## Key Insight
//!
//! When combined with `#[autodiff]`, GPU offload means:
//! 1. Enzyme generates the backward function at compile time
//! 2. `core::intrinsics::offload` compiles both forward+backward to a single GPU kernel
//! 3. The GPU executes the entire training step without CPU involvement
//! 4. No tape, no per-op kernel launches, no CPU-GPU synchronization
//!
//! ## Building
//!
//! Requires nightly Rust with GPU offload support:
//! ```bash
//! RUSTFLAGS="-C lto=fat -Z unstable-options -Z offload=Enable" \
//!     cargo +nightly build --features nightly-offload
//! ```

// ============================================================================
// SECTION 1: What the nightly offload API looks like (cfg-gated)
// ============================================================================

#[cfg(feature = "nightly-offload")]
mod nightly {
    #![feature(gpu_offload, core_intrinsics, autodiff)]
    use std::autodiff::autodiff_reverse;

    /// A differentiable function that runs entirely on GPU.
    ///
    /// The pipeline:
    /// 1. `#[autodiff_reverse]` generates `d_gpu_matmul` at compile time
    /// 2. `core::intrinsics::offload` compiles `d_gpu_matmul` to a GPU kernel
    /// 3. At runtime: single kernel launch, no tape, no CPU in the loop
    #[autodiff_reverse(d_gpu_matmul, Duplicated, Duplicated, Duplicated, Active)]
    fn gpu_matmul(a: &[f64], b: &[f64], c: &mut [f64]) -> f64 {
        let n = c.len();
        let k = a.len() / n; // infer inner dimension
        for i in 0..n {
            c[i] = 0.0;
            for j in 0..k {
                c[i] += a[i * k + j] * b[j];
            }
        }
        c.iter().sum() // scalar output for loss
    }

    /// Training step: forward + backward on GPU
    fn gpu_train_step(
        input: &mut [f64],
        weight: &mut [f64],
        output: &mut [f64],
        grad_input: &mut [f64],
        grad_weight: &mut [f64],
        grad_output: &mut [f64],
    ) {
        // This entire call is offloaded to GPU:
        // - The Enzyme-generated backward function runs on GPU
        // - Data transfer (host ↔ device) is managed by the offload infrastructure
        // - No cudaMalloc, no cudaMemcpy, no kernel launch API
        core::intrinsics::offload(
            _gpu_train_kernel,
            (input, weight, output, grad_input, grad_weight, grad_output),
        );
    }

    fn _gpu_train_kernel(
        input: &mut [f64],
        weight: &mut [f64],
        output: &mut [f64],
        grad_input: &mut [f64],
        grad_weight: &mut [f64],
        grad_output: &mut [f64],
    ) {
        // Enzyme-generated d_gpu_matmul runs here on GPU
        d_gpu_matmul(
            input, grad_input,
            weight, grad_weight,
            output, grad_output,
            1.0, // seed = 1.0 for reverse mode
        );
    }
}

// ============================================================================
// SECTION 2: Stable Rust simulation of the offload pipeline
// ============================================================================

/// Simulates what `core::intrinsics::offload` does under the hood.
///
/// In the real implementation, the compiler:
/// 1. Compiles `kernel_fn` to PTX/AMDGPU ISA (via LLVM)
/// 2. At runtime, copies `args` to device memory
/// 3. Launches the kernel
/// 4. Copies results back to host memory
///
/// For Neo Theano, this means the entire forward+backward can be a single
/// kernel launch with automatic data movement.
pub fn simulate_offload<F, Args>(kernel_fn: F, args: Args)
where
    F: Fn(Args),
{
    // In real offload: compile to GPU, transfer data, execute
    // Here: just execute on CPU (same computation, different execution target)
    println!("[simulated offload] executing kernel on CPU (would be GPU)");
    kernel_fn(args);
}

/// Matrix multiply + backward simulation (what would run on GPU)
pub fn matmul_forward_backward(
    input: &[f64],
    weight: &[f64],
    input_dim: usize,
    output_dim: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Forward
    let mut output = vec![0.0; output_dim];
    for o in 0..output_dim {
        for i in 0..input_dim {
            output[o] += input[i] * weight[o * input_dim + i];
        }
    }

    // Backward (Enzyme would generate this)
    let grad_output = vec![1.0; output_dim]; // assume scalar loss sum
    let mut grad_input = vec![0.0; input_dim];
    let mut grad_weight = vec![0.0; weight.len()];

    for o in 0..output_dim {
        for i in 0..input_dim {
            grad_input[i] += grad_output[o] * weight[o * input_dim + i];
            grad_weight[o * input_dim + i] += grad_output[o] * input[i];
        }
    }

    (output, grad_input, grad_weight)
}

// ============================================================================
// SECTION 3: Architecture for Neo Theano GPU offload integration
// ============================================================================

/// How compiled AD + GPU offload integrates with Neo Theano:
///
/// ```text
/// Current Architecture (tape-based):
/// ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
/// │ CPU: Build   │────▶│ GPU: matmul  │────▶│ CPU: Tape    │
/// │ tape node    │     │ kernel #1    │     │ next node    │
/// └──────────────┘     └──────────────┘     └──────────────┘
///        │                                         │
///        ▼                                         ▼
/// ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
/// │ CPU: Build   │────▶│ GPU: relu    │────▶│ CPU: Tape    │
/// │ tape node    │     │ kernel #2    │     │ next node    │
/// └──────────────┘     └──────────────┘     └──────────────┘
///        │                                         │
///       ...                                       ...
///  (N kernel launches, N CPU-GPU syncs)
///
/// Proposed Architecture (compiled AD + offload):
/// ┌──────────────┐     ┌──────────────────────────────────┐
/// │ CPU: Launch  │────▶│ GPU: Entire forward+backward     │
/// │ single kernel│     │ (fused matmul+relu+loss+grads)   │
/// └──────────────┘     └──────────────────────────────────┘
///  (1 kernel launch, 1 CPU-GPU sync)
/// ```
///
/// The `OffloadManager` coordinates this:
pub struct OffloadManager {
    /// Whether GPU offload is available (checked at runtime)
    pub gpu_available: bool,
    /// Target device: "cuda", "rocm", "metal", etc.
    pub target_device: String,
}

impl OffloadManager {
    pub fn new() -> Self {
        Self {
            gpu_available: false, // Would check at runtime
            target_device: "cpu".to_string(),
        }
    }

    /// Execute a compiled forward+backward on the best available device.
    ///
    /// With `core::intrinsics::offload`, this becomes:
    /// ```rust,ignore
    /// if self.gpu_available {
    ///     core::intrinsics::offload(compiled_fn, args);
    /// } else {
    ///     compiled_fn(args);
    /// }
    /// ```
    pub fn execute_training_step(
        &self,
        input: &[f64],
        weight: &[f64],
        input_dim: usize,
        output_dim: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        if self.gpu_available {
            // In nightly: core::intrinsics::offload(...)
            // Here: same computation, simulated offload
            println!("[offload] executing on {}", self.target_device);
        }
        matmul_forward_backward(input, weight, input_dim, output_dim)
    }
}

impl Default for OffloadManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SECTION 4: Memory transfer analysis
// ============================================================================

/// Comparison of memory transfer patterns between tape-based and compiled AD.
///
/// Tape-based (current):
/// - Forward: transfer input → GPU, get output ← GPU (per layer)
/// - Each intermediate: stays on GPU but needs separate allocation
/// - Backward: gradient tensors allocated per op, transferred per op
/// - Total transfers: O(2 * n_layers) for a sequential model
///
/// Compiled AD + offload:
/// - Transfer: input + all weights → GPU (once)
/// - Execute: entire forward+backward on GPU (no transfers)
/// - Transfer: gradients ← GPU (once)
/// - Total transfers: O(1) regardless of model depth
///
/// This is where the real speedup comes from for deep models.
pub struct MemoryTransferAnalysis {
    pub n_layers: usize,
    pub param_size_bytes: usize,
    pub activation_size_bytes: usize,
}

impl MemoryTransferAnalysis {
    pub fn tape_based_transfers(&self) -> usize {
        // Each layer: 1 input transfer + 1 output transfer (forward)
        // + 1 grad transfer + 1 grad output transfer (backward)
        self.n_layers * 4
    }

    pub fn compiled_transfers(&self) -> usize {
        // Input + weights in, gradients out — that's it
        2
    }

    pub fn transfer_reduction_factor(&self) -> f64 {
        self.tape_based_transfers() as f64 / self.compiled_transfers() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_forward_backward() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2x3

        let (output, grad_input, grad_weight) =
            matmul_forward_backward(&input, &weight, 3, 2);

        // Forward: output[0] = 1*0.1 + 2*0.2 + 3*0.3 = 1.4
        //          output[1] = 1*0.4 + 2*0.5 + 3*0.6 = 3.2
        assert!((output[0] - 1.4).abs() < 1e-10);
        assert!((output[1] - 3.2).abs() < 1e-10);

        // Backward with grad_output = [1, 1]:
        // grad_input[0] = 1*0.1 + 1*0.4 = 0.5
        assert!((grad_input[0] - 0.5).abs() < 1e-10);

        // grad_weight[0] = 1*1.0 = 1.0 (grad_output[0] * input[0])
        assert!((grad_weight[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_offload_manager() {
        let manager = OffloadManager::new();
        let input = vec![1.0, 2.0];
        let weight = vec![0.5, 0.5, 0.5, 0.5]; // 2x2

        let (output, grad_input, grad_weight) =
            manager.execute_training_step(&input, &weight, 2, 2);

        assert_eq!(output.len(), 2);
        assert_eq!(grad_input.len(), 2);
        assert_eq!(grad_weight.len(), 4);
    }

    #[test]
    fn test_transfer_analysis() {
        let analysis = MemoryTransferAnalysis {
            n_layers: 50, // ResNet-50 depth
            param_size_bytes: 100_000_000,
            activation_size_bytes: 50_000_000,
        };

        // Tape-based: 200 transfers for 50-layer model
        assert_eq!(analysis.tape_based_transfers(), 200);
        // Compiled: always 2
        assert_eq!(analysis.compiled_transfers(), 2);
        // 100x reduction
        assert!((analysis.transfer_reduction_factor() - 100.0).abs() < 1e-10);
    }
}
