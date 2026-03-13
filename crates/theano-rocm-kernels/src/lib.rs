//! ROCm/HIP kernel source management.
//!
//! This crate contains the HIP kernel source files (.hip) and handles
//! compilation when the ROCm toolkit is available. When ROCm is not
//! available, it provides the kernel sources as embedded strings for
//! runtime compilation.

/// Embedded HIP kernel sources for runtime compilation.
pub mod sources {
    /// Elementwise operation kernels (unary and binary).
    pub const ELEMENTWISE: &str = include_str!("../../../kernels/hip/elementwise.hip");

    /// Reduction kernels (sum, max, etc.).
    pub const REDUCE: &str = include_str!("../../../kernels/hip/reduce.hip");

    /// Softmax kernel (numerically stable).
    pub const SOFTMAX: &str = include_str!("../../../kernels/hip/softmax.hip");
}

/// Kernel function names for dispatch.
///
/// HIP is source-compatible with CUDA, so the kernel names are identical.
pub mod kernel_names {
    // Unary
    pub const NEG_F32: &str = "neg_f32";
    pub const ABS_F32: &str = "abs_f32";
    pub const EXP_F32: &str = "exp_f32";
    pub const LOG_F32: &str = "log_f32";
    pub const SQRT_F32: &str = "sqrt_f32";
    pub const SIN_F32: &str = "sin_f32";
    pub const COS_F32: &str = "cos_f32";
    pub const TANH_F32: &str = "tanh_f32";
    pub const SIGMOID_F32: &str = "sigmoid_f32";
    pub const RELU_F32: &str = "relu_f32";
    pub const GELU_F32: &str = "gelu_f32";
    pub const SILU_F32: &str = "silu_f32";
    pub const SQUARE_F32: &str = "square_f32";
    pub const RECIPROCAL_F32: &str = "reciprocal_f32";

    // Binary
    pub const ADD_F32: &str = "add_f32";
    pub const SUB_F32: &str = "sub_f32";
    pub const MUL_F32: &str = "mul_f32";
    pub const DIV_F32: &str = "div_f32";

    // Scalar
    pub const ADD_SCALAR_F32: &str = "add_scalar_f32";
    pub const MUL_SCALAR_F32: &str = "mul_scalar_f32";

    // Fill
    pub const FILL_F32: &str = "fill_f32";

    // Cast
    pub const CAST_F32_TO_F64: &str = "cast_f32_to_f64";
    pub const CAST_F64_TO_F32: &str = "cast_f64_to_f32";

    // Reduce
    pub const REDUCE_SUM_F32: &str = "reduce_sum_f32";
    pub const REDUCE_MAX_F32: &str = "reduce_max_f32";

    // Softmax
    pub const SOFTMAX_F32: &str = "softmax_f32";
}

/// Compute grid/block dimensions for a 1D kernel launch.
///
/// Returns `(grid_size, block_size)` suitable for HIP kernel launches.
/// Uses a block size of 256 threads (standard for AMD GCN/CDNA wavefronts).
pub fn compute_launch_config(n: usize) -> (u32, u32) {
    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    let grid_size = grid_size.min(65535);
    (grid_size, block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_sources_embedded() {
        assert!(sources::ELEMENTWISE.contains("neg_f32"));
        assert!(sources::ELEMENTWISE.contains("relu_f32"));
        assert!(sources::ELEMENTWISE.contains("add_f32"));
        assert!(sources::ELEMENTWISE.contains("sigmoid_f32"));
        assert!(sources::REDUCE.contains("reduce_sum_f32"));
        assert!(sources::REDUCE.contains("reduce_max_f32"));
        assert!(sources::SOFTMAX.contains("softmax_f32"));
    }

    #[test]
    fn test_kernel_sources_are_hip() {
        // HIP kernels use the same compat.h macros as CUDA
        assert!(sources::ELEMENTWISE.contains("THEANO_GLOBAL_FUNC"));
        assert!(sources::REDUCE.contains("THEANO_GLOBAL_FUNC"));
        assert!(sources::SOFTMAX.contains("THEANO_GLOBAL_FUNC"));
    }

    #[test]
    fn test_launch_config() {
        let (grid, block) = compute_launch_config(1024);
        assert_eq!(block, 256);
        assert_eq!(grid, 4);

        let (grid, _) = compute_launch_config(1);
        assert_eq!(grid, 1);

        let (grid, _) = compute_launch_config(256);
        assert_eq!(grid, 1);

        let (grid, _) = compute_launch_config(257);
        assert_eq!(grid, 2);
    }

    #[test]
    fn test_launch_config_large() {
        let (grid, block) = compute_launch_config(100_000_000);
        assert_eq!(block, 256);
        assert_eq!(grid, 65535); // clamped to max
    }

    #[test]
    fn test_kernel_names() {
        assert_eq!(kernel_names::NEG_F32, "neg_f32");
        assert_eq!(kernel_names::ADD_F32, "add_f32");
        assert_eq!(kernel_names::REDUCE_SUM_F32, "reduce_sum_f32");
        assert_eq!(kernel_names::SOFTMAX_F32, "softmax_f32");
    }
}
