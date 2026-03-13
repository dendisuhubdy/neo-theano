//! CUDA kernel source management and PTX compilation.
//!
//! This crate contains the CUDA kernel source files (.cu) and handles
//! compilation to PTX at build time when the CUDA toolkit is available.
//! When CUDA is not available, it provides the kernel sources as embedded
//! strings for runtime compilation.

/// Embedded CUDA kernel sources for runtime compilation.
pub mod sources {
    /// Elementwise operation kernels (unary and binary).
    pub const ELEMENTWISE: &str = include_str!("../../../kernels/cuda/elementwise.cu");

    /// Reduction kernels (sum, max, etc.).
    pub const REDUCE: &str = include_str!("../../../kernels/cuda/reduce.cu");

    /// Softmax kernel (numerically stable).
    pub const SOFTMAX: &str = include_str!("../../../kernels/cuda/softmax.cu");
}

/// Kernel function names for dispatch.
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
        assert!(sources::REDUCE.contains("reduce_sum_f32"));
        assert!(sources::SOFTMAX.contains("softmax_f32"));
    }

    #[test]
    fn test_launch_config() {
        let (grid, block) = compute_launch_config(1024);
        assert_eq!(block, 256);
        assert_eq!(grid, 4);

        let (grid, _) = compute_launch_config(1);
        assert_eq!(grid, 1);
    }
}
