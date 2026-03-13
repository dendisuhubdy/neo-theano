//! Metal Shading Language (MSL) compute kernels.
//!
//! These are embedded as string constants and compiled at runtime via
//! `MTLDevice::newLibraryWithSource` when using the real Metal API.
//! The kernels target Apple GPU architecture with unified memory.

/// Elementwise operations in MSL.
pub const ELEMENTWISE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void neg_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = -input[id];
}

kernel void relu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = max(input[id], 0.0f);
}

kernel void sigmoid_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}

kernel void exp_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = exp(input[id]);
}

kernel void tanh_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = tanh(input[id]);
}

kernel void abs_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = abs(input[id]);
}

kernel void sqrt_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = sqrt(input[id]);
}

kernel void rsqrt_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = rsqrt(input[id]);
}

kernel void log_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = log(input[id]);
}

kernel void sin_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = sin(input[id]);
}

kernel void cos_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = cos(input[id]);
}

kernel void floor_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = floor(input[id]);
}

kernel void ceil_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = ceil(input[id]);
}

kernel void round_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = round(input[id]);
}

kernel void square_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id] * input[id];
}

kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    output[id] = x / (1.0f + exp(-x));
}

kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    float c = 0.7978845608f; // sqrt(2/pi)
    output[id] = 0.5f * x * (1.0f + tanh(c * (x + 0.044715f * x * x * x)));
}

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] + b[id];
}

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] - b[id];
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] * b[id];
}

kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] / b[id];
}

kernel void pow_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = pow(a[id], b[id]);
}
"#;

/// Softmax kernel in MSL.
pub const SOFTMAX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Simplified softmax for a single row
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dim_size [[buffer(2)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint offset = row_id * dim_size;

    // Find max
    float max_val = -INFINITY;
    for (uint i = tid; i < dim_size; i += 256) {
        max_val = max(max_val, input[offset + i]);
    }
    // (Would need threadgroup reduction here for multi-thread)

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = tid; i < dim_size; i += 256) {
        float val = exp(input[offset + i] - max_val);
        output[offset + i] = val;
        sum += val;
    }

    // Normalize
    for (uint i = tid; i < dim_size; i += 256) {
        output[offset + i] /= sum;
    }
}
"#;

/// Reduction kernels in MSL.
pub const REDUCE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tcount [[threads_per_grid]]
) {
    float sum = 0.0f;
    for (uint i = tid; i < count; i += tcount) {
        sum += input[i];
    }
    // Atomic add to output (simplified; real version uses threadgroup reduction)
    output[0] = sum;
}

kernel void reduce_max_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tcount [[threads_per_grid]]
) {
    float max_val = -INFINITY;
    for (uint i = tid; i < count; i += tcount) {
        max_val = max(max_val, input[i]);
    }
    output[0] = max_val;
}

kernel void reduce_min_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tcount [[threads_per_grid]]
) {
    float min_val = INFINITY;
    for (uint i = tid; i < count; i += tcount) {
        min_val = min(min_val, input[i]);
    }
    output[0] = min_val;
}
"#;

// ---------- Kernel name constants ----------

/// Unary kernel names (must match the function names in ELEMENTWISE_MSL).
pub mod kernel_names {
    pub const NEG_F32: &str = "neg_f32";
    pub const RELU_F32: &str = "relu_f32";
    pub const SIGMOID_F32: &str = "sigmoid_f32";
    pub const EXP_F32: &str = "exp_f32";
    pub const TANH_F32: &str = "tanh_f32";
    pub const ABS_F32: &str = "abs_f32";
    pub const SQRT_F32: &str = "sqrt_f32";
    pub const RSQRT_F32: &str = "rsqrt_f32";
    pub const LOG_F32: &str = "log_f32";
    pub const SIN_F32: &str = "sin_f32";
    pub const COS_F32: &str = "cos_f32";
    pub const FLOOR_F32: &str = "floor_f32";
    pub const CEIL_F32: &str = "ceil_f32";
    pub const ROUND_F32: &str = "round_f32";
    pub const SQUARE_F32: &str = "square_f32";
    pub const SILU_F32: &str = "silu_f32";
    pub const GELU_F32: &str = "gelu_f32";

    pub const ADD_F32: &str = "add_f32";
    pub const SUB_F32: &str = "sub_f32";
    pub const MUL_F32: &str = "mul_f32";
    pub const DIV_F32: &str = "div_f32";
    pub const POW_F32: &str = "pow_f32";

    pub const SOFTMAX_F32: &str = "softmax_f32";
    pub const REDUCE_SUM_F32: &str = "reduce_sum_f32";
    pub const REDUCE_MAX_F32: &str = "reduce_max_f32";
    pub const REDUCE_MIN_F32: &str = "reduce_min_f32";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_msl_contains_kernels() {
        assert!(ELEMENTWISE_MSL.contains("kernel void neg_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void relu_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void sigmoid_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void exp_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void tanh_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void abs_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void sqrt_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void rsqrt_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void log_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void sin_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void cos_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void floor_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void ceil_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void round_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void square_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void silu_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void gelu_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void add_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void sub_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void mul_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void div_f32"));
        assert!(ELEMENTWISE_MSL.contains("kernel void pow_f32"));
    }

    #[test]
    fn test_softmax_msl_contains_kernel() {
        assert!(SOFTMAX_MSL.contains("kernel void softmax_f32"));
        assert!(SOFTMAX_MSL.contains("dim_size"));
    }

    #[test]
    fn test_reduce_msl_contains_kernels() {
        assert!(REDUCE_MSL.contains("kernel void reduce_sum_f32"));
        assert!(REDUCE_MSL.contains("kernel void reduce_max_f32"));
        assert!(REDUCE_MSL.contains("kernel void reduce_min_f32"));
    }

    #[test]
    fn test_kernel_names_match_msl() {
        // Verify that kernel name constants match actual function names in MSL source
        assert!(ELEMENTWISE_MSL.contains(&format!("kernel void {}", kernel_names::NEG_F32)));
        assert!(ELEMENTWISE_MSL.contains(&format!("kernel void {}", kernel_names::RELU_F32)));
        assert!(ELEMENTWISE_MSL.contains(&format!("kernel void {}", kernel_names::SIGMOID_F32)));
        assert!(ELEMENTWISE_MSL.contains(&format!("kernel void {}", kernel_names::EXP_F32)));
        assert!(ELEMENTWISE_MSL.contains(&format!("kernel void {}", kernel_names::TANH_F32)));
        assert!(ELEMENTWISE_MSL.contains(&format!("kernel void {}", kernel_names::ADD_F32)));
        assert!(ELEMENTWISE_MSL.contains(&format!("kernel void {}", kernel_names::MUL_F32)));
        assert!(SOFTMAX_MSL.contains(&format!("kernel void {}", kernel_names::SOFTMAX_F32)));
        assert!(REDUCE_MSL.contains(&format!("kernel void {}", kernel_names::REDUCE_SUM_F32)));
        assert!(REDUCE_MSL.contains(&format!("kernel void {}", kernel_names::REDUCE_MAX_F32)));
        assert!(REDUCE_MSL.contains(&format!("kernel void {}", kernel_names::REDUCE_MIN_F32)));
    }

    #[test]
    fn test_msl_uses_metal_stdlib() {
        assert!(ELEMENTWISE_MSL.contains("#include <metal_stdlib>"));
        assert!(SOFTMAX_MSL.contains("#include <metal_stdlib>"));
        assert!(REDUCE_MSL.contains("#include <metal_stdlib>"));
    }

    #[test]
    fn test_msl_uses_proper_buffer_attributes() {
        // Metal kernels use [[buffer(N)]] attributes
        assert!(ELEMENTWISE_MSL.contains("[[buffer(0)]]"));
        assert!(ELEMENTWISE_MSL.contains("[[buffer(1)]]"));
        assert!(ELEMENTWISE_MSL.contains("[[thread_position_in_grid]]"));
    }

    #[test]
    fn test_kernel_name_constants_are_valid() {
        // Ensure kernel names don't contain spaces or special chars
        let names = [
            kernel_names::NEG_F32,
            kernel_names::RELU_F32,
            kernel_names::ADD_F32,
            kernel_names::MUL_F32,
            kernel_names::SOFTMAX_F32,
            kernel_names::REDUCE_SUM_F32,
        ];
        for name in &names {
            assert!(!name.is_empty());
            assert!(!name.contains(' '));
            assert!(name.chars().all(|c| c.is_alphanumeric() || c == '_'));
        }
    }
}
