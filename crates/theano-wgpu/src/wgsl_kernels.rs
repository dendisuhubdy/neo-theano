//! WGSL (WebGPU Shading Language) compute shaders for the WebGPU backend.
//!
//! These shaders are compiled at runtime by the `wgpu` device and dispatched
//! as compute pipelines. Each entry point operates on `storage` buffers of f32.

/// WGSL source for elementwise unary operations.
pub const ELEMENTWISE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn neg_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = -input[idx];
}

@compute @workgroup_size(256)
fn abs_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = abs(input[idx]);
}

@compute @workgroup_size(256)
fn relu_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = max(input[idx], 0.0);
}

@compute @workgroup_size(256)
fn exp_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = exp(input[idx]);
}

@compute @workgroup_size(256)
fn log_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = log(input[idx]);
}

@compute @workgroup_size(256)
fn sqrt_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = sqrt(input[idx]);
}

@compute @workgroup_size(256)
fn sigmoid_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

@compute @workgroup_size(256)
fn tanh_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = tanh(input[idx]);
}

@compute @workgroup_size(256)
fn sin_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = sin(input[idx]);
}

@compute @workgroup_size(256)
fn cos_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = cos(input[idx]);
}

@compute @workgroup_size(256)
fn floor_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = floor(input[idx]);
}

@compute @workgroup_size(256)
fn ceil_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = ceil(input[idx]);
}

@compute @workgroup_size(256)
fn round_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = round(input[idx]);
}

@compute @workgroup_size(256)
fn square_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = input[idx] * input[idx];
}

@compute @workgroup_size(256)
fn reciprocal_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input) { return; }
    output[idx] = 1.0 / input[idx];
}
"#;

/// WGSL source for elementwise binary operations.
pub const BINARY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn add_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&a) { return; }
    out[idx] = a[idx] + b[idx];
}

@compute @workgroup_size(256)
fn sub_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&a) { return; }
    out[idx] = a[idx] - b[idx];
}

@compute @workgroup_size(256)
fn mul_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&a) { return; }
    out[idx] = a[idx] * b[idx];
}

@compute @workgroup_size(256)
fn div_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&a) { return; }
    out[idx] = a[idx] / b[idx];
}

@compute @workgroup_size(256)
fn max_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&a) { return; }
    out[idx] = max(a[idx], b[idx]);
}

@compute @workgroup_size(256)
fn min_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&a) { return; }
    out[idx] = min(a[idx], b[idx]);
}

@compute @workgroup_size(256)
fn pow_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&a) { return; }
    out[idx] = pow(a[idx], b[idx]);
}
"#;

/// WGSL source for reduction operations.
pub const REDUCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum_f32(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>,
                  @builtin(workgroup_id) wid: vec3<u32>) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load into shared memory
    if global_idx < arrayLength(&input) {
        shared_data[local_idx] = input[global_idx];
    } else {
        shared_data[local_idx] = 0.0;
    }
    workgroupBarrier();

    // Tree reduction
    var stride: u32 = 128u;
    loop {
        if stride == 0u { break; }
        if local_idx < stride {
            shared_data[local_idx] = shared_data[local_idx] + shared_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Write result
    if local_idx == 0u {
        output[wid.x] = shared_data[0];
    }
}
"#;

// Kernel entry point names for dispatching.

/// Unary kernel entry points.
pub mod unary {
    pub const NEG_F32: &str = "neg_f32";
    pub const ABS_F32: &str = "abs_f32";
    pub const RELU_F32: &str = "relu_f32";
    pub const EXP_F32: &str = "exp_f32";
    pub const LOG_F32: &str = "log_f32";
    pub const SQRT_F32: &str = "sqrt_f32";
    pub const SIGMOID_F32: &str = "sigmoid_f32";
    pub const TANH_F32: &str = "tanh_f32";
    pub const SIN_F32: &str = "sin_f32";
    pub const COS_F32: &str = "cos_f32";
    pub const FLOOR_F32: &str = "floor_f32";
    pub const CEIL_F32: &str = "ceil_f32";
    pub const ROUND_F32: &str = "round_f32";
    pub const SQUARE_F32: &str = "square_f32";
    pub const RECIPROCAL_F32: &str = "reciprocal_f32";
}

/// Binary kernel entry points.
pub mod binary {
    pub const ADD_F32: &str = "add_f32";
    pub const SUB_F32: &str = "sub_f32";
    pub const MUL_F32: &str = "mul_f32";
    pub const DIV_F32: &str = "div_f32";
    pub const MAX_F32: &str = "max_f32";
    pub const MIN_F32: &str = "min_f32";
    pub const POW_F32: &str = "pow_f32";
}

/// Reduction kernel entry points.
pub mod reduce {
    pub const SUM_F32: &str = "reduce_sum_f32";
}

/// Default workgroup size for compute shaders.
pub const DEFAULT_WORKGROUP_SIZE: u32 = 256;

/// Calculate the number of workgroups needed for a given element count.
pub fn num_workgroups(num_elements: usize) -> u32 {
    ((num_elements as u32) + DEFAULT_WORKGROUP_SIZE - 1) / DEFAULT_WORKGROUP_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_sources_not_empty() {
        assert!(!ELEMENTWISE_WGSL.is_empty());
        assert!(!BINARY_WGSL.is_empty());
        assert!(!REDUCE_WGSL.is_empty());
    }

    #[test]
    fn test_kernel_names() {
        assert_eq!(unary::NEG_F32, "neg_f32");
        assert_eq!(binary::ADD_F32, "add_f32");
        assert_eq!(reduce::SUM_F32, "reduce_sum_f32");
    }

    #[test]
    fn test_num_workgroups() {
        assert_eq!(num_workgroups(1), 1);
        assert_eq!(num_workgroups(256), 1);
        assert_eq!(num_workgroups(257), 2);
        assert_eq!(num_workgroups(512), 2);
        assert_eq!(num_workgroups(1024), 4);
    }

    #[test]
    fn test_elementwise_contains_entry_points() {
        assert!(ELEMENTWISE_WGSL.contains("fn neg_f32"));
        assert!(ELEMENTWISE_WGSL.contains("fn relu_f32"));
        assert!(ELEMENTWISE_WGSL.contains("fn exp_f32"));
        assert!(ELEMENTWISE_WGSL.contains("fn sigmoid_f32"));
        assert!(ELEMENTWISE_WGSL.contains("fn tanh_f32"));
    }

    #[test]
    fn test_binary_contains_entry_points() {
        assert!(BINARY_WGSL.contains("fn add_f32"));
        assert!(BINARY_WGSL.contains("fn mul_f32"));
        assert!(BINARY_WGSL.contains("fn sub_f32"));
        assert!(BINARY_WGSL.contains("fn div_f32"));
    }

    #[test]
    fn test_reduce_contains_entry_points() {
        assert!(REDUCE_WGSL.contains("fn reduce_sum_f32"));
    }

    #[test]
    fn test_bounds_checking_in_kernels() {
        // All kernels should have bounds checking via arrayLength
        assert!(ELEMENTWISE_WGSL.contains("arrayLength"));
        assert!(BINARY_WGSL.contains("arrayLength"));
        assert!(REDUCE_WGSL.contains("arrayLength"));
    }
}
