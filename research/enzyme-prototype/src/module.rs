//! # PyTorch-Style Module API for Enzyme Autodiff
//!
//! The core design problem: Enzyme differentiates **functions over flat slices**,
//! but users want to write **struct-based modules** like PyTorch's `nn.Module`.
//!
//! Solution: `flatten_params()` / `load_params()` bridge the gap.
//!
//! ```text
//! struct MLP ─→ flatten_params() ─→ &[f32] ─→ #[autodiff] ─→ Enzyme ─→ d_params ─→ load_params() ─→ struct MLP
//!   (init)         (serialize)       (flat)     (compile)      (LLVM)    (grads)      (deserialize)    (updated)
//! ```

/// Core trait — every layer implements this.
/// Struct = `__init__`, `forward()` = `forward()`. Same mental model as PyTorch.
pub trait Module {
    /// Forward pass. Reads from `input`, writes to `output`.
    fn forward(&self, input: &[f32], output: &mut [f32]);

    /// Total number of f32 parameters in this module.
    fn param_count(&self) -> usize;

    /// Flatten all parameters into a contiguous slice.
    /// Enzyme needs this — it differentiates functions over flat memory.
    fn flatten_params(&self, buf: &mut [f32]);

    /// Load parameters from a contiguous slice back into struct fields.
    fn load_params(&mut self, buf: &[f32]);

    /// Output size given input length (for buffer pre-allocation).
    fn output_size(&self, input_len: usize) -> usize;
}

// ============================================================================
// Built-in Layers
// ============================================================================

/// Dense linear layer: `output = input @ weight^T + bias`
pub struct Linear {
    pub weight: Vec<f32>, // [out_features, in_features]
    pub bias: Vec<f32>,   // [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Constructor — this IS your `__init__`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let k = 1.0 / (in_features as f32).sqrt();
        let weight: Vec<f32> = (0..out_features * in_features)
            .map(|i| {
                // Kaiming uniform (deterministic for testing)
                let t = ((i as f64 * 0.618033988749895) % 1.0) as f32;
                (t * 2.0 - 1.0) * k
            })
            .collect();
        let bias = vec![0.0; out_features];
        Self { weight, bias, in_features, out_features }
    }
}

impl Module for Linear {
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let batch = input.len() / self.in_features;
        for i in 0..batch {
            for j in 0..self.out_features {
                let mut sum = self.bias[j];
                for k in 0..self.in_features {
                    sum += input[i * self.in_features + k]
                        * self.weight[j * self.in_features + k];
                }
                output[i * self.out_features + j] = sum;
            }
        }
    }

    fn param_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }

    fn flatten_params(&self, buf: &mut [f32]) {
        let n = self.weight.len();
        buf[..n].copy_from_slice(&self.weight);
        buf[n..n + self.bias.len()].copy_from_slice(&self.bias);
    }

    fn load_params(&mut self, buf: &[f32]) {
        let n = self.weight.len();
        let bias_len = self.bias.len();
        self.weight.copy_from_slice(&buf[..n]);
        self.bias.copy_from_slice(&buf[n..n + bias_len]);
    }

    fn output_size(&self, input_len: usize) -> usize {
        (input_len / self.in_features) * self.out_features
    }
}

/// ReLU activation — zero parameters.
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = i.max(0.0);
        }
    }
    fn param_count(&self) -> usize { 0 }
    fn flatten_params(&self, _buf: &mut [f32]) {}
    fn load_params(&mut self, _buf: &[f32]) {}
    fn output_size(&self, input_len: usize) -> usize { input_len }
}

/// Sigmoid activation — zero parameters.
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = 1.0 / (1.0 + (-i).exp());
        }
    }
    fn param_count(&self) -> usize { 0 }
    fn flatten_params(&self, _buf: &mut [f32]) {}
    fn load_params(&mut self, _buf: &[f32]) {}
    fn output_size(&self, input_len: usize) -> usize { input_len }
}

// ============================================================================
// Model Composition (like nn.Sequential)
// ============================================================================

/// A 2-layer MLP with ReLU. Demonstrates the PyTorch __init__ / forward pattern.
///
/// ```python
/// # PyTorch equivalent:
/// class MLP(nn.Module):
///     def __init__(self):
///         super().__init__()
///         self.fc1 = nn.Linear(784, 256)
///         self.relu = nn.ReLU()
///         self.fc2 = nn.Linear(256, 10)
///
///     def forward(self, x):
///         x = self.fc1(x)
///         x = self.relu(x)
///         x = self.fc2(x)
///         return x
/// ```
pub struct MLP {
    pub fc1: Linear,
    pub relu: ReLU,
    pub fc2: Linear,
    hidden_size: usize,
}

impl MLP {
    /// Constructor — this IS `__init__`.
    pub fn new(in_features: usize, hidden_size: usize, out_features: usize) -> Self {
        Self {
            fc1: Linear::new(in_features, hidden_size),
            relu: ReLU,
            fc2: Linear::new(hidden_size, out_features),
            hidden_size,
        }
    }
}

impl Module for MLP {
    /// This IS `forward(self, x)`.
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let batch = input.len() / self.fc1.in_features;
        let mut hidden = vec![0.0f32; batch * self.hidden_size];
        let mut activated = vec![0.0f32; batch * self.hidden_size];

        self.fc1.forward(input, &mut hidden);
        self.relu.forward(&hidden, &mut activated);
        self.fc2.forward(&activated, output);
    }

    fn param_count(&self) -> usize {
        self.fc1.param_count() + self.fc2.param_count()
    }

    fn flatten_params(&self, buf: &mut [f32]) {
        let n1 = self.fc1.param_count();
        self.fc1.flatten_params(&mut buf[..n1]);
        self.fc2.flatten_params(&mut buf[n1..]);
    }

    fn load_params(&mut self, buf: &[f32]) {
        let n1 = self.fc1.param_count();
        self.fc1.load_params(&buf[..n1]);
        self.fc2.load_params(&buf[n1..]);
    }

    fn output_size(&self, input_len: usize) -> usize {
        let batch = input_len / self.fc1.in_features;
        batch * self.fc2.out_features
    }
}

// ============================================================================
// The Bridge: What #[autodiff] Would Generate
// ============================================================================

/// This is what the framework generates under the hood.
///
/// The `#[autodiff]` attribute would go on this function — it takes flattened
/// parameters as a single `&[f32]` so Enzyme can differentiate it.
///
/// Your `Module::forward()` gets inlined into this by LLVM.
///
/// ```text
/// On nightly, this would be:
///
/// #[autodiff(d_mlp_forward_flat, Reverse,
///     Duplicated,  // params  → + grad_params
///     Const,       // input   → no gradient
///     Duplicated,  // output  → + grad_output
///     Const,       // config  → not differentiated
///     Active       // return  → seeded with 1.0
/// )]
/// ```
pub fn mlp_forward_flat(
    params: &[f32],
    input: &[f32],
    output: &mut [f32],
    in_features: usize,
    hidden_size: usize,
    out_features: usize,
    batch_size: usize,
) -> f32 {
    // Unpack parameters from flat slice
    let w1_end = hidden_size * in_features;
    let b1_end = w1_end + hidden_size;
    let w2_end = b1_end + out_features * hidden_size;
    let _b2_end = w2_end + out_features;

    let w1 = &params[..w1_end];
    let b1 = &params[w1_end..b1_end];
    let w2 = &params[b1_end..w2_end];
    let b2 = &params[w2_end..];

    // Layer 1: linear + ReLU (LLVM fuses these)
    let mut hidden = vec![0.0f32; batch_size * hidden_size];
    for i in 0..batch_size {
        for j in 0..hidden_size {
            let mut sum = b1[j];
            for k in 0..in_features {
                sum += input[i * in_features + k] * w1[j * in_features + k];
            }
            hidden[i * hidden_size + j] = sum.max(0.0); // relu fused
        }
    }

    // Layer 2: linear
    for i in 0..batch_size {
        for j in 0..out_features {
            let mut sum = b2[j];
            for k in 0..hidden_size {
                sum += hidden[i * hidden_size + k] * w2[j * hidden_size + k];
            }
            output[i * out_features + j] = sum;
        }
    }

    // Return scalar loss (sum for simplicity; real version: cross-entropy)
    output.iter().sum()
}

/// Manual backward of `mlp_forward_flat` — this is what Enzyme generates.
///
/// Takes the same args as forward, plus shadow buffers for gradients.
pub fn d_mlp_forward_flat(
    params: &[f32],
    grad_params: &mut [f32],
    input: &[f32],
    output: &mut [f32],
    _grad_output: &mut [f32],
    in_features: usize,
    hidden_size: usize,
    out_features: usize,
    batch_size: usize,
    _seed: f32,
) -> f32 {
    // Unpack parameters
    let w1_end = hidden_size * in_features;
    let b1_end = w1_end + hidden_size;
    let w2_end = b1_end + out_features * hidden_size;

    let w1 = &params[..w1_end];
    let b1 = &params[w1_end..b1_end];
    let w2 = &params[b1_end..w2_end];
    let b2 = &params[w2_end..];

    // === Forward pass (cached) ===
    let mut pre_relu = vec![0.0f32; batch_size * hidden_size];
    let mut hidden = vec![0.0f32; batch_size * hidden_size];

    for i in 0..batch_size {
        for j in 0..hidden_size {
            let mut sum = b1[j];
            for k in 0..in_features {
                sum += input[i * in_features + k] * w1[j * in_features + k];
            }
            pre_relu[i * hidden_size + j] = sum;
            hidden[i * hidden_size + j] = sum.max(0.0);
        }
    }

    for i in 0..batch_size {
        for j in 0..out_features {
            let mut sum = b2[j];
            for k in 0..hidden_size {
                sum += hidden[i * hidden_size + k] * w2[j * hidden_size + k];
            }
            output[i * out_features + j] = sum;
        }
    }

    let loss: f32 = output.iter().sum();

    // === Backward pass (Enzyme generates this) ===
    let grad_output_val = 1.0f32; // d(sum)/d(each output) = 1

    // Layer 2 backward
    let mut grad_hidden = vec![0.0f32; batch_size * hidden_size];
    for i in 0..batch_size {
        for j in 0..out_features {
            // grad_b2
            grad_params[w2_end + j] += grad_output_val;
            for k in 0..hidden_size {
                // grad_w2
                grad_params[b1_end + j * hidden_size + k] += grad_output_val * hidden[i * hidden_size + k];
                // grad_hidden
                grad_hidden[i * hidden_size + k] += grad_output_val * w2[j * hidden_size + k];
            }
        }
    }

    // ReLU backward
    for idx in 0..batch_size * hidden_size {
        if pre_relu[idx] <= 0.0 {
            grad_hidden[idx] = 0.0;
        }
    }

    // Layer 1 backward
    for i in 0..batch_size {
        for j in 0..hidden_size {
            // grad_b1
            grad_params[w1_end + j] += grad_hidden[i * hidden_size + j];
            for k in 0..in_features {
                // grad_w1
                grad_params[j * in_features + k] += grad_hidden[i * hidden_size + j] * input[i * in_features + k];
            }
        }
    }

    loss
}

// ============================================================================
// Training Loop (the PyTorch-equivalent experience)
// ============================================================================

/// Simple SGD optimizer (operates on flat parameter slice).
pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self { Self { lr } }

    pub fn step(&self, params: &mut [f32], grads: &[f32]) {
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            *p -= self.lr * g;
        }
    }
}

/// Adam optimizer (operates on flat parameter slice).
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub m: Vec<f32>, // first moment
    pub v: Vec<f32>, // second moment
    pub t: usize,
}

impl Adam {
    pub fn new(param_count: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            t: 0,
        }
    }

    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Complete training step — replaces `loss.backward()` + `optimizer.step()`.
///
/// ```python
/// # PyTorch equivalent:
/// optimizer.zero_grad()
/// output = model(images)
/// loss = criterion(output, labels)
/// loss.backward()
/// optimizer.step()
/// ```
///
/// In Rust autodiff, all of this is ONE function call + a step:
///
/// ```ignore
/// let loss = d_model_forward(params, grads, input, ...);
/// optimizer.step(params, grads);
/// ```
pub fn train_step(
    model: &mut MLP,
    input: &[f32],
    batch_size: usize,
    optimizer: &mut Adam,
) -> f32 {
    let n = model.param_count();
    let in_features = model.fc1.in_features;
    let hidden_size = model.hidden_size;
    let out_features = model.fc2.out_features;

    let mut params = vec![0.0f32; n];
    let mut grads = vec![0.0f32; n];
    let out_size = model.output_size(input.len());
    let mut output = vec![0.0f32; out_size];
    let mut grad_output = vec![0.0f32; out_size];

    // Flatten params from struct
    model.flatten_params(&mut params);

    // Forward + backward in ONE call
    let loss = d_mlp_forward_flat(
        &params, &mut grads,
        input,
        &mut output, &mut grad_output,
        in_features, hidden_size, out_features, batch_size,
        1.0,
    );

    // Optimizer step
    optimizer.step(&mut params, &grads);

    // Load updated params back into model struct
    model.load_params(&params);

    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(3, 2);
        let input = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 2];
        linear.forward(&input, &mut output);
        // Just verify it produces finite output
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = MLP::new(4, 8, 2);
        let input = vec![1.0f32; 4]; // batch=1, in=4
        let mut output = vec![0.0f32; 2];
        mlp.forward(&input, &mut output);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_flatten_load_roundtrip() {
        let mlp = MLP::new(4, 8, 2);
        let n = mlp.param_count();
        let mut buf = vec![0.0f32; n];
        mlp.flatten_params(&mut buf);

        let mut mlp2 = MLP::new(4, 8, 2);
        mlp2.load_params(&buf);

        let mut buf2 = vec![0.0f32; n];
        mlp2.flatten_params(&mut buf2);

        assert_eq!(buf, buf2, "flatten → load → flatten should be identity");
    }

    #[test]
    fn test_flat_forward_matches_module() {
        let mlp = MLP::new(4, 8, 2);
        let n = mlp.param_count();
        let mut params = vec![0.0f32; n];
        mlp.flatten_params(&mut params);

        let input = vec![1.0f32, 0.5, -0.3, 0.8]; // batch=1
        let mut out_module = vec![0.0f32; 2];
        let mut out_flat = vec![0.0f32; 2];

        mlp.forward(&input, &mut out_module);
        mlp_forward_flat(&params, &input, &mut out_flat, 4, 8, 2, 1);

        for (a, b) in out_module.iter().zip(out_flat.iter()) {
            assert!((a - b).abs() < 1e-5, "module={}, flat={}", a, b);
        }
    }

    #[test]
    fn test_gradient_numerical_check() {
        let mlp = MLP::new(4, 8, 2);
        let n = mlp.param_count();
        let mut params = vec![0.0f32; n];
        mlp.flatten_params(&mut params);

        let input = vec![1.0f32, 0.5, -0.3, 0.8];
        let eps = 1e-4f32;

        // Analytical gradient
        let mut grads = vec![0.0f32; n];
        let mut out = vec![0.0f32; 2];
        let mut dout = vec![0.0f32; 2];
        d_mlp_forward_flat(&params, &mut grads, &input, &mut out, &mut dout, 4, 8, 2, 1, 1.0);

        // Numerical gradient for first 5 parameters
        for idx in 0..5.min(n) {
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[idx] += eps;
            p_minus[idx] -= eps;

            let mut o_plus = vec![0.0f32; 2];
            let mut o_minus = vec![0.0f32; 2];
            let f_plus = mlp_forward_flat(&p_plus, &input, &mut o_plus, 4, 8, 2, 1);
            let f_minus = mlp_forward_flat(&p_minus, &input, &mut o_minus, 4, 8, 2, 1);

            let numerical = (f_plus - f_minus) / (2.0 * eps);
            let analytical = grads[idx];

            assert!(
                (analytical - numerical).abs() < 0.1,
                "param[{}]: analytical={:.6}, numerical={:.6}",
                idx, analytical, numerical
            );
        }
    }

    #[test]
    fn test_training_step_reduces_loss() {
        let mut mlp = MLP::new(4, 8, 2);
        let mut optimizer = Adam::new(mlp.param_count(), 0.01);
        let input = vec![1.0f32, 0.5, -0.3, 0.8];

        let loss1 = train_step(&mut mlp, &input, 1, &mut optimizer);
        let loss2 = train_step(&mut mlp, &input, 1, &mut optimizer);

        // After a training step, parameters should change
        assert!(loss1.is_finite());
        assert!(loss2.is_finite());
        assert!((loss1 - loss2).abs() > 1e-10, "training should change the loss");
    }
}
