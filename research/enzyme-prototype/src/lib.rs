//! # Enzyme-Based Compiler-Level Autodiff Prototype
//!
//! This crate demonstrates how Rust's `#[autodiff]` (powered by Enzyme)
//! can replace tape-based autograd for static computation graphs.
//!
//! ## How It Works
//!
//! 1. Write a normal Rust function (the "primal")
//! 2. Annotate it with `#[autodiff_reverse(...)]` or `#[autodiff_forward(...)]`
//! 3. The compiler generates a derivative function at LLVM IR level
//! 4. No tape, no runtime overhead, no per-op dispatch
//!
//! ## Building
//!
//! Requires nightly Rust with Enzyme support:
//! ```bash
//! rustup toolchain install nightly
//! RUSTFLAGS="-C lto=fat" cargo +nightly build --features nightly-autodiff
//! ```

// ============================================================================
// SECTION 1: What the nightly API looks like (cfg-gated)
// ============================================================================

#[cfg(feature = "nightly-autodiff")]
mod nightly {
    #![feature(autodiff)]
    use std::autodiff::autodiff_reverse;

    /// A simple scalar function: f(x, y) = x² + 3xy + y²
    ///
    /// Enzyme generates `d_quadratic` which computes:
    /// - ∂f/∂x = 2x + 3y
    /// - ∂f/∂y = 3x + 2y
    #[autodiff_reverse(d_quadratic, Active, Active, Active)]
    pub fn quadratic(x: f64, y: f64) -> f64 {
        x * x + 3.0 * x * y + y * y
    }

    /// Linear layer forward: output = input @ weight^T + bias
    ///
    /// `Duplicated` means: the argument gets a "shadow" buffer for gradient accumulation.
    /// Enzyme generates `d_linear_forward` that computes the forward pass AND fills
    /// the shadow buffers with gradients — all in one fused function.
    #[autodiff_reverse(d_linear_forward, Duplicated, Duplicated, Duplicated, Active)]
    pub fn linear_forward(input: &[f64], weight: &[f64], bias: &[f64]) -> f64 {
        let input_size = input.len();
        let output_size = bias.len();
        let mut total = 0.0;
        for o in 0..output_size {
            let mut sum = bias[o];
            for i in 0..input_size {
                sum += input[i] * weight[o * input_size + i];
            }
            total += sum; // sum outputs for scalar loss
        }
        total
    }

    /// Two-layer MLP: relu(x @ W1 + b1) @ W2 + b2
    ///
    /// Enzyme differentiates through:
    /// - Matrix multiplications
    /// - ReLU activation (piecewise linear — Enzyme handles branches)
    /// - Multiple layers composed together
    ///
    /// The generated backward function is a SINGLE compiled unit — LLVM can:
    /// - Fuse operations across layers
    /// - Decide optimal checkpointing (cache vs. recompute intermediates)
    /// - Vectorize the entire backward pass
    #[autodiff_reverse(d_mlp_forward, Duplicated, Duplicated, Duplicated, Duplicated, Duplicated, Active)]
    pub fn mlp_forward(
        input: &[f64],      // [batch_size * input_dim]
        w1: &[f64],         // [hidden_dim * input_dim]
        b1: &[f64],         // [hidden_dim]
        w2: &[f64],         // [output_dim * hidden_dim]
        b2: &[f64],         // [output_dim]
    ) -> f64 {
        let input_dim = input.len();
        let hidden_dim = b1.len();
        let output_dim = b2.len();

        // Layer 1: linear + ReLU
        let mut hidden = vec![0.0; hidden_dim];
        for h in 0..hidden_dim {
            let mut sum = b1[h];
            for i in 0..input_dim {
                sum += input[i] * w1[h * input_dim + i];
            }
            hidden[h] = sum.max(0.0); // ReLU — Enzyme handles this branch
        }

        // Layer 2: linear
        let mut output = 0.0;
        for o in 0..output_dim {
            let mut sum = b2[o];
            for h in 0..hidden_dim {
                sum += hidden[h] * w2[o * hidden_dim + h];
            }
            output += sum;
        }

        output
    }

    /// Softmax + cross-entropy loss
    ///
    /// This is numerically stable (log-sum-exp trick).
    /// Enzyme differentiates through the entire thing, including the
    /// data-dependent max() and the log/exp — no special backward rules needed.
    #[autodiff_reverse(d_cross_entropy, Duplicated, Const, Active)]
    pub fn cross_entropy_loss(logits: &[f64], target: usize) -> f64 {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let log_sum_exp = max_logit + sum_exp.ln();
        -(logits[target] - log_sum_exp) // negative log likelihood
    }

    /// Forward-mode AD example: computing a Jacobian-vector product (JVP)
    ///
    /// Forward mode is efficient when #inputs < #outputs.
    /// The tangent vector `dx` propagates through the function alongside the primal.
    use std::autodiff::autodiff_forward;

    #[autodiff_forward(jvp_transform, Dual, Dual)]
    pub fn nonlinear_transform(x: &[f64]) -> f64 {
        x.iter().map(|&xi| (xi * xi).sin()).sum()
    }
}

// ============================================================================
// SECTION 2: Stable Rust simulation of what Enzyme does
//
// This shows the SAME computations but implemented manually, so you can
// run and test without nightly. The point is to demonstrate the structure
// of the generated code.
// ============================================================================

/// Manual implementation of forward + backward for f(x,y) = x² + 3xy + y²
///
/// This is what Enzyme generates automatically from `quadratic()`.
pub mod manual_reference {
    /// Forward pass
    pub fn quadratic(x: f64, y: f64) -> f64 {
        x * x + 3.0 * x * y + y * y
    }

    /// Backward pass (what Enzyme generates)
    /// Returns (df/dx, df/dy)
    pub fn d_quadratic(x: f64, y: f64) -> (f64, f64) {
        let dfdx = 2.0 * x + 3.0 * y;
        let dfdy = 3.0 * x + 2.0 * y;
        (dfdx, dfdy)
    }

    /// Forward + backward for a linear layer
    /// Returns (output, grad_input, grad_weight, grad_bias)
    pub fn d_linear_forward(
        input: &[f64],
        weight: &[f64],
        bias: &[f64],
        grad_output: f64, // upstream gradient (scalar for simplicity)
    ) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        let input_size = input.len();
        let output_size = bias.len();

        // Forward
        let mut output = 0.0;
        for o in 0..output_size {
            let mut sum = bias[o];
            for i in 0..input_size {
                sum += input[i] * weight[o * input_size + i];
            }
            output += sum;
        }

        // Backward (what Enzyme synthesizes by reversing the IR)
        let mut grad_input = vec![0.0; input_size];
        let mut grad_weight = vec![0.0; weight.len()];
        let mut grad_bias = vec![0.0; output_size];

        for o in 0..output_size {
            grad_bias[o] = grad_output;
            for i in 0..input_size {
                grad_input[i] += grad_output * weight[o * input_size + i];
                grad_weight[o * input_size + i] += grad_output * input[i];
            }
        }

        (output, grad_input, grad_weight, grad_bias)
    }

    /// Forward + backward for 2-layer MLP with ReLU
    /// This is the FUSED version — everything in one function, no tape.
    pub fn d_mlp_forward(
        input: &[f64],
        w1: &[f64],
        b1: &[f64],
        w2: &[f64],
        b2: &[f64],
        grad_output: f64,
    ) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let input_dim = input.len();
        let hidden_dim = b1.len();
        let output_dim = b2.len();

        // === Forward pass (cached for backward) ===
        let mut pre_relu = vec![0.0; hidden_dim];
        let mut hidden = vec![0.0; hidden_dim];
        for h in 0..hidden_dim {
            let mut sum = b1[h];
            for i in 0..input_dim {
                sum += input[i] * w1[h * input_dim + i];
            }
            pre_relu[h] = sum;
            hidden[h] = sum.max(0.0);
        }

        let mut output = 0.0;
        for o in 0..output_dim {
            let mut sum = b2[o];
            for h in 0..hidden_dim {
                sum += hidden[h] * w2[o * hidden_dim + h];
            }
            output += sum;
        }

        // === Backward pass (Enzyme generates this by reversing the IR) ===

        // Gradient through layer 2
        let mut grad_hidden = vec![0.0; hidden_dim];
        let mut grad_w2 = vec![0.0; w2.len()];
        let mut grad_b2 = vec![0.0; output_dim];

        for o in 0..output_dim {
            grad_b2[o] = grad_output;
            for h in 0..hidden_dim {
                grad_hidden[h] += grad_output * w2[o * hidden_dim + h];
                grad_w2[o * hidden_dim + h] += grad_output * hidden[h];
            }
        }

        // Gradient through ReLU (Enzyme handles this branch natively)
        for h in 0..hidden_dim {
            if pre_relu[h] <= 0.0 {
                grad_hidden[h] = 0.0;
            }
        }

        // Gradient through layer 1
        let mut grad_input = vec![0.0; input_dim];
        let mut grad_w1 = vec![0.0; w1.len()];
        let mut grad_b1 = vec![0.0; hidden_dim];

        for h in 0..hidden_dim {
            grad_b1[h] = grad_hidden[h];
            for i in 0..input_dim {
                grad_input[i] += grad_hidden[h] * w1[h * input_dim + i];
                grad_w1[h * input_dim + i] += grad_hidden[h] * input[i];
            }
        }

        (output, grad_input, grad_w1, grad_b1, grad_w2, grad_b2)
    }
}

// ============================================================================
// SECTION 3: Integration trait — how this connects to Neo Theano
// ============================================================================

/// Trait for modules that support compiler-level AD.
///
/// When a module implements this trait, the training loop can bypass
/// the dynamic tape entirely and call the Enzyme-generated backward
/// function directly.
///
/// ```text
/// if module.use_compiled_ad() {
///     // Single fused GPU kernel for forward+backward
///     let (output, param_grads) = module.forward_backward(input, grad_output);
///     optimizer.step_with_grads(param_grads);
/// } else {
///     // Dynamic tape (existing behavior)
///     let output = module.forward(input);
///     output.backward();
///     optimizer.step();
/// }
/// ```
pub trait CompiledAD {
    /// Run forward and backward in a single compiled function.
    /// Returns (output, gradient_for_each_parameter).
    ///
    /// The backward computation is NOT tape-based — it's a compiler-generated
    /// function that Enzyme synthesized from the forward pass LLVM IR.
    fn forward_backward(&self, input: &[f64], grad_output: f64) -> (f64, Vec<Vec<f64>>);

    /// Whether this module should use compiled AD.
    /// Default: true. Override to false for modules with dynamic behavior.
    fn use_compiled_ad(&self) -> bool {
        true
    }
}

/// Example: A compiled linear layer
pub struct CompiledLinear {
    pub weight: Vec<f64>,
    pub bias: Vec<f64>,
    pub input_size: usize,
    pub output_size: usize,
}

impl CompiledLinear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Kaiming uniform initialization
        let bound = (1.0 / input_size as f64).sqrt();
        let weight: Vec<f64> = (0..output_size * input_size)
            .map(|i| {
                // Deterministic pseudo-random for reproducibility
                let t = (i as f64 * 0.618033988749895) % 1.0;
                (t * 2.0 - 1.0) * bound
            })
            .collect();
        let bias = vec![0.0; output_size];

        Self { weight, bias, input_size, output_size }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.output_size];
        for o in 0..self.output_size {
            output[o] = self.bias[o];
            for i in 0..self.input_size {
                output[o] += input[i] * self.weight[o * self.input_size + i];
            }
        }
        output
    }
}

impl CompiledAD for CompiledLinear {
    fn forward_backward(&self, input: &[f64], grad_output: f64) -> (f64, Vec<Vec<f64>>) {
        let (output, _grad_input, grad_weight, grad_bias) =
            manual_reference::d_linear_forward(input, &self.weight, &self.bias, grad_output);
        (output, vec![grad_weight, grad_bias])
    }
}

/// Example: A compiled 2-layer MLP
pub struct CompiledMLP {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl CompiledMLP {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let init = |size: usize, fan_in: usize| -> Vec<f64> {
            let bound = (1.0 / fan_in as f64).sqrt();
            (0..size)
                .map(|i| {
                    let t = (i as f64 * 0.618033988749895) % 1.0;
                    (t * 2.0 - 1.0) * bound
                })
                .collect()
        };

        Self {
            w1: init(hidden_dim * input_dim, input_dim),
            b1: vec![0.0; hidden_dim],
            w2: init(output_dim * hidden_dim, hidden_dim),
            b2: vec![0.0; output_dim],
            input_dim,
            hidden_dim,
            output_dim,
        }
    }
}

impl CompiledAD for CompiledMLP {
    fn forward_backward(&self, input: &[f64], grad_output: f64) -> (f64, Vec<Vec<f64>>) {
        let (output, _grad_input, grad_w1, grad_b1, grad_w2, grad_b2) =
            manual_reference::d_mlp_forward(input, &self.w1, &self.b1, &self.w2, &self.b2, grad_output);
        (output, vec![grad_w1, grad_b1, grad_w2, grad_b2])
    }
}

// ============================================================================
// SECTION 4: Training loop comparison
// ============================================================================

/// Demonstrates a training step using compiled AD (no tape).
///
/// Compare this with the dynamic tape version:
/// ```text
/// // Dynamic tape (current Neo Theano):
/// let x = Variable::new(input, true);
/// let output = model.forward(&x);
/// let loss = loss_fn(&output, &target);
/// loss.backward();                    // walks the tape
/// optimizer.step();                   // reads .grad from each param
///
/// // Compiled AD (this approach):
/// let (loss, grads) = model.forward_backward(&input, 1.0);
/// optimizer.step_with_grads(&grads);  // no tape involved
/// ```
pub fn compiled_training_step(
    model: &CompiledMLP,
    input: &[f64],
    learning_rate: f64,
) -> (f64, Vec<Vec<f64>>) {
    // Forward + backward in a single compiled function call
    let (loss, param_grads) = model.forward_backward(input, 1.0);

    // param_grads[0] = grad_w1, param_grads[1] = grad_b1, etc.
    // An optimizer would apply: param -= lr * grad
    let updated_grads: Vec<Vec<f64>> = param_grads
        .iter()
        .map(|g| g.iter().map(|&gi| gi * learning_rate).collect())
        .collect();

    (loss, updated_grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_gradient() {
        let (dfdx, dfdy) = manual_reference::d_quadratic(2.0, 3.0);
        // f(x,y) = x² + 3xy + y²
        // df/dx = 2x + 3y = 4 + 9 = 13
        // df/dy = 3x + 2y = 6 + 6 = 12
        assert!((dfdx - 13.0).abs() < 1e-10);
        assert!((dfdy - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_gradient() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2x3
        let bias = vec![0.01, 0.02];

        let (output, grad_input, grad_weight, grad_bias) =
            manual_reference::d_linear_forward(&input, &weight, &bias, 1.0);

        // Verify forward: output = sum of (input @ weight^T + bias)
        // neuron 0: 1*0.1 + 2*0.2 + 3*0.3 + 0.01 = 1.41
        // neuron 1: 1*0.4 + 2*0.5 + 3*0.6 + 0.02 = 3.22
        // total = 4.63
        assert!((output - 4.63).abs() < 1e-10);

        // grad_bias should be [1.0, 1.0] (upstream gradient flows through)
        assert!((grad_bias[0] - 1.0).abs() < 1e-10);
        assert!((grad_bias[1] - 1.0).abs() < 1e-10);

        // grad_input[i] = sum_o(grad_output * weight[o, i])
        // grad_input[0] = 1.0 * 0.1 + 1.0 * 0.4 = 0.5
        assert!((grad_input[0] - 0.5).abs() < 1e-10);

        // grad_weight[o*input_size+i] = grad_output * input[i]
        // grad_weight[0] = 1.0 * 1.0 = 1.0
        assert!((grad_weight[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mlp_gradient_shapes() {
        let input_dim = 4;
        let hidden_dim = 8;
        let output_dim = 2;

        let mlp = CompiledMLP::new(input_dim, hidden_dim, output_dim);
        let input = vec![1.0; input_dim];

        let (loss, grads) = mlp.forward_backward(&input, 1.0);

        // Should produce 4 gradient tensors: w1, b1, w2, b2
        assert_eq!(grads.len(), 4);
        assert_eq!(grads[0].len(), hidden_dim * input_dim);  // grad_w1
        assert_eq!(grads[1].len(), hidden_dim);               // grad_b1
        assert_eq!(grads[2].len(), output_dim * hidden_dim);  // grad_w2
        assert_eq!(grads[3].len(), output_dim);                // grad_b2

        // Loss should be finite
        assert!(loss.is_finite());
    }

    #[test]
    fn test_compiled_training_step() {
        let mlp = CompiledMLP::new(4, 8, 2);
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let lr = 0.01;

        let (loss, scaled_grads) = compiled_training_step(&mlp, &input, lr);

        assert!(loss.is_finite());
        // All gradients should be scaled by learning rate
        for grad in &scaled_grads {
            for &g in grad {
                assert!(g.is_finite());
            }
        }
    }

    /// Numerical gradient check — verifies that our analytical gradients
    /// match finite-difference approximations.
    #[test]
    fn test_gradient_numerical_check() {
        let eps = 1e-7;

        // Check df/dx at (2.0, 3.0)
        let x = 2.0;
        let y = 3.0;
        let (analytical_dfdx, analytical_dfdy) = manual_reference::d_quadratic(x, y);

        let numerical_dfdx = (manual_reference::quadratic(x + eps, y) - manual_reference::quadratic(x - eps, y)) / (2.0 * eps);
        let numerical_dfdy = (manual_reference::quadratic(x, y + eps) - manual_reference::quadratic(x, y - eps)) / (2.0 * eps);

        assert!((analytical_dfdx - numerical_dfdx).abs() < 1e-5, "dfdx: analytical={}, numerical={}", analytical_dfdx, numerical_dfdx);
        assert!((analytical_dfdy - numerical_dfdy).abs() < 1e-5, "dfdy: analytical={}, numerical={}", analytical_dfdy, numerical_dfdy);
    }
}
