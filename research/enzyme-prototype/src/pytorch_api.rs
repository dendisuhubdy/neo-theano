//! # Zero-Change PyTorch API with Enzyme Autodiff Under the Hood
//!
//! The key insight from the user: **don't change the PyTorch API at all**.
//! Just make it faster by using `#[autodiff]` and `core::intrinsics::offload`
//! internally, transparently.
//!
//! The user writes the EXACT same code as current Neo Theano:
//! ```ignore
//! let model = MLP {
//!     fc1: Linear::new(784, 256),
//!     relu: ReLU,
//!     fc2: Linear::new(256, 10),
//! };
//!
//! fn forward(&self, x: &Tensor) -> Tensor {
//!     let h = self.fc1.forward(x);
//!     let h = self.relu.forward(&h);
//!     self.fc2.forward(&h)
//! }
//!
//! let loss = model.train_step(&batch, &labels, &mut optimizer);
//! ```
//!
//! The framework detects at compile/runtime whether to use:
//! - **Enzyme path**: when the graph is static and nightly features are available
//! - **Tape path**: when the graph is dynamic (existing behavior, unchanged)

// ============================================================================
// SECTION 1: The Unchanged API (same as current Neo Theano)
// ============================================================================

/// Minimal tensor type for demonstration.
/// In the real framework, this is `theano_core::Tensor`.
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<f32>>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape, requires_grad: false, grad: None }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::new(vec![0.0; n], shape.to_vec())
    }

    pub fn requires_grad_(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// The Module trait — IDENTICAL to what exists in `theano-nn`.
/// Users implement `forward()` and `parameters()`. That's it.
pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}

// ============================================================================
// SECTION 2: Layers (unchanged API)
// ============================================================================

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let k = 1.0 / (in_features as f32).sqrt();
        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| {
                let t = ((i as f64 * 0.618033988749895) % 1.0) as f32;
                (t * 2.0 - 1.0) * k
            })
            .collect();
        Self {
            weight: Tensor::new(weight_data, vec![out_features, in_features]).requires_grad_(),
            bias: Tensor::new(vec![0.0; out_features], vec![out_features]).requires_grad_(),
            in_features,
            out_features,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let batch = input.data.len() / self.in_features;
        let mut out_data = vec![0.0f32; batch * self.out_features];
        for i in 0..batch {
            for j in 0..self.out_features {
                let mut sum = self.bias.data[j];
                for k in 0..self.in_features {
                    sum += input.data[i * self.in_features + k]
                        * self.weight.data[j * self.in_features + k];
                }
                out_data[i * self.out_features + j] = sum;
            }
        }
        Tensor::new(out_data, vec![batch, self.out_features])
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let data: Vec<f32> = input.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(data, input.shape.clone())
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

// ============================================================================
// SECTION 3: Model Definition (EXACT PyTorch pattern)
// ============================================================================

/// ```python
/// # PyTorch:
/// class MLP(nn.Module):
///     def __init__(self):
///         self.fc1 = nn.Linear(784, 256)
///         self.relu = nn.ReLU()
///         self.fc2 = nn.Linear(256, 10)
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
}

impl MLP {
    pub fn new() -> Self {
        MLP {
            fc1: Linear::new(784, 256),
            relu: ReLU,
            fc2: Linear::new(256, 10),
        }
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.forward(x);
        let h = self.relu.forward(&h);
        self.fc2.forward(&h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}

// ============================================================================
// SECTION 4: The Magic — Transparent Enzyme Integration
//
// The user's code above is UNCHANGED. The framework adds Enzyme
// acceleration by:
//
// 1. Detecting static graphs (model structure doesn't change between forward calls)
// 2. Generating a flattened #[autodiff] function at compile time via proc-macro
// 3. Using that compiled backward instead of tape replay
//
// The user NEVER sees this — they just call model.forward() and loss.backward()
// as before, but it runs faster.
// ============================================================================

/// Describes how the framework transparently chooses Enzyme vs tape.
///
/// When a model is wrapped in `CompiledModel`, the framework:
/// 1. On first forward: traces the graph to learn the structure
/// 2. Generates a flat #[autodiff] function matching that structure
/// 3. On subsequent forwards: uses the compiled path
///
/// If the graph changes (dynamic model), it falls back to tape automatically.
pub struct CompiledModel<M: Module> {
    model: M,
    /// Cached flattened parameters for Enzyme path
    param_buf: Vec<f32>,
    /// Cached gradient buffer (reused, zero-alloc in hot path)
    grad_buf: Vec<f32>,
    /// Whether compiled path is active
    compiled_ready: bool,
}

impl<M: Module> CompiledModel<M> {
    pub fn new(model: M) -> Self {
        let params = model.parameters();
        let total_params: usize = params.iter().map(|t| t.numel()).sum();
        Self {
            model,
            param_buf: vec![0.0; total_params],
            grad_buf: vec![0.0; total_params],
            compiled_ready: false,
        }
    }

    /// The user calls this exactly like they would with dynamic autograd:
    /// `let output = compiled_model.forward(input);`
    ///
    /// But internally, if Enzyme is available, the forward+backward
    /// is a single compiled function.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Same API, same behavior — just potentially faster under the hood
        self.model.forward(input)
    }

    /// Replaces the `loss.backward()` + `optimizer.step()` pattern.
    ///
    /// ```ignore
    /// // User code (unchanged from PyTorch style):
    /// let output = model.forward(&input);
    /// let loss = criterion(&output, &target);
    /// loss.backward();           // ← this is where Enzyme kicks in
    /// optimizer.step();
    ///
    /// // Or with the convenience wrapper:
    /// let loss = model.train_step(&input, &target, &mut optimizer);
    /// ```
    pub fn train_step(
        &mut self,
        input: &Tensor,
        _target: &Tensor,
        _lr: f32,
    ) -> f32 {
        // Step 1: Flatten params
        let params = self.model.parameters();
        let mut offset = 0;
        for p in &params {
            self.param_buf[offset..offset + p.numel()].copy_from_slice(&p.data);
            offset += p.numel();
        }

        // Step 2: Zero gradients (just memset — no param group walking)
        self.grad_buf.fill(0.0);

        // Step 3: Forward + backward
        // On nightly with Enzyme: calls d_model_forward (compiled, single function)
        // On stable: falls back to tape-based autograd (existing behavior)
        let output = self.model.forward(input);
        let loss: f32 = output.data.iter().sum(); // placeholder loss

        // In the Enzyme path, grad_buf would be filled by d_model_forward.
        // Here we simulate with manual backward:
        self._simulate_backward(input);

        // Step 4: SGD update
        for (p, g) in self.param_buf.iter_mut().zip(self.grad_buf.iter()) {
            *p -= _lr * g;
        }

        self.compiled_ready = true;
        loss
    }

    fn _simulate_backward(&mut self, _input: &Tensor) {
        // In real implementation with nightly:
        //   d_model_forward(&self.param_buf, &mut self.grad_buf, input, ..., 1.0);
        // Here: fill with placeholder gradients
        for g in self.grad_buf.iter_mut() {
            *g = 0.001; // placeholder
        }
    }
}

// ============================================================================
// SECTION 5: What the nightly implementation would look like internally
//
// This is the code the FRAMEWORK generates — NOT what the user writes.
// The user's API is unchanged. This is the optimization layer.
// ============================================================================

/// What the framework generates when it sees a static model structure.
///
/// On nightly, this would have #[autodiff_reverse(...)] applied:
/// ```ignore
/// #[autodiff_reverse(d_static_mlp, Duplicated, Const, Duplicated, Active)]
/// ```
///
/// The user never sees or calls this directly.
fn _static_mlp_forward(
    params: &[f32],      // flattened [w1, b1, w2, b2]
    input: &[f32],       // [batch, 784]
    output: &mut [f32],  // [batch, 10]
    batch_size: usize,
) -> f32 {
    // Unpack (compiler optimizes to direct offset reads)
    let w1 = &params[..256 * 784];
    let b1 = &params[256 * 784..256 * 784 + 256];
    let w2 = &params[256 * 784 + 256..256 * 784 + 256 + 10 * 256];
    let b2 = &params[256 * 784 + 256 + 10 * 256..];

    // Forward — user's forward() inlined by LLVM
    let mut hidden = vec![0.0f32; batch_size * 256];
    for i in 0..batch_size {
        for j in 0..256 {
            let mut sum = b1[j];
            for k in 0..784 {
                sum += input[i * 784 + k] * w1[j * 784 + k];
            }
            hidden[i * 256 + j] = sum.max(0.0); // relu fused
        }
    }

    for i in 0..batch_size {
        for j in 0..10 {
            let mut sum = b2[j];
            for k in 0..256 {
                sum += hidden[i * 256 + k] * w2[j * 256 + k];
            }
            output[i * 10 + j] = sum;
        }
    }

    output.iter().sum()
}

// ============================================================================
// SECTION 6: How the user's code looks end-to-end
// ============================================================================

/// Complete training program — nearly identical to PyTorch.
///
/// ```ignore
/// // ─── __init__ ───
/// let model = MLP::new();               // PyTorch: model = MLP()
///
/// // ─── Wrap for Enzyme (optional, transparent) ───
/// let mut model = CompiledModel::new(model);
///
/// // ─── Training loop ───
/// for epoch in 0..10 {
///     for batch in dataloader.iter() {
///         // forward(self, x)
///         let output = model.forward(&batch.images);
///
///         // loss + backward + step (Enzyme does this in ONE call)
///         let loss = model.train_step(
///             &batch.images,
///             &batch.labels,
///             0.001, // lr
///         );
///
///         println!("loss: {:.4}", loss);
///     }
/// }
/// ```
///
/// **What changed from the PyTorch API?** Nothing in the model definition.
/// The only addition is `CompiledModel::new(model)` which wraps the existing
/// model for Enzyme acceleration. Without the wrapper, everything works
/// exactly as before with tape-based autograd.
pub fn _example_training_program() {
    // This function exists only for documentation.
    // The actual implementation would integrate with theano-core::Tensor,
    // theano-autograd::Variable, and theano-optim::Optimizer.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unchanged_api_forward() {
        let model = MLP::new();
        let input = Tensor::new(vec![0.5f32; 784], vec![1, 784]);
        let output = model.forward(&input);
        assert_eq!(output.shape, vec![1, 10]);
        assert!(output.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_compiled_model_wraps_transparently() {
        let model = MLP::new();
        let compiled = CompiledModel::new(model);
        let input = Tensor::new(vec![0.5f32; 784], vec![1, 784]);

        // Same forward() call — user doesn't know Enzyme is underneath
        let output = compiled.forward(&input);
        assert_eq!(output.shape, vec![1, 10]);
    }

    #[test]
    fn test_train_step() {
        let model = MLP::new();
        let mut compiled = CompiledModel::new(model);
        let input = Tensor::new(vec![0.5f32; 784], vec![1, 784]);
        let target = Tensor::new(vec![1.0f32; 10], vec![1, 10]);

        let loss = compiled.train_step(&input, &target, 0.001);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_parameters_collected() {
        let model = MLP::new();
        let params = model.parameters();

        // fc1: weight(256*784) + bias(256) = 200960
        // fc2: weight(10*256) + bias(10) = 2570
        let total: usize = params.iter().map(|t| t.numel()).sum();
        assert_eq!(total, 256 * 784 + 256 + 10 * 256 + 10);
    }
}
