# Compiler-Level Autodiff for Neo Theano

## Problem Statement

Neo Theano currently implements **tape-based reverse-mode AD** (dynamic autograd), which mirrors PyTorch's eager execution model. While flexible, this approach has fundamental efficiency limitations on GPUs:

1. **Per-operation kernel dispatch** — each op is a separate GPU kernel launch (~5-10μs overhead each)
2. **CPU-GPU synchronization** — the tape lives on CPU, creating ping-pong between host and device
3. **No cross-operation fusion** — `relu(matmul(x, w) + b)` becomes 3 separate kernel launches instead of 1
4. **Tape memory overhead** — storing every intermediate for backward pass

These costs are tolerable in PyTorch because operations are coarse-grained (large matrix ops amortize dispatch overhead). But the trend line is clear: PyTorch itself is moving toward compilation (`torch.compile`, Dynamo/Inductor, CUDA Graphs) to eliminate dynamism.

## Proposed Architecture: Dual-Mode AD

We propose a **hybrid system** where:

- **Static graphs** (fixed architecture, known at compile time) → **Enzyme/`#[autodiff]`** for compiler-level AD
- **Dynamic graphs** (architecture varies per input) → **Existing tape-based autograd** (unchanged)

This gives users the best of both worlds with an explicit opt-in mechanism.

```
                    ┌─────────────────────────┐
                    │      User Code          │
                    │  fn model(x: &[f64])    │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   Compilation Mode?      │
                    └──┬───────────────────┬──┘
                       │                   │
              #[autodiff_reverse]     Variable::new()
                       │                   │
              ┌────────▼────────┐  ┌───────▼────────┐
              │  Enzyme (LLVM)  │  │  Dynamic Tape   │
              │  compile-time   │  │  runtime graph   │
              │  AD synthesis   │  │  construction    │
              └────────┬────────┘  └───────┬─────────┘
                       │                   │
              ┌────────▼────────┐  ┌───────▼─────────┐
              │ Fused GPU       │  │ Per-op kernel    │
              │ forward+backward│  │ launches         │
              │ single kernel   │  │ (current model)  │
              └─────────────────┘  └─────────────────┘
```

## Integration Strategy

### Layer 1: `#[autodiff]` on Pure Numerical Functions

The simplest integration — annotate pure numerical functions that implement forward passes:

```rust
#![feature(autodiff)]
use std::autodiff::autodiff_reverse;

// The forward function — pure numerical computation
#[autodiff_reverse(d_linear, Duplicated, Duplicated, Duplicated, Active)]
fn linear_forward(input: &[f64], weight: &[f64], bias: &[f64]) -> f64 {
    // matmul + bias — Enzyme sees through all of this
    let mut out = 0.0;
    for i in 0..input.len() {
        out += input[i] * weight[i];
    }
    out + bias[0]
}
```

Enzyme generates `d_linear` at compile time — a function that computes both the forward pass and the gradient w.r.t. all `Duplicated` inputs. No tape, no runtime overhead, no per-op dispatch.

### Layer 2: GPU Offload via `core::intrinsics::offload`

Once we have the differentiated function, we can offload it to GPU:

```rust
#![feature(gpu_offload, core_intrinsics)]

fn train_step(input: &mut [f64], weight: &mut [f64], grad_input: &mut [f64], grad_weight: &mut [f64]) {
    core::intrinsics::offload(gpu_train_kernel, (input, weight, grad_input, grad_weight))
}

fn gpu_train_kernel(input: &mut [f64], weight: &mut [f64], grad_input: &mut [f64], grad_weight: &mut [f64]) {
    // forward + backward in a single GPU kernel
    d_linear(input, grad_input, weight, grad_weight, &[0.0], &mut [0.0], 1.0);
}
```

The compiler generates a single GPU kernel for the entire forward+backward pass. No tape walking on CPU, no kernel launch overhead per operation.

### Layer 3: Integration with Neo Theano's Module System

The key design question: how do `#[autodiff]` functions compose with our existing `Module` trait?

```rust
/// A module that uses compiler-level AD for its forward+backward
pub trait CompiledModule: Module {
    /// The Enzyme-generated backward function
    fn backward_compiled(&self, input: &Tensor, grad_output: &Tensor) -> (Tensor, Vec<Tensor>);

    /// Whether to use compiled AD (true) or dynamic tape (false)
    fn use_compiled_ad(&self) -> bool { true }
}

/// Linear layer with dual-mode AD
pub struct CompiledLinear {
    inner: Linear,  // reuse existing Linear
}

impl Module for CompiledLinear {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.inner.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.inner.parameters()
    }
}

impl CompiledModule for CompiledLinear {
    fn backward_compiled(&self, input: &Tensor, grad_output: &Tensor) -> (Tensor, Vec<Tensor>) {
        // Call Enzyme-generated derivative
        // This is a single fused operation, not tape replay
        let (grad_input, grad_weight, grad_bias) = d_linear_batched(
            input.data(), grad_output.data(),
            self.inner.weight().data(), self.inner.bias().data()
        );
        (Tensor::from(grad_input), vec![Tensor::from(grad_weight), Tensor::from(grad_bias)])
    }
}
```

### Layer 4: Whole-Model Compilation

The ultimate goal — differentiate an entire model as a single compilation unit:

```rust
#[autodiff_reverse(d_mlp, Duplicated, Duplicated, Duplicated, Duplicated, Active)]
fn mlp_forward(
    input: &[f64],
    w1: &[f64], b1: &[f64],
    w2: &[f64], b2: &[f64],
) -> f64 {
    // Layer 1: linear + relu
    let hidden: Vec<f64> = (0..HIDDEN_SIZE).map(|i| {
        let sum: f64 = (0..INPUT_SIZE).map(|j| input[j] * w1[i * INPUT_SIZE + j]).sum();
        (sum + b1[i]).max(0.0) // relu
    }).collect();

    // Layer 2: linear
    let out: f64 = (0..OUTPUT_SIZE).map(|i| {
        let sum: f64 = (0..HIDDEN_SIZE).map(|j| hidden[j] * w2[i * HIDDEN_SIZE + j]).sum();
        sum + b2[i]
    }).sum();

    out
}
```

Enzyme generates a **single backward function** for the entire MLP. LLVM then:
- Fuses all operations (matmul + relu + matmul + loss)
- Decides which intermediates to cache vs. recompute (automatic checkpointing)
- Vectorizes and optimizes the backward pass as a whole
- Can target GPU via offload

## Performance Expectations

| Metric | Dynamic Tape | Compiler AD (Enzyme) | Speedup |
|---|---|---|---|
| Kernel launches per forward+backward | O(n_ops) | O(1) | 100-1000x fewer |
| CPU-GPU synchronization points | O(n_ops) | O(1) | Eliminated |
| Memory for tape | O(n_ops) | 0 (compiler decides) | Eliminated |
| Cross-op fusion | Manual only | Automatic via LLVM | ~2-3x memory BW |
| Gradient correctness | Manual backward impl | Proven correct by construction | N/A |

Expected wall-clock speedup for typical models:
- **Small models** (< 1M params): **5-20x** — dispatch overhead dominates at this scale
- **Large models** (> 100M params): **1.5-3x** — compute dominates, but fusion still helps
- **Custom ops / fine-grained ops**: **10-50x** — these suffer most from dispatch overhead

## Limitations and When NOT to Use Compiler AD

1. **Dynamic architectures** — if the computation graph changes per input (tree-RNNs, dynamic routing), use tape-based AD
2. **Very large models** — Enzyme compilation time scales with model size; for billion-parameter models, compile times may be prohibitive
3. **Rapid prototyping** — changing a layer requires recompilation, not just re-running
4. **Black-box operations** — FFI calls to libraries not compiled with Enzyme need custom derivative rules
5. **Unsafe code with raw pointers** — Enzyme needs alias information; prefer slices and references

## Implementation Phases

### Phase A: Foundation (This PR)
- Research documentation and design
- Prototype crate demonstrating `#[autodiff]` on numerical functions
- Prototype crate demonstrating `core::intrinsics::offload`
- Benchmark harness for comparing tape vs. compiled AD

### Phase B: Core Integration
- `CompiledModule` trait in `theano-nn`
- Enzyme-backed `CompiledLinear`, `CompiledConv2d`
- Integration with existing `Optimizer` (compiled modules produce gradients differently)
- Feature flag: `theano = { features = ["enzyme"] }`

### Phase C: GPU Offload Pipeline
- `core::intrinsics::offload` integration for compiled forward+backward
- Memory management: automatic host↔device transfer via offload infrastructure
- Kernel caching: compile once, execute many times

### Phase D: Whole-Model Compilation
- Proc-macro `#[compile_model]` that generates a single `#[autodiff]` function from a model definition
- Automatic checkpointing strategy selection
- Mixed-precision support

## Nightly Rust Requirements

Both features require nightly Rust with specific flags:

```toml
# rust-toolchain.toml
[toolchain]
channel = "nightly"
components = ["rust-src", "llvm-tools"]
```

```bash
# Build flags
RUSTFLAGS="-C lto=fat -Z unstable-options"
```

For GPU offload, additionally:
```bash
RUSTFLAGS="-C lto=fat -Z unstable-options -Z offload=Enable"
```

## References

- [Enzyme: High-Performance Automatic Differentiation of LLVM](https://enzyme.mit.edu/)
- [Rust autodiff tracking issue #124509](https://github.com/rust-lang/rust/issues/124509)
- [Rust GPU offload tracking issue #131513](https://github.com/rust-lang/rust/issues/131513)
- [Rust GPU kernel ABI tracking issue #135467](https://github.com/rust-lang/rust/issues/135467)
- [Manuel Drehwald's PhD thesis on Enzyme+Rust](https://github.com/ZuseZ4)
- [JAX: Composable Transformations](https://github.com/google/jax) — prior art for compiled AD
