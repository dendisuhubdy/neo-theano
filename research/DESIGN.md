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

### Empirical Benchmark Results (CPU, tape overhead simulation)

All benchmarks run via Criterion.rs (`cargo bench` in `research/benchmarks/`).
Measures the overhead of tape construction, node allocation, indirect calls, and topological sort
vs direct fused computation (what Enzyme would generate).

#### MLP (Linear + ReLU layers)

| Model | Tape-based | Compiled (no tape) | Speedup |
|---|---|---|---|
| MLP 4x8x2 | 5.85μs | 1.31μs | **4.5x** |
| MLP 16x32x4 | 280μs | 30.1μs | **9.3x** |
| MLP 64x128x8 | 7.28ms | 1.26ms | **5.8x** |
| MLP 128x256x16 | 64.8ms | 11.4ms | **5.7x** |

#### Conv2d (Convolution + ReLU)

| Config | Tape-based | Compiled (no tape) | Speedup |
|---|---|---|---|
| 1c->4c 8x8 k3 | 103μs | 6.82μs | **15.1x** |
| 3c->16c 16x16 k3 | 6.70ms | 291μs | **23.0x** |
| 3c->32c 16x16 k5 | 29.4ms | 1.13ms | **26.0x** |
| 16c->32c 8x8 k3 | 13.2ms | 652μs | **20.2x** |

#### LSTM Cell (4 gates: input, forget, cell, output)

| Config | Tape-based | Compiled (no tape) | Speedup |
|---|---|---|---|
| in=4 h=8 seq=4 | 135μs | 7.69μs | **17.5x** |
| in=8 h=16 seq=8 | 1.08ms | 36.9μs | **29.3x** |
| in=16 h=32 seq=8 | 4.55ms | 145μs | **31.5x** |
| in=32 h=64 seq=4 | 9.41ms | 410μs | **22.9x** |

#### BatchNorm (mean, variance, normalize, scale+shift)

| Config | Tape-based | Compiled (no tape) | Speedup |
|---|---|---|---|
| batch=8 features=16 | 38.3μs | 1.56μs | **24.6x** |
| batch=16 features=32 | 157μs | 5.89μs | **26.7x** |
| batch=32 features=64 | 609μs | 32.3μs | **18.9x** |
| batch=64 features=128 | 2.82ms | 188μs | **15.0x** |

#### Multi-Head Attention (Q/K/V projection + scaled dot-product)

| Config | Tape-based | Compiled (no tape) | Speedup |
|---|---|---|---|
| seq=4 d=8 heads=2 | 118μs | 2.82μs | **41.7x** |
| seq=8 d=16 heads=4 | 923μs | 10.7μs | **86.0x** |
| seq=8 d=32 heads=4 | 3.63ms | 35.7μs | **101.7x** |

#### Dynamic Graph (tape rebuilt each iteration) vs Static Graph (fixed topology, replay only)

| Model | Dynamic (tape) | Static graph | Speedup |
|---|---|---|---|
| MLP 4x8x2 | 5.79μs | 761ns | **7.6x** |
| MLP 16x32x4 | 280μs | 26.2μs | **10.7x** |
| MLP 64x128x8 | 7.44ms | 742μs | **10.0x** |
| MLP 128x256x16 | 69.9ms | 6.23ms | **11.2x** |

**Key observations:**
- **Attention sees the largest speedups** (42-102x) because it has the highest ratio of small ops (Q/K/V projections, score computation, softmax, weighted sum) per "logical step"
- **LSTM and BatchNorm** (17-31x) also benefit heavily from many fine-grained ops per cell/feature
- **Conv2d** (15-26x) benefits from eliminating per-spatial-position tape overhead in the nested convolution loops
- **MLP** (4.5-9.3x) shows the baseline benefit from eliminating tape allocation and indirect calls
- **Static vs dynamic graph** (7.6-11.2x) isolates the cost of graph *construction* — even without switching to compiled AD, avoiding graph rebuild gives a 10x win on medium models
- Speedups generally peak at medium sizes where operation count is high enough for overhead to dominate but not so large that raw compute takes over

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
