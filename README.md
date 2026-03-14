<p align="center">
  <h1 align="center">Neo Theano</h1>
  <p align="center">A PyTorch-compatible deep learning framework in Rust</p>
</p>

<p align="center">
  <a href="https://github.com/dendisuhubdy/theano/actions"><img src="https://img.shields.io/github/actions/workflow/status/dendisuhubdy/theano/ci.yml?branch=main&style=flat-square&logo=github" alt="Build Status"></a>
  <a href="https://crates.io/crates/theano"><img src="https://img.shields.io/crates/v/theano.svg?style=flat-square&logo=rust" alt="crates.io"></a>
  <a href="https://docs.rs/theano"><img src="https://img.shields.io/docsrs/theano?style=flat-square&logo=docs.rs" alt="docs.rs"></a>
  <a href="https://github.com/dendisuhubdy/theano/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="License"></a>
  <a href="https://github.com/dendisuhubdy/theano"><img src="https://img.shields.io/github/stars/dendisuhubdy/theano?style=flat-square&logo=github" alt="GitHub Stars"></a>
</p>

---

Neo Theano brings the full PyTorch deep learning experience to Rust. It provides a familiar, ergonomic API for tensors, automatic differentiation, neural network layers, and optimizers -- all with zero-cost abstractions and multi-backend GPU acceleration.

Born from the legacy of the original [Theano](https://github.com/Theano/Theano) project at [MILA](https://mila.quebec/en), Neo Theano aims for 100% PyTorch API parity while leveraging Rust's type safety, fearless concurrency, and performance.

## Features

- **PyTorch-compatible API** -- `Tensor`, `Variable`, `Module`, `Optimizer` all mirror their PyTorch counterparts
- **Reverse-mode autograd** -- dynamic computational graph rebuilt every forward pass, just like PyTorch
- **Comprehensive nn layers** -- Linear, Conv2d, Conv1d, BatchNorm, LayerNorm, RNN, LSTM, Transformer attention, and more
- **All major optimizers** -- SGD (with momentum/Nesterov), Adam, AdamW, RMSprop, Adagrad
- **Learning rate schedulers** -- StepLR, CosineAnnealing, ExponentialLR, ReduceLROnPlateau, and more
- **Loss functions** -- MSE, CrossEntropy, BCE, NLL, L1, SmoothL1, KLDiv
- **Multi-backend GPU support** -- CPU, CUDA, ROCm, Metal, WebGPU, oneAPI
- **Weight initialization** -- Kaiming, Xavier/Glorot (uniform and normal variants)
- **Serialization** -- save and load model weights
- **Data loading** -- dataset and dataloader abstractions
- **JIT compilation** -- graph-level optimizations (experimental)
- **Quantization** -- INT8/INT4 model quantization (experimental)
- **Python bindings** -- PyO3-based Python interface (coming soon)

## Quick Start

Add Neo Theano to your project:

```toml
[dependencies]
theano = "0.1"
```

### Tensor Operations

```rust
use theano::prelude::*;

// Create tensors
let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
let b = Tensor::ones(&[2, 3]);

// Arithmetic (operator overloading)
let c = &a + &b;
let d = &a * &b;

// Reductions
let total = a.sum().unwrap();
let avg = a.mean().unwrap();

// Matrix multiplication
let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
let y = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
let z = x.matmul(&y).unwrap();

// Reshaping and transposing
let reshaped = a.reshape(&[3, 2]).unwrap();
let transposed = a.transpose(0, 1).unwrap();
```

### Automatic Differentiation

```rust
use theano::prelude::*;

// f(x) = x^2, df/dx = 2x
let x = Variable::requires_grad(Tensor::scalar(3.0));
let y = x.mul(&x).unwrap();
y.backward();

let grad = x.grad().unwrap(); // grad = 6.0
```

### Training a Neural Network

```rust
use theano::prelude::*;
use theano::nn::{Linear, ReLU, Sequential, MSELoss, Module};
use theano::optim::{SGD, Optimizer};

// Define model
let model = Sequential::new(vec![])
    .add(Linear::new(2, 16))
    .add(ReLU)
    .add(Linear::new(16, 1));

// Set up optimizer
let params = model.parameters();
let mut optimizer = SGD::new(params, 0.01).momentum(0.9);
let loss_fn = MSELoss::new();

// Training step
optimizer.zero_grad();
let input = Variable::new(Tensor::from_slice(&[1.0, 0.0], &[1, 2]));
let target = Variable::new(Tensor::from_slice(&[1.0], &[1, 1]));
let output = model.forward(&input);
let loss = loss_fn.forward(&output, &target);
loss.backward();
optimizer.step();
```

### CNN for Image Classification

```rust
use theano::prelude::*;
use theano::nn::*;
use theano::optim::{Adam, Optimizer};

// Define a CNN
let conv1 = Conv2d::with_options(1, 32, (3, 3), (1, 1), (1, 1), true);
let conv2 = Conv2d::with_options(32, 64, (3, 3), (1, 1), (1, 1), true);
let pool = MaxPool2d::new(2);
let fc1 = Linear::new(64 * 7 * 7, 128);
let fc2 = Linear::new(128, 10);

// Forward pass
let x = conv1.forward(&input);
let x = ReLU.forward(&x);
let x = pool.forward(&x);
// ... continue through the network
```

## Compiler-Level Autodiff (Research)

Neo Theano is exploring **compiler-level automatic differentiation** via Rust's experimental [`#[autodiff]`](https://github.com/rust-lang/rust/issues/124509) attribute (powered by [Enzyme](https://enzyme.mit.edu/)). Instead of building a tape at runtime, Enzyme generates backward functions at compile time from LLVM IR — eliminating tape allocation, indirect calls, and enabling whole-model fusion into a single GPU kernel.

The framework maintains a **dual-mode architecture**: static graphs use Enzyme (compiled AD), dynamic graphs keep using the existing tape. Users opt in explicitly — zero change to the PyTorch API.

### Benchmark Results (tape overhead, CPU)

Measured via [Criterion.rs](https://bheisler.github.io/criterion.rs/) in `research/benchmarks/`. Compares tape-based AD (dynamic graph with node allocation, topological sort, indirect calls) vs compiled AD simulation (direct fused computation, no tape).

#### Tape-based vs Compiled AD by module type

| Module | Tape-based | Compiled (no tape) | Speedup |
|---|---|---|---|
| **MLP** 4x8x2 | 5.85μs | 1.31μs | **4.5x** |
| **MLP** 16x32x4 | 280μs | 30.1μs | **9.3x** |
| **MLP** 64x128x8 | 7.28ms | 1.26ms | **5.8x** |
| **MLP** 128x256x16 | 64.8ms | 11.4ms | **5.7x** |
| **Conv2d** 1c->4c 8x8 k3 | 103μs | 6.82μs | **15.1x** |
| **Conv2d** 3c->16c 16x16 k3 | 6.70ms | 291μs | **23.0x** |
| **Conv2d** 3c->32c 16x16 k5 | 29.4ms | 1.13ms | **26.0x** |
| **LSTM** in=4 h=8 seq=4 | 135μs | 7.69μs | **17.5x** |
| **LSTM** in=8 h=16 seq=8 | 1.08ms | 36.9μs | **29.3x** |
| **LSTM** in=16 h=32 seq=8 | 4.55ms | 145μs | **31.5x** |
| **BatchNorm** b=16 f=32 | 157μs | 5.89μs | **26.7x** |
| **BatchNorm** b=64 f=128 | 2.82ms | 188μs | **15.0x** |
| **Attention** seq=4 d=8 h=2 | 118μs | 2.82μs | **41.7x** |
| **Attention** seq=8 d=16 h=4 | 923μs | 10.7μs | **86.0x** |
| **Attention** seq=8 d=32 h=4 | 3.63ms | 35.7μs | **101.7x** |

#### Dynamic graph (rebuild each iteration) vs Static graph (fixed topology)

| Model | Dynamic (tape) | Static graph | Speedup |
|---|---|---|---|
| MLP 4x8x2 | 5.79μs | 761ns | **7.6x** |
| MLP 16x32x4 | 280μs | 26.2μs | **10.7x** |
| MLP 64x128x8 | 7.44ms | 742μs | **10.0x** |
| MLP 128x256x16 | 69.9ms | 6.23ms | **11.2x** |

Attention sees the largest speedups (42-102x) due to the highest ratio of small operations per logical step. Even without compiled AD, avoiding dynamic graph rebuild alone yields ~10x on medium models. Full analysis in [`research/DESIGN.md`](research/DESIGN.md).

### Rust Autodiff vs PyTorch: Where Each Wins

The benchmark numbers above measure one dimension — tape overhead — but the real comparison between compiled AD in Rust and PyTorch's interpreter-based approach spans several axes:

#### Summary

| Dimension | Rust Autodiff (Enzyme) | PyTorch (Eager + torch.compile) | Winner |
|---|---|---|---|
| Kernel launches per training step | 1-2 (fused) | 10-100+ (per-op dispatch) | Rust |
| Memory predictability | Deterministic at compile time | Dynamic allocation, GC pressure | Rust |
| GPU utilization | Near-peak (no bubbles between ops) | Bubble-filled (launch gaps, sync points) | Rust |
| Gradient correctness | Impossible to write a wrong backward (you never write one) | Manual `backward()` impls, silent shape bugs | Rust |
| Ecosystem breadth | Small (new framework) | Every paper has a PyTorch implementation | PyTorch |
| Iteration speed | Recompile on every change | Change a line, re-run instantly | PyTorch |
| Dynamic architectures | Requires explicit opt-out to tape mode | Graphs that change shape per-input work naturally | PyTorch |
| Debugging | Compile errors, no runtime graph inspection | `pdb`, hooks, `register_hook`, tensor printing | PyTorch |
| Multi-GPU / distributed | Manual (early stage) | DDP/FSDP/DeepSpeed, battle-tested at scale | PyTorch |
| Pretrained model zoo | Minimal | Hugging Face, timm, torchvision — thousands of models | PyTorch |

#### Where Rust autodiff wins decisively

**Kernel launch overhead.** PyTorch dispatches each operation as a separate CUDA kernel — a `Linear` layer alone is a matmul kernel + bias add kernel + activation kernel, each with ~5-10μs launch overhead. Enzyme fuses the entire forward+backward pass into 1-2 kernels. For models with hundreds of small operations (attention, LSTMs), this is the difference between the GPU sitting idle 40% of the time and running at near-peak throughput.

**Memory predictability.** PyTorch's tape grows dynamically during the forward pass, allocating `GradFn` nodes on the heap. Peak memory depends on input shape, batch size, and which operations trigger saves — it's hard to predict and hard to bound. Enzyme decides at compile time what to cache vs. recompute (automatic checkpointing). You know your memory footprint before you run.

**GPU utilization.** Every CPU-GPU synchronization point (tape management, Python GIL, allocator) creates a bubble where the GPU has no work. PyTorch mitigates this with CUDA graphs and operator fusion, but the fundamental architecture has the CPU in the critical path. Compiled AD removes the CPU from the loop entirely for the forward+backward computation.

**Correctness guarantees.** In PyTorch, every new operation needs a hand-written `backward()` function. Get the Jacobian wrong, and gradients silently diverge — the model trains but converges to garbage. Enzyme derives the backward pass from the forward pass by differentiating LLVM IR. If the forward pass is correct, the backward pass is correct by construction. This eliminates an entire class of bugs that are notoriously hard to catch (gradient errors often look like "the model just doesn't train well").

**Inference latency.** For deployment, compiled AD produces a single optimized binary with no runtime framework overhead. No Python interpreter, no dynamic dispatch, no GC pauses. Latency is deterministic and minimal — critical for real-time applications (autonomous driving, trading, robotics).

**Energy efficiency.** Fewer kernel launches, no interpreter overhead, and tighter GPU utilization translate directly to lower energy per training step. At scale (thousands of GPUs for weeks), this is a meaningful cost and carbon difference.

#### Where PyTorch still wins

**Ecosystem is the moat.** Every ML paper published in the last 5 years ships with PyTorch code. Hugging Face has 200k+ pretrained models. torchvision, torchaudio, torchtext provide ready-made data pipelines. This ecosystem took a decade to build and cannot be replicated by any new framework on technical merit alone.

**Iteration speed is the lifeblood of research.** Change a layer, re-run in seconds. Add a `print(tensor.shape)` anywhere. Set a breakpoint in the backward pass. PyTorch's eager execution makes the gap between "idea" and "experiment" nearly zero. Compiled AD requires recompilation — fast in Rust, but fundamentally slower than "just re-run the script."

**Dynamic architectures work naturally.** Tree-structured RNNs, pointer networks, models with input-dependent control flow — these rebuild a different computation graph every forward pass. PyTorch's tape handles this without any special annotation. Enzyme requires the computation to be statically known at compile time, so dynamic architectures must fall back to tape mode.

**Debugging is interactive.** PyTorch lets you `print()` any tensor, set breakpoints in the backward pass via `register_hook`, inspect the computation graph at runtime, and use standard Python debugging tools. Compiled AD is a black box — you get the correct answer, but you can't step through the backward pass to understand why a particular gradient has a particular value.

**Distributed training is battle-tested.** PyTorch's DDP, FSDP, and integrations with DeepSpeed/Megatron have been proven at billion-parameter scale across thousands of GPUs. This infrastructure took years of engineering to stabilize. New frameworks must rebuild this from scratch.

**Community and hiring.** Practically every ML engineer knows PyTorch. Practically none know Rust autodiff. For organizations, this means PyTorch projects can hire from a deep talent pool, find answers on StackOverflow, and get support from a massive community.

#### The convergence point

PyTorch is moving toward what Rust autodiff does natively. `torch.compile` traces dynamic Python code into static graphs. Triton fuses operations into custom GPU kernels. CUDA Graphs eliminate kernel launch overhead. FlexAttention compiles attention variants at runtime. Each of these is an engineering effort to recover the performance that a compiled language gets for free.

But PyTorch is doing this by **layering compilation on top of an interpreter**, which is inherently more fragile than starting from a compiled language:
- `torch.compile` has graph breaks when it hits unsupported Python constructs
- Triton kernels require manual tuning and a separate language
- CUDA Graphs require careful management of memory addresses and stream ordering
- Dynamic shapes force recompilation or fallback to eager mode

Rust autodiff starts where PyTorch is trying to end up: a compiled language where the compiler sees the entire computation, can differentiate it, fuse it, and target any hardware backend. The tradeoff is that you lose the interpreter's flexibility — but if your model architecture is fixed (which it is for most production deployments), you never needed that flexibility in the first place.

#### Pitfalls and honest limitations

**Compile times scale with model size.** Enzyme differentiates the entire forward pass as a single LLVM function. For large models (billions of parameters, deeply nested layers), this can push compile times from seconds to minutes. PyTorch has no compile step for eager mode. `torch.compile` has similar scaling issues, but at least offers a fallback.

**Nightly Rust dependency is a hard requirement.** `#[autodiff]` and `gpu_offload` are unstable features behind nightly-only flags. Nightly Rust can break between releases. For production systems that need stability guarantees, this is a real risk — you may need to pin a specific nightly version and maintain that pin.

**Enzyme's Rust support is early-stage.** While Enzyme itself is mature (battle-tested in C/C++ for years at MIT), the Rust frontend is new. Edge cases in Rust's type system (traits, enums, complex generics) may not be fully supported. You may hit ICEs (internal compiler errors) on valid code.

**No automatic mixed precision.** PyTorch's `autocast` transparently converts operations to FP16/BF16 where safe. Enzyme operates on whatever types the code uses. Mixed precision requires manual annotation — cast your tensors to `f16` explicitly, know which operations need `f32` for numerical stability.

**Limited hardware backend support.** PyTorch runs on CUDA, ROCm, XLA (TPUs), MPS (Apple Silicon), and more — with vendor-optimized kernels for each. Enzyme targets LLVM backends, which covers NVPTX (CUDA) and AMDGPU but lacks the hand-tuned kernel libraries (cuDNN, cuBLAS) that PyTorch calls for critical operations like convolution and attention.

**Debugging compiled gradients is hard.** When PyTorch's autograd gives a wrong gradient, you can set hooks, print intermediates, and trace the tape. When Enzyme gives a wrong gradient (rare, but possible due to aliasing issues or unsupported constructs), you're debugging LLVM IR. This requires a fundamentally different skillset.

**The "two-language problem" reappears.** Research happens in Python/PyTorch. If you develop a new architecture in Rust autodiff, you can't share it with the broader ML community via a Jupyter notebook. You can't use it in Hugging Face. You can't compare against baselines that only exist in PyTorch. The friction of translating between ecosystems is real.

**Graph-changing models require fallback.** Any model where the computation graph depends on the input (adaptive computation, early exit networks, mixture-of-experts routing) cannot use compiled AD for the dynamic portion. You need a fallback to tape-based AD, which means maintaining two code paths and understanding when each applies.

## Architecture

Neo Theano is organized as a Rust workspace with focused crates:

```
theano/
├── crates/
│   ├── theano/              # Facade crate -- re-exports everything
│   ├── theano-types/        # Core types: DType, Device, Shape, Error
│   ├── theano-core/         # Tensor implementation and operations
│   ├── theano-backend/      # Backend trait abstraction
│   ├── theano-cpu/          # CPU backend (BLAS-accelerated)
│   ├── theano-cuda/         # NVIDIA CUDA backend
│   ├── theano-cuda-kernels/ # CUDA kernel implementations
│   ├── theano-rocm/         # AMD ROCm backend
│   ├── theano-rocm-kernels/ # ROCm kernel implementations
│   ├── theano-barracuda/    # BarraCUDA backend
│   ├── theano-metal/        # Apple Metal backend
│   ├── theano-wgpu/         # WebGPU backend (cross-platform)
│   ├── theano-oneapi/       # Intel oneAPI backend
│   ├── theano-autograd/     # Reverse-mode automatic differentiation
│   ├── theano-nn/           # Neural network layers and loss functions
│   ├── theano-optim/        # Optimizers and LR schedulers
│   ├── theano-data/         # Dataset and DataLoader
│   ├── theano-serialize/    # Model serialization (save/load)
│   ├── theano-distributed/  # Distributed training
│   ├── theano-jit/          # JIT compilation and graph optimization
│   ├── theano-quantize/     # Model quantization (INT8/INT4)
│   └── theano-python/       # Python bindings via PyO3
└── examples/
```

## Supported Backends

| Backend | Feature Flag | Status | Notes |
|---------|-------------|--------|-------|
| CPU | `cpu` (default) | Stable | BLAS-accelerated via `gemm` crate |
| NVIDIA CUDA | `cuda` | Implemented | cudarc FFI, caching allocator, cuBLAS, custom .cu kernels |
| AMD ROCm | `rocm` | Implemented | HIP FFI, mirrors CUDA architecture, .hip kernels |
| BarraCUDA | -- | Experimental | Direct .cu → AMD GFX11 compilation (no HIP translation) |
| Apple Metal | `metal` | Implemented | MSL shaders, MPS, Apple M1/M2/M3/M4 |
| WebGPU | `wgpu-backend` | Implemented | WGSL shaders, Vulkan/Metal/DX12/WebGPU/WASM |
| Intel oneAPI | `oneapi` | Implemented | Level Zero, Intel Arc/Gaudi/Ponte Vecchio |

Enable backends via Cargo features:

```toml
[dependencies]
theano = { version = "0.1", features = ["cpu", "cuda"] }
```

## Neural Network Layers

| Layer | Module | PyTorch Equivalent |
|-------|--------|--------------------|
| Linear | `Linear` | `nn.Linear` |
| Conv2d | `Conv2d` | `nn.Conv2d` |
| Conv1d | `Conv1d` | `nn.Conv1d` |
| BatchNorm1d | `BatchNorm1d` | `nn.BatchNorm1d` |
| LayerNorm | `LayerNorm` | `nn.LayerNorm` |
| GroupNorm | `GroupNorm` | `nn.GroupNorm` |
| Dropout | `Dropout` | `nn.Dropout` |
| Embedding | `Embedding` | `nn.Embedding` |
| RNNCell | `RNNCell` | `nn.RNNCell` |
| LSTMCell | `LSTMCell` | `nn.LSTMCell` |
| MultiheadAttention | `MultiheadAttention` | `nn.MultiheadAttention` |
| MaxPool2d | `MaxPool2d` | `nn.MaxPool2d` |
| AvgPool2d | `AvgPool2d` | `nn.AvgPool2d` |
| AdaptiveAvgPool2d | `AdaptiveAvgPool2d` | `nn.AdaptiveAvgPool2d` |
| Flatten | `Flatten` | `nn.Flatten` |
| ZeroPad2d | `ZeroPad2d` | `nn.ZeroPad2d` |
| Sequential | `Sequential` | `nn.Sequential` |

## Activation Functions

| Activation | Module | Description |
|-----------|--------|-------------|
| ReLU | `ReLU` | Rectified Linear Unit |
| Sigmoid | `Sigmoid` | Logistic sigmoid |
| Tanh | `Tanh` | Hyperbolic tangent |
| GELU | `GELU` | Gaussian Error Linear Unit |
| SiLU | `SiLU` | Sigmoid Linear Unit (Swish) |
| Softmax | `Softmax` | Normalized exponential |
| LogSoftmax | `LogSoftmax` | Log of softmax |

## Optimizers

| Optimizer | Module | Description |
|-----------|--------|-------------|
| SGD | `SGD` | Stochastic Gradient Descent (momentum, Nesterov, weight decay) |
| Adam | `Adam` | Adaptive Moment Estimation (AMSGrad variant) |
| AdamW | `AdamW` | Adam with decoupled weight decay |
| RMSprop | `RMSprop` | Root Mean Square Propagation |
| Adagrad | `Adagrad` | Adaptive Gradient |

## Loss Functions

| Loss | Module | Description |
|------|--------|-------------|
| MSELoss | `MSELoss` | Mean Squared Error |
| CrossEntropyLoss | `CrossEntropyLoss` | Cross-entropy (log-softmax + NLL) |
| BCELoss | `BCELoss` | Binary Cross-Entropy |
| BCEWithLogitsLoss | `BCEWithLogitsLoss` | BCE with built-in sigmoid |
| NLLLoss | `NLLLoss` | Negative Log-Likelihood |
| L1Loss | `L1Loss` | Mean Absolute Error |
| SmoothL1Loss | `SmoothL1Loss` | Huber Loss |
| KLDivLoss | `KLDivLoss` | Kullback-Leibler Divergence |

## Learning Rate Schedulers

| Scheduler | Module | Description |
|-----------|--------|-------------|
| StepLR | `StepLR` | Decay LR by gamma every N steps |
| ExponentialLR | `ExponentialLR` | Exponential decay |
| CosineAnnealingLR | `CosineAnnealingLR` | Cosine annealing to eta_min |
| MultiStepLR | `MultiStepLR` | Decay at specified milestones |
| LinearLR | `LinearLR` | Linear warmup/decay |
| ReduceLROnPlateau | `ReduceLROnPlateau` | Reduce when metric plateaus |

## Building from Source

```bash
git clone https://github.com/dendisuhubdy/theano.git
cd theano

# Build with default (CPU) backend
cargo build --release

# Build with CUDA support
cargo build --release --features cuda

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

### Requirements

- Rust 1.75+ (2021 edition)
- For CUDA: NVIDIA CUDA Toolkit 12+
- For ROCm: AMD ROCm 6+

## Running Examples

```bash
# Basic tensor operations
cargo run -p theano --example basic_tensor

# Automatic differentiation
cargo run -p theano --example autograd

# Neural network training (MLP on XOR)
cargo run -p theano --example neural_network

# CNN architecture demo (MNIST-style)
cargo run -p theano --example mnist_cnn
```

## Roadmap

### Phase 1 -- Core Foundation (done)
- [x] Tensor type with dtype/device/shape support
- [x] CPU backend with BLAS-accelerated matmul
- [x] Reverse-mode autograd engine
- [x] Full operator set: arithmetic, reductions, unary math, view ops
- [x] Weight initialization (Kaiming, Xavier)

### Phase 2 -- Neural Network Layers (done)
- [x] Linear, Conv2d, Conv1d
- [x] BatchNorm, LayerNorm, GroupNorm
- [x] Dropout, Embedding
- [x] RNN/LSTM cells
- [x] MultiheadAttention (Transformer building block)
- [x] Pooling (Max, Average, Adaptive)
- [x] All major activation functions
- [x] Sequential container

### Phase 3 -- Optimizers and Training (done)
- [x] SGD with momentum/Nesterov
- [x] Adam, AdamW
- [x] RMSprop, Adagrad
- [x] Learning rate schedulers
- [x] Loss functions (MSE, CrossEntropy, BCE, NLL, L1, KLDiv)
- [x] Data loading abstractions

### Phase 4 -- GPU Backends (done)
- [x] NVIDIA CUDA backend (cudarc, caching allocator, .cu kernels)
- [x] AMD ROCm/HIP backend (mirror of CUDA architecture)
- [x] BarraCUDA backend (direct .cu to AMD GFX11 compilation)
- [x] Apple Metal backend (MSL shaders, M1-M4 support)
- [x] WebGPU backend (wgpu, WGSL shaders, cross-platform + WASM)
- [x] Intel oneAPI backend (Level Zero, Arc/Gaudi/Ponte Vecchio)

### Phase 5 -- Advanced Features (done)
- [x] Distributed training (ProcessGroup, DDP, FSDP, NCCL/RCCL/Gloo)
- [x] JIT compilation and graph optimization (SSA IR, dead code elimination)
- [x] INT8/INT4/FP8 quantization (PTQ, QAT, observers, fake quantize)
- [x] Model serialization (SafeTensors format, state_dict save/load)
- [x] Python bindings via PyO3 (tensor, nn, dtype API surface)
- [x] ONNX export stubs

### Phase 6 -- Ecosystem (done)
- [x] 22 PyTorch-equivalent examples (MNIST, ResNet, VAE, DCGAN, ViT, GCN, GAT, RL, etc.)
- [x] Comprehensive documentation (architecture, getting-started, API reference, backends, contributing)
- [x] GitHub Actions CI/CD (test matrix, clippy, fmt, coverage, release, benchmarks)
- [x] Jenkinsfile for enterprise CI
- [x] [theano-vision](https://github.com/Neo-Theano/multimodal) — torchvision-equivalent image transforms and pretrained models
- [x] [theano-audio](https://github.com/Neo-Theano/multimodal) — torchaudio-equivalent audio processing
- [x] [theano-text](https://github.com/Neo-Theano/multimodal) — torchtext-equivalent NLP utilities
- [x] [theano-multimodal](https://github.com/Neo-Theano/multimodal) — multimodal model zoo (ResNet, ViT, GPT-2, etc.)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork the repo, then:
git checkout -b feature/my-feature
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all
# Open a pull request
```

Areas where help is especially appreciated:
- GPU backend implementations (CUDA, ROCm, Metal)
- Additional neural network layers
- Performance optimizations and benchmarks
- Documentation and examples
- Python binding improvements

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Acknowledgments

- **[PyTorch](https://pytorch.org/)** -- the API design and semantics that Neo Theano strives to match
- **[Theano](https://github.com/Theano/Theano)** -- the pioneering deep learning framework from MILA that inspired this project's name and mission
- **[MILA](https://mila.quebec/en)** -- Montreal Institute for Learning Algorithms, where it all started
- **[Yoshua Bengio](https://yoshuabengio.org/)** and the Theano team -- for laying the foundations of modern deep learning frameworks
