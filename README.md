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
| NVIDIA CUDA | `cuda` | In progress | CUDA 12+ |
| AMD ROCm | `rocm` | In progress | ROCm 6+ |
| BarraCUDA | -- | Planned | High-performance CUDA alternative |
| Apple Metal | `metal` | Planned | macOS/iOS GPU acceleration |
| WebGPU | `wgpu-backend` | Planned | Cross-platform via `wgpu` |
| Intel oneAPI | `oneapi` | Planned | Intel GPU/accelerator support |

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

### Phase 4 -- GPU Backends (in progress)
- [ ] NVIDIA CUDA backend
- [ ] AMD ROCm backend
- [ ] Apple Metal backend
- [ ] WebGPU backend
- [ ] Intel oneAPI backend

### Phase 5 -- Advanced Features (planned)
- [ ] Distributed training (data parallel, model parallel)
- [ ] JIT compilation and graph optimization
- [ ] INT8/INT4 quantization
- [ ] Model serialization (save/load checkpoints)
- [ ] Python bindings via PyO3
- [ ] ONNX import/export

### Phase 6 -- Ecosystem (planned)
- [ ] Pre-trained model zoo (ResNet, GPT-2, ViT, etc.)
- [ ] torchvision-equivalent image transforms
- [ ] torchaudio-equivalent audio processing
- [ ] Hugging Face Hub integration

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
