# Changelog

All notable changes to Neo Theano will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Research: Compiler-level autodiff** (`research/`) — Enzyme (`#[autodiff]`) and GPU offload (`core::intrinsics::offload`) exploration
  - `enzyme-prototype/` — Module trait with `flatten_params`/`load_params` bridge for Enzyme, PyTorch-style API with transparent `CompiledModel` wrapper, numerical gradient verification (15 tests)
  - `offload-prototype/` — GPU offload integration patterns, memory transfer analysis (3 tests)
  - `benchmarks/` — Criterion benchmarks across 6 module types: MLP (4.5–9.3x), Conv2d (15–26x), LSTM (17–31x), BatchNorm (15–27x), Attention (42–102x), plus dynamic-vs-static graph comparison (7.6–11.2x); larger gains expected on GPU from kernel fusion
  - `DESIGN.md` — Full architecture document: dual-mode AD (static graphs → Enzyme, dynamic graphs → existing tape)
  - Key design principle: **zero change to PyTorch API** — Enzyme is a transparent optimization layer underneath existing `Module` trait

## [0.1.0] - 2026-03-14

### Added
- **Phase 1: Core tensor library** (`theano-types`, `theano-core`, `theano-backend`, `theano-cpu`)
  - N-dimensional tensor with Arc-based shared ownership
  - 100+ tensor operations (unary, binary, reductions, matmul, view ops)
  - CPU backend with BLAS integration
  - Type system: DType, Device, Layout abstractions

- **Phase 2: Autograd engine** (`theano-autograd`)
  - Reverse-mode automatic differentiation with topological sort
  - Variable wrapper for computational graph construction
  - 30+ GradFn implementations with broadcasting support
  - Thread-local no-grad context

- **Phase 3: CUDA backend** (`theano-cuda`, `theano-cuda-kernels`)
  - CudaDevice with cudarc wrapper, stream management, cuBLAS handle
  - CachingAllocator (PyTorch-style memory pooling)
  - Dynamic library loading (no build-time CUDA dependency)
  - CUDA kernels: elementwise, reduction, softmax

- **Phase 4: Training infrastructure** (`theano-nn`, `theano-optim`, `theano-data`)
  - Module trait with forward/parameters/num_parameters
  - 18 layer types: Linear, Conv2d, Conv1d, RNN, LSTM, BatchNorm, LayerNorm, MultiheadAttention, etc.
  - 5 optimizers: SGD (momentum/Nesterov), Adam, AdamW, RMSprop, Adagrad
  - 6 LR schedulers: StepLR, CosineAnnealing, ReduceLROnPlateau, etc.
  - DataLoader with batching, shuffling, parallel workers

- **Phase 5: Extended nn/optim** — Full layer coverage, loss functions (MSE, CrossEntropy, L1, SmoothL1, BCE, KLDiv)

- **Phase 6: Multi-GPU** (`theano-rocm`, `theano-barracuda`, `theano-distributed`)
  - ROCm/HIP backend for AMD GPUs
  - BarraCUDA compiler infrastructure
  - Distributed training: DDP, FSDP, all-reduce collectives

- **Phase 7: Cross-platform backends** (`theano-metal`, `theano-wgpu`, `theano-oneapi`)
  - Apple Metal GPU backend
  - WebGPU backend (cross-platform)
  - Intel oneAPI backend

- **Phase 8: Ecosystem** (`theano-serialize`, `theano-jit`, `theano-quantize`, `theano-python`)
  - Serialization: SafeTensors, StateDict, ONNX export
  - JIT: SSA-based IR, graph optimization passes (DCE, constant folding)
  - Quantization: INT8/INT4 with calibration
  - Python bindings via PyO3

- **Phase 9: Documentation & CI**
  - README with full API documentation
  - Example programs (basic tensor, autograd, training)
  - CI/CD workflows (GitHub Actions, Jenkinsfile)
  - 22 example programs in separate repository (Neo-Theano/examples)
