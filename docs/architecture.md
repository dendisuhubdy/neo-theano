# Architecture Overview

This document describes the internal architecture of the Neo Theano deep learning framework.

## Workspace Structure

Neo Theano is organized as a Cargo workspace with 23 crates under `crates/`. Each crate has a focused responsibility, keeping compilation units small and enabling feature-gated GPU backends.

### Crate Dependency DAG

```
theano  (facade crate — re-exports everything)
├── theano-types         (DType, Device, Shape, Layout, Error — zero dependencies)
├── theano-core          (Tensor, Storage, creation, ops, views, formatting)
│   ├── theano-types
│   └── theano-backend
├── theano-backend       (Backend, BackendStorage, UnaryOp/BinaryOp/ReduceOp traits)
│   └── theano-types
├── theano-cpu           (CPU backend: BLAS matmul via `gemm`, rayon parallelism)
│   ├── theano-types
│   └── theano-backend
├── theano-cuda          (NVIDIA GPU backend via cudarc, caching allocator)
│   ├── theano-types
│   ├── theano-backend
│   └── theano-cuda-kernels
├── theano-cuda-kernels  (Compiled CUDA .cu kernels)
├── theano-rocm          (AMD GPU backend via HIP runtime)
│   ├── theano-types
│   ├── theano-backend
│   └── theano-rocm-kernels
├── theano-rocm-kernels  (Compiled ROCm/HIP kernels)
├── theano-barracuda     (Experimental: compile .cu directly for AMD via source translation)
│   ├── theano-types
│   └── theano-backend
├── theano-metal         (Apple Silicon backend via Metal Shading Language)
│   ├── theano-types
│   └── theano-backend
├── theano-wgpu          (Cross-platform GPU via WebGPU/WGSL)
│   ├── theano-types
│   └── theano-backend
├── theano-oneapi        (Intel GPU/accelerator backend via oneAPI)
│   ├── theano-types
│   └── theano-backend
├── theano-autograd      (Variable, GradFn, backward engine, no_grad)
│   ├── theano-types
│   └── theano-core
├── theano-nn            (Module trait, Linear, Conv2d, BatchNorm, Transformer, losses)
│   ├── theano-types
│   ├── theano-core
│   └── theano-autograd
├── theano-optim         (Optimizer trait, SGD, Adam, AdamW, RMSprop, schedulers)
│   ├── theano-types
│   ├── theano-core
│   └── theano-autograd
├── theano-data          (Dataset trait, TensorDataset, DataLoader, samplers)
│   └── theano-core
├── theano-serialize     (SafeTensors, state_dict, ONNX import/export)
│   └── theano-core
├── theano-distributed   (DistributedDataParallel, ProcessGroup, collectives)
│   ├── theano-core
│   └── theano-autograd
├── theano-jit           (JIT compilation / graph optimization — planned)
│   └── theano-core
├── theano-quantize      (INT8/INT4 quantization — planned)
│   └── theano-core
└── theano-python        (PyO3 Python bindings)
    └── theano
```

## Core Design Decisions

### Arc-based Tensors

`Tensor` is defined as a thin wrapper around `Arc<TensorInner>`:

```rust
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
}
```

This design mirrors PyTorch's reference-counted tensor model:

- **Cheap cloning**: Cloning a `Tensor` increments an atomic reference count. No data is copied.
- **Shared storage**: Multiple tensors (e.g., after `view()`, `transpose()`, `slice()`) can share the same underlying data buffer while having different shapes, strides, and offsets.
- **Thread safety**: `Arc` provides `Send + Sync`, enabling safe sharing across threads for data loading and distributed training.
- **Autograd compatibility**: The autograd engine stores saved tensors for backward without deep copies.

### Dynamic Rank

Tensor shape and strides are stored as `Vec<usize>` rather than const generics:

```rust
pub(crate) struct TensorInner {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    // ...
}
```

This matches PyTorch's runtime-determined rank model. Advantages:

- Operations like `reshape`, `squeeze`, `unsqueeze` can change rank without type-level gymnastics.
- Neural network layers can handle batched and unbatched inputs uniformly.
- Compatibility with dynamically shaped models (variable-length sequences, etc.).

### The Backend Trait

All compute is abstracted behind two traits in `theano-backend`:

```rust
pub trait BackendStorage: Sized + Send + Sync {
    fn unary_op(&self, op: UnaryOp, shape: &[usize], strides: &[usize]) -> Result<Self>;
    fn binary_op(&self, rhs: &Self, op: BinaryOp, ...) -> Result<Self>;
    fn reduce_op(&self, op: ReduceOp, ...) -> Result<Self>;
    fn matmul(&self, rhs: &Self, ...) -> Result<Self>;
    fn fill(value: f64, len: usize, dtype: DType, device: &Device) -> Result<Self>;
    // ...
}

pub trait Backend: Sized + Clone + Send + Sync + 'static {
    type Storage: BackendStorage;
    fn name() -> &'static str;
    fn device_type() -> DeviceType;
}
```

Each backend (CPU, CUDA, ROCm, Metal, WebGPU, oneAPI) provides its own `Storage` type that implements `BackendStorage`. The `Storage` enum in `theano-core` provides runtime dispatch:

```rust
pub enum Storage {
    Cpu(Box<dyn BackendStorageBoxed>),
    // Future: Cuda(...), Rocm(...), Metal(...), etc.
}
```

## Data Flow

### Forward Pass

```
User code
    │
    ▼
theano::Tensor API  (tensor.add(&other), tensor.matmul(&w), etc.)
    │
    ▼
theano-core tensor_ops.rs  (shape validation, broadcast computation, stride logic)
    │
    ▼
Storage dispatch  (match on Storage enum variant)
    │
    ▼
BackendStorage::binary_op / matmul / ...  (backend-specific kernel)
    │
    ▼
New Tensor returned with result storage
```

### Autograd (Backward Pass)

```
User code
    │
    ▼
Variable wraps Tensor, records grad_fn for each operation
    │
    ▼
variable.backward()
    │
    ▼
theano-autograd engine.rs:
    1. Topological sort of the computation graph (DFS from loss)
    2. Iterate in reverse order
    3. For each node, call grad_fn.backward(grad_output)
    4. Accumulate gradients for inputs
    5. Store final gradients on leaf tensors via tensor.set_grad()
    │
    ▼
Optimizer reads tensor.grad(), updates parameters
```

Key details:

- `Variable` stores both the output `Tensor` and a `Vec<Variable>` of inputs, forming an explicit DAG.
- `GradFn` implementations (AddBackward, MulBackward, MatmulBackward, etc.) compute local Jacobian-vector products.
- Gradients are accumulated additively when a variable is used multiple times (fan-out).
- `no_grad()` / `NoGradGuard` disables graph construction for inference.

## Memory Management

### CPU

The CPU backend uses standard Rust `Vec<T>` allocations. No custom allocator is needed since the system allocator handles this well.

### CUDA Caching Allocator

`theano-cuda` includes a `CachingAllocator` modeled after PyTorch's `CUDACachingAllocator`:

- **Two pools**: Small (blocks <= 1 MB) and Large (blocks > 1 MB).
- **Best-fit search**: `BTreeMap<usize, Vec<Block>>` keyed by block size for O(log n) lookup of the smallest sufficient cached block.
- **Block splitting**: If a cached block is significantly larger than requested, it is split and the remainder is returned to the pool.
- **OOM recovery**: On allocation failure, the allocator frees all cached blocks and retries.
- **Statistics tracking**: Tracks allocated bytes, cached bytes, peak usage, cache hit/miss rates.

Size rounding:
- Small allocations: rounded up to 512-byte granularity.
- Large allocations: rounded up to 2 MB granularity.

### Other GPU Backends

ROCm, Metal, WebGPU, and oneAPI backends each manage their own device memory through their respective APIs, following similar caching patterns where applicable.

## How Backends Are Plugged In

### Compile-Time Feature Gates

Each GPU backend is an optional dependency of the `theano` facade crate, gated behind a Cargo feature:

```toml
[features]
default = ["cpu"]
cpu = ["theano-cpu"]
cuda = ["theano-cuda"]
rocm = ["theano-rocm"]
metal = ["theano-metal"]
wgpu-backend = ["theano-wgpu"]
oneapi = ["theano-oneapi"]
full = ["cpu", "cuda", "rocm", "metal", "wgpu-backend", "oneapi"]
```

Users enable the backend they need:

```bash
cargo add theano --features cuda
```

### Adding a New Backend

1. Create a new crate `crates/theano-<name>/`.
2. Depend on `theano-types` and `theano-backend`.
3. Implement `BackendStorage` for your storage type.
4. Implement `Backend` for your backend marker type.
5. Add the crate as a workspace member and optional dependency.
6. Add a `Storage` enum variant in `theano-core` for runtime dispatch.
7. Wire up feature flags in the `theano` facade crate.

See [backends.md](backends.md) for detailed per-backend documentation and [contributing.md](contributing.md) for the full process.
