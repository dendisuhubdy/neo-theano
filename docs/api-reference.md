# API Reference

This document provides an overview of the public API surface of Neo Theano. For full Rustdoc documentation, run `cargo doc --workspace --open`.

## theano::Tensor

The fundamental data structure. A multi-dimensional array of numbers on a compute device. Cheaply clonable (shared ownership via `Arc`).

### Creation

| Function | Description |
|---|---|
| `Tensor::zeros(shape)` | All zeros |
| `Tensor::ones(shape)` | All ones |
| `Tensor::full(shape, value)` | Filled with a constant |
| `Tensor::from_slice(data, shape)` | From an `&[f64]` with a given shape |
| `Tensor::scalar(value)` | 0-dimensional scalar tensor |
| `Tensor::randn(shape)` | Standard normal distribution |
| `Tensor::rand(shape)` | Uniform distribution [0, 1) |
| `Tensor::arange(start, end, step)` | Evenly spaced values |
| `Tensor::eye(n)` | Identity matrix |

### Metadata

| Method | Returns | Description |
|---|---|---|
| `.shape()` | `&[usize]` | Dimensions of the tensor |
| `.size()` | `Shape` | Shape as a `Shape` object |
| `.ndim()` / `.dim()` | `usize` | Number of dimensions |
| `.numel()` | `usize` | Total number of elements |
| `.strides()` | `&[usize]` | Strides in elements per dimension |
| `.dtype()` | `DType` | Element data type |
| `.device()` | `&Device` | Device the data lives on |
| `.layout()` | `Layout` | Memory layout |
| `.is_contiguous()` | `bool` | Whether data is C-contiguous |
| `.is_scalar()` | `bool` | Whether 0-dimensional |

### Arithmetic Operations

All operations support broadcasting following NumPy/PyTorch rules.

| Method | Operator | Description |
|---|---|---|
| `a.add(&b)` | `&a + &b` | Elementwise addition |
| `a.sub(&b)` | `&a - &b` | Elementwise subtraction |
| `a.mul(&b)` | `&a * &b` | Elementwise multiplication |
| `a.div(&b)` | `&a / &b` | Elementwise division |
| `a.pow(&b)` | — | Elementwise power |
| `a.neg()` | `-&a` | Elementwise negation |
| `a.add_scalar(s)` | — | Add scalar to all elements |
| `a.mul_scalar(s)` | — | Multiply all elements by scalar |

### Unary Math Operations

| Method | Description |
|---|---|
| `.exp()` | Exponential |
| `.log()` | Natural logarithm |
| `.sqrt()` | Square root |
| `.abs()` | Absolute value |
| `.sin()` / `.cos()` / `.tan()` | Trigonometric functions |
| `.tanh()` / `.sigmoid()` | Activation functions |
| `.relu()` | Rectified linear unit |
| `.clamp(min, max)` | Clamp values to range |
| `.floor()` / `.ceil()` / `.round()` | Rounding |
| `.reciprocal()` | 1/x |
| `.sign()` | Sign function |

### Reductions

| Method | Description |
|---|---|
| `.sum()` | Sum of all elements (returns scalar) |
| `.mean()` | Mean of all elements (returns scalar) |
| `.sum_dim(dim, keep_dim)` | Sum along a dimension |
| `.mean_dim(dim, keep_dim)` | Mean along a dimension |
| `.max()` / `.min()` | Global max/min |
| `.argmax(dim)` / `.argmin(dim)` | Index of max/min along dimension |

### View and Shape Operations

| Method | Description |
|---|---|
| `.reshape(shape)` | Reshape to a new shape (same number of elements) |
| `.view(shape)` | Alias for reshape |
| `.transpose(dim0, dim1)` | Swap two dimensions |
| `.t()` | Transpose last two dimensions (matrix transpose) |
| `.unsqueeze(dim)` | Add a dimension of size 1 |
| `.squeeze(dim)` | Remove a dimension of size 1 |
| `.select(dim, index)` | Select a slice along a dimension |
| `.contiguous()` | Return a contiguous copy if not already |

### Matrix Operations

| Method | Description |
|---|---|
| `.matmul(&other)` | Matrix multiplication (supports batched) |
| `.mm(&other)` | 2D matrix multiply |
| `.bmm(&other)` | Batched 3D matrix multiply |

### Data Type and Device

| Method | Description |
|---|---|
| `.to_dtype(dtype)` | Cast to a different data type |
| `.to_device(device)` | Move to a different device |
| `.to_vec_f64()` | Copy data to a `Vec<f64>` on the host |
| `.item()` | Extract scalar value from a 1-element tensor |

### Autograd

| Method | Description |
|---|---|
| `.requires_grad()` | Whether this tensor tracks gradients |
| `.requires_grad_(flag)` | Set requires_grad flag (returns new tensor) |
| `.grad()` | Get accumulated gradient |
| `.grad_fn()` | Get the autograd function that produced this tensor |
| `.is_leaf()` | Whether this is a user-created (not computed) tensor |
| `.detach()` | Detach from autograd graph |

---

## theano::Variable

Autograd wrapper around `Tensor`. When `requires_grad` is true, operations on Variables build a computational graph that enables `backward()` to compute gradients.

### Creation

```rust
// Leaf variable (no gradient tracking)
let v = Variable::new(tensor);

// Leaf variable with gradient tracking
let v = Variable::requires_grad(tensor);
```

### Core Methods

| Method | Description |
|---|---|
| `.tensor()` | Get reference to underlying `Tensor` |
| `.into_tensor()` | Consume and return the underlying `Tensor` |
| `.requires_grad_flag()` | Whether this variable requires gradient |
| `.grad()` | Get the accumulated gradient as a `Tensor` |
| `.grad_fn()` | Get the autograd function |
| `.detach()` | Detach from computation graph |
| `.backward()` | Run backward pass (must be scalar) |

### Operations

Variable supports the same operations as Tensor (add, sub, mul, div, matmul, etc.) but with automatic gradient tracking. Operator overloads (`+`, `-`, `*`, `/`) are available for `&Variable`.

### No-Grad Context

```rust
// Closure-based
let result = no_grad(|| {
    // Operations here do not build the autograd graph
    variable.mul_scalar(2.0).unwrap()
});

// RAII guard
let _guard = NoGradGuard::new();
```

---

## theano::nn

Neural network building blocks, mirroring `torch.nn`.

### Module Trait

```rust
pub trait Module: Send + Sync {
    fn forward(&self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<Variable>;
    fn named_parameters(&self) -> Vec<(String, Variable)>;
    fn num_parameters(&self) -> usize;
}
```

### Layers

| Struct | Description | Constructor |
|---|---|---|
| `Linear` | Fully connected layer | `Linear::new(in_features, out_features, bias)` |
| `Conv2d` | 2D convolution | `Conv2d::new(in_ch, out_ch, kernel, stride, padding)` |
| `Conv1d` | 1D convolution | `Conv1d::new(in_ch, out_ch, kernel, stride, padding)` |
| `BatchNorm1d` | 1D batch normalization | `BatchNorm1d::new(num_features)` |
| `LayerNorm` | Layer normalization | `LayerNorm::new(normalized_shape)` |
| `GroupNorm` | Group normalization | `GroupNorm::new(num_groups, num_channels)` |
| `Dropout` | Dropout regularization | `Dropout::new(p)` |
| `Embedding` | Embedding lookup table | `Embedding::new(num_embeddings, embedding_dim)` |
| `MaxPool2d` | 2D max pooling | `MaxPool2d::new(kernel_size, stride, padding)` |
| `AvgPool2d` | 2D average pooling | `AvgPool2d::new(kernel_size, stride, padding)` |
| `AdaptiveAvgPool2d` | Adaptive average pooling | `AdaptiveAvgPool2d::new(output_size)` |
| `ZeroPad2d` | Zero padding | `ZeroPad2d::new(padding)` |
| `Flatten` | Flatten dimensions | `Flatten::new(start_dim, end_dim)` |
| `MultiheadAttention` | Multi-head attention | `MultiheadAttention::new(embed_dim, num_heads)` |

### Activations

| Struct | Description |
|---|---|
| `ReLU` | Rectified linear unit |
| `Sigmoid` | Sigmoid activation |
| `Tanh` | Hyperbolic tangent |
| `GELU` | Gaussian error linear unit |
| `SiLU` | Sigmoid linear unit (Swish) |
| `Softmax` | Softmax (specify dim) |
| `LogSoftmax` | Log-softmax (specify dim) |

### Recurrent Layers

| Struct | Description |
|---|---|
| `RNNCell` | Vanilla RNN cell |
| `LSTMCell` | Long short-term memory cell |

### Loss Functions

| Struct | Description |
|---|---|
| `MSELoss` | Mean squared error |
| `CrossEntropyLoss` | Cross-entropy (combines log-softmax + NLL) |
| `L1Loss` | L1 / mean absolute error |
| `SmoothL1Loss` | Smooth L1 / Huber loss |
| `BCELoss` | Binary cross-entropy |
| `BCEWithLogitsLoss` | BCE with built-in sigmoid |
| `KLDivLoss` | Kullback-Leibler divergence |
| `NLLLoss` | Negative log likelihood |

### Containers

| Struct | Description |
|---|---|
| `Sequential` | Ordered sequence of modules |

### Initialization

| Function | Description |
|---|---|
| `kaiming_uniform(tensor, fan_in)` | Kaiming/He uniform initialization |
| `kaiming_normal(tensor, fan_in)` | Kaiming/He normal initialization |
| `xavier_uniform(tensor, fan_in, fan_out)` | Xavier/Glorot uniform initialization |
| `xavier_normal(tensor, fan_in, fan_out)` | Xavier/Glorot normal initialization |

---

## theano::optim

Optimization algorithms, mirroring `torch.optim`.

### Optimizer Trait

```rust
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn params(&self) -> &[Variable];
}
```

### Optimizers

| Struct | Description | Key Parameters |
|---|---|---|
| `SGD` | Stochastic gradient descent | `lr`, `momentum`, `nesterov` |
| `Adam` | Adam optimizer | `lr`, `betas`, `eps`, `weight_decay` |
| `AdamW` | Adam with decoupled weight decay | `lr`, `betas`, `eps`, `weight_decay` |
| `RMSprop` | RMSprop optimizer | `lr`, `alpha`, `eps`, `momentum` |
| `Adagrad` | Adagrad optimizer | `lr`, `eps`, `lr_decay` |

### Learning Rate Schedulers

| Struct | Description |
|---|---|
| `StepLR` | Decay LR by gamma every step_size epochs |
| `ExponentialLR` | Decay LR by gamma every epoch |
| `CosineAnnealingLR` | Cosine annealing schedule |
| `MultiStepLR` | Decay at specified milestones |
| `LinearLR` | Linear warmup/decay |
| `ReduceLROnPlateau` | Reduce LR when a metric plateaus |

---

## theano::data

Data loading utilities, mirroring `torch.utils.data`.

### Dataset Trait

```rust
pub trait Dataset: Send + Sync {
    type Item;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get(&self, index: usize) -> Self::Item;
}
```

### Structs

| Struct | Description |
|---|---|
| `TensorDataset` | In-memory dataset of (data, label) tensor pairs |
| `DataLoader` | Batched iteration over a dataset with optional shuffling |

---

## Backend Selection and Device Management

### Device Enum

```rust
pub enum Device {
    Cpu,
    Cuda(usize),    // NVIDIA GPU (ordinal)
    Rocm(usize),    // AMD GPU (ordinal)
    Metal(usize),   // Apple Silicon GPU
    Wgpu(usize),    // WebGPU device
    OneApi(usize),  // Intel accelerator
}
```

### DType Enum

Supported element types:

| Variant | Rust Type | Size |
|---|---|---|
| `DType::Float16` | `half::f16` | 2 bytes |
| `DType::BFloat16` | `half::bf16` | 2 bytes |
| `DType::Float32` | `f32` | 4 bytes |
| `DType::Float64` | `f64` | 8 bytes |
| `DType::Int8` | `i8` | 1 byte |
| `DType::Int16` | `i16` | 2 bytes |
| `DType::Int32` | `i32` | 4 bytes |
| `DType::Int64` | `i64` | 8 bytes |
| `DType::Bool` | `bool` | 1 byte |

### Feature Flags

Enable backends via Cargo features:

```toml
theano = { version = "0.1", features = ["cpu"] }          # CPU only (default)
theano = { version = "0.1", features = ["cuda"] }         # NVIDIA GPU
theano = { version = "0.1", features = ["rocm"] }         # AMD GPU
theano = { version = "0.1", features = ["metal"] }        # Apple Metal
theano = { version = "0.1", features = ["wgpu-backend"] } # WebGPU
theano = { version = "0.1", features = ["oneapi"] }       # Intel oneAPI
theano = { version = "0.1", features = ["full"] }         # All backends
```
