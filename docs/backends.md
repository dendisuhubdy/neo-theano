# Backends Guide

Neo Theano supports multiple compute backends through a unified `Backend` / `BackendStorage` trait system. Each backend is a separate crate behind a Cargo feature flag, so only the backends you need are compiled.

## CPU Backend (Default)

**Crate**: `theano-cpu`
**Feature**: `cpu` (enabled by default)

The CPU backend is always available and requires no additional setup.

### BLAS Acceleration

Matrix multiplication uses the [`gemm`](https://crates.io/crates/gemm) crate, which provides highly optimized BLAS-level matmul with automatic CPU feature detection (AVX2, AVX-512, NEON, etc.).

For additional BLAS options, you can link against system BLAS libraries:

- **OpenBLAS**: `sudo apt install libopenblas-dev` (Linux) or `brew install openblas` (macOS)
- **Intel MKL**: Available through oneAPI toolkit
- **Apple Accelerate**: Automatically available on macOS

### Parallelism

The CPU backend uses [Rayon](https://crates.io/crates/rayon) for parallel elementwise operations. Thread count is controlled by the `RAYON_NUM_THREADS` environment variable:

```bash
RAYON_NUM_THREADS=8 cargo run --release
```

### Usage

```rust
use theano::prelude::*;

// CPU is the default device
let t = Tensor::ones(&[1024, 1024]);
assert_eq!(t.device(), &Device::Cpu);
```

---

## CUDA Backend (NVIDIA GPU)

**Crate**: `theano-cuda`
**Feature**: `cuda`

### Prerequisites

1. NVIDIA GPU with compute capability >= 7.0 (Volta or newer recommended)
2. NVIDIA driver >= 525
3. CUDA Toolkit 12.x installed
4. `nvcc` on your `PATH`

### Setup

```bash
# Linux (Ubuntu/Debian)
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi
```

Enable in `Cargo.toml`:

```toml
[dependencies]
theano = { version = "0.1", features = ["cuda"] }
```

### Architecture

The CUDA backend is built on [`cudarc`](https://crates.io/crates/cudarc), a safe Rust wrapper around the CUDA driver API. Key components:

- **`CudaStorage`**: Device memory buffer with async copy support.
- **`CachingAllocator`**: PyTorch-style caching allocator with small/large pools, block splitting, and OOM recovery.
- **`theano-cuda-kernels`**: Pre-compiled CUDA kernels for elementwise, reduction, and GEMM operations.

### Memory Management

The caching allocator reduces the overhead of frequent `cudaMalloc`/`cudaFree` calls:

- Freed blocks are returned to a cache pool instead of being released to the driver.
- Subsequent allocations check the cache first (best-fit search via `BTreeMap`).
- Two pools: small blocks (<= 1 MB, 512-byte granularity) and large blocks (> 1 MB, 2 MB granularity).
- Call `empty_cache()` to release all cached memory back to the driver.

### Usage

```rust
use theano::prelude::*;

let gpu = Device::Cuda(0);
let a = Tensor::randn_device(&[512, 512], &gpu);
let b = Tensor::randn_device(&[512, 512], &gpu);
let c = a.matmul(&b).unwrap(); // Runs on GPU
```

---

## ROCm/HIP Backend (AMD GPU)

**Crate**: `theano-rocm`
**Feature**: `rocm`

### Prerequisites

1. AMD GPU (RDNA 2 or newer, or CDNA/MI-series)
2. ROCm 6.x installed
3. HIP runtime available

### Setup

```bash
# Linux (Ubuntu)
# Follow AMD's official ROCm installation guide:
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

# Verify installation
rocm-smi
hipcc --version
```

Enable in `Cargo.toml`:

```toml
[dependencies]
theano = { version = "0.1", features = ["rocm"] }
```

### Architecture

The ROCm backend mirrors the CUDA backend structure:

- **`RocmStorage`**: HIP device memory buffer.
- **`CachingAllocator`**: Same two-pool design as the CUDA allocator, using `hipMalloc`/`hipFree`.
- **`theano-rocm-kernels`**: HIP kernels compiled via `hipcc`.

### Usage

```rust
use theano::prelude::*;

let gpu = Device::Rocm(0);
let t = Tensor::ones_device(&[256, 256], &gpu);
```

---

## BarraCUDA Backend (Experimental)

**Crate**: `theano-barracuda`
**Feature**: (not yet in the default feature set)

### Overview

BarraCUDA is an experimental backend that compiles CUDA `.cu` source files directly for AMD GPUs through source-level translation. Instead of maintaining separate CUDA and HIP kernel codebases, BarraCUDA translates CUDA API calls to HIP equivalents and compiles with `hipcc`.

### Architecture

- **`BarraCudaCompiler`**: Parses `.cu` files, performs CUDA-to-HIP API translation, and invokes `hipcc`.
- **`BarraCudaDevice`**: Manages the translated kernels on AMD hardware.
- **`BarraCudaBackend`**: Implements the `Backend` trait by delegating to translated kernels.

### Status

This backend is experimental. It works for simple kernels but complex CUDA features (cooperative groups, tensor cores) may not translate correctly. Contributions are welcome.

### Usage

```rust
use theano_barracuda::BarraCudaBackend;
// Compile CUDA kernels for AMD GPU
let device = BarraCudaDevice::new(0)?;
```

---

## Metal Backend (Apple Silicon)

**Crate**: `theano-metal`
**Feature**: `metal`

### Prerequisites

1. macOS 13+ (Ventura or later)
2. Apple Silicon (M1/M2/M3/M4) or AMD GPU on Mac

### Setup

No additional installation needed. The Metal framework is part of macOS.

```toml
[dependencies]
theano = { version = "0.1", features = ["metal"] }
```

### Architecture

- **`MetalStorage`**: Metal buffer wrapper.
- **`MetalDevice`**: Manages `MTLDevice`, command queues, and pipeline states.
- **`MetalBackend`**: Implements `Backend` using Metal Shading Language (MSL) compute kernels.
- **MSL Kernels**: Elementwise, reduction, and matmul kernels written in Metal Shading Language, compiled at runtime.

### Usage

```rust
use theano::prelude::*;

let gpu = Device::Metal(0);
let t = Tensor::randn_device(&[512, 512], &gpu);
```

---

## WebGPU Backend (Cross-Platform)

**Crate**: `theano-wgpu`
**Feature**: `wgpu-backend`

### Overview

The WebGPU backend provides cross-platform GPU compute using the [wgpu](https://crates.io/crates/wgpu) crate. It works on:

- Windows (DirectX 12 / Vulkan)
- Linux (Vulkan)
- macOS (Metal, via wgpu's Metal backend)
- Web browsers (WebGPU API, via wasm)

### Prerequisites

- A GPU with Vulkan 1.1+, DirectX 12, or Metal support
- For web: A browser with WebGPU support (Chrome 113+, Firefox Nightly)

### Setup

```toml
[dependencies]
theano = { version = "0.1", features = ["wgpu-backend"] }
```

### Architecture

- **`WgpuStorage`**: `wgpu::Buffer` wrapper.
- **`WgpuDevice`**: Manages `wgpu::Device`, queue, and compute pipelines.
- **`WgpuBackend`**: Implements `Backend` using WGSL (WebGPU Shading Language) compute shaders.
- **WGSL Kernels**: Compute shaders for elementwise ops, reductions, and matmul.

### Usage

```rust
use theano::prelude::*;

let gpu = Device::Wgpu(0);
let t = Tensor::ones_device(&[256, 256], &gpu);
```

---

## Intel oneAPI Backend

**Crate**: `theano-oneapi`
**Feature**: `oneapi`

### Prerequisites

1. Intel GPU (Arc series, Data Center GPU Max, or integrated)
2. Intel oneAPI Base Toolkit installed
3. Level Zero runtime

### Setup

```bash
# Install Intel oneAPI Base Toolkit
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

source /opt/intel/oneapi/setvars.sh
```

```toml
[dependencies]
theano = { version = "0.1", features = ["oneapi"] }
```

### Architecture

- **`OneApiStorage`**: SYCL/Level Zero device memory.
- **`OneApiDevice`**: Manages SYCL queue and device context.
- **`OneApiBackend`**: Implements `Backend` using oneAPI DPC++/SYCL kernels.

### Usage

```rust
use theano::prelude::*;

let gpu = Device::OneApi(0);
let t = Tensor::ones_device(&[256, 256], &gpu);
```

---

## Writing a Custom Backend

To add a new backend to Neo Theano:

### 1. Create the Crate

```bash
cargo init crates/theano-mybackend --lib
```

Add to workspace `Cargo.toml`:

```toml
[workspace]
members = [
    # ...
    "crates/theano-mybackend",
]
```

### 2. Define Your Storage

```rust
// crates/theano-mybackend/src/lib.rs
use theano_backend::{BackendStorage, Backend, UnaryOp, BinaryOp, ReduceOp};
use theano_types::{DType, Device, DeviceType, Result};

pub struct MyStorage {
    // Your device memory representation
    data: Vec<u8>,
    dtype: DType,
    len: usize,
}

impl BackendStorage for MyStorage {
    fn device(&self) -> Device { /* ... */ }
    fn dtype(&self) -> DType { self.dtype }
    fn len(&self) -> usize { self.len }

    fn unary_op(&self, op: UnaryOp, shape: &[usize], strides: &[usize]) -> Result<Self> {
        // Dispatch to your kernel
        todo!()
    }

    fn binary_op(&self, rhs: &Self, op: BinaryOp, /* ... */) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, op: ReduceOp, /* ... */) -> Result<Self> {
        todo!()
    }

    fn matmul(&self, rhs: &Self, /* ... */) -> Result<Self> {
        todo!()
    }

    fn fill(value: f64, len: usize, dtype: DType, device: &Device) -> Result<Self> {
        todo!()
    }

    fn from_f64_slice(data: &[f64], dtype: DType, device: &Device) -> Result<Self> {
        todo!()
    }

    fn to_f64_vec(&self, shape: &[usize], strides: &[usize]) -> Result<Vec<f64>> {
        todo!()
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> { todo!() }
    fn contiguous(&self, shape: &[usize], strides: &[usize]) -> Result<Self> { todo!() }
    fn where_cond(&self, cond: &Self, other: &Self, shape: &[usize], strides: &[usize]) -> Result<Self> { todo!() }
    fn index_select(&self, dim: usize, indices: &Self, shape: &[usize], strides: &[usize]) -> Result<Self> { todo!() }
}
```

### 3. Define Your Backend

```rust
#[derive(Clone)]
pub struct MyBackend;

impl Backend for MyBackend {
    type Storage = MyStorage;

    fn name() -> &'static str { "mybackend" }
    fn device_type() -> DeviceType { DeviceType::Custom }
}
```

### 4. Wire Into theano-core

Add a new variant to the `Storage` enum in `crates/theano-core/src/storage.rs` and implement dispatch for all methods.

### 5. Add Feature Flags

In `crates/theano/Cargo.toml`:

```toml
[features]
mybackend = ["theano-mybackend"]
```

### Priority of Operations to Implement

Start with these operations in order of importance:

1. `fill`, `from_f64_slice`, `to_f64_vec` (creation and data transfer)
2. `binary_op` for Add, Sub, Mul, Div (basic arithmetic)
3. `unary_op` for Neg, Exp, Log, Relu (basic math and activations)
4. `matmul` (critical for neural networks)
5. `reduce_op` for Sum, Mean (loss computation)
6. `to_dtype`, `contiguous` (type casting and memory layout)
7. Everything else (comparisons, advanced reductions, index_select, where_cond)

See [contributing.md](contributing.md) for the development workflow and testing requirements.
