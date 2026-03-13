# Getting Started

This guide walks you through installing Neo Theano and building your first neural network in Rust.

## Prerequisites

- **Rust 1.75 or later** — Install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup update stable
  ```

- **Optional: CUDA Toolkit 12.x** — For NVIDIA GPU acceleration. Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

- **Optional: ROCm 6.x** — For AMD GPU acceleration. See [AMD ROCm Installation](https://rocm.docs.amd.com/).

## Installation

Add Theano to your `Cargo.toml`:

```toml
[dependencies]
theano = "0.1"
```

### With GPU support

For NVIDIA CUDA:

```toml
[dependencies]
theano = { version = "0.1", features = ["cuda"] }
```

For AMD ROCm:

```toml
[dependencies]
theano = { version = "0.1", features = ["rocm"] }
```

For Apple Metal:

```toml
[dependencies]
theano = { version = "0.1", features = ["metal"] }
```

For all backends:

```toml
[dependencies]
theano = { version = "0.1", features = ["full"] }
```

## First Tensor Operations

```rust
use theano::prelude::*;

fn main() {
    // Create tensors from data
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::ones(&[2, 2]);

    // Arithmetic operations
    let c = &a + &b;
    println!("a + b = {:?}", c.to_vec_f64().unwrap());

    // Matrix multiplication
    let d = a.matmul(&b).unwrap();
    println!("a @ b = {:?}", d.to_vec_f64().unwrap());

    // Common creation functions
    let zeros = Tensor::zeros(&[3, 3]);
    let ones = Tensor::ones(&[3, 3]);
    let randn = Tensor::randn(&[2, 3]); // Standard normal

    // Shape and metadata
    println!("shape: {:?}", a.shape());    // [2, 2]
    println!("ndim: {}", a.ndim());         // 2
    println!("numel: {}", a.numel());       // 4
    println!("dtype: {:?}", a.dtype());     // Float64
}
```

## Autograd: Computing Gradients

Theano provides automatic differentiation through the `Variable` wrapper:

```rust
use theano::prelude::*;

fn main() {
    // Create variables that track gradients
    let x = Variable::requires_grad(Tensor::from_slice(&[2.0, 3.0], &[2]));
    let w = Variable::requires_grad(Tensor::from_slice(&[0.5, -0.5], &[2]));

    // Forward pass: f(x, w) = sum(x * w)
    let y = x.mul(&w).unwrap();
    let loss = y.sum().unwrap();

    // Backward pass: compute gradients
    loss.backward();

    // Access gradients
    println!("dL/dx = {:?}", x.grad().unwrap().to_vec_f64().unwrap());
    // [0.5, -0.5] — the values of w
    println!("dL/dw = {:?}", w.grad().unwrap().to_vec_f64().unwrap());
    // [2.0, 3.0] — the values of x
}
```

## Building a Simple Neural Network

Define a model by implementing the `Module` trait:

```rust
use theano::prelude::*;
use theano::nn::{Linear, ReLU, Module, Sequential, MSELoss};
use theano::optim::{SGD, Optimizer};

// Define a two-layer network
fn build_model() -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(2, 64, true)),    // input -> hidden
        Box::new(ReLU),
        Box::new(Linear::new(64, 1, true)),    // hidden -> output
    ])
}

fn main() {
    let model = build_model();
    let criterion = MSELoss::new();
    let mut optimizer = SGD::new(model.parameters(), 0.01, 0.0, false);

    // Dummy training data
    let x = Variable::new(Tensor::randn(&[32, 2]));
    let y_true = Variable::new(Tensor::randn(&[32, 1]));

    // Training step
    optimizer.zero_grad();
    let y_pred = model.forward(&x);
    let loss = criterion.forward(&y_pred, &y_true);
    loss.backward();
    optimizer.step();

    println!("loss = {}", loss.tensor().item().unwrap());
}
```

## Training Loop

A typical training loop follows the same pattern as PyTorch:

```rust
use theano::prelude::*;
use theano::nn::{Module, CrossEntropyLoss};
use theano::optim::{Adam, Optimizer};
use theano::data::{Dataset, DataLoader};

fn train(
    model: &impl Module,
    optimizer: &mut impl Optimizer,
    dataloader: &DataLoader,
    epochs: usize,
) {
    let criterion = CrossEntropyLoss::new();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (batch_x, batch_y) in dataloader.iter() {
            let input = Variable::new(batch_x);
            let target = Variable::new(batch_y);

            // Forward
            optimizer.zero_grad();
            let output = model.forward(&input);
            let loss = criterion.forward(&output, &target);

            // Backward
            loss.backward();
            optimizer.step();

            total_loss += loss.tensor().item().unwrap();
            num_batches += 1;
        }

        println!(
            "Epoch {}/{}: avg_loss = {:.4}",
            epoch + 1,
            epochs,
            total_loss / num_batches as f64
        );
    }
}
```

## Using Different Backends

### CPU (Default)

The CPU backend is enabled by default. No additional setup is required.

```rust
use theano::prelude::*;

let t = Tensor::ones(&[3, 3]); // Uses CPU backend
assert_eq!(t.device(), &Device::Cpu);
```

### CUDA (NVIDIA GPU)

Enable the `cuda` feature and specify the device:

```toml
# Cargo.toml
theano = { version = "0.1", features = ["cuda"] }
```

```rust
use theano::prelude::*;

// Check CUDA availability
if Device::cuda_is_available() {
    let gpu = Device::Cuda(0); // First GPU

    // Create tensor on GPU
    let t = Tensor::ones_device(&[1024, 1024], &gpu);

    // Move tensor to GPU
    let cpu_tensor = Tensor::randn(&[256, 256]);
    let gpu_tensor = cpu_tensor.to_device(&gpu).unwrap();

    // Operations run on GPU automatically
    let result = gpu_tensor.matmul(&gpu_tensor).unwrap();
}
```

### Disabling Gradient Computation

For inference, disable autograd to save memory and computation:

```rust
use theano::prelude::*;

// Using no_grad closure
let output = no_grad(|| {
    model.forward(&input)
});

// Using NoGradGuard (RAII)
{
    let _guard = NoGradGuard::new();
    let output = model.forward(&input);
    // grad tracking is disabled in this scope
}
```

## Next Steps

- [API Reference](api-reference.md) — Complete API documentation
- [Backends Guide](backends.md) — Detailed backend setup and configuration
- [Architecture](architecture.md) — Internal design and data flow
- [Contributing](contributing.md) — How to contribute to Neo Theano
