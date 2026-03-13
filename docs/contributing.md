# Contributing to Neo Theano

Thank you for your interest in contributing to Neo Theano. This guide covers the development workflow, coding conventions, and process for submitting changes.

## Setting Up the Development Environment

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Git
- A C compiler (for BLAS dependencies)

### Clone and Build

```bash
git clone https://github.com/dendisuhubdy/theano.git
cd theano

# Build the entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Build documentation
cargo doc --workspace --no-deps --open
```

### Optional: GPU Development

For CUDA backend development:
```bash
# Ensure CUDA toolkit is installed and nvcc is on PATH
nvcc --version

# Build with CUDA
cargo build --workspace --features theano/cuda
```

For ROCm backend development:
```bash
# Ensure ROCm is installed
hipcc --version

# Build with ROCm
cargo build --workspace --features theano/rocm
```

### IDE Setup

We recommend VS Code with the `rust-analyzer` extension, or CLion with the Rust plugin.

For `rust-analyzer`, add to `.vscode/settings.json`:
```json
{
    "rust-analyzer.cargo.features": ["cpu"]
}
```

## Code Style and Conventions

### Formatting

All code must be formatted with `rustfmt`:

```bash
cargo fmt --all
```

Check formatting without modifying files:

```bash
cargo fmt --all -- --check
```

### Linting

All code must pass `clippy` with no warnings:

```bash
cargo clippy --workspace -- -D warnings
```

### Naming Conventions

- **Crate names**: `theano-<module>` (e.g., `theano-core`, `theano-cuda`)
- **Types**: `PascalCase` (e.g., `BackendStorage`, `CachingAllocator`)
- **Functions/methods**: `snake_case` (e.g., `from_slice`, `binary_op`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `SMALL_POOL_THRESHOLD`)
- **Feature flags**: `lowercase-with-dashes` (e.g., `wgpu-backend`)

### PyTorch Parity

We aim for 100% API parity with PyTorch where it makes sense in Rust:

- `torch.Tensor` maps to `theano::Tensor`
- `torch.autograd.Variable` maps to `theano::Variable`
- `torch.nn.Module` maps to `theano::nn::Module` (trait)
- `torch.optim.Optimizer` maps to `theano::optim::Optimizer` (trait)
- `torch.utils.data.Dataset` maps to `theano::data::Dataset` (trait)

Method names should match PyTorch where possible:
- `tensor.shape()` not `tensor.dimensions()`
- `tensor.numel()` not `tensor.num_elements()`
- `tensor.item()` not `tensor.scalar_value()`
- `tensor.t()` not `tensor.matrix_transpose()`

### Documentation

All public APIs must have doc comments:

```rust
/// Perform a single optimization step (parameter update).
///
/// This reads the `.grad()` of each parameter and updates
/// the parameter values according to the optimizer algorithm.
///
/// # Examples
///
/// ```
/// optimizer.zero_grad();
/// let loss = model.forward(&input);
/// loss.backward();
/// optimizer.step();
/// ```
fn step(&mut self);
```

### Error Handling

- Use `theano_types::Result<T>` (which is `Result<T, TheanoError>`) for fallible operations.
- Use `TheanoError` variants for specific error kinds (shape mismatch, device mismatch, out of memory, etc.).
- Avoid `.unwrap()` in library code. Use `.expect()` only for invariants that are guaranteed by construction.
- Panics are acceptable only for programming errors (e.g., calling `backward()` on a non-scalar).

## Running Tests

### All Tests

```bash
cargo test --workspace
```

### Specific Crate

```bash
cargo test -p theano-core
cargo test -p theano-autograd
cargo test -p theano-nn
```

### Doc Tests

```bash
cargo test --workspace --doc
```

### With Output

```bash
cargo test --workspace -- --nocapture
```

### GPU Tests

GPU tests are gated behind feature flags and will be skipped if the hardware is not present:

```bash
cargo test --workspace --features theano/cuda
cargo test --workspace --features theano/rocm
```

### Benchmarks

```bash
cargo bench --workspace
```

## Adding New Operations

### 1. Add to Backend Trait (if needed)

If the operation requires a new kernel type, add it to the appropriate enum in `crates/theano-backend/src/ops.rs`:

```rust
pub enum UnaryOp {
    // ... existing ops
    MyNewOp,
}
```

### 2. Implement in CPU Backend

Add the kernel implementation in `crates/theano-cpu/src/cpu_kernels.rs`:

```rust
UnaryOp::MyNewOp => {
    // Implement the operation
    for i in 0..len {
        output[i] = my_new_op(input[i]);
    }
}
```

### 3. Add Tensor Method

In `crates/theano-core/src/tensor_ops.rs`:

```rust
impl Tensor {
    pub fn my_new_op(&self) -> Result<Tensor> {
        // Validate inputs
        // Dispatch to storage
        // Return new tensor
    }
}
```

### 4. Add Variable Method with GradFn

In `crates/theano-autograd/src/grad_fns.rs`, define the backward:

```rust
pub struct MyNewOpBackward {
    pub saved_input: SavedTensor,
}

impl GradFn for MyNewOpBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // Compute and return gradient
    }

    fn name(&self) -> &str {
        "MyNewOpBackward"
    }
}
```

In `crates/theano-autograd/src/variable.rs`:

```rust
impl Variable {
    pub fn my_new_op(&self) -> Result<Variable> {
        let result = self.tensor.my_new_op()?;
        let grad_fn = Arc::new(MyNewOpBackward {
            saved_input: SavedTensor(self.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }
}
```

### 5. Add Tests

Add unit tests for both the forward computation and the gradient:

```rust
#[test]
fn test_my_new_op() {
    let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
    let result = t.my_new_op().unwrap();
    // Assert correctness
}

#[test]
fn test_my_new_op_backward() {
    let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0], &[2]));
    let y = x.my_new_op().unwrap();
    let loss = y.sum().unwrap();
    loss.backward();
    // Assert gradient correctness
}
```

### 6. Implement in GPU Backends

Add the kernel to each GPU backend crate (`theano-cuda`, `theano-rocm`, etc.).

## Adding New Backends

See the "Writing a Custom Backend" section in [backends.md](backends.md) for the full process. In summary:

1. Create `crates/theano-<name>/` with `BackendStorage` implementation.
2. Add `Storage` variant in `theano-core`.
3. Add feature flag in `theano` facade crate.
4. Add tests that mirror the CPU backend test suite.
5. Add documentation in `docs/backends.md`.

## PR Process

### Before Submitting

1. **Format**: `cargo fmt --all`
2. **Lint**: `cargo clippy --workspace -- -D warnings`
3. **Test**: `cargo test --workspace`
4. **Doc**: `cargo doc --workspace --no-deps` (no warnings)

### PR Guidelines

- Keep PRs focused on a single change.
- Write a clear description of what and why.
- Include tests for new functionality.
- Include doc comments for new public APIs.
- Reference related issues in the PR description.
- Ensure CI passes before requesting review.

### Commit Messages

Use conventional commit style:

```
feat(nn): add GroupNorm layer

Implements group normalization following PyTorch's nn.GroupNorm API.
Supports configurable number of groups and affine parameters.

Closes #42
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `perf`: Performance improvements
- `chore`: Build, CI, dependency updates

### Review Process

1. Open a PR against `main`.
2. CI must pass (format, lint, test, doc).
3. At least one maintainer review is required.
4. Squash-merge into `main`.

## Getting Help

- Open an issue for bugs or feature requests.
- Start a discussion for questions or design proposals.
- Tag `@dendisuhubdy` for maintainer attention on critical issues.
