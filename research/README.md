# Autodiff Research: Compiler-Level AD for Neo Theano

## Motivation

Neo Theano currently uses **tape-based reverse-mode AD** (like PyTorch eager mode). This works but has fundamental GPU efficiency limitations: per-op kernel dispatch, CPU-GPU synchronization per operation, and no cross-operation fusion.

Rust nightly now has two experimental features that can eliminate these limitations entirely:

1. **`#[autodiff]`** — Enzyme-based compiler-level AD that generates backward functions at compile time from LLVM IR
2. **`core::intrinsics::offload`** — GPU offload that compiles Rust functions to GPU kernels with automatic data movement

Together, these allow a single GPU kernel for an entire forward+backward pass — no tape, no per-op dispatch, no CPU in the loop.

## Contents

```
research/
├── DESIGN.md                    # Full architecture document
├── enzyme-prototype/            # Enzyme autodiff integration prototype
│   └── src/lib.rs               # CompiledAD trait, manual reference impl, nightly API examples
├── offload-prototype/           # GPU offload integration prototype
│   └── src/lib.rs               # OffloadManager, memory transfer analysis
└── benchmarks/                  # Tape vs compiled AD benchmarks
    ├── src/lib.rs               # Benchmark framework
    └── benches/autodiff_comparison.rs  # Criterion benchmarks
```

## Key Design Decision

**Dual-mode AD**: Static graphs use Enzyme (compiler AD), dynamic graphs keep using the existing tape. Users opt in explicitly:

```rust
// Static (compiled AD — single fused GPU kernel)
#[autodiff_reverse(d_model, Duplicated, Active)]
fn model_forward(params: &[f64]) -> f64 { ... }

// Dynamic (tape-based — existing behavior, unchanged)
let x = Variable::new(input, true);
let output = model.forward(&x);
output.backward();
```

## Running

```bash
# Tests (stable Rust)
cd research/enzyme-prototype && cargo test
cd research/offload-prototype && cargo test

# Benchmarks
cd research/benchmarks && cargo bench

# With nightly + Enzyme (when available)
RUSTFLAGS="-C lto=fat" cargo +nightly test --features nightly-autodiff
```

## Nightly Requirements

| Feature | Flag | Tracking Issue | Status |
|---|---|---|---|
| `#[autodiff]` | `#![feature(autodiff)]` | [#124509](https://github.com/rust-lang/rust/issues/124509) | Experimental, active |
| GPU offload | `#![feature(gpu_offload)]` | [#131513](https://github.com/rust-lang/rust/issues/131513) | Experimental, active |
| GPU kernel ABI | `#![feature(abi_gpu_kernel)]` | [#135467](https://github.com/rust-lang/rust/issues/135467) | Experimental |

All three are driven by Manuel Drehwald (ZuseZ4) as part of Rust's scientific computing initiative.
