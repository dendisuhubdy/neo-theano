#!/usr/bin/env bash
#
# bench_autograd.sh — Run the autograd overhead benchmark.
#
# Compares tape-based runtime AD (current Variable::backward()) against
# compile-time autodiff (Enzyme target, estimated as forward-only cost).
#
# Usage:
#   ./scripts/bench_autograd.sh            # default: release mode
#   ./scripts/bench_autograd.sh --debug    # debug mode (slower, useful for profiling)
#   ./scripts/bench_autograd.sh --save     # save output to benchmarks/results/
#
# Requirements:
#   - Rust toolchain (stable, 1.75+)
#   - No additional dependencies beyond the workspace Cargo.toml
#
# What it measures:
#   - "forward + runtime backward (tape)": actual Variable::backward() with
#     GradFn dispatch, topological sort, HashMap gradient accumulation
#   - "forward + compile-time autodiff (est.)": forward pass under no_grad;
#     this is the computation Enzyme would fuse into a single compiled function,
#     minus the backward FLOPs (roughly 1x forward)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROFILE="--release"
SAVE=false

for arg in "$@"; do
    case "$arg" in
        --debug) PROFILE="" ;;
        --save)  SAVE=true ;;
        --help|-h)
            echo "Usage: $0 [--debug] [--save]"
            echo "  --debug   Run in debug mode (slower, for profiling)"
            echo "  --save    Save output to benchmarks/results/autograd_<timestamp>.txt"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Run $0 --help for usage"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "Building benchmark ($([ -n "$PROFILE" ] && echo "release" || echo "debug"))..."
cargo build --example autograd_bench $PROFILE 2>/dev/null

if [ "$SAVE" = true ]; then
    mkdir -p benchmarks/results
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTFILE="benchmarks/results/autograd_${TIMESTAMP}.txt"

    {
        echo "# Neo Theano Autograd Benchmark"
        echo "# Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
        echo "# Platform: $(uname -srm)"
        echo "# Rust: $(rustc --version)"
        echo "# CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: //' || echo 'unknown')"
        echo "# Profile: $([ -n "$PROFILE" ] && echo "release" || echo "debug")"
        echo "#"
        cargo run --example autograd_bench $PROFILE 2>/dev/null
    } | tee "$OUTFILE"

    echo ""
    echo "Results saved to: $OUTFILE"
else
    cargo run --example autograd_bench $PROFILE 2>/dev/null
fi
