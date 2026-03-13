use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use theano_autodiff_benchmarks::{tape_based, compiled};

fn bench_tape_mlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("tape_based_mlp");
    for &(input, hidden, layers) in &[
        (4, 8, 2),
        (16, 32, 4),
        (64, 128, 8),
        (128, 256, 16),
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("{}x{}x{}", input, hidden, layers)),
            &(input, hidden, layers),
            |b, &(i, h, l)| b.iter(|| tape_based::bench_mlp(i, h, l)),
        );
    }
    group.finish();
}

fn bench_compiled_mlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_mlp");
    for &(input, hidden, layers) in &[
        (4, 8, 2),
        (16, 32, 4),
        (64, 128, 8),
        (128, 256, 16),
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("{}x{}x{}", input, hidden, layers)),
            &(input, hidden, layers),
            |b, &(i, h, l)| b.iter(|| compiled::bench_mlp(i, h, l)),
        );
    }
    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("overhead_comparison");
    for &(input, hidden, layers) in &[
        (8, 16, 3),
        (32, 64, 6),
    ] {
        group.bench_with_input(
            BenchmarkId::new("tape", format!("{}x{}x{}", input, hidden, layers)),
            &(input, hidden, layers),
            |b, &(i, h, l)| b.iter(|| tape_based::bench_mlp(i, h, l)),
        );
        group.bench_with_input(
            BenchmarkId::new("compiled", format!("{}x{}x{}", input, hidden, layers)),
            &(input, hidden, layers),
            |b, &(i, h, l)| b.iter(|| compiled::bench_mlp(i, h, l)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_tape_mlp, bench_compiled_mlp, bench_comparison);
criterion_main!(benches);
