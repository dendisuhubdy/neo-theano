use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use theano_autodiff_benchmarks::{tape_based, compiled, dynamic_vs_static};

// ─── MLP benchmarks (tape vs compiled) ───────────────────────────────────────

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

// ─── Conv2d benchmarks (tape vs compiled) ────────────────────────────────────

fn bench_tape_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("tape_based_conv2d");
    for &(ci, co, spatial, ks) in &[
        (1, 4, 8, 3),     // small: 1->4 channels, 8x8, 3x3 kernel
        (3, 16, 16, 3),   // medium: RGB->16 channels, 16x16
        (3, 32, 16, 5),   // medium: RGB->32 channels, 5x5 kernel
        (16, 32, 8, 3),   // deeper: 16->32 channels, 8x8
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("{}c->{}c_{}x{}_k{}", ci, co, spatial, spatial, ks)),
            &(ci, co, spatial, ks),
            |b, &(ci, co, s, k)| b.iter(|| tape_based::bench_conv2d(ci, co, s, k)),
        );
    }
    group.finish();
}

fn bench_compiled_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_conv2d");
    for &(ci, co, spatial, ks) in &[
        (1, 4, 8, 3),
        (3, 16, 16, 3),
        (3, 32, 16, 5),
        (16, 32, 8, 3),
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("{}c->{}c_{}x{}_k{}", ci, co, spatial, spatial, ks)),
            &(ci, co, spatial, ks),
            |b, &(ci, co, s, k)| b.iter(|| compiled::bench_conv2d(ci, co, s, k)),
        );
    }
    group.finish();
}

// ─── LSTM benchmarks (tape vs compiled) ──────────────────────────────────────

fn bench_tape_lstm(c: &mut Criterion) {
    let mut group = c.benchmark_group("tape_based_lstm");
    for &(input_size, hidden_size, seq_len) in &[
        (4, 8, 4),      // small LSTM
        (8, 16, 8),     // medium LSTM
        (16, 32, 8),    // larger LSTM
        (32, 64, 4),    // wide LSTM, short seq
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("in{}_h{}_seq{}", input_size, hidden_size, seq_len)),
            &(input_size, hidden_size, seq_len),
            |b, &(is, hs, sl)| b.iter(|| tape_based::bench_lstm(is, hs, sl)),
        );
    }
    group.finish();
}

fn bench_compiled_lstm(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_lstm");
    for &(input_size, hidden_size, seq_len) in &[
        (4, 8, 4),
        (8, 16, 8),
        (16, 32, 8),
        (32, 64, 4),
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("in{}_h{}_seq{}", input_size, hidden_size, seq_len)),
            &(input_size, hidden_size, seq_len),
            |b, &(is, hs, sl)| b.iter(|| compiled::bench_lstm(is, hs, sl)),
        );
    }
    group.finish();
}

// ─── BatchNorm benchmarks (tape vs compiled) ─────────────────────────────────

fn bench_tape_batchnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("tape_based_batchnorm");
    for &(batch, features) in &[
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("b{}_f{}", batch, features)),
            &(batch, features),
            |b, &(bs, f)| b.iter(|| tape_based::bench_batchnorm(bs, f)),
        );
    }
    group.finish();
}

fn bench_compiled_batchnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_batchnorm");
    for &(batch, features) in &[
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("b{}_f{}", batch, features)),
            &(batch, features),
            |b, &(bs, f)| b.iter(|| compiled::bench_batchnorm(bs, f)),
        );
    }
    group.finish();
}

// ─── Attention benchmarks (tape vs compiled) ─────────────────────────────────

fn bench_tape_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("tape_based_attention");
    for &(seq_len, embed_dim, num_heads) in &[
        (4, 8, 2),     // small attention
        (8, 16, 4),    // medium attention
        (8, 32, 4),    // wider attention
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("seq{}_d{}_h{}", seq_len, embed_dim, num_heads)),
            &(seq_len, embed_dim, num_heads),
            |b, &(s, d, h)| b.iter(|| tape_based::bench_attention(s, d, h)),
        );
    }
    group.finish();
}

fn bench_compiled_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_attention");
    for &(seq_len, embed_dim, num_heads) in &[
        (4, 8, 2),
        (8, 16, 4),
        (8, 32, 4),
    ] {
        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("seq{}_d{}_h{}", seq_len, embed_dim, num_heads)),
            &(seq_len, embed_dim, num_heads),
            |b, &(s, d, h)| b.iter(|| compiled::bench_attention(s, d, h)),
        );
    }
    group.finish();
}

// ─── Dynamic vs Static graph benchmarks ──────────────────────────────────────

fn bench_dynamic_vs_static(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_vs_static_graph");

    for &(input, hidden, layers) in &[
        (4, 8, 2),
        (16, 32, 4),
        (64, 128, 8),
        (128, 256, 16),
    ] {
        let graph = dynamic_vs_static::StaticGraph::build_mlp(input, hidden, layers);
        let input_values: Vec<f64> = (0..input).map(|i| (i as f64 + 1.0) * 0.1).collect();

        // Dynamic: rebuild graph + forward + backward every iteration
        group.bench_with_input(
            BenchmarkId::new("dynamic_tape", format!("{}x{}x{}", input, hidden, layers)),
            &(input, hidden, layers),
            |b, &(i, h, l)| b.iter(|| tape_based::bench_mlp(i, h, l)),
        );

        // Static: forward + backward on pre-built graph (no graph construction)
        group.bench_with_input(
            BenchmarkId::new("static_graph", format!("{}x{}x{}", input, hidden, layers)),
            &input_values,
            |b, iv| b.iter(|| dynamic_vs_static::bench_static_forward_backward(&graph, iv)),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tape_mlp,
    bench_compiled_mlp,
    bench_comparison,
    bench_tape_conv2d,
    bench_compiled_conv2d,
    bench_tape_lstm,
    bench_compiled_lstm,
    bench_tape_batchnorm,
    bench_compiled_batchnorm,
    bench_tape_attention,
    bench_compiled_attention,
    bench_dynamic_vs_static,
);
criterion_main!(benches);
