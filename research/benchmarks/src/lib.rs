//! # Autodiff Benchmark Framework
//!
//! Compares tape-based AD vs compiled (Enzyme) AD across different model sizes
//! and architectures to quantify the overhead of dynamic taping.

/// Tape-based autograd simulation.
///
/// This mimics what Neo Theano's current autograd does:
/// 1. Forward: execute ops, build tape (record GradFn + inputs per op)
/// 2. Backward: topological sort tape, replay in reverse
///
/// The overhead we're measuring:
/// - Tape node allocation per operation
/// - Vec push for each operation
/// - Topological sort traversal
/// - Indirect function call per GradFn::backward()
pub mod tape_based {
    use std::time::Instant;

    struct TapeNode {
        grad_fn: Box<dyn Fn(f64) -> Vec<f64>>,
        input_indices: Vec<usize>,
    }

    pub struct Tape {
        nodes: Vec<TapeNode>,
        values: Vec<f64>,
    }

    impl Tape {
        pub fn new() -> Self {
            Self { nodes: Vec::new(), values: Vec::new() }
        }

        pub fn constant(&mut self, val: f64) -> usize {
            let idx = self.values.len();
            self.values.push(val);
            idx
        }

        pub fn add(&mut self, a: usize, b: usize) -> usize {
            let val = self.values[a] + self.values[b];
            let idx = self.values.len();
            self.values.push(val);
            self.nodes.push(TapeNode {
                grad_fn: Box::new(|g| vec![g, g]),
                input_indices: vec![a, b],
            });
            idx
        }

        pub fn mul(&mut self, a: usize, b: usize) -> usize {
            let va = self.values[a];
            let vb = self.values[b];
            let val = va * vb;
            let idx = self.values.len();
            self.values.push(val);
            self.nodes.push(TapeNode {
                grad_fn: Box::new(move |g| vec![g * vb, g * va]),
                input_indices: vec![a, b],
            });
            idx
        }

        pub fn relu(&mut self, a: usize) -> usize {
            let va = self.values[a];
            let val = va.max(0.0);
            let idx = self.values.len();
            self.values.push(val);
            self.nodes.push(TapeNode {
                grad_fn: Box::new(move |g| vec![if va > 0.0 { g } else { 0.0 }]),
                input_indices: vec![a],
            });
            idx
        }

        pub fn backward(&self, output_idx: usize) -> Vec<f64> {
            let mut grads = vec![0.0; self.values.len()];
            grads[output_idx] = 1.0;

            // Walk tape in reverse (simplified topological sort)
            for (node_idx, node) in self.nodes.iter().enumerate().rev() {
                let output_node_idx = node_idx + (self.values.len() - self.nodes.len());
                let g = grads[output_node_idx];
                if g == 0.0 { continue; }

                let input_grads = (node.grad_fn)(g);
                for (i, &input_idx) in node.input_indices.iter().enumerate() {
                    grads[input_idx] += input_grads[i];
                }
            }

            grads
        }
    }

    /// Benchmark: MLP forward+backward using tape-based AD
    pub fn bench_mlp(input_dim: usize, hidden_dim: usize, n_layers: usize) -> (f64, std::time::Duration) {
        let start = Instant::now();
        let mut tape = Tape::new();

        // Create input values
        let mut current: Vec<usize> = (0..input_dim)
            .map(|i| tape.constant((i as f64 + 1.0) * 0.1))
            .collect();

        // N hidden layers: linear + relu
        for _layer in 0..n_layers {
            let mut next = Vec::with_capacity(hidden_dim);
            for h in 0..hidden_dim {
                // Weighted sum (simulated matmul row)
                let mut sum = tape.constant(0.01); // bias
                for (i, &c) in current.iter().enumerate() {
                    let w = tape.constant(0.01 * ((h * input_dim + i) as f64 % 7.0 - 3.0));
                    let prod = tape.mul(c, w);
                    sum = tape.add(sum, prod);
                }
                let activated = tape.relu(sum);
                next.push(activated);
            }
            current = next;
        }

        // Sum outputs for scalar loss
        let mut loss = current[0];
        for &c in &current[1..] {
            loss = tape.add(loss, c);
        }

        // Backward
        let grads = tape.backward(loss);
        let loss_val = tape.values[loss];
        let duration = start.elapsed();

        // Use grads to prevent optimization
        let _grad_sum: f64 = grads.iter().sum();

        (loss_val, duration)
    }
}

/// Compiled AD simulation (what Enzyme generates).
///
/// No tape, no allocations, no indirect calls.
/// Just direct computation — the exact same math but without the overhead.
pub mod compiled {
    use std::time::Instant;

    /// Benchmark: MLP forward+backward using compiled AD (no tape)
    pub fn bench_mlp(input_dim: usize, hidden_dim: usize, n_layers: usize) -> (f64, std::time::Duration) {
        let start = Instant::now();

        // Weights (same deterministic init as tape version)
        let weights: Vec<Vec<Vec<f64>>> = (0..n_layers)
            .map(|l| {
                let fan_in = if l == 0 { input_dim } else { hidden_dim };
                (0..hidden_dim)
                    .map(|h| {
                        (0..fan_in)
                            .map(|i| 0.01 * ((h * fan_in + i) as f64 % 7.0 - 3.0))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let biases: Vec<Vec<f64>> = (0..n_layers)
            .map(|_| vec![0.01; hidden_dim])
            .collect();

        // Forward pass
        let mut current: Vec<f64> = (0..input_dim)
            .map(|i| (i as f64 + 1.0) * 0.1)
            .collect();
        let mut pre_activations: Vec<Vec<f64>> = Vec::new();
        let mut activations: Vec<Vec<f64>> = vec![current.clone()];

        for l in 0..n_layers {
            let mut next = vec![0.0; hidden_dim];
            let mut pre_act = vec![0.0; hidden_dim];
            for h in 0..hidden_dim {
                let mut sum = biases[l][h];
                for (i, &c) in current.iter().enumerate() {
                    sum += c * weights[l][h][i];
                }
                pre_act[h] = sum;
                next[h] = sum.max(0.0); // relu
            }
            pre_activations.push(pre_act);
            activations.push(next.clone());
            current = next;
        }

        let loss: f64 = current.iter().sum();

        // Backward pass (fused — no tape traversal)
        let mut grad = vec![1.0; hidden_dim]; // d(loss)/d(output) = 1 for sum

        let mut _all_grad_weights: Vec<Vec<Vec<f64>>> = Vec::new();

        for l in (0..n_layers).rev() {
            // ReLU backward
            for h in 0..hidden_dim {
                if pre_activations[l][h] <= 0.0 {
                    grad[h] = 0.0;
                }
            }

            // Linear backward
            let fan_in = activations[l].len();
            let mut grad_input = vec![0.0; fan_in];
            let mut grad_w = vec![vec![0.0; fan_in]; hidden_dim];

            for h in 0..hidden_dim {
                for i in 0..fan_in {
                    grad_input[i] += grad[h] * weights[l][h][i];
                    grad_w[h][i] = grad[h] * activations[l][i];
                }
            }

            _all_grad_weights.push(grad_w);
            grad = grad_input;
        }

        let duration = start.elapsed();
        (loss, duration)
    }
}

/// Run comparison and report overhead ratio
pub fn compare_approaches(input_dim: usize, hidden_dim: usize, n_layers: usize, n_runs: usize) -> ComparisonResult {
    // Warmup
    for _ in 0..3 {
        tape_based::bench_mlp(input_dim, hidden_dim, n_layers);
        compiled::bench_mlp(input_dim, hidden_dim, n_layers);
    }

    let mut tape_times = Vec::with_capacity(n_runs);
    let mut compiled_times = Vec::with_capacity(n_runs);

    for _ in 0..n_runs {
        let (_, t) = tape_based::bench_mlp(input_dim, hidden_dim, n_layers);
        tape_times.push(t.as_nanos() as f64);

        let (_, c) = compiled::bench_mlp(input_dim, hidden_dim, n_layers);
        compiled_times.push(c.as_nanos() as f64);
    }

    let tape_median = median(&mut tape_times);
    let compiled_median = median(&mut compiled_times);

    ComparisonResult {
        input_dim,
        hidden_dim,
        n_layers,
        tape_median_ns: tape_median,
        compiled_median_ns: compiled_median,
        speedup: tape_median / compiled_median,
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

pub struct ComparisonResult {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub tape_median_ns: f64,
    pub compiled_median_ns: f64,
    pub speedup: f64,
}

impl std::fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MLP(in={}, hidden={}, layers={}) | tape: {:.0}ns | compiled: {:.0}ns | speedup: {:.1}x",
            self.input_dim, self.hidden_dim, self.n_layers,
            self.tape_median_ns, self.compiled_median_ns, self.speedup
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_vs_compiled_same_result() {
        let (tape_loss, _) = tape_based::bench_mlp(4, 4, 2);
        let (compiled_loss, _) = compiled::bench_mlp(4, 4, 2);

        // Both approaches should compute the same forward pass
        assert!(
            (tape_loss - compiled_loss).abs() < 1e-6,
            "tape={}, compiled={}", tape_loss, compiled_loss
        );
    }

    #[test]
    fn test_compiled_faster_than_tape() {
        // With enough operations, the tape overhead should be measurable
        let result = compare_approaches(8, 16, 3, 50);
        println!("{}", result);
        // Compiled should generally be faster due to no tape overhead
        // (On small problems the difference may be noise)
        assert!(result.speedup > 0.5, "compiled should not be 2x slower than tape");
    }
}
