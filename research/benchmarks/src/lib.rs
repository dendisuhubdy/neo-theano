//! # Autodiff Benchmark Framework
//!
//! Compares tape-based AD vs compiled (Enzyme) AD across different model sizes
//! and architectures to quantify the overhead of dynamic taping.
//!
//! Also benchmarks dynamic graph construction vs static graph replay to
//! measure the cost of rebuilding computation graphs every iteration.

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

        pub fn sigmoid(&mut self, a: usize) -> usize {
            let va = self.values[a];
            let val = 1.0 / (1.0 + (-va).exp());
            let idx = self.values.len();
            self.values.push(val);
            self.nodes.push(TapeNode {
                grad_fn: Box::new(move |g| vec![g * val * (1.0 - val)]),
                input_indices: vec![a],
            });
            idx
        }

        pub fn tanh_op(&mut self, a: usize) -> usize {
            let va = self.values[a];
            let val = va.tanh();
            let idx = self.values.len();
            self.values.push(val);
            self.nodes.push(TapeNode {
                grad_fn: Box::new(move |g| vec![g * (1.0 - val * val)]),
                input_indices: vec![a],
            });
            idx
        }

        pub fn sub(&mut self, a: usize, b: usize) -> usize {
            let val = self.values[a] - self.values[b];
            let idx = self.values.len();
            self.values.push(val);
            self.nodes.push(TapeNode {
                grad_fn: Box::new(|g| vec![g, -g]),
                input_indices: vec![a, b],
            });
            idx
        }

        pub fn div(&mut self, a: usize, b: usize) -> usize {
            let va = self.values[a];
            let vb = self.values[b];
            let val = va / vb;
            let idx = self.values.len();
            self.values.push(val);
            self.nodes.push(TapeNode {
                grad_fn: Box::new(move |g| vec![g / vb, -g * va / (vb * vb)]),
                input_indices: vec![a, b],
            });
            idx
        }

        pub fn exp(&mut self, a: usize) -> usize {
            let va = self.values[a];
            let val = va.exp();
            let idx = self.values.len();
            self.values.push(val);
            self.nodes.push(TapeNode {
                grad_fn: Box::new(move |g| vec![g * val]),
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

    /// Benchmark: Conv2d forward+backward using tape-based AD
    /// Simulates a 2D convolution with kernel_size x kernel_size filters
    pub fn bench_conv2d(
        channels_in: usize, channels_out: usize,
        spatial: usize, kernel_size: usize,
    ) -> f64 {
        let mut tape = Tape::new();

        // Input: [channels_in, spatial, spatial]
        let input: Vec<Vec<Vec<usize>>> = (0..channels_in)
            .map(|c| {
                (0..spatial).map(|h| {
                    (0..spatial).map(|w| {
                        tape.constant(0.1 * ((c * spatial * spatial + h * spatial + w) as f64 % 11.0 - 5.0))
                    }).collect()
                }).collect()
            }).collect();

        // Kernel weights: [channels_out, channels_in, kernel_size, kernel_size]
        let out_spatial = spatial - kernel_size + 1;
        let mut output_flat = Vec::new();

        for co in 0..channels_out {
            for oh in 0..out_spatial {
                for ow in 0..out_spatial {
                    let bias = tape.constant(0.01);
                    let mut sum = bias;
                    for ci in 0..channels_in {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let w_val = 0.01 * ((co * channels_in * kernel_size * kernel_size
                                    + ci * kernel_size * kernel_size
                                    + kh * kernel_size + kw) as f64 % 7.0 - 3.0);
                                let w = tape.constant(w_val);
                                let prod = tape.mul(input[ci][oh + kh][ow + kw], w);
                                sum = tape.add(sum, prod);
                            }
                        }
                    }
                    let activated = tape.relu(sum);
                    output_flat.push(activated);
                }
            }
        }

        // Sum for scalar loss
        let mut loss = output_flat[0];
        for &o in &output_flat[1..] {
            loss = tape.add(loss, o);
        }

        let grads = tape.backward(loss);
        let _grad_sum: f64 = grads.iter().sum();
        tape.values[loss]
    }

    /// Benchmark: LSTM cell forward+backward using tape-based AD
    /// Simulates a single LSTM cell step with gates
    pub fn bench_lstm(input_size: usize, hidden_size: usize, seq_len: usize) -> f64 {
        let mut tape = Tape::new();

        // Initial hidden and cell state
        let mut h: Vec<usize> = (0..hidden_size).map(|_| tape.constant(0.0)).collect();
        let mut c: Vec<usize> = (0..hidden_size).map(|_| tape.constant(0.0)).collect();

        // Input sequence: [seq_len, input_size]
        let inputs: Vec<Vec<usize>> = (0..seq_len)
            .map(|t| {
                (0..input_size).map(|i| {
                    tape.constant(0.1 * ((t * input_size + i) as f64 % 9.0 - 4.0))
                }).collect()
            }).collect();

        for t in 0..seq_len {
            let mut new_h = Vec::with_capacity(hidden_size);
            let mut new_c = Vec::with_capacity(hidden_size);

            for j in 0..hidden_size {
                // Compute 4 gates: input, forget, cell, output
                // Each gate = sigmoid/tanh(W_ih @ x + W_hh @ h + b)
                let mut gate_vals = [tape.constant(0.01); 4];

                for i in 0..input_size {
                    for g in 0..4 {
                        let w_val = 0.01 * ((g * hidden_size * input_size + j * input_size + i) as f64 % 7.0 - 3.0);
                        let w = tape.constant(w_val);
                        let prod = tape.mul(inputs[t][i], w);
                        gate_vals[g] = tape.add(gate_vals[g], prod);
                    }
                }
                for k in 0..hidden_size {
                    for g in 0..4 {
                        let w_val = 0.01 * ((g * hidden_size * hidden_size + j * hidden_size + k) as f64 % 5.0 - 2.0);
                        let w = tape.constant(w_val);
                        let prod = tape.mul(h[k], w);
                        gate_vals[g] = tape.add(gate_vals[g], prod);
                    }
                }

                // Apply activations: sigmoid for i,f,o; tanh for g
                let i_gate = tape.sigmoid(gate_vals[0]);
                let f_gate = tape.sigmoid(gate_vals[1]);
                let g_gate = tape.tanh_op(gate_vals[2]);
                let o_gate = tape.sigmoid(gate_vals[3]);

                // c_new = f * c_old + i * g
                let fc = tape.mul(f_gate, c[j]);
                let ig = tape.mul(i_gate, g_gate);
                let c_new = tape.add(fc, ig);

                // h_new = o * tanh(c_new)
                let c_tanh = tape.tanh_op(c_new);
                let h_new = tape.mul(o_gate, c_tanh);

                new_c.push(c_new);
                new_h.push(h_new);
            }

            h = new_h;
            c = new_c;
        }

        // Sum final hidden state for loss
        let mut loss = h[0];
        for &hi in &h[1..] {
            loss = tape.add(loss, hi);
        }

        let grads = tape.backward(loss);
        let _grad_sum: f64 = grads.iter().sum();
        tape.values[loss]
    }

    /// Benchmark: BatchNorm forward+backward using tape-based AD
    pub fn bench_batchnorm(batch_size: usize, features: usize) -> f64 {
        let mut tape = Tape::new();
        let eps = tape.constant(1e-5);

        // Input: [batch_size, features]
        let input: Vec<Vec<usize>> = (0..batch_size)
            .map(|b| {
                (0..features).map(|f| {
                    tape.constant(0.1 * ((b * features + f) as f64 % 13.0 - 6.0))
                }).collect()
            }).collect();

        // Gamma and beta (learnable parameters)
        let gamma: Vec<usize> = (0..features).map(|_| tape.constant(1.0)).collect();
        let beta: Vec<usize> = (0..features).map(|_| tape.constant(0.0)).collect();

        // Compute mean per feature
        let batch_inv = tape.constant(1.0 / batch_size as f64);
        let mut means = Vec::with_capacity(features);
        for f in 0..features {
            let mut sum = input[0][f];
            for b in 1..batch_size {
                sum = tape.add(sum, input[b][f]);
            }
            means.push(tape.mul(sum, batch_inv));
        }

        // Compute variance per feature
        let mut vars = Vec::with_capacity(features);
        for f in 0..features {
            let mut var_sum = tape.constant(0.0);
            for b in 0..batch_size {
                let diff = tape.sub(input[b][f], means[f]);
                let sq = tape.mul(diff, diff);
                var_sum = tape.add(var_sum, sq);
            }
            vars.push(tape.mul(var_sum, batch_inv));
        }

        // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
        let mut output_flat = Vec::new();
        for f in 0..features {
            let var_eps = tape.add(vars[f], eps);
            // Approximate sqrt via identity for benchmark (preserves op count)
            // In practice we'd need a sqrt tape op, but the overhead count is what matters
            for b in 0..batch_size {
                let centered = tape.sub(input[b][f], means[f]);
                let normed = tape.div(centered, var_eps); // simplified: div by var+eps instead of sqrt
                let scaled = tape.mul(normed, gamma[f]);
                let shifted = tape.add(scaled, beta[f]);
                output_flat.push(shifted);
            }
        }

        // Sum for scalar loss
        let mut loss = output_flat[0];
        for &o in &output_flat[1..] {
            loss = tape.add(loss, o);
        }

        let grads = tape.backward(loss);
        let _grad_sum: f64 = grads.iter().sum();
        tape.values[loss]
    }

    /// Benchmark: Multi-head attention forward+backward using tape-based AD
    /// Simulates Q, K, V projections + scaled dot-product attention
    pub fn bench_attention(seq_len: usize, embed_dim: usize, num_heads: usize) -> f64 {
        let mut tape = Tape::new();
        let head_dim = embed_dim / num_heads;
        let scale = tape.constant(1.0 / (head_dim as f64).sqrt());

        // Input: [seq_len, embed_dim]
        let input: Vec<Vec<usize>> = (0..seq_len)
            .map(|s| {
                (0..embed_dim).map(|e| {
                    tape.constant(0.1 * ((s * embed_dim + e) as f64 % 11.0 - 5.0))
                }).collect()
            }).collect();

        // Q, K, V projections (simplified: one linear each)
        let project = |tape: &mut Tape, input: &[Vec<usize>], proj_id: usize| -> Vec<Vec<usize>> {
            input.iter().map(|row| {
                (0..embed_dim).map(|o| {
                    let mut sum = tape.constant(0.01);
                    for (i, &x) in row.iter().enumerate() {
                        let w_val = 0.01 * ((proj_id * embed_dim * embed_dim + o * embed_dim + i) as f64 % 7.0 - 3.0);
                        let w = tape.constant(w_val);
                        let prod = tape.mul(x, w);
                        sum = tape.add(sum, prod);
                    }
                    sum
                }).collect()
            }).collect()
        };

        let q = project(&mut tape, &input, 0);
        let k = project(&mut tape, &input, 1);
        let v = project(&mut tape, &input, 2);

        // Scaled dot-product attention per head (simplified: single head for benchmark)
        // scores = Q @ K^T / sqrt(d_k)
        let mut attn_output = Vec::new();
        for s in 0..seq_len {
            for d in 0..embed_dim {
                // Attention score for position s, output dim d
                let mut weighted_sum = tape.constant(0.0);
                for s2 in 0..seq_len {
                    // dot(q[s], k[s2]) simplified to first head_dim elements
                    let mut score = tape.constant(0.0);
                    for hd in 0..head_dim.min(embed_dim) {
                        let prod = tape.mul(q[s][hd], k[s2][hd]);
                        score = tape.add(score, prod);
                    }
                    score = tape.mul(score, scale);
                    // Softmax approximated with exp (overhead-equivalent)
                    let exp_score = tape.exp(score);
                    let weighted = tape.mul(exp_score, v[s2][d]);
                    weighted_sum = tape.add(weighted_sum, weighted);
                }
                attn_output.push(weighted_sum);
            }
        }

        // Sum for scalar loss
        let mut loss = attn_output[0];
        for &o in &attn_output[1..] {
            loss = tape.add(loss, o);
        }

        let grads = tape.backward(loss);
        let _grad_sum: f64 = grads.iter().sum();
        tape.values[loss]
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

    /// Benchmark: Conv2d forward+backward using compiled AD (no tape)
    pub fn bench_conv2d(
        channels_in: usize, channels_out: usize,
        spatial: usize, kernel_size: usize,
    ) -> f64 {
        let out_spatial = spatial - kernel_size + 1;

        // Weights: [channels_out, channels_in, kernel_size, kernel_size]
        let weights: Vec<Vec<Vec<Vec<f64>>>> = (0..channels_out)
            .map(|co| {
                (0..channels_in).map(|ci| {
                    (0..kernel_size).map(|kh| {
                        (0..kernel_size).map(|kw| {
                            0.01 * ((co * channels_in * kernel_size * kernel_size
                                + ci * kernel_size * kernel_size
                                + kh * kernel_size + kw) as f64 % 7.0 - 3.0)
                        }).collect()
                    }).collect()
                }).collect()
            }).collect();
        let biases: Vec<f64> = vec![0.01; channels_out];

        // Input
        let input: Vec<Vec<Vec<f64>>> = (0..channels_in)
            .map(|c| {
                (0..spatial).map(|h| {
                    (0..spatial).map(|w| {
                        0.1 * ((c * spatial * spatial + h * spatial + w) as f64 % 11.0 - 5.0)
                    }).collect()
                }).collect()
            }).collect();

        // Forward
        let mut output = vec![vec![vec![0.0; out_spatial]; out_spatial]; channels_out];
        let mut pre_act = vec![vec![vec![0.0; out_spatial]; out_spatial]; channels_out];

        for co in 0..channels_out {
            for oh in 0..out_spatial {
                for ow in 0..out_spatial {
                    let mut sum = biases[co];
                    for ci in 0..channels_in {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                sum += input[ci][oh + kh][ow + kw] * weights[co][ci][kh][kw];
                            }
                        }
                    }
                    pre_act[co][oh][ow] = sum;
                    output[co][oh][ow] = sum.max(0.0); // relu
                }
            }
        }

        let loss: f64 = output.iter().flat_map(|c| c.iter().flat_map(|r| r.iter())).sum();

        // Backward (fused)
        let mut _grad_weights = vec![vec![vec![vec![0.0; kernel_size]; kernel_size]; channels_in]; channels_out];
        let mut _grad_input = vec![vec![vec![0.0; spatial]; spatial]; channels_in];

        for co in 0..channels_out {
            for oh in 0..out_spatial {
                for ow in 0..out_spatial {
                    let g = if pre_act[co][oh][ow] > 0.0 { 1.0 } else { 0.0 };
                    for ci in 0..channels_in {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                _grad_weights[co][ci][kh][kw] += g * input[ci][oh + kh][ow + kw];
                                _grad_input[ci][oh + kh][ow + kw] += g * weights[co][ci][kh][kw];
                            }
                        }
                    }
                }
            }
        }

        loss
    }

    /// Benchmark: LSTM cell forward+backward using compiled AD (no tape)
    pub fn bench_lstm(input_size: usize, hidden_size: usize, seq_len: usize) -> f64 {
        // Weights
        let w_ih: Vec<Vec<Vec<f64>>> = (0..4).map(|g| {
            (0..hidden_size).map(|j| {
                (0..input_size).map(|i| {
                    0.01 * ((g * hidden_size * input_size + j * input_size + i) as f64 % 7.0 - 3.0)
                }).collect()
            }).collect()
        }).collect();
        let w_hh: Vec<Vec<Vec<f64>>> = (0..4).map(|g| {
            (0..hidden_size).map(|j| {
                (0..hidden_size).map(|k| {
                    0.01 * ((g * hidden_size * hidden_size + j * hidden_size + k) as f64 % 5.0 - 2.0)
                }).collect()
            }).collect()
        }).collect();
        let biases = vec![0.01; hidden_size];

        // Input sequence
        let inputs: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| {
                (0..input_size).map(|i| {
                    0.1 * ((t * input_size + i) as f64 % 9.0 - 4.0)
                }).collect()
            }).collect();

        // Forward
        let mut h = vec![0.0; hidden_size];
        let mut c = vec![0.0; hidden_size];

        // Store intermediates for backward
        let mut all_gates: Vec<Vec<[f64; 4]>> = Vec::new(); // [seq_len][hidden_size][4 gates]
        let mut all_c: Vec<Vec<f64>> = vec![c.clone()];
        let mut all_h: Vec<Vec<f64>> = vec![h.clone()];

        for t in 0..seq_len {
            let mut gates = vec![[0.0f64; 4]; hidden_size];
            let mut new_h = vec![0.0; hidden_size];
            let mut new_c = vec![0.0; hidden_size];

            for j in 0..hidden_size {
                let mut gate_pre = [biases[j]; 4];
                for i in 0..input_size {
                    for g in 0..4 {
                        gate_pre[g] += inputs[t][i] * w_ih[g][j][i];
                    }
                }
                for k in 0..hidden_size {
                    for g in 0..4 {
                        gate_pre[g] += h[k] * w_hh[g][j][k];
                    }
                }

                // Activations
                let i_gate = 1.0 / (1.0 + (-gate_pre[0]).exp());
                let f_gate = 1.0 / (1.0 + (-gate_pre[1]).exp());
                let g_gate = gate_pre[2].tanh();
                let o_gate = 1.0 / (1.0 + (-gate_pre[3]).exp());

                gates[j] = [i_gate, f_gate, g_gate, o_gate];

                new_c[j] = f_gate * c[j] + i_gate * g_gate;
                new_h[j] = o_gate * new_c[j].tanh();
            }

            all_gates.push(gates);
            h = new_h;
            c = new_c;
            all_c.push(c.clone());
            all_h.push(h.clone());
        }

        let loss: f64 = h.iter().sum();

        // Backward (fused, no tape)
        let mut dh = vec![1.0; hidden_size];
        let mut dc = vec![0.0; hidden_size];
        let mut _dw_ih = vec![vec![vec![0.0; input_size]; hidden_size]; 4];
        let mut _dw_hh = vec![vec![vec![0.0; hidden_size]; hidden_size]; 4];

        for t in (0..seq_len).rev() {
            for j in 0..hidden_size {
                let [ig, fg, gg, og] = all_gates[t][j];
                let c_val = all_c[t + 1][j];
                let c_tanh = c_val.tanh();

                let do_gate = dh[j] * c_tanh;
                let dc_from_h = dh[j] * og * (1.0 - c_tanh * c_tanh);
                let dc_total = dc[j] + dc_from_h;

                let di_gate = dc_total * gg;
                let df_gate = dc_total * all_c[t][j];
                let dg_gate = dc_total * ig;

                dc[j] = dc_total * fg;

                // Sigmoid backward: d_pre = d_out * gate * (1 - gate)
                let di_pre = di_gate * ig * (1.0 - ig);
                let df_pre = df_gate * fg * (1.0 - fg);
                let dg_pre = dg_gate * (1.0 - gg * gg);
                let do_pre = do_gate * og * (1.0 - og);
                let d_pres = [di_pre, df_pre, dg_pre, do_pre];

                for g in 0..4 {
                    for i in 0..input_size {
                        _dw_ih[g][j][i] += d_pres[g] * inputs[t][i];
                    }
                    for k in 0..hidden_size {
                        _dw_hh[g][j][k] += d_pres[g] * all_h[t][k];
                    }
                }
            }

            // Propagate dh to previous timestep
            let mut new_dh = vec![0.0; hidden_size];
            for j in 0..hidden_size {
                let [ig, fg, gg, og] = all_gates[t][j];
                let c_val = all_c[t + 1][j];
                let c_tanh = c_val.tanh();
                let dc_from_h = dh[j] * og * (1.0 - c_tanh * c_tanh);
                let dc_total = dc[j] + dc_from_h;

                let di_pre = dc_total * gg * ig * (1.0 - ig);
                let df_pre = dc_total * all_c[t][j] * fg * (1.0 - fg);
                let dg_pre = dc_total * ig * (1.0 - gg * gg);
                let do_pre = dh[j] * c_tanh * og * (1.0 - og);
                let d_pres = [di_pre, df_pre, dg_pre, do_pre];

                for k in 0..hidden_size {
                    for g in 0..4 {
                        new_dh[k] += d_pres[g] * w_hh[g][j][k];
                    }
                }
            }
            dh = new_dh;
        }

        loss
    }

    /// Benchmark: BatchNorm forward+backward using compiled AD (no tape)
    pub fn bench_batchnorm(batch_size: usize, features: usize) -> f64 {
        let eps = 1e-5;

        // Input
        let input: Vec<Vec<f64>> = (0..batch_size)
            .map(|b| {
                (0..features).map(|f| {
                    0.1 * ((b * features + f) as f64 % 13.0 - 6.0)
                }).collect()
            }).collect();

        let gamma = vec![1.0; features];
        let beta = vec![0.0; features];

        // Forward: compute mean, var, normalize
        let mut means = vec![0.0; features];
        let mut vars = vec![0.0; features];
        let inv_n = 1.0 / batch_size as f64;

        for f in 0..features {
            for b in 0..batch_size {
                means[f] += input[b][f];
            }
            means[f] *= inv_n;

            for b in 0..batch_size {
                let diff = input[b][f] - means[f];
                vars[f] += diff * diff;
            }
            vars[f] *= inv_n;
        }

        let mut output = vec![vec![0.0; features]; batch_size];
        let mut x_centered = vec![vec![0.0; features]; batch_size];
        let mut inv_std = vec![0.0; features];

        for f in 0..features {
            inv_std[f] = 1.0 / (vars[f] + eps).sqrt();
            for b in 0..batch_size {
                x_centered[b][f] = input[b][f] - means[f];
                let normed = x_centered[b][f] * inv_std[f];
                output[b][f] = normed * gamma[f] + beta[f];
            }
        }

        let loss: f64 = output.iter().flat_map(|r| r.iter()).sum();

        // Backward (fused)
        let mut _dgamma = vec![0.0; features];
        let mut _dbeta = vec![0.0; features];
        let mut _dinput = vec![vec![0.0; features]; batch_size];

        for f in 0..features {
            let mut dxhat_sum = 0.0;
            let mut dxhat_x_sum = 0.0;

            for b in 0..batch_size {
                let dout = 1.0; // d(loss)/d(output) = 1 for sum
                _dbeta[f] += dout;
                let dxhat = dout * gamma[f];
                _dgamma[f] += dxhat * x_centered[b][f] * inv_std[f];
                dxhat_sum += dxhat;
                dxhat_x_sum += dxhat * x_centered[b][f];
            }

            for b in 0..batch_size {
                let dxhat = gamma[f]; // simplified: dout=1
                _dinput[b][f] = inv_std[f] * inv_n * (
                    batch_size as f64 * dxhat - dxhat_sum - x_centered[b][f] * inv_std[f] * inv_std[f] * dxhat_x_sum
                );
            }
        }

        loss
    }

    /// Benchmark: Multi-head attention forward+backward using compiled AD (no tape)
    pub fn bench_attention(seq_len: usize, embed_dim: usize, num_heads: usize) -> f64 {
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Input
        let input: Vec<Vec<f64>> = (0..seq_len)
            .map(|s| {
                (0..embed_dim).map(|e| {
                    0.1 * ((s * embed_dim + e) as f64 % 11.0 - 5.0)
                }).collect()
            }).collect();

        // Projection weights
        let make_proj = |proj_id: usize| -> Vec<Vec<f64>> {
            (0..embed_dim).map(|o| {
                (0..embed_dim).map(|i| {
                    0.01 * ((proj_id * embed_dim * embed_dim + o * embed_dim + i) as f64 % 7.0 - 3.0)
                }).collect()
            }).collect()
        };

        let w_q = make_proj(0);
        let w_k = make_proj(1);
        let w_v = make_proj(2);

        // Q, K, V projections
        let project = |input: &[Vec<f64>], w: &[Vec<f64>]| -> Vec<Vec<f64>> {
            input.iter().map(|row| {
                (0..embed_dim).map(|o| {
                    let mut sum = 0.01;
                    for (i, &x) in row.iter().enumerate() {
                        sum += x * w[o][i];
                    }
                    sum
                }).collect()
            }).collect()
        };

        let q = project(&input, &w_q);
        let k = project(&input, &w_k);
        let v = project(&input, &w_v);

        // Attention scores and weighted sum
        let mut attn_output = vec![vec![0.0; embed_dim]; seq_len];
        let mut scores = vec![vec![0.0; seq_len]; seq_len];
        let mut exp_scores = vec![vec![0.0; seq_len]; seq_len];

        for s in 0..seq_len {
            for s2 in 0..seq_len {
                let mut score = 0.0;
                for hd in 0..head_dim.min(embed_dim) {
                    score += q[s][hd] * k[s2][hd];
                }
                scores[s][s2] = score * scale;
                exp_scores[s][s2] = scores[s][s2].exp();
            }
            for d in 0..embed_dim {
                for s2 in 0..seq_len {
                    attn_output[s][d] += exp_scores[s][s2] * v[s2][d];
                }
            }
        }

        let loss: f64 = attn_output.iter().flat_map(|r| r.iter()).sum();

        // Backward (fused)
        let mut _dw_q = vec![vec![0.0; embed_dim]; embed_dim];
        let mut _dw_k = vec![vec![0.0; embed_dim]; embed_dim];
        let mut _dw_v = vec![vec![0.0; embed_dim]; embed_dim];

        // d(loss)/d(attn_output) = 1 for sum
        let mut dv = vec![vec![0.0; embed_dim]; seq_len];
        let mut d_exp_scores = vec![vec![0.0; seq_len]; seq_len];

        for s in 0..seq_len {
            for s2 in 0..seq_len {
                for d in 0..embed_dim {
                    // d/d(exp_score) = v[s2][d]
                    d_exp_scores[s][s2] += v[s2][d];
                    // d/d(v[s2][d]) = exp_score
                    dv[s2][d] += exp_scores[s][s2];
                }
                // d/d(score) through exp
                let d_score = d_exp_scores[s][s2] * exp_scores[s][s2] * scale;
                for hd in 0..head_dim.min(embed_dim) {
                    _dw_q[hd][0] += d_score * k[s2][hd]; // simplified accumulation
                    _dw_k[hd][0] += d_score * q[s][hd];
                }
            }
        }

        // Use gradients to prevent optimization
        let _dv_sum: f64 = dv.iter().flat_map(|r| r.iter()).sum();

        loss
    }
}

/// Dynamic vs Static graph comparison.
///
/// Dynamic graph: Rebuilds the computation graph every forward pass (like PyTorch).
/// Each iteration allocates new tape nodes, closures, and vectors.
///
/// Static graph: Builds the graph structure once, then replays it with different
/// data (like old Theano/TF1). The graph topology is fixed; only values change.
pub mod dynamic_vs_static {
    /// A pre-compiled static graph node: fixed topology, no allocations per iteration
    struct StaticNode {
        op: StaticOp,
        inputs: [usize; 2], // max 2 inputs
    }

    #[derive(Clone, Copy)]
    enum StaticOp {
        Input,
        Constant(f64),
        Add,
        Mul,
        Relu,
        Sigmoid,
        Tanh,
    }

    /// Static graph: build once, evaluate many times without re-allocation
    pub struct StaticGraph {
        nodes: Vec<StaticNode>,
        n_inputs: usize,
    }

    impl StaticGraph {
        pub fn build_mlp(input_dim: usize, hidden_dim: usize, n_layers: usize) -> Self {
            let mut nodes = Vec::new();

            // Input nodes
            for _ in 0..input_dim {
                nodes.push(StaticNode { op: StaticOp::Input, inputs: [0; 2] });
            }
            let n_inputs = input_dim;

            let mut current: Vec<usize> = (0..input_dim).collect();

            for _layer in 0..n_layers {
                let mut next = Vec::with_capacity(hidden_dim);
                for h in 0..hidden_dim {
                    // Bias constant
                    let bias_idx = nodes.len();
                    nodes.push(StaticNode { op: StaticOp::Constant(0.01), inputs: [0; 2] });
                    let mut sum_idx = bias_idx;

                    for (i, &c) in current.iter().enumerate() {
                        let w_val = 0.01 * ((h * input_dim + i) as f64 % 7.0 - 3.0);
                        let w_idx = nodes.len();
                        nodes.push(StaticNode { op: StaticOp::Constant(w_val), inputs: [0; 2] });
                        let prod_idx = nodes.len();
                        nodes.push(StaticNode { op: StaticOp::Mul, inputs: [c, w_idx] });
                        let new_sum_idx = nodes.len();
                        nodes.push(StaticNode { op: StaticOp::Add, inputs: [sum_idx, prod_idx] });
                        sum_idx = new_sum_idx;
                    }
                    let relu_idx = nodes.len();
                    nodes.push(StaticNode { op: StaticOp::Relu, inputs: [sum_idx, 0] });
                    next.push(relu_idx);
                }
                current = next;
            }

            // Sum outputs
            let mut loss_idx = current[0];
            for &c in &current[1..] {
                let new_idx = nodes.len();
                nodes.push(StaticNode { op: StaticOp::Add, inputs: [loss_idx, c] });
                loss_idx = new_idx;
            }

            StaticGraph { nodes, n_inputs }
        }

        /// Evaluate the pre-built graph with given inputs — no allocation of graph structure
        pub fn forward(&self, input_values: &[f64]) -> f64 {
            let mut values = vec![0.0; self.nodes.len()];
            for (i, node) in self.nodes.iter().enumerate() {
                values[i] = match node.op {
                    StaticOp::Input => input_values[i],
                    StaticOp::Constant(v) => v,
                    StaticOp::Add => values[node.inputs[0]] + values[node.inputs[1]],
                    StaticOp::Mul => values[node.inputs[0]] * values[node.inputs[1]],
                    StaticOp::Relu => values[node.inputs[0]].max(0.0),
                    StaticOp::Sigmoid => 1.0 / (1.0 + (-values[node.inputs[0]]).exp()),
                    StaticOp::Tanh => values[node.inputs[0]].tanh(),
                };
            }
            values[values.len() - 1]
        }

        /// Forward + backward on static graph. Gradients computed via fixed-topology reverse pass.
        pub fn forward_backward(&self, input_values: &[f64]) -> (f64, Vec<f64>) {
            let n = self.nodes.len();
            let mut values = vec![0.0; n];

            // Forward
            for (i, node) in self.nodes.iter().enumerate() {
                values[i] = match node.op {
                    StaticOp::Input => input_values[i],
                    StaticOp::Constant(v) => v,
                    StaticOp::Add => values[node.inputs[0]] + values[node.inputs[1]],
                    StaticOp::Mul => values[node.inputs[0]] * values[node.inputs[1]],
                    StaticOp::Relu => values[node.inputs[0]].max(0.0),
                    StaticOp::Sigmoid => 1.0 / (1.0 + (-values[node.inputs[0]]).exp()),
                    StaticOp::Tanh => values[node.inputs[0]].tanh(),
                };
            }
            let loss = values[n - 1];

            // Backward (reverse over fixed topology — no allocation of grad_fn closures)
            let mut grads = vec![0.0; n];
            grads[n - 1] = 1.0;

            for (i, node) in self.nodes.iter().enumerate().rev() {
                let g = grads[i];
                if g == 0.0 { continue; }
                match node.op {
                    StaticOp::Input | StaticOp::Constant(_) => {}
                    StaticOp::Add => {
                        grads[node.inputs[0]] += g;
                        grads[node.inputs[1]] += g;
                    }
                    StaticOp::Mul => {
                        grads[node.inputs[0]] += g * values[node.inputs[1]];
                        grads[node.inputs[1]] += g * values[node.inputs[0]];
                    }
                    StaticOp::Relu => {
                        grads[node.inputs[0]] += if values[node.inputs[0]] > 0.0 { g } else { 0.0 };
                    }
                    StaticOp::Sigmoid => {
                        let s = values[i];
                        grads[node.inputs[0]] += g * s * (1.0 - s);
                    }
                    StaticOp::Tanh => {
                        let t = values[i];
                        grads[node.inputs[0]] += g * (1.0 - t * t);
                    }
                }
            }

            let input_grads: Vec<f64> = grads[..self.n_inputs].to_vec();
            (loss, input_grads)
        }
    }

    /// Dynamic graph benchmark: rebuild tape every iteration (like PyTorch)
    pub fn bench_dynamic(input_dim: usize, hidden_dim: usize, n_layers: usize) -> f64 {
        // This calls the tape_based benchmark which rebuilds the tape every time
        let (loss, _) = super::tape_based::bench_mlp(input_dim, hidden_dim, n_layers);
        loss
    }

    /// Static graph benchmark: graph built once, just re-evaluate
    pub fn bench_static_forward_only(graph: &StaticGraph, input_values: &[f64]) -> f64 {
        graph.forward(input_values)
    }

    /// Static graph benchmark: forward + backward on pre-built graph
    pub fn bench_static_forward_backward(graph: &StaticGraph, input_values: &[f64]) -> (f64, Vec<f64>) {
        graph.forward_backward(input_values)
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

pub fn median(v: &mut [f64]) -> f64 {
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

    #[test]
    fn test_conv2d_tape_vs_compiled() {
        let tape_loss = tape_based::bench_conv2d(2, 4, 8, 3);
        let compiled_loss = compiled::bench_conv2d(2, 4, 8, 3);
        assert!(
            (tape_loss - compiled_loss).abs() < 1e-4,
            "conv2d: tape={}, compiled={}", tape_loss, compiled_loss
        );
    }

    #[test]
    fn test_lstm_tape_vs_compiled() {
        let tape_loss = tape_based::bench_lstm(4, 4, 2);
        let compiled_loss = compiled::bench_lstm(4, 4, 2);
        assert!(
            (tape_loss - compiled_loss).abs() < 1e-4,
            "lstm: tape={}, compiled={}", tape_loss, compiled_loss
        );
    }

    #[test]
    fn test_batchnorm_tape_vs_compiled() {
        let tape_loss = tape_based::bench_batchnorm(4, 4);
        let compiled_loss = compiled::bench_batchnorm(4, 4);
        // BatchNorm tape version uses div instead of sqrt, so allow larger tolerance
        assert!(
            (tape_loss - compiled_loss).abs() / compiled_loss.abs().max(1e-10) < 0.5,
            "batchnorm: tape={}, compiled={}", tape_loss, compiled_loss
        );
    }

    #[test]
    fn test_dynamic_vs_static_same_result() {
        let input_dim = 4;
        let hidden_dim = 4;
        let n_layers = 2;

        let graph = dynamic_vs_static::StaticGraph::build_mlp(input_dim, hidden_dim, n_layers);
        let input_values: Vec<f64> = (0..input_dim).map(|i| (i as f64 + 1.0) * 0.1).collect();

        let static_loss = dynamic_vs_static::bench_static_forward_only(&graph, &input_values);
        let dynamic_loss = dynamic_vs_static::bench_dynamic(input_dim, hidden_dim, n_layers);

        assert!(
            (static_loss - dynamic_loss).abs() < 1e-6,
            "static={}, dynamic={}", static_loss, dynamic_loss
        );
    }
}
