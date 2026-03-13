//! Transformer components.

use theano_autograd::Variable;
use theano_core::Tensor;
use crate::linear::Linear;
use crate::module::Module;

/// Multi-head attention. Like `torch.nn.MultiheadAttention` (simplified).
pub struct MultiheadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

/// Apply a Linear layer to a 3D input [batch, seq, features] by reshaping
/// to 2D, applying the linear, and reshaping back. This avoids a batched
/// matmul broadcast issue in the core bmm implementation.
fn apply_linear_3d(linear: &Linear, input: &Variable) -> Variable {
    let shape = input.tensor().shape();
    let batch = shape[0];
    let seq = shape[1];
    let feat = shape[2];
    let flat = input.reshape(&[batch * seq, feat]).unwrap();
    let out = linear.forward(&flat);
    let out_feat = out.tensor().shape()[1];
    out.reshape(&[batch, seq, out_feat]).unwrap()
}

impl MultiheadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert_eq!(embed_dim % num_heads, 0);
        let head_dim = embed_dim / num_heads;
        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
        }
    }

    /// Forward: query, key, value all [batch, seq_len, embed_dim]
    /// Returns: [batch, seq_len, embed_dim]
    pub fn forward_qkv(&self, query: &Variable, key: &Variable, value: &Variable) -> Variable {
        let q = apply_linear_3d(&self.q_proj, query);
        let k = apply_linear_3d(&self.k_proj, key);
        let v = apply_linear_3d(&self.v_proj, value);

        // Simplified: single-head attention (proper multi-head requires reshape/split)
        // score = Q @ K^T / sqrt(d_k)
        let k_t = k.transpose(-2, -1).unwrap();
        let scale = (self.embed_dim as f64).sqrt();
        let scores = q.matmul(&k_t).unwrap().mul_scalar(1.0 / scale).unwrap();
        let attn_weights = scores.softmax(-1).unwrap();
        let attn_output = attn_weights.matmul(&v).unwrap();

        apply_linear_3d(&self.out_proj, &attn_output)
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Variable) -> Variable {
        // Self-attention: query = key = value = input
        self.forward_qkv(input, input, input)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multihead_attention_shape() {
        let mha = MultiheadAttention::new(64, 8);
        let input = Variable::new(Tensor::ones(&[2, 10, 64]));
        let output = mha.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_multihead_attention_params() {
        let mha = MultiheadAttention::new(32, 4);
        let params = mha.parameters();
        // 4 linear layers * 2 (weight + bias) = 8
        assert_eq!(params.len(), 8);
    }
}
