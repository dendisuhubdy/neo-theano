//! Transformer components.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_types::{Device, Result};
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

    /// Reconstruct a MultiheadAttention from pre-trained Linear layers.
    pub fn from_linears(num_heads: usize, q_proj: Linear, k_proj: Linear, v_proj: Linear, out_proj: Linear) -> Self {
        let embed_dim = q_proj.in_features();
        let head_dim = embed_dim / num_heads;
        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        }
    }

    /// Move this layer to a different device, returning a new MultiheadAttention.
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            embed_dim: self.embed_dim,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            q_proj: self.q_proj.to(device)?,
            k_proj: self.k_proj.to(device)?,
            v_proj: self.v_proj.to(device)?,
            out_proj: self.out_proj.to(device)?,
        })
    }

    /// Move to CPU.
    pub fn cpu(&self) -> Result<Self> {
        self.to(&Device::Cpu)
    }

    /// Move to CUDA device 0.
    pub fn cuda(&self) -> Result<Self> {
        self.to(&Device::Cuda(0))
    }

    /// Forward with optional attention mask.
    ///
    /// query, key, value: [batch, seq_len, embed_dim]
    /// mask: optional additive mask [batch, seq_len, seq_len] or broadcastable.
    ///       Added to attention scores before softmax (like PyTorch).
    ///       Use `-inf` values to mask out positions.
    ///
    /// Returns: [batch, seq_len, embed_dim]
    pub fn forward_with_mask(&self, query: &Variable, key: &Variable, value: &Variable, mask: Option<&Variable>) -> Variable {
        let q = apply_linear_3d(&self.q_proj, query);
        let k = apply_linear_3d(&self.k_proj, key);
        let v = apply_linear_3d(&self.v_proj, value);

        // score = Q @ K^T / sqrt(d_k)
        let k_t = k.transpose(-2, -1).unwrap();
        let scale = (self.embed_dim as f64).sqrt();
        let scores = q.matmul(&k_t).unwrap().mul_scalar(1.0 / scale).unwrap();

        // Apply additive mask before softmax
        let masked_scores = match mask {
            Some(m) => scores.add(m).unwrap(),
            None => scores,
        };

        let attn_weights = masked_scores.softmax(-1).unwrap();
        let attn_output = attn_weights.matmul(&v).unwrap();

        apply_linear_3d(&self.out_proj, &attn_output)
    }

    /// Forward: query, key, value all [batch, seq_len, embed_dim]
    /// Returns: [batch, seq_len, embed_dim]
    pub fn forward_qkv(&self, query: &Variable, key: &Variable, value: &Variable) -> Variable {
        self.forward_with_mask(query, key, value, None)
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

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        let mut params = Vec::new();
        for (name, var) in self.q_proj.named_parameters() {
            params.push((format!("q_proj.{name}"), var));
        }
        for (name, var) in self.k_proj.named_parameters() {
            params.push((format!("k_proj.{name}"), var));
        }
        for (name, var) in self.v_proj.named_parameters() {
            params.push((format!("v_proj.{name}"), var));
        }
        for (name, var) in self.out_proj.named_parameters() {
            params.push((format!("out_proj.{name}"), var));
        }
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

    #[test]
    fn test_multihead_attention_to_device() {
        let mha = MultiheadAttention::new(32, 4);
        let mha_gpu = mha.to(&Device::Cuda(0)).unwrap();
        for param in mha_gpu.parameters() {
            assert_eq!(param.device(), &Device::Cuda(0));
        }

        let mha_cpu = mha_gpu.cpu().unwrap();
        for param in mha_cpu.parameters() {
            assert_eq!(param.device(), &Device::Cpu);
        }
    }

    #[test]
    fn test_multihead_attention_named_parameters() {
        let mha = MultiheadAttention::new(32, 4);
        let named = mha.named_parameters();
        assert_eq!(named.len(), 8);
        assert_eq!(named[0].0, "q_proj.weight");
        assert_eq!(named[1].0, "q_proj.bias");
        assert_eq!(named[2].0, "k_proj.weight");
        assert_eq!(named[3].0, "k_proj.bias");
    }

    #[test]
    fn test_multihead_attention_with_mask() {
        let mha = MultiheadAttention::new(16, 4);
        let input = Variable::new(Tensor::ones(&[1, 4, 16]));

        // Create a causal mask (upper triangular = -inf, rest = 0)
        let seq_len = 4;
        let mut mask_data = vec![0.0f64; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f64::NEG_INFINITY;
            }
        }
        let mask = Variable::new(Tensor::from_slice(&mask_data, &[1, seq_len, seq_len]));

        let output_masked = mha.forward_with_mask(&input, &input, &input, Some(&mask));
        assert_eq!(output_masked.tensor().shape(), &[1, 4, 16]);

        // Without mask should also work
        let output_no_mask = mha.forward_with_mask(&input, &input, &input, None);
        assert_eq!(output_no_mask.tensor().shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_multihead_attention_forward_qkv_unchanged() {
        // Verify forward_qkv still works (backward compat)
        let mha = MultiheadAttention::new(16, 4);
        let input = Variable::new(Tensor::ones(&[1, 4, 16]));
        let output = mha.forward_qkv(&input, &input, &input);
        assert_eq!(output.tensor().shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_multihead_attention_state_dict() {
        let mha = MultiheadAttention::new(32, 4);
        let sd = mha.state_dict();
        assert!(sd.contains_key("q_proj.weight"));
        assert!(sd.contains_key("q_proj.bias"));
        assert!(sd.contains_key("out_proj.weight"));
        assert_eq!(sd.len(), 8);
    }
}
