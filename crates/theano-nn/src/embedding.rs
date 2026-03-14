//! Embedding layer.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_types::{Device, Result};
use crate::init;
use crate::module::Module;

/// Embedding layer. Like `torch.nn.Embedding`.
/// Maps integer indices to dense vectors.
pub struct Embedding {
    num_embeddings: usize,
    embedding_dim: usize,
    weight: Variable,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = init::normal_init(&[num_embeddings, embedding_dim], 0.0, 1.0);
        Self {
            num_embeddings,
            embedding_dim,
            weight,
        }
    }

    pub fn num_embeddings(&self) -> usize { self.num_embeddings }
    pub fn embedding_dim(&self) -> usize { self.embedding_dim }

    /// Move this layer to a different device, returning a new Embedding.
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
            weight: self.weight.to(device)?,
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

    /// Reconstruct an Embedding layer from a pre-trained weight tensor.
    pub fn from_tensors(weight: Tensor) -> Self {
        let shape = weight.shape().to_vec();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];
        Self {
            num_embeddings,
            embedding_dim,
            weight: Variable::requires_grad(weight),
        }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Variable) -> Variable {
        // input: integer indices tensor, output: [*input_shape, embedding_dim]
        let input_shape = input.tensor().shape().to_vec();
        let num_indices: usize = input.tensor().numel();

        // Flatten indices to 1D for index_select, which selects rows from weight
        let flat_indices = if input_shape.len() == 1 {
            input.clone()
        } else {
            Variable::new(input.tensor().reshape(&[num_indices]).unwrap())
        };

        // Use index_select through autograd so gradients flow back to self.weight
        let selected = self.weight.index_select(0, &flat_indices).unwrap();

        // Reshape to [*input_shape, embedding_dim]
        let mut out_shape = input_shape;
        out_shape.push(self.embedding_dim);
        selected.reshape(&out_shape).unwrap()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_shape() {
        let emb = Embedding::new(100, 32);
        let input = Variable::new(Tensor::from_slice(&[0.0, 5.0, 10.0], &[3]));
        let output = emb.forward(&input);
        assert_eq!(output.tensor().shape(), &[3, 32]);
    }

    #[test]
    fn test_embedding_batch() {
        let emb = Embedding::new(50, 16);
        let input = Variable::new(Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]));
        let output = emb.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 2, 16]);
    }

    #[test]
    fn test_embedding_params() {
        let emb = Embedding::new(100, 32);
        assert_eq!(emb.parameters().len(), 1);
        assert_eq!(emb.parameters()[0].tensor().shape(), &[100, 32]);
    }

    #[test]
    fn test_embedding_to_device() {
        let emb = Embedding::new(50, 16);
        let emb_gpu = emb.to(&Device::Cuda(0)).unwrap();
        assert_eq!(emb_gpu.parameters()[0].device(), &Device::Cuda(0));

        // Verify metadata preserved
        assert_eq!(emb_gpu.num_embeddings(), 50);
        assert_eq!(emb_gpu.embedding_dim(), 16);

        let emb_cpu = emb_gpu.cpu().unwrap();
        assert_eq!(emb_cpu.parameters()[0].device(), &Device::Cpu);
    }

    #[test]
    fn test_embedding_named_parameters() {
        let emb = Embedding::new(50, 16);
        let named = emb.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    #[test]
    fn test_embedding_state_dict() {
        let emb = Embedding::new(50, 16);
        let sd = emb.state_dict();
        assert!(sd.contains_key("weight"));
        assert_eq!(sd["weight"].shape(), &[50, 16]);
    }
}
