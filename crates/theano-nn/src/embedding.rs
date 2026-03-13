//! Embedding layer.

use theano_autograd::Variable;
use theano_core::Tensor;
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
}

impl Module for Embedding {
    fn forward(&self, input: &Variable) -> Variable {
        // input: integer indices tensor, output: [*input_shape, embedding_dim]
        let indices = input.tensor().to_vec_f64().unwrap();
        let weight_data = self.weight.tensor().to_vec_f64().unwrap();
        let input_shape = input.tensor().shape().to_vec();

        let num_indices: usize = indices.len();
        let mut output = vec![0.0f64; num_indices * self.embedding_dim];

        for (i, &idx) in indices.iter().enumerate() {
            let idx = idx as usize;
            assert!(idx < self.num_embeddings, "index {} out of range for embedding of size {}", idx, self.num_embeddings);
            let src_offset = idx * self.embedding_dim;
            let dst_offset = i * self.embedding_dim;
            output[dst_offset..dst_offset + self.embedding_dim]
                .copy_from_slice(&weight_data[src_offset..src_offset + self.embedding_dim]);
        }

        let mut out_shape = input_shape;
        out_shape.push(self.embedding_dim);

        Variable::new(Tensor::from_slice(&output, &out_shape))
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone()]
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
}
