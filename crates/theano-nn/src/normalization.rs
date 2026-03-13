//! Normalization layers.

use theano_autograd::Variable;
use theano_core::Tensor;
use crate::init;
use crate::module::Module;

/// Layer Normalization. Like `torch.nn.LayerNorm`.
/// Normalizes over the last D dimensions.
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    eps: f64,
    weight: Variable,
    bias: Variable,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let weight = Variable::requires_grad(Tensor::ones(&normalized_shape));
        let bias = init::zeros(&normalized_shape);
        Self {
            normalized_shape,
            eps: 1e-5,
            weight,
            bias,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Variable) -> Variable {
        // Normalize over the last len(normalized_shape) dimensions
        let ndim = input.tensor().ndim();
        let norm_dims = self.normalized_shape.len();
        let reduce_start = ndim - norm_dims;

        // Flatten the normalized dims, compute mean and variance
        let data = input.tensor().to_vec_f64().unwrap();
        let shape = input.tensor().shape().to_vec();

        let outer_size: usize = shape[..reduce_start].iter().product();
        let inner_size: usize = shape[reduce_start..].iter().product();

        let mut output = vec![0.0f64; data.len()];
        let weight_data = self.weight.tensor().to_vec_f64().unwrap();
        let bias_data = self.bias.tensor().to_vec_f64().unwrap();

        for i in 0..outer_size {
            let offset = i * inner_size;
            let slice = &data[offset..offset + inner_size];

            let mean: f64 = slice.iter().sum::<f64>() / inner_size as f64;
            let var: f64 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / inner_size as f64;
            let std = (var + self.eps).sqrt();

            for j in 0..inner_size {
                let normalized = (slice[j] - mean) / std;
                output[offset + j] = normalized * weight_data[j] + bias_data[j];
            }
        }

        Variable::new(Tensor::from_slice(&output, &shape))
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// Group Normalization. Like `torch.nn.GroupNorm`.
/// Divides channels into groups and normalizes within each group.
pub struct GroupNorm {
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    weight: Variable,
    bias: Variable,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Self {
        assert_eq!(num_channels % num_groups, 0, "num_channels must be divisible by num_groups");
        Self {
            num_groups,
            num_channels,
            eps: 1e-5,
            weight: Variable::requires_grad(Tensor::ones(&[num_channels])),
            bias: init::zeros(&[num_channels]),
        }
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Variable) -> Variable {
        let shape = input.tensor().shape().to_vec();
        assert!(shape.len() >= 2);
        let n = shape[0];
        let c = shape[1];
        assert_eq!(c, self.num_channels);

        let spatial: usize = shape[2..].iter().product();
        let channels_per_group = c / self.num_groups;

        let data = input.tensor().to_vec_f64().unwrap();
        let weight_data = self.weight.tensor().to_vec_f64().unwrap();
        let bias_data = self.bias.tensor().to_vec_f64().unwrap();
        let mut output = vec![0.0f64; data.len()];

        for batch in 0..n {
            for g in 0..self.num_groups {
                let ch_start = g * channels_per_group;
                let ch_end = ch_start + channels_per_group;

                // Gather all values in this group
                let mut sum = 0.0f64;
                let mut count = 0;
                for ch in ch_start..ch_end {
                    for s in 0..spatial {
                        let idx = batch * c * spatial + ch * spatial + s;
                        sum += data[idx];
                        count += 1;
                    }
                }
                let mean = sum / count as f64;

                let mut var_sum = 0.0f64;
                for ch in ch_start..ch_end {
                    for s in 0..spatial {
                        let idx = batch * c * spatial + ch * spatial + s;
                        var_sum += (data[idx] - mean) * (data[idx] - mean);
                    }
                }
                let var = var_sum / count as f64;
                let std = (var + self.eps).sqrt();

                for ch in ch_start..ch_end {
                    for s in 0..spatial {
                        let idx = batch * c * spatial + ch * spatial + s;
                        let normalized = (data[idx] - mean) / std;
                        output[idx] = normalized * weight_data[ch] + bias_data[ch];
                    }
                }
            }
        }

        Variable::new(Tensor::from_slice(&output, &shape))
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_shape() {
        let ln = LayerNorm::new(vec![4]);
        let input = Variable::new(Tensor::ones(&[2, 4]));
        let output = ln.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 4]);
    }

    #[test]
    fn test_layernorm_params() {
        let ln = LayerNorm::new(vec![8]);
        assert_eq!(ln.parameters().len(), 2);
    }

    #[test]
    fn test_groupnorm_shape() {
        let gn = GroupNorm::new(2, 4);
        let input = Variable::new(Tensor::ones(&[2, 4, 3, 3]));
        let output = gn.forward(&input);
        assert_eq!(output.tensor().shape(), &[2, 4, 3, 3]);
    }
}
