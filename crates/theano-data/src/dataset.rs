/// Trait for datasets, mirroring PyTorch's Dataset.
pub trait Dataset: Send + Sync {
    type Item;

    /// Number of samples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single sample by index.
    fn get(&self, index: usize) -> Self::Item;
}

/// A simple in-memory dataset that holds data and labels as Tensors.
/// Like TensorDataset in PyTorch.
pub struct TensorDataset {
    data: theano_core::Tensor,
    labels: theano_core::Tensor,
}

impl TensorDataset {
    pub fn new(data: theano_core::Tensor, labels: theano_core::Tensor) -> Self {
        assert_eq!(data.shape()[0], labels.shape()[0],
            "data and labels must have same number of samples");
        Self { data, labels }
    }
}

impl Dataset for TensorDataset {
    type Item = (theano_core::Tensor, theano_core::Tensor);

    fn len(&self) -> usize {
        self.data.shape()[0]
    }

    fn get(&self, index: usize) -> Self::Item {
        let sample = self.data.select(0, index as i64).unwrap();
        let label = self.labels.select(0, index as i64).unwrap();
        (sample, label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theano_core::Tensor;

    #[test]
    fn test_tensor_dataset() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let labels = Tensor::from_slice(&[0.0, 1.0, 2.0], &[3]);
        let ds = TensorDataset::new(data, labels);

        assert_eq!(ds.len(), 3);
        let (sample, label) = ds.get(1);
        assert_eq!(sample.to_vec_f64().unwrap(), vec![3.0, 4.0]);
        assert_eq!(label.item().unwrap(), 1.0);
    }
}
