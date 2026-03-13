use std::sync::Arc;

use crate::dataset::Dataset;
use crate::sampler::{Sampler, SequentialSampler, RandomSampler};

/// DataLoader that iterates over a dataset in batches.
/// Like PyTorch's DataLoader.
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size,
            shuffle: false,
            drop_last: false,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Get the number of batches.
    pub fn len(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create an iterator over batches.
    /// Each batch is a Vec of dataset items.
    pub fn iter(&self) -> DataLoaderIter<'_, D> {
        let sampler: Box<dyn Sampler> = if self.shuffle {
            Box::new(RandomSampler)
        } else {
            Box::new(SequentialSampler)
        };
        let indices = sampler.indices(self.dataset.len());

        DataLoaderIter {
            dataset: &self.dataset,
            indices,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            pos: 0,
        }
    }
}

pub struct DataLoaderIter<'a, D: Dataset> {
    dataset: &'a D,
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    pos: usize,
}

impl<'a, D: Dataset> Iterator for DataLoaderIter<'a, D> {
    type Item = Vec<D::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let remaining = self.indices.len() - self.pos;
        let batch_len = remaining.min(self.batch_size);

        if batch_len < self.batch_size && self.drop_last {
            return None;
        }

        let batch: Vec<D::Item> = self.indices[self.pos..self.pos + batch_len]
            .iter()
            .map(|&idx| self.dataset.get(idx))
            .collect();

        self.pos += batch_len;
        Some(batch)
    }
}

// Allow for-in loops
impl<'a, D: Dataset> IntoIterator for &'a DataLoader<D> {
    type Item = Vec<D::Item>;
    type IntoIter = DataLoaderIter<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;
    use theano_core::Tensor;

    #[test]
    fn test_dataloader_sequential() {
        let data = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[5, 2],
        );
        let labels = Tensor::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0], &[5]);
        let ds = TensorDataset::new(data, labels);
        let loader = DataLoader::new(ds, 2);

        assert_eq!(loader.len(), 3); // 5/2 = 2 full + 1 partial
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[2].len(), 1); // last batch is partial
    }

    #[test]
    fn test_dataloader_drop_last() {
        let data = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[5, 2],
        );
        let labels = Tensor::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0], &[5]);
        let ds = TensorDataset::new(data, labels);
        let loader = DataLoader::new(ds, 2).drop_last(true);

        assert_eq!(loader.len(), 2); // drops the partial batch
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_dataloader_for_loop() {
        let data = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let labels = Tensor::from_slice(&[0.0, 1.0], &[2]);
        let ds = TensorDataset::new(data, labels);
        let loader = DataLoader::new(ds, 1);

        let mut count = 0;
        for batch in &loader {
            assert_eq!(batch.len(), 1);
            count += 1;
        }
        assert_eq!(count, 2);
    }
}
