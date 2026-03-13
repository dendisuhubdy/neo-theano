use rand::seq::SliceRandom;
use rand::thread_rng;

pub trait Sampler: Send + Sync {
    fn indices(&self, dataset_len: usize) -> Vec<usize>;
}

pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    fn indices(&self, dataset_len: usize) -> Vec<usize> {
        (0..dataset_len).collect()
    }
}

pub struct RandomSampler;

impl Sampler for RandomSampler {
    fn indices(&self, dataset_len: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..dataset_len).collect();
        indices.shuffle(&mut thread_rng());
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let indices = SequentialSampler.indices(5);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_sampler() {
        let indices = RandomSampler.indices(10);
        assert_eq!(indices.len(), 10);
        // All indices should be present
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}
