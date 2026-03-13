use theano_core::Tensor;
use crate::process_group::ProcessGroup;

/// Reduction operations for collectives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
    Avg,
}

/// All-reduce: reduce tensor across all processes and distribute result.
/// Like `torch.distributed.all_reduce`.
///
/// In mock (single-process) mode, this is a no-op.
pub fn all_reduce(tensor: &Tensor, _op: ReduceOp, pg: &ProcessGroup) -> Tensor {
    if pg.world_size() == 1 {
        // Single process: no-op
        return tensor.clone();
    }
    // In a real implementation:
    // - NCCL: ncclAllReduce
    // - Gloo: gloo::allreduce
    // For now, return the tensor unchanged (mock)
    tensor.clone()
}

/// Broadcast tensor from src_rank to all processes.
/// Like `torch.distributed.broadcast`.
pub fn broadcast(tensor: &Tensor, _src_rank: usize, pg: &ProcessGroup) -> Tensor {
    if pg.world_size() == 1 {
        return tensor.clone();
    }
    // Mock: return unchanged
    tensor.clone()
}

/// All-gather: gather tensors from all processes.
/// Like `torch.distributed.all_gather`.
/// Returns a vector of tensors, one per process.
pub fn all_gather(tensor: &Tensor, pg: &ProcessGroup) -> Vec<Tensor> {
    if pg.world_size() == 1 {
        return vec![tensor.clone()];
    }
    // Mock: return just this process's tensor
    vec![tensor.clone(); pg.world_size()]
}

/// Reduce-scatter: reduce then scatter.
/// Like `torch.distributed.reduce_scatter`.
pub fn reduce_scatter(tensors: &[Tensor], _op: ReduceOp, pg: &ProcessGroup) -> Tensor {
    if pg.world_size() == 1 || tensors.is_empty() {
        return tensors.first().cloned().unwrap_or_else(|| Tensor::scalar(0.0));
    }
    // Mock: return first tensor
    tensors[0].clone()
}

/// Barrier: synchronize all processes.
/// Like `torch.distributed.barrier`.
pub fn barrier(pg: &ProcessGroup) {
    if pg.world_size() == 1 {
        return;
    }
    // Mock: no-op for single process
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_group::ProcessGroup;

    #[test]
    fn test_all_reduce_single() {
        let pg = ProcessGroup::single();
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = all_reduce(&tensor, ReduceOp::Sum, &pg);
        assert_eq!(result.to_vec_f64().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_single() {
        let pg = ProcessGroup::single();
        let tensor = Tensor::from_slice(&[1.0, 2.0], &[2]);
        let result = broadcast(&tensor, 0, &pg);
        assert_eq!(result.to_vec_f64().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_all_gather_single() {
        let pg = ProcessGroup::single();
        let tensor = Tensor::from_slice(&[1.0, 2.0], &[2]);
        let result = all_gather(&tensor, &pg);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_barrier_single() {
        let pg = ProcessGroup::single();
        barrier(&pg); // should not panic
    }
}
