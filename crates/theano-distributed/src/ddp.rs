//! Distributed Data Parallel (DDP) wrapper.
//! Like `torch.nn.parallel.DistributedDataParallel`.

use std::sync::Arc;

use theano_core::Tensor;
use crate::process_group::ProcessGroup;
use crate::collectives::{all_reduce, ReduceOp};

/// Distributed Data Parallel wrapper.
///
/// Wraps a model and synchronizes gradients across processes during
/// backward pass via all-reduce.
pub struct DistributedDataParallel {
    /// The process group for communication.
    process_group: Arc<ProcessGroup>,
    /// Whether to broadcast parameters from rank 0 on construction.
    broadcast_buffers: bool,
    /// Bucket size for gradient bucketing (bytes).
    bucket_size_mb: usize,
}

impl DistributedDataParallel {
    /// Create a new DDP wrapper.
    pub fn new(process_group: Arc<ProcessGroup>) -> Self {
        Self {
            process_group,
            broadcast_buffers: true,
            bucket_size_mb: 25,
        }
    }

    pub fn broadcast_buffers(mut self, v: bool) -> Self {
        self.broadcast_buffers = v;
        self
    }

    pub fn bucket_size_mb(mut self, mb: usize) -> Self {
        self.bucket_size_mb = mb;
        self
    }

    /// Synchronize gradients across all processes.
    /// Call this after backward() and before optimizer.step().
    pub fn sync_gradients(&self, gradients: &[Tensor]) -> Vec<Tensor> {
        let world_size = self.process_group.world_size();
        if world_size == 1 {
            return gradients.to_vec();
        }

        // All-reduce each gradient and average
        gradients.iter().map(|grad| {
            let reduced = all_reduce(grad, ReduceOp::Sum, &self.process_group);
            // Average by dividing by world_size
            reduced.mul_scalar(1.0 / world_size as f64).unwrap()
        }).collect()
    }

    /// Get the process group.
    pub fn process_group(&self) -> &ProcessGroup {
        &self.process_group
    }
}

/// Fully Sharded Data Parallel configuration.
/// Like `torch.distributed.fsdp.FullyShardedDataParallel`.
pub struct FSDPConfig {
    /// Sharding strategy.
    pub sharding_strategy: ShardingStrategy,
    /// CPU offload for parameters.
    pub cpu_offload: bool,
    /// Mixed precision policy.
    pub mixed_precision: bool,
}

/// FSDP sharding strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// Shard parameters, gradients, and optimizer states.
    FullShard,
    /// Shard only gradients and optimizer states.
    ShardGradOp,
    /// No sharding (equivalent to DDP).
    NoShard,
}

impl Default for FSDPConfig {
    fn default() -> Self {
        Self {
            sharding_strategy: ShardingStrategy::FullShard,
            cpu_offload: false,
            mixed_precision: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddp_single_process() {
        let pg = Arc::new(ProcessGroup::single());
        let ddp = DistributedDataParallel::new(pg);

        let grads = vec![
            Tensor::from_slice(&[1.0, 2.0], &[2]),
            Tensor::from_slice(&[3.0, 4.0], &[2]),
        ];

        let synced = ddp.sync_gradients(&grads);
        assert_eq!(synced.len(), 2);
        assert_eq!(synced[0].to_vec_f64().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_fsdp_config() {
        let config = FSDPConfig::default();
        assert_eq!(config.sharding_strategy, ShardingStrategy::FullShard);
        assert!(!config.cpu_offload);
    }
}
