/// Communication backend type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// NCCL backend (NVIDIA GPUs).
    Nccl,
    /// RCCL backend (AMD GPUs).
    Rccl,
    /// Gloo backend (CPU, TCP-based).
    Gloo,
    /// MPI backend.
    Mpi,
}

/// A process group represents a set of processes that can communicate.
/// Like `torch.distributed.ProcessGroup`.
pub struct ProcessGroup {
    rank: usize,
    world_size: usize,
    backend: Backend,
    // In a real implementation, this would hold NCCL/RCCL communicators
    // For now, mock single-process
}

impl ProcessGroup {
    /// Create a new process group.
    pub fn new(rank: usize, world_size: usize, backend: Backend) -> Self {
        Self { rank, world_size, backend }
    }

    /// Create a single-process group (for local development/testing).
    pub fn single() -> Self {
        Self { rank: 0, world_size: 1, backend: Backend::Gloo }
    }

    /// Get the rank of this process.
    pub fn rank(&self) -> usize { self.rank }

    /// Get the total number of processes.
    pub fn world_size(&self) -> usize { self.world_size }

    /// Get the communication backend.
    pub fn backend(&self) -> Backend { self.backend }

    /// Whether this is the master (rank 0) process.
    pub fn is_master(&self) -> bool { self.rank == 0 }
}

/// Initialize the default process group.
/// Like `torch.distributed.init_process_group`.
pub fn init_process_group(backend: Backend, rank: usize, world_size: usize) -> ProcessGroup {
    ProcessGroup::new(rank, world_size, backend)
}

/// Get the rank of the current process.
pub fn get_rank(pg: &ProcessGroup) -> usize {
    pg.rank()
}

/// Get the world size.
pub fn get_world_size(pg: &ProcessGroup) -> usize {
    pg.world_size()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_group_single() {
        let pg = ProcessGroup::single();
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 1);
        assert!(pg.is_master());
    }

    #[test]
    fn test_init_process_group() {
        let pg = init_process_group(Backend::Gloo, 0, 4);
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 4);
        assert_eq!(pg.backend(), Backend::Gloo);
    }

    #[test]
    fn test_non_master() {
        let pg = ProcessGroup::new(2, 4, Backend::Nccl);
        assert!(!pg.is_master());
        assert_eq!(pg.rank(), 2);
    }
}
