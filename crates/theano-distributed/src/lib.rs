//! Distributed training support for Theano.
//!
//! Provides process group abstractions, collective operations (all_reduce,
//! broadcast, etc.), and distributed training wrappers (DDP, FSDP).

pub mod process_group;
pub mod collectives;
pub mod ddp;

pub use process_group::{ProcessGroup, Backend as DistBackend};
pub use collectives::{ReduceOp as CollReduceOp, all_reduce, broadcast, all_gather, barrier};
pub use ddp::DistributedDataParallel;
