//! CUDA backend implementation.

use theano_backend::Backend;
use theano_types::DeviceType;

use crate::storage::CudaStorage;

/// CUDA GPU backend.
#[derive(Clone, Debug)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    type Storage = CudaStorage;

    fn name() -> &'static str {
        "cuda"
    }

    fn device_type() -> DeviceType {
        DeviceType::Cuda
    }
}
