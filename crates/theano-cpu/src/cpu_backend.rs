use theano_backend::Backend;
use theano_types::DeviceType;

use crate::cpu_storage::CpuStorage;

/// CPU backend implementation.
#[derive(Clone, Debug)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Storage = CpuStorage;

    fn name() -> &'static str {
        "cpu"
    }

    fn device_type() -> DeviceType {
        DeviceType::Cpu
    }
}
