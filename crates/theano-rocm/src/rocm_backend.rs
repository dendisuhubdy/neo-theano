//! ROCm backend implementation.

use theano_backend::Backend;
use theano_types::DeviceType;

use crate::storage::RocmStorage;

/// ROCm/HIP GPU backend.
#[derive(Clone, Debug)]
pub struct RocmBackend;

impl Backend for RocmBackend {
    type Storage = RocmStorage;

    fn name() -> &'static str {
        "rocm"
    }

    fn device_type() -> DeviceType {
        DeviceType::Rocm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        assert_eq!(RocmBackend::name(), "rocm");
    }

    #[test]
    fn test_backend_device_type() {
        assert_eq!(RocmBackend::device_type(), DeviceType::Rocm);
    }
}
