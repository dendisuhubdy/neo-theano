//! Intel oneAPI backend implementation.

use theano_backend::Backend;
use theano_types::DeviceType;

use crate::storage::OneApiStorage;

/// Intel oneAPI backend — targets Intel Arc GPUs, Gaudi, and data center GPUs.
#[derive(Clone, Debug)]
pub struct OneApiBackend;

impl Backend for OneApiBackend {
    type Storage = OneApiStorage;

    fn name() -> &'static str {
        "oneapi"
    }

    fn device_type() -> DeviceType {
        DeviceType::OneApi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        assert_eq!(OneApiBackend::name(), "oneapi");
    }

    #[test]
    fn test_backend_device_type() {
        assert_eq!(OneApiBackend::device_type(), DeviceType::OneApi);
    }
}
