//! Metal backend implementation.

use theano_backend::Backend;
use theano_types::DeviceType;

use crate::storage::MetalStorage;

/// Apple Metal GPU backend.
///
/// Uses Metal Performance Shaders (MPS) for optimized ML primitives
/// on Apple Silicon (M1/M2/M3/M4) GPUs with unified memory.
#[derive(Clone, Debug)]
pub struct MetalBackend;

impl Backend for MetalBackend {
    type Storage = MetalStorage;

    fn name() -> &'static str {
        "metal"
    }

    fn device_type() -> DeviceType {
        DeviceType::Metal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        assert_eq!(MetalBackend::name(), "metal");
    }

    #[test]
    fn test_backend_device_type() {
        assert_eq!(MetalBackend::device_type(), DeviceType::Metal);
    }
}
