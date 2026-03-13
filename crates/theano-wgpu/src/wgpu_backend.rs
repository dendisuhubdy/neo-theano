//! WebGPU backend implementation.

use theano_backend::Backend;
use theano_types::DeviceType;

use crate::storage::WgpuStorage;

/// WebGPU backend — cross-platform GPU compute via Vulkan/Metal/DX12/WebGPU.
#[derive(Clone, Debug)]
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    type Storage = WgpuStorage;

    fn name() -> &'static str {
        "wgpu"
    }

    fn device_type() -> DeviceType {
        DeviceType::Wgpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        assert_eq!(WgpuBackend::name(), "wgpu");
    }

    #[test]
    fn test_backend_device_type() {
        assert_eq!(WgpuBackend::device_type(), DeviceType::Wgpu);
    }
}
