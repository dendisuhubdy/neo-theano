//! WebGPU device management.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::WgpuError;

/// The native graphics API backing the WebGPU device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WgpuBackendType {
    /// Vulkan (Linux, Android, Windows).
    Vulkan,
    /// Metal (macOS, iOS).
    Metal,
    /// DirectX 12 (Windows).
    Dx12,
    /// Native WebGPU (WASM in browsers).
    WebGpu,
    /// Unknown / fallback.
    Unknown,
}

/// Properties of a WebGPU device.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub backend_type: WgpuBackendType,
    pub max_buffer_size: usize,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_storage_buffers_per_shader_stage: u32,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        let backend_type = Self::detect_backend_type();
        Self {
            name: format!("Mock WebGPU Device ({:?})", backend_type),
            backend_type,
            max_buffer_size: 256 * 1024 * 1024, // 256 MB
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffers_per_shader_stage: 8,
        }
    }
}

impl DeviceProperties {
    /// Detect the native backend based on the current platform.
    fn detect_backend_type() -> WgpuBackendType {
        if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
            WgpuBackendType::Metal
        } else if cfg!(target_os = "windows") {
            WgpuBackendType::Dx12
        } else if cfg!(target_arch = "wasm32") {
            WgpuBackendType::WebGpu
        } else {
            WgpuBackendType::Vulkan
        }
    }
}

/// Represents a single WebGPU device.
///
/// Manages the adapter, device handle, queue, and shader module cache.
/// When compiled without real `wgpu`, operates in mock mode for testing.
#[derive(Clone)]
pub struct WgpuDevice {
    /// Device ordinal (0, 1, 2, ...).
    ordinal: usize,
    /// Device properties.
    props: Arc<DeviceProperties>,
}

impl WgpuDevice {
    /// Create a new WebGPU device handle.
    ///
    /// In mock mode (default), creates a device with platform-appropriate
    /// backend type and reasonable default limits.
    pub fn new(ordinal: usize) -> Result<Self, WgpuError> {
        let props = DeviceProperties {
            name: format!("Mock WebGPU Device {}", ordinal),
            ..DeviceProperties::default()
        };

        Ok(Self {
            ordinal,
            props: Arc::new(props),
        })
    }

    /// Get the device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get the device name.
    pub fn name(&self) -> &str {
        &self.props.name
    }

    /// Get the native backend type (Vulkan, Metal, DX12, WebGPU).
    pub fn backend_type(&self) -> WgpuBackendType {
        self.props.backend_type
    }

    /// Get the maximum buffer size in bytes.
    pub fn max_buffer_size(&self) -> usize {
        self.props.max_buffer_size
    }

    /// Get device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.props
    }

    /// Synchronize this device (wait for all pending operations).
    pub fn synchronize(&self) -> Result<(), WgpuError> {
        // In mock mode, no-op. With real wgpu, would call device.poll(Maintain::Wait).
        Ok(())
    }
}

/// Global device manager — tracks all WebGPU devices.
pub struct DeviceManager {
    devices: Mutex<Vec<Option<Arc<WgpuDevice>>>>,
}

impl DeviceManager {
    /// Create a new device manager.
    pub fn new() -> Self {
        Self {
            devices: Mutex::new(Vec::new()),
        }
    }

    /// Get or create a WebGPU device by ordinal.
    pub fn get_device(&self, ordinal: usize) -> Result<Arc<WgpuDevice>, WgpuError> {
        let mut devices = self.devices.lock();

        // Extend the vec if needed
        while devices.len() <= ordinal {
            devices.push(None);
        }

        if let Some(dev) = &devices[ordinal] {
            return Ok(dev.clone());
        }

        let dev = Arc::new(WgpuDevice::new(ordinal)?);
        devices[ordinal] = Some(dev.clone());
        Ok(dev)
    }

    /// Get the number of available WebGPU devices.
    pub fn device_count(&self) -> usize {
        // Mock: report 1 device for testing.
        // With real wgpu, would enumerate adapters.
        1
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global singleton device manager.
static DEVICE_MANAGER: std::sync::LazyLock<DeviceManager> =
    std::sync::LazyLock::new(DeviceManager::new);

/// Get the global device manager.
pub fn device_manager() -> &'static DeviceManager {
    &DEVICE_MANAGER
}

/// Convenience: get a WebGPU device by ordinal.
pub fn get_device(ordinal: usize) -> Result<Arc<WgpuDevice>, WgpuError> {
    device_manager().get_device(ordinal)
}

/// Get the number of available WebGPU devices.
pub fn device_count() -> usize {
    device_manager().device_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_device() {
        let dev = WgpuDevice::new(0).unwrap();
        assert_eq!(dev.ordinal(), 0);
        assert!(dev.name().contains("Mock"));
        assert!(dev.max_buffer_size() > 0);
    }

    #[test]
    fn test_backend_type_detection() {
        let dev = WgpuDevice::new(0).unwrap();
        let bt = dev.backend_type();
        // On macOS should be Metal, on Linux should be Vulkan, etc.
        assert_ne!(bt, WgpuBackendType::Unknown);
    }

    #[test]
    fn test_device_properties() {
        let dev = WgpuDevice::new(0).unwrap();
        let props = dev.properties();
        assert!(props.max_compute_workgroup_size_x > 0);
        assert!(props.max_storage_buffers_per_shader_stage > 0);
    }

    #[test]
    fn test_device_synchronize() {
        let dev = WgpuDevice::new(0).unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_device_manager() {
        let mgr = DeviceManager::new();
        let dev0 = mgr.get_device(0).unwrap();
        let dev0_again = mgr.get_device(0).unwrap();
        // Same Arc
        assert!(Arc::ptr_eq(&dev0, &dev0_again));
    }

    #[test]
    fn test_device_count() {
        let count = device_count();
        assert!(count >= 1); // Mock reports 1
    }

    #[test]
    fn test_global_get_device() {
        let dev = get_device(0).unwrap();
        assert_eq!(dev.ordinal(), 0);
    }
}
