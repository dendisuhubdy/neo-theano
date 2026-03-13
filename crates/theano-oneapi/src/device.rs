//! Intel oneAPI device management.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::OneApiError;

/// Intel GPU device family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntelDeviceFamily {
    /// Intel Arc (Alchemist) — consumer discrete GPUs (A770, A750, etc.).
    Arc,
    /// Intel Data Center GPU Flex Series (formerly Arctic Sound).
    Flex,
    /// Intel Data Center GPU Max Series (Ponte Vecchio) — HPC/AI.
    PonteVecchio,
    /// Intel Gaudi AI accelerator (Habana Labs).
    Gaudi,
    /// Integrated Intel UHD/Iris Xe graphics.
    Integrated,
    /// Unknown device family.
    Unknown,
}

/// Properties of an Intel GPU device.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub family: IntelDeviceFamily,
    pub total_memory: usize,
    pub eu_count: u32,
    pub max_compute_units: u32,
    pub max_workgroup_size: u32,
    pub max_memory_allocation_size: usize,
    pub supports_fp16: bool,
    pub supports_fp64: bool,
    pub supports_int8: bool,
    pub driver_version: String,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            name: "Mock Intel Arc GPU".to_string(),
            family: IntelDeviceFamily::Arc,
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GB
            eu_count: 512,
            max_compute_units: 512,
            max_workgroup_size: 1024,
            max_memory_allocation_size: 4 * 1024 * 1024 * 1024, // 4 GB
            supports_fp16: true,
            supports_fp64: false, // Arc GPUs have limited FP64
            supports_int8: true,
            driver_version: "mock-1.0.0".to_string(),
        }
    }
}

/// Represents a single Intel GPU device.
///
/// Manages the Level Zero context, command queue, and device properties.
/// When compiled without real Level Zero / SYCL, operates in mock mode.
#[derive(Clone)]
pub struct OneApiDevice {
    /// Device ordinal (0, 1, 2, ...).
    ordinal: usize,
    /// Device properties.
    props: Arc<DeviceProperties>,
}

impl OneApiDevice {
    /// Create a new Intel GPU device handle.
    ///
    /// In mock mode (default), creates a device with Arc GPU properties.
    pub fn new(ordinal: usize) -> Result<Self, OneApiError> {
        let props = DeviceProperties {
            name: format!("Mock Intel Arc GPU {}", ordinal),
            ..DeviceProperties::default()
        };

        Ok(Self {
            ordinal,
            props: Arc::new(props),
        })
    }

    /// Create a device with a specific device family (for testing).
    pub fn with_family(ordinal: usize, family: IntelDeviceFamily) -> Result<Self, OneApiError> {
        let (name, total_memory, eu_count, supports_fp64) = match family {
            IntelDeviceFamily::Arc => (
                format!("Mock Intel Arc A770 {}", ordinal),
                16 * 1024 * 1024 * 1024_usize,
                512,
                false,
            ),
            IntelDeviceFamily::PonteVecchio => (
                format!("Mock Intel Data Center GPU Max {}", ordinal),
                128 * 1024 * 1024 * 1024_usize,
                1024,
                true,
            ),
            IntelDeviceFamily::Gaudi => (
                format!("Mock Intel Gaudi2 {}", ordinal),
                96 * 1024 * 1024 * 1024_usize,
                0, // Gaudi uses TPC cores, not EUs
                false,
            ),
            IntelDeviceFamily::Flex => (
                format!("Mock Intel Flex 170 {}", ordinal),
                12 * 1024 * 1024 * 1024_usize,
                512,
                false,
            ),
            IntelDeviceFamily::Integrated => (
                format!("Mock Intel Iris Xe {}", ordinal),
                0, // Shared memory
                96,
                false,
            ),
            IntelDeviceFamily::Unknown => (
                format!("Mock Intel GPU {}", ordinal),
                8 * 1024 * 1024 * 1024_usize,
                256,
                false,
            ),
        };

        let props = DeviceProperties {
            name,
            family,
            total_memory,
            eu_count,
            max_compute_units: eu_count,
            supports_fp64,
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

    /// Get the device family.
    pub fn family(&self) -> IntelDeviceFamily {
        self.props.family
    }

    /// Get total device memory in bytes.
    pub fn total_memory(&self) -> usize {
        self.props.total_memory
    }

    /// Get the number of Execution Units (EUs).
    pub fn eu_count(&self) -> u32 {
        self.props.eu_count
    }

    /// Get device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.props
    }

    /// Synchronize this device (wait for all pending operations).
    pub fn synchronize(&self) -> Result<(), OneApiError> {
        // In mock mode, no-op. With real Level Zero, would call zeCommandQueueSynchronize.
        Ok(())
    }

    /// Get memory info: (free, total).
    pub fn memory_info(&self) -> Result<(usize, usize), OneApiError> {
        // Mock: report full memory as free.
        Ok((self.props.total_memory, self.props.total_memory))
    }
}

/// Global device manager — tracks all Intel GPU devices.
pub struct DeviceManager {
    devices: Mutex<Vec<Option<Arc<OneApiDevice>>>>,
}

impl DeviceManager {
    /// Create a new device manager.
    pub fn new() -> Self {
        Self {
            devices: Mutex::new(Vec::new()),
        }
    }

    /// Get or create an Intel GPU device by ordinal.
    pub fn get_device(&self, ordinal: usize) -> Result<Arc<OneApiDevice>, OneApiError> {
        let mut devices = self.devices.lock();

        // Extend the vec if needed
        while devices.len() <= ordinal {
            devices.push(None);
        }

        if let Some(dev) = &devices[ordinal] {
            return Ok(dev.clone());
        }

        let dev = Arc::new(OneApiDevice::new(ordinal)?);
        devices[ordinal] = Some(dev.clone());
        Ok(dev)
    }

    /// Get the number of available Intel GPU devices.
    pub fn device_count(&self) -> usize {
        // Mock: report 1 device for testing.
        // With real Level Zero, would call zeDeviceGet.
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

/// Convenience: get an Intel GPU device by ordinal.
pub fn get_device(ordinal: usize) -> Result<Arc<OneApiDevice>, OneApiError> {
    device_manager().get_device(ordinal)
}

/// Get the number of available Intel GPU devices.
pub fn device_count() -> usize {
    device_manager().device_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_device() {
        let dev = OneApiDevice::new(0).unwrap();
        assert_eq!(dev.ordinal(), 0);
        assert!(dev.name().contains("Mock"));
        assert!(dev.total_memory() > 0);
    }

    #[test]
    fn test_device_family() {
        let dev = OneApiDevice::new(0).unwrap();
        assert_eq!(dev.family(), IntelDeviceFamily::Arc);
    }

    #[test]
    fn test_device_eu_count() {
        let dev = OneApiDevice::new(0).unwrap();
        assert!(dev.eu_count() > 0);
    }

    #[test]
    fn test_device_properties() {
        let dev = OneApiDevice::new(0).unwrap();
        let props = dev.properties();
        assert!(props.supports_fp16);
        assert!(props.supports_int8);
        assert!(props.max_workgroup_size > 0);
    }

    #[test]
    fn test_device_with_family_arc() {
        let dev = OneApiDevice::with_family(0, IntelDeviceFamily::Arc).unwrap();
        assert_eq!(dev.family(), IntelDeviceFamily::Arc);
        assert!(dev.name().contains("Arc"));
    }

    #[test]
    fn test_device_with_family_ponte_vecchio() {
        let dev = OneApiDevice::with_family(0, IntelDeviceFamily::PonteVecchio).unwrap();
        assert_eq!(dev.family(), IntelDeviceFamily::PonteVecchio);
        assert!(dev.properties().supports_fp64);
        assert!(dev.eu_count() > 512); // Data center GPUs have more EUs
    }

    #[test]
    fn test_device_with_family_gaudi() {
        let dev = OneApiDevice::with_family(0, IntelDeviceFamily::Gaudi).unwrap();
        assert_eq!(dev.family(), IntelDeviceFamily::Gaudi);
        assert!(dev.total_memory() > 64 * 1024 * 1024 * 1024); // 96 GB
    }

    #[test]
    fn test_device_synchronize() {
        let dev = OneApiDevice::new(0).unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_device_memory_info() {
        let dev = OneApiDevice::new(0).unwrap();
        let (free, total) = dev.memory_info().unwrap();
        assert!(free <= total);
        assert!(total > 0);
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
