//! ROCm/HIP device management.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::allocator::CachingAllocator;
use crate::error::RocmError;

/// Represents a single AMD ROCm/HIP GPU device.
///
/// Manages the HIP context, default stream, rocBLAS handle, RNG,
/// and the caching memory allocator.
#[derive(Clone)]
pub struct RocmDevice {
    /// Device ordinal (0, 1, 2, ...).
    ordinal: usize,
    /// Caching memory allocator for this device.
    allocator: Arc<CachingAllocator>,
    /// Device properties.
    props: Arc<DeviceProperties>,

    // When hip feature is enabled, these would hold the real HIP handles:
    // context: Arc<hip_sys::HipDevice>,
    // blas: Arc<rocblas_sys::RocblasHandle>,
    // rng: Arc<Mutex<hiprand_sys::HiprandGenerator>>,
}

/// ROCm device properties (mirrors hipDeviceProp_t).
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub gcn_arch: u32,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_multiprocessor: u32,
    pub warp_size: u32,
    pub max_shared_memory_per_block: usize,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            name: "Mock ROCm Device".to_string(),
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GB (typical MI250/MI300)
            gcn_arch: 908, // gfx908 (MI100)
            multiprocessor_count: 120,
            max_threads_per_block: 1024,
            max_threads_per_multiprocessor: 2048,
            warp_size: 64, // AMD wavefront size is 64 (vs NVIDIA 32)
            max_shared_memory_per_block: 64 * 1024, // 64 KB LDS
        }
    }
}

impl RocmDevice {
    /// Create a new ROCm device handle.
    ///
    /// When the `hip` feature is enabled, this initializes the HIP context.
    /// Otherwise, it creates a mock device for development/testing.
    pub fn new(ordinal: usize) -> Result<Self, RocmError> {
        #[cfg(feature = "hip")]
        {
            return Self::new_hip(ordinal);
        }

        #[cfg(not(feature = "hip"))]
        {
            Self::new_mock(ordinal)
        }
    }

    /// Create a mock device (no HIP required).
    #[cfg(not(feature = "hip"))]
    fn new_mock(ordinal: usize) -> Result<Self, RocmError> {
        Ok(Self {
            ordinal,
            allocator: Arc::new(CachingAllocator::new(ordinal)),
            props: Arc::new(DeviceProperties {
                name: format!("Mock ROCm Device {ordinal}"),
                ..DeviceProperties::default()
            }),
        })
    }

    /// Create a real HIP device.
    #[cfg(feature = "hip")]
    fn new_hip(ordinal: usize) -> Result<Self, RocmError> {
        // Real HIP device initialization would go here:
        // hip_sys::hipSetDevice(ordinal as i32)
        //     .map_err(|e| RocmError::driver(format!("failed to set HIP device {ordinal}: {e}")))?;

        Ok(Self {
            ordinal,
            allocator: Arc::new(CachingAllocator::new(ordinal)),
            props: Arc::new(DeviceProperties::default()), // TODO: query real props via hipGetDeviceProperties
        })
    }

    /// Get the device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.props
    }

    /// Get the device name.
    pub fn name(&self) -> &str {
        &self.props.name
    }

    /// Get total device memory in bytes.
    pub fn total_memory(&self) -> usize {
        self.props.total_memory
    }

    /// Get the GCN architecture version (e.g. 908 for gfx908).
    pub fn gcn_arch(&self) -> u32 {
        self.props.gcn_arch
    }

    /// Get the caching allocator for this device.
    pub fn allocator(&self) -> &CachingAllocator {
        &self.allocator
    }

    /// Synchronize this device (wait for all pending operations).
    pub fn synchronize(&self) -> Result<(), RocmError> {
        #[cfg(feature = "hip")]
        {
            // hip_sys::hipDeviceSynchronize()
            //     .map_err(|e| RocmError::driver(format!("hipDeviceSynchronize failed: {e}")))?;
        }
        Ok(())
    }

    /// Get memory info: (free, total).
    pub fn memory_info(&self) -> Result<(usize, usize), RocmError> {
        let stats = self.allocator.stats();
        let used = stats.allocated_bytes + stats.cached_bytes;
        let total = self.props.total_memory;
        let free = total.saturating_sub(used);
        Ok((free, total))
    }

    /// Release all cached memory back to the driver.
    pub fn empty_cache(&self) {
        self.allocator.empty_cache();
    }
}

/// Global device manager -- tracks all ROCm devices.
pub struct DeviceManager {
    devices: Mutex<Vec<Option<Arc<RocmDevice>>>>,
}

impl DeviceManager {
    /// Create a new device manager.
    pub fn new() -> Self {
        Self {
            devices: Mutex::new(Vec::new()),
        }
    }

    /// Get or create a ROCm device by ordinal.
    pub fn get_device(&self, ordinal: usize) -> Result<Arc<RocmDevice>, RocmError> {
        let mut devices = self.devices.lock();

        // Extend the vec if needed
        while devices.len() <= ordinal {
            devices.push(None);
        }

        if let Some(dev) = &devices[ordinal] {
            return Ok(dev.clone());
        }

        let dev = Arc::new(RocmDevice::new(ordinal)?);
        devices[ordinal] = Some(dev.clone());
        Ok(dev)
    }

    /// Get the number of available ROCm devices.
    pub fn device_count(&self) -> usize {
        #[cfg(feature = "hip")]
        {
            // let mut count: i32 = 0;
            // hip_sys::hipGetDeviceCount(&mut count);
            // count as usize
            1
        }
        #[cfg(not(feature = "hip"))]
        {
            // Mock: report 1 device for testing
            1
        }
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

/// Convenience: get a ROCm device by ordinal.
pub fn get_device(ordinal: usize) -> Result<Arc<RocmDevice>, RocmError> {
    device_manager().get_device(ordinal)
}

/// Get the number of available ROCm devices.
pub fn device_count() -> usize {
    device_manager().device_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_device() {
        let dev = RocmDevice::new(0).unwrap();
        assert_eq!(dev.ordinal(), 0);
        assert!(dev.name().contains("Mock"));
        assert!(dev.total_memory() > 0);
    }

    #[test]
    fn test_device_properties() {
        let dev = RocmDevice::new(0).unwrap();
        let props = dev.properties();
        assert_eq!(props.warp_size, 64); // AMD wavefront = 64
        assert!(props.gcn_arch > 0);
        assert!(props.max_shared_memory_per_block > 0);
    }

    #[test]
    fn test_device_memory_info() {
        let dev = RocmDevice::new(0).unwrap();
        let (free, total) = dev.memory_info().unwrap();
        assert!(free <= total);
        assert!(total > 0);
    }

    #[test]
    fn test_device_synchronize() {
        let dev = RocmDevice::new(0).unwrap();
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
    fn test_device_allocator() {
        let dev = RocmDevice::new(0).unwrap();
        let block = dev.allocator().allocate(4096).unwrap();
        assert!(block.size >= 4096);
        dev.allocator().free(block);
    }

    #[test]
    fn test_device_empty_cache() {
        let dev = RocmDevice::new(0).unwrap();
        let block = dev.allocator().allocate(4096).unwrap();
        dev.allocator().free(block);
        dev.empty_cache();
        let stats = dev.allocator().stats();
        assert_eq!(stats.cached_bytes, 0);
    }

    #[test]
    fn test_global_device_manager() {
        let dev = get_device(0).unwrap();
        let dev_again = get_device(0).unwrap();
        assert!(Arc::ptr_eq(&dev, &dev_again));
    }
}
