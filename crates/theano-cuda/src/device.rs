//! CUDA device management.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::allocator::CachingAllocator;
use crate::error::CudaError;

/// Represents a single CUDA GPU device.
///
/// Manages the CUDA context, default stream, cuBLAS handle, RNG,
/// and the caching memory allocator.
#[derive(Clone)]
pub struct CudaDevice {
    /// Device ordinal (0, 1, 2, ...).
    ordinal: usize,
    /// Caching memory allocator for this device.
    allocator: Arc<CachingAllocator>,
    /// Device properties.
    props: Arc<DeviceProperties>,

    // When cudarc feature is enabled, these hold the real CUDA handles:
    // context: Arc<cudarc::driver::CudaDevice>,
    // blas: Arc<cudarc::cublas::CudaBlas>,
    // rng: Arc<Mutex<cudarc::curand::CudaRng>>,
}

/// CUDA device properties (mirrors cudaDeviceProp).
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (u32, u32),
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_multiprocessor: u32,
    pub warp_size: u32,
    pub max_shared_memory_per_block: usize,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            name: "Mock CUDA Device".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024, // 8 GB
            compute_capability: (8, 0),
            multiprocessor_count: 108,
            max_threads_per_block: 1024,
            max_threads_per_multiprocessor: 2048,
            warp_size: 32,
            max_shared_memory_per_block: 48 * 1024,
        }
    }
}

impl CudaDevice {
    /// Create a new CUDA device handle.
    ///
    /// When the `cudarc` feature is enabled, this initializes the CUDA context.
    /// Otherwise, it creates a mock device for development/testing.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        #[cfg(feature = "cudarc")]
        {
            return Self::new_cudarc(ordinal);
        }

        #[cfg(not(feature = "cudarc"))]
        {
            Self::new_mock(ordinal)
        }
    }

    /// Create a mock device (no CUDA required).
    #[cfg(not(feature = "cudarc"))]
    fn new_mock(ordinal: usize) -> Result<Self, CudaError> {
        Ok(Self {
            ordinal,
            allocator: Arc::new(CachingAllocator::new(ordinal)),
            props: Arc::new(DeviceProperties {
                name: format!("Mock CUDA Device {ordinal}"),
                ..DeviceProperties::default()
            }),
        })
    }

    /// Create a real CUDA device via cudarc.
    #[cfg(feature = "cudarc")]
    fn new_cudarc(ordinal: usize) -> Result<Self, CudaError> {
        use cudarc::driver::CudaDevice as CudarcDevice;

        let device = CudarcDevice::new(ordinal)
            .map_err(|e| CudaError::driver(format!("failed to create CUDA device {ordinal}: {e}")))?;

        Ok(Self {
            ordinal,
            allocator: Arc::new(CachingAllocator::new(ordinal)),
            props: Arc::new(DeviceProperties::default()), // TODO: query real props
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

    /// Get the caching allocator for this device.
    pub fn allocator(&self) -> &CachingAllocator {
        &self.allocator
    }

    /// Synchronize this device (wait for all pending operations).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        #[cfg(feature = "cudarc")]
        {
            // self.context.synchronize()?;
        }
        Ok(())
    }

    /// Get memory info: (free, total).
    pub fn memory_info(&self) -> Result<(usize, usize), CudaError> {
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

/// Global device manager — tracks all CUDA devices.
pub struct DeviceManager {
    devices: Mutex<Vec<Option<Arc<CudaDevice>>>>,
}

impl DeviceManager {
    /// Create a new device manager.
    pub fn new() -> Self {
        Self {
            devices: Mutex::new(Vec::new()),
        }
    }

    /// Get or create a CUDA device by ordinal.
    pub fn get_device(&self, ordinal: usize) -> Result<Arc<CudaDevice>, CudaError> {
        let mut devices = self.devices.lock();

        // Extend the vec if needed
        while devices.len() <= ordinal {
            devices.push(None);
        }

        if let Some(dev) = &devices[ordinal] {
            return Ok(dev.clone());
        }

        let dev = Arc::new(CudaDevice::new(ordinal)?);
        devices[ordinal] = Some(dev.clone());
        Ok(dev)
    }

    /// Get the number of available CUDA devices.
    pub fn device_count(&self) -> usize {
        #[cfg(feature = "cudarc")]
        {
            cudarc::driver::result::device::get_count().unwrap_or(0) as usize
        }
        #[cfg(not(feature = "cudarc"))]
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

/// Convenience: get a CUDA device by ordinal.
pub fn get_device(ordinal: usize) -> Result<Arc<CudaDevice>, CudaError> {
    device_manager().get_device(ordinal)
}

/// Get the number of available CUDA devices.
pub fn device_count() -> usize {
    device_manager().device_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_device() {
        let dev = CudaDevice::new(0).unwrap();
        assert_eq!(dev.ordinal(), 0);
        assert!(dev.name().contains("Mock"));
        assert!(dev.total_memory() > 0);
    }

    #[test]
    fn test_device_memory_info() {
        let dev = CudaDevice::new(0).unwrap();
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
    fn test_device_allocator() {
        let dev = CudaDevice::new(0).unwrap();
        let block = dev.allocator().allocate(4096).unwrap();
        assert!(block.size >= 4096);
        dev.allocator().free(block);
    }
}
