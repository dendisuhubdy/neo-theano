//! Metal device management for Apple Silicon GPUs.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::MetalError;

/// Apple Silicon generation identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppleSiliconGen {
    /// Apple M1 (2020)
    M1,
    /// Apple M2 (2022)
    M2,
    /// Apple M3 (2023)
    M3,
    /// Apple M4 (2024)
    M4,
    /// Unknown or future generation
    Unknown,
}

impl AppleSiliconGen {
    /// Detect generation from the device name string.
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("m4") {
            AppleSiliconGen::M4
        } else if lower.contains("m3") {
            AppleSiliconGen::M3
        } else if lower.contains("m2") {
            AppleSiliconGen::M2
        } else if lower.contains("m1") {
            AppleSiliconGen::M1
        } else {
            AppleSiliconGen::Unknown
        }
    }

    /// Maximum number of GPU cores for this generation (base model).
    pub fn base_gpu_cores(&self) -> u32 {
        match self {
            AppleSiliconGen::M1 => 8,
            AppleSiliconGen::M2 => 10,
            AppleSiliconGen::M3 => 10,
            AppleSiliconGen::M4 => 10,
            AppleSiliconGen::Unknown => 8,
        }
    }
}

/// Metal device properties (Apple Silicon specific).
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Human-readable device name.
    pub name: String,
    /// Total unified memory in bytes.
    pub total_memory: usize,
    /// Apple Silicon generation.
    pub silicon_gen: AppleSiliconGen,
    /// Number of GPU cores.
    pub gpu_core_count: u32,
    /// Maximum threadgroup (workgroup) size.
    pub max_threadgroup_size: u32,
    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: u32,
    /// Maximum buffer length in bytes.
    pub max_buffer_length: usize,
    /// Whether the device supports unified memory (always true for Apple Silicon).
    pub has_unified_memory: bool,
    /// Maximum recommended working set size.
    pub recommended_max_working_set_size: usize,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            name: "Mock Apple GPU".to_string(),
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GB unified memory
            silicon_gen: AppleSiliconGen::M1,
            gpu_core_count: 8,
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024, // 256 MB
            has_unified_memory: true,
            recommended_max_working_set_size: 12 * 1024 * 1024 * 1024, // 12 GB
        }
    }
}

/// Represents a single Metal GPU device.
///
/// Manages the Metal device handle, command queue, and shader library.
/// On Apple Silicon, GPU and CPU share unified memory, which allows
/// zero-copy data sharing when appropriate.
#[derive(Clone)]
pub struct MetalDevice {
    /// Device ordinal (0 for the default GPU, typically only one on Apple Silicon).
    ordinal: usize,
    /// Device properties.
    props: Arc<DeviceProperties>,
    /// Tracks current memory usage for the mock allocator.
    allocated_bytes: Arc<std::sync::atomic::AtomicUsize>,
    /// Mock address counter.
    mock_addr: Arc<std::sync::atomic::AtomicUsize>,

    // When metal-api feature is enabled, these hold the real Metal handles:
    // device: metal::Device,
    // command_queue: metal::CommandQueue,
    // library: metal::Library,
}

impl MetalDevice {
    /// Create a new Metal device handle.
    ///
    /// When the `metal-api` feature is enabled, this initializes the real Metal device.
    /// Otherwise, it creates a mock device for development/testing.
    pub fn new(ordinal: usize) -> Result<Self, MetalError> {
        #[cfg(feature = "metal-api")]
        {
            return Self::new_metal(ordinal);
        }

        #[cfg(not(feature = "metal-api"))]
        {
            Self::new_mock(ordinal)
        }
    }

    /// Create a mock device (no Metal API required).
    #[cfg(not(feature = "metal-api"))]
    fn new_mock(ordinal: usize) -> Result<Self, MetalError> {
        let gen = AppleSiliconGen::M1;
        Ok(Self {
            ordinal,
            props: Arc::new(DeviceProperties {
                name: format!("Mock Apple M1 GPU {ordinal}"),
                silicon_gen: gen,
                gpu_core_count: gen.base_gpu_cores(),
                ..DeviceProperties::default()
            }),
            allocated_bytes: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            mock_addr: Arc::new(std::sync::atomic::AtomicUsize::new(0x2000_0000)),
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

    /// Get total device memory in bytes (unified memory on Apple Silicon).
    pub fn total_memory(&self) -> usize {
        self.props.total_memory
    }

    /// Get the Apple Silicon generation.
    pub fn silicon_gen(&self) -> AppleSiliconGen {
        self.props.silicon_gen
    }

    /// Whether the device has unified memory (always true on Apple Silicon).
    pub fn has_unified_memory(&self) -> bool {
        self.props.has_unified_memory
    }

    /// Synchronize this device (wait for all pending GPU operations).
    pub fn synchronize(&self) -> Result<(), MetalError> {
        #[cfg(feature = "metal-api")]
        {
            // command_queue.insertDebugCaptureBoundary() or wait on command buffer
        }
        Ok(())
    }

    /// Get memory info: (free, total).
    pub fn memory_info(&self) -> Result<(usize, usize), MetalError> {
        let used = self.allocated_bytes.load(std::sync::atomic::Ordering::Relaxed);
        let total = self.props.total_memory;
        let free = total.saturating_sub(used);
        Ok((free, total))
    }

    /// Allocate a mock GPU buffer, returning a mock pointer.
    pub(crate) fn mock_alloc(&self, size: usize) -> usize {
        self.allocated_bytes.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        self.mock_addr.fetch_add(size, std::sync::atomic::Ordering::Relaxed)
    }

    /// Free a mock GPU buffer.
    pub(crate) fn mock_free(&self, size: usize) {
        self.allocated_bytes.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Global device manager -- tracks all Metal devices.
pub struct DeviceManager {
    devices: Mutex<Vec<Option<Arc<MetalDevice>>>>,
}

impl DeviceManager {
    /// Create a new device manager.
    pub fn new() -> Self {
        Self {
            devices: Mutex::new(Vec::new()),
        }
    }

    /// Get or create a Metal device by ordinal.
    pub fn get_device(&self, ordinal: usize) -> Result<Arc<MetalDevice>, MetalError> {
        let mut devices = self.devices.lock();

        // Extend the vec if needed
        while devices.len() <= ordinal {
            devices.push(None);
        }

        if let Some(dev) = &devices[ordinal] {
            return Ok(dev.clone());
        }

        let dev = Arc::new(MetalDevice::new(ordinal)?);
        devices[ordinal] = Some(dev.clone());
        Ok(dev)
    }

    /// Get the number of available Metal devices.
    pub fn device_count(&self) -> usize {
        #[cfg(feature = "metal-api")]
        {
            // Use MTLCopyAllDevices() or similar
            1
        }
        #[cfg(not(feature = "metal-api"))]
        {
            // Mock: report 1 device for testing (Apple Silicon typically has 1 GPU)
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

/// Convenience: get a Metal device by ordinal.
pub fn get_device(ordinal: usize) -> Result<Arc<MetalDevice>, MetalError> {
    device_manager().get_device(ordinal)
}

/// Get the number of available Metal devices.
pub fn device_count() -> usize {
    device_manager().device_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_device() {
        let dev = MetalDevice::new(0).unwrap();
        assert_eq!(dev.ordinal(), 0);
        assert!(dev.name().contains("Mock"));
        assert!(dev.name().contains("Apple"));
        assert!(dev.total_memory() > 0);
    }

    #[test]
    fn test_device_properties() {
        let dev = MetalDevice::new(0).unwrap();
        let props = dev.properties();
        assert!(props.has_unified_memory);
        assert!(props.max_threadgroup_size > 0);
        assert!(props.gpu_core_count > 0);
    }

    #[test]
    fn test_device_silicon_gen() {
        let dev = MetalDevice::new(0).unwrap();
        assert_eq!(dev.silicon_gen(), AppleSiliconGen::M1);
        assert!(dev.has_unified_memory());
    }

    #[test]
    fn test_device_memory_info() {
        let dev = MetalDevice::new(0).unwrap();
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
    fn test_silicon_gen_detection() {
        assert_eq!(AppleSiliconGen::from_name("Apple M1 Pro"), AppleSiliconGen::M1);
        assert_eq!(AppleSiliconGen::from_name("Apple M2 Max"), AppleSiliconGen::M2);
        assert_eq!(AppleSiliconGen::from_name("Apple M3 Ultra"), AppleSiliconGen::M3);
        assert_eq!(AppleSiliconGen::from_name("Apple M4"), AppleSiliconGen::M4);
        assert_eq!(AppleSiliconGen::from_name("Intel Iris"), AppleSiliconGen::Unknown);
    }

    #[test]
    fn test_silicon_gen_gpu_cores() {
        assert_eq!(AppleSiliconGen::M1.base_gpu_cores(), 8);
        assert_eq!(AppleSiliconGen::M2.base_gpu_cores(), 10);
        assert_eq!(AppleSiliconGen::M3.base_gpu_cores(), 10);
        assert_eq!(AppleSiliconGen::M4.base_gpu_cores(), 10);
    }

    #[test]
    fn test_device_synchronize() {
        let dev = MetalDevice::new(0).unwrap();
        dev.synchronize().unwrap();
    }

    #[test]
    fn test_mock_alloc_free() {
        let dev = MetalDevice::new(0).unwrap();
        let ptr = dev.mock_alloc(4096);
        assert!(ptr > 0);
        let (free1, total) = dev.memory_info().unwrap();
        assert_eq!(total - free1, 4096);
        dev.mock_free(4096);
        let (free2, _) = dev.memory_info().unwrap();
        assert_eq!(free2, total);
    }

    #[test]
    fn test_global_device_manager() {
        let dev = get_device(0).unwrap();
        assert_eq!(dev.ordinal(), 0);
    }
}
