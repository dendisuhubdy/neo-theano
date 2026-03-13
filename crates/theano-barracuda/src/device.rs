//! BarraCUDA device management for AMD GPUs.

use std::sync::Arc;

/// Represents an AMD GPU device accessed via BarraCUDA-compiled kernels.
#[derive(Clone)]
pub struct BarracudaDevice {
    ordinal: usize,
    name: String,
    total_memory: usize,
}

impl BarracudaDevice {
    /// Create a new BarraCUDA device handle.
    pub fn new(ordinal: usize) -> Self {
        Self {
            ordinal,
            name: format!("AMD GPU {} (BarraCUDA)", ordinal),
            total_memory: 8 * 1024 * 1024 * 1024, // 8 GB default
        }
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn total_memory(&self) -> usize {
        self.total_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barracuda_device() {
        let dev = BarracudaDevice::new(0);
        assert_eq!(dev.ordinal(), 0);
        assert!(dev.name().contains("BarraCUDA"));
    }
}
