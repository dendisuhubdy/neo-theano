use std::fmt;

/// The type of compute device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Wgpu,
    OneApi,
}

/// Represents a compute device, mirroring `torch.device`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Rocm(usize),
    Metal(usize),
    Wgpu(usize),
    OneApi(usize),
}

impl Device {
    /// Get the device type.
    pub fn device_type(&self) -> DeviceType {
        match self {
            Device::Cpu => DeviceType::Cpu,
            Device::Cuda(_) => DeviceType::Cuda,
            Device::Rocm(_) => DeviceType::Rocm,
            Device::Metal(_) => DeviceType::Metal,
            Device::Wgpu(_) => DeviceType::Wgpu,
            Device::OneApi(_) => DeviceType::OneApi,
        }
    }

    /// Get the device ordinal (0 for CPU).
    pub fn ordinal(&self) -> usize {
        match self {
            Device::Cpu => 0,
            Device::Cuda(i) | Device::Rocm(i) | Device::Metal(i) | Device::Wgpu(i) | Device::OneApi(i) => *i,
        }
    }

    /// Whether this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Whether this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
            Device::Rocm(i) => write!(f, "rocm:{i}"),
            Device::Metal(i) => write!(f, "metal:{i}"),
            Device::Wgpu(i) => write!(f, "wgpu:{i}"),
            Device::OneApi(i) => write!(f, "oneapi:{i}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Rocm(1).to_string(), "rocm:1");
    }

    #[test]
    fn test_device_type() {
        assert_eq!(Device::Cpu.device_type(), DeviceType::Cpu);
        assert_eq!(Device::Cuda(0).device_type(), DeviceType::Cuda);
    }
}
