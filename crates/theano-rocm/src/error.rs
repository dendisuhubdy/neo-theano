/// ROCm/HIP-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    #[error("ROCm/HIP not available (driver not found)")]
    NotAvailable,

    #[error("invalid ROCm device ordinal {ordinal}: only {count} devices found")]
    InvalidDevice { ordinal: usize, count: usize },

    #[error("ROCm out of memory on device {device}: requested {requested} bytes, {free} free of {total} total")]
    OutOfMemory {
        device: usize,
        requested: usize,
        free: usize,
        total: usize,
    },

    #[error("HIP driver error: {msg}")]
    DriverError { msg: String },

    #[error("HIP kernel launch error: {msg}")]
    KernelError { msg: String },

    #[error("rocBLAS error: {msg}")]
    RocblasError { msg: String },

    #[error("MIOpen error: {msg}")]
    MiopenError { msg: String },

    #[error("ROCm error: {msg}")]
    Other { msg: String },
}

impl RocmError {
    pub fn driver(msg: impl Into<String>) -> Self {
        Self::DriverError { msg: msg.into() }
    }

    pub fn kernel(msg: impl Into<String>) -> Self {
        Self::KernelError { msg: msg.into() }
    }

    pub fn rocblas(msg: impl Into<String>) -> Self {
        Self::RocblasError { msg: msg.into() }
    }

    pub fn miopen(msg: impl Into<String>) -> Self {
        Self::MiopenError { msg: msg.into() }
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other { msg: msg.into() }
    }
}

impl From<RocmError> for theano_types::TheanoError {
    fn from(e: RocmError) -> Self {
        match e {
            RocmError::OutOfMemory {
                device,
                requested: _,
                free: _,
                total: _,
            } => theano_types::TheanoError::OutOfMemory {
                device: theano_types::Device::Rocm(device),
                msg: e.to_string(),
            },
            _ => theano_types::TheanoError::RuntimeError {
                msg: e.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = RocmError::NotAvailable;
        assert!(e.to_string().contains("ROCm/HIP not available"));

        let e = RocmError::InvalidDevice { ordinal: 5, count: 2 };
        assert!(e.to_string().contains("5"));
        assert!(e.to_string().contains("2"));

        let e = RocmError::OutOfMemory {
            device: 0,
            requested: 1024,
            free: 100,
            total: 8192,
        };
        assert!(e.to_string().contains("1024"));
    }

    #[test]
    fn test_error_conversion() {
        let e = RocmError::OutOfMemory {
            device: 0,
            requested: 4096,
            free: 100,
            total: 8192,
        };
        let theano_err: theano_types::TheanoError = e.into();
        match theano_err {
            theano_types::TheanoError::OutOfMemory { device, .. } => {
                assert_eq!(device, theano_types::Device::Rocm(0));
            }
            _ => panic!("expected OutOfMemory variant"),
        }
    }

    #[test]
    fn test_error_conversion_runtime() {
        let e = RocmError::driver("test driver error");
        let theano_err: theano_types::TheanoError = e.into();
        match theano_err {
            theano_types::TheanoError::RuntimeError { msg } => {
                assert!(msg.contains("driver error"));
            }
            _ => panic!("expected RuntimeError variant"),
        }
    }

    #[test]
    fn test_convenience_constructors() {
        let e = RocmError::driver("bad driver");
        assert!(matches!(e, RocmError::DriverError { .. }));

        let e = RocmError::kernel("launch failed");
        assert!(matches!(e, RocmError::KernelError { .. }));

        let e = RocmError::rocblas("gemm error");
        assert!(matches!(e, RocmError::RocblasError { .. }));

        let e = RocmError::miopen("conv error");
        assert!(matches!(e, RocmError::MiopenError { .. }));

        let e = RocmError::other("something");
        assert!(matches!(e, RocmError::Other { .. }));
    }
}
