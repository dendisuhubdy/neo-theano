//! Intel oneAPI-specific error types.

/// Errors specific to the Intel oneAPI backend.
#[derive(Debug, thiserror::Error)]
pub enum OneApiError {
    #[error("Intel oneAPI runtime not available")]
    NotAvailable,

    #[error("invalid device ordinal {ordinal}: only {count} Intel GPU devices found")]
    InvalidDevice { ordinal: usize, count: usize },

    #[error("Intel GPU out of memory on device {device}: requested {requested} bytes")]
    OutOfMemory {
        device: usize,
        requested: usize,
    },

    #[error("Level Zero driver error: {msg}")]
    DriverError { msg: String },

    #[error("oneAPI kernel error: {msg}")]
    KernelError { msg: String },

    #[error("oneMKL error: {msg}")]
    MklError { msg: String },

    #[error("oneDNN error: {msg}")]
    DnnError { msg: String },

    #[error("Intel oneAPI error: {msg}")]
    Other { msg: String },
}

impl OneApiError {
    pub fn driver(msg: impl Into<String>) -> Self {
        Self::DriverError { msg: msg.into() }
    }

    pub fn kernel(msg: impl Into<String>) -> Self {
        Self::KernelError { msg: msg.into() }
    }

    pub fn mkl(msg: impl Into<String>) -> Self {
        Self::MklError { msg: msg.into() }
    }

    pub fn dnn(msg: impl Into<String>) -> Self {
        Self::DnnError { msg: msg.into() }
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other { msg: msg.into() }
    }
}

impl From<OneApiError> for theano_types::TheanoError {
    fn from(e: OneApiError) -> Self {
        match e {
            OneApiError::OutOfMemory { device, .. } => theano_types::TheanoError::OutOfMemory {
                device: theano_types::Device::OneApi(device),
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
        let e = OneApiError::NotAvailable;
        assert_eq!(e.to_string(), "Intel oneAPI runtime not available");

        let e = OneApiError::driver("test error");
        assert!(e.to_string().contains("test error"));

        let e = OneApiError::kernel("bad kernel");
        assert!(e.to_string().contains("bad kernel"));
    }

    #[test]
    fn test_error_conversion() {
        let e = OneApiError::Other { msg: "fail".to_string() };
        let te: theano_types::TheanoError = e.into();
        assert!(matches!(te, theano_types::TheanoError::RuntimeError { .. }));
    }

    #[test]
    fn test_oom_conversion() {
        let e = OneApiError::OutOfMemory { device: 0, requested: 1024 };
        let te: theano_types::TheanoError = e.into();
        assert!(matches!(te, theano_types::TheanoError::OutOfMemory { .. }));
    }

    #[test]
    fn test_invalid_device_display() {
        let e = OneApiError::InvalidDevice { ordinal: 5, count: 2 };
        assert!(e.to_string().contains("5"));
        assert!(e.to_string().contains("2"));
    }
}
