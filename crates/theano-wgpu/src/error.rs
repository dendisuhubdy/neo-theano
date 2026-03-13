//! WebGPU-specific error types.

/// Errors specific to the WebGPU backend.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("WebGPU not available")]
    NotAvailable,

    #[error("WebGPU adapter not found")]
    AdapterNotFound,

    #[error("WebGPU device error: {msg}")]
    DeviceError { msg: String },

    #[error("WebGPU shader error: {msg}")]
    ShaderError { msg: String },

    #[error("WebGPU buffer error: {msg}")]
    BufferError { msg: String },

    #[error("WebGPU out of memory on device {device}: requested {requested} bytes")]
    OutOfMemory {
        device: usize,
        requested: usize,
    },

    #[error("WebGPU error: {msg}")]
    Other { msg: String },
}

impl WgpuError {
    pub fn device(msg: impl Into<String>) -> Self {
        Self::DeviceError { msg: msg.into() }
    }

    pub fn shader(msg: impl Into<String>) -> Self {
        Self::ShaderError { msg: msg.into() }
    }

    pub fn buffer(msg: impl Into<String>) -> Self {
        Self::BufferError { msg: msg.into() }
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other { msg: msg.into() }
    }
}

impl From<WgpuError> for theano_types::TheanoError {
    fn from(e: WgpuError) -> Self {
        match e {
            WgpuError::OutOfMemory { device, .. } => theano_types::TheanoError::OutOfMemory {
                device: theano_types::Device::Wgpu(device),
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
        let e = WgpuError::NotAvailable;
        assert_eq!(e.to_string(), "WebGPU not available");

        let e = WgpuError::device("test error");
        assert!(e.to_string().contains("test error"));
    }

    #[test]
    fn test_error_conversion() {
        let e = WgpuError::Other { msg: "fail".to_string() };
        let te: theano_types::TheanoError = e.into();
        assert!(matches!(te, theano_types::TheanoError::RuntimeError { .. }));
    }

    #[test]
    fn test_oom_conversion() {
        let e = WgpuError::OutOfMemory { device: 0, requested: 1024 };
        let te: theano_types::TheanoError = e.into();
        assert!(matches!(te, theano_types::TheanoError::OutOfMemory { .. }));
    }
}
