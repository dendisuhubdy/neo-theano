/// Metal-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("Metal not available (requires macOS with Apple GPU)")]
    NotAvailable,

    #[error("Metal device error: {msg}")]
    DeviceError { msg: String },

    #[error("Metal command buffer error: {msg}")]
    CommandBufferError { msg: String },

    #[error("Metal shader compilation error: {msg}")]
    ShaderError { msg: String },

    #[error("Metal out of memory: {msg}")]
    OutOfMemory { msg: String },

    #[error("Metal error: {msg}")]
    Other { msg: String },
}

impl MetalError {
    pub fn device(msg: impl Into<String>) -> Self {
        Self::DeviceError { msg: msg.into() }
    }

    pub fn command_buffer(msg: impl Into<String>) -> Self {
        Self::CommandBufferError { msg: msg.into() }
    }

    pub fn shader(msg: impl Into<String>) -> Self {
        Self::ShaderError { msg: msg.into() }
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other { msg: msg.into() }
    }
}

impl From<MetalError> for theano_types::TheanoError {
    fn from(e: MetalError) -> Self {
        match e {
            MetalError::OutOfMemory { msg } => theano_types::TheanoError::OutOfMemory {
                device: theano_types::Device::Metal(0),
                msg,
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
        let e = MetalError::NotAvailable;
        assert!(e.to_string().contains("Metal not available"));
    }

    #[test]
    fn test_error_device() {
        let e = MetalError::device("GPU hung");
        assert!(e.to_string().contains("GPU hung"));
    }

    #[test]
    fn test_error_shader() {
        let e = MetalError::shader("syntax error at line 5");
        assert!(e.to_string().contains("syntax error"));
    }

    #[test]
    fn test_error_command_buffer() {
        let e = MetalError::command_buffer("commit failed");
        assert!(e.to_string().contains("commit failed"));
    }

    #[test]
    fn test_error_conversion_to_theano() {
        let e = MetalError::Other { msg: "test".into() };
        let te: theano_types::TheanoError = e.into();
        assert!(te.to_string().contains("Metal error: test"));
    }

    #[test]
    fn test_oom_conversion() {
        let e = MetalError::OutOfMemory { msg: "16GB exhausted".into() };
        let te: theano_types::TheanoError = e.into();
        match te {
            theano_types::TheanoError::OutOfMemory { device, msg } => {
                assert_eq!(device, theano_types::Device::Metal(0));
                assert!(msg.contains("16GB"));
            }
            _ => panic!("Expected OutOfMemory variant"),
        }
    }
}
