use crate::{DType, Device};

/// Result type alias for Theano operations.
pub type Result<T> = std::result::Result<T, TheanoError>;

/// Unified error type for all Theano operations.
#[derive(Debug, thiserror::Error)]
pub enum TheanoError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("invalid shape: {msg}")]
    InvalidShape { msg: String },

    #[error("dtype mismatch: expected {expected}, got {got}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("unsupported dtype {dtype} for operation {op}")]
    UnsupportedDType { dtype: DType, op: String },

    #[error("device mismatch: expected {expected}, got {got}")]
    DeviceMismatch { expected: Device, got: Device },

    #[error("unsupported device {device} for operation {op}")]
    UnsupportedDevice { device: Device, op: String },

    #[error("index out of bounds: index {index} for dimension of size {size}")]
    IndexOutOfBounds { index: i64, size: usize },

    #[error("dimension out of range: expected [{min}, {max}), got {got}")]
    DimensionOutOfRange { got: i64, min: i64, max: i64 },

    #[error("not contiguous: operation requires contiguous tensor")]
    NotContiguous,

    #[error("shapes cannot be broadcast together: {a:?} and {b:?}")]
    BroadcastError { a: Vec<usize>, b: Vec<usize> },

    #[error("autograd error: {msg}")]
    AutogradError { msg: String },

    #[error("runtime error: {msg}")]
    RuntimeError { msg: String },

    #[error("out of memory on device {device}: {msg}")]
    OutOfMemory { device: Device, msg: String },

    #[error("not implemented: {msg}")]
    NotImplemented { msg: String },

    #[error("invalid argument: {msg}")]
    InvalidArgument { msg: String },
}

impl TheanoError {
    pub fn shape_mismatch(expected: &[usize], got: &[usize]) -> Self {
        Self::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        }
    }

    pub fn broadcast_error(a: &[usize], b: &[usize]) -> Self {
        Self::BroadcastError {
            a: a.to_vec(),
            b: b.to_vec(),
        }
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        Self::RuntimeError { msg: msg.into() }
    }

    pub fn not_implemented(msg: impl Into<String>) -> Self {
        Self::NotImplemented { msg: msg.into() }
    }

    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Self::InvalidArgument { msg: msg.into() }
    }
}
