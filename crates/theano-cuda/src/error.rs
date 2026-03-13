/// CUDA-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA not available (driver not found)")]
    NotAvailable,

    #[error("invalid device ordinal {ordinal}: only {count} CUDA devices found")]
    InvalidDevice { ordinal: usize, count: usize },

    #[error("CUDA out of memory on device {device}: requested {requested} bytes, {free} free of {total} total")]
    OutOfMemory {
        device: usize,
        requested: usize,
        free: usize,
        total: usize,
    },

    #[error("CUDA driver error: {msg}")]
    DriverError { msg: String },

    #[error("CUDA kernel launch error: {msg}")]
    KernelError { msg: String },

    #[error("cuBLAS error: {msg}")]
    CublasError { msg: String },

    #[error("cuDNN error: {msg}")]
    CudnnError { msg: String },

    #[error("CUDA error: {msg}")]
    Other { msg: String },
}

impl CudaError {
    pub fn driver(msg: impl Into<String>) -> Self {
        Self::DriverError { msg: msg.into() }
    }

    pub fn kernel(msg: impl Into<String>) -> Self {
        Self::KernelError { msg: msg.into() }
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other { msg: msg.into() }
    }
}

impl From<CudaError> for theano_types::TheanoError {
    fn from(e: CudaError) -> Self {
        match e {
            CudaError::OutOfMemory {
                device,
                requested: _,
                free: _,
                total: _,
            } => theano_types::TheanoError::OutOfMemory {
                device: theano_types::Device::Cuda(device),
                msg: e.to_string(),
            },
            _ => theano_types::TheanoError::RuntimeError {
                msg: e.to_string(),
            },
        }
    }
}
