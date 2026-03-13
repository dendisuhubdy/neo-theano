use theano_backend::BackendStorage;
use theano_types::{DType, Device, Result};

/// Type-erased storage wrapper that dispatches to backend-specific storage at runtime.
///
/// This is the runtime-dispatch path (like PyTorch). For compile-time dispatch,
/// use `Tensor<B: Backend>` directly.
pub enum Storage {
    Cpu(Box<dyn BackendStorageBoxed>),
}

/// Object-safe wrapper around BackendStorage for dynamic dispatch.
///
/// We need this because `BackendStorage` uses `Sized` return types.
pub trait BackendStorageBoxed: Send + Sync {
    fn device(&self) -> Device;
    fn dtype(&self) -> DType;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn to_f64_vec(&self, shape: &[usize], strides: &[usize], offset: usize) -> Result<Vec<f64>>;
    fn as_any(&self) -> &dyn std::any::Any;
    fn clone_box(&self) -> Box<dyn BackendStorageBoxed>;
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(s) => Storage::Cpu(s.clone_box()),
        }
    }
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(s) => s.device(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(s) => s.dtype(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Storage::Cpu(s) => s.is_empty(),
        }
    }

    pub fn to_f64_vec(&self, shape: &[usize], strides: &[usize], offset: usize) -> Result<Vec<f64>> {
        match self {
            Storage::Cpu(s) => s.to_f64_vec(shape, strides, offset),
        }
    }

    pub fn as_any(&self) -> &dyn std::any::Any {
        match self {
            Storage::Cpu(s) => s.as_any(),
        }
    }
}
