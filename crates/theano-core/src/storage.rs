use theano_backend::BackendStorage;
use theano_types::{DType, Device, DeviceType, Result, TheanoError};

/// Type-erased storage wrapper that dispatches to backend-specific storage at runtime.
///
/// This is the runtime-dispatch path (like PyTorch). For compile-time dispatch,
/// use `Tensor<B: Backend>` directly.
pub enum Storage {
    Cpu(Box<dyn BackendStorageBoxed>),
    Cuda(Box<dyn BackendStorageBoxed>),
    Rocm(Box<dyn BackendStorageBoxed>),
    Metal(Box<dyn BackendStorageBoxed>),
    Wgpu(Box<dyn BackendStorageBoxed>),
    OneApi(Box<dyn BackendStorageBoxed>),
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
    /// Transfer data to a different device. Returns new storage on the target device.
    ///
    /// The default implementation supports CPU↔CPU (clone) and materializes data
    /// through f64 for cross-device transfers. Backend-specific implementations
    /// can override this for efficient device-to-device copies (e.g. cudaMemcpy).
    fn to_device(&self, device: &Device, shape: &[usize], strides: &[usize], offset: usize) -> Result<Box<dyn BackendStorageBoxed>>;
}

impl Clone for Storage {
    fn clone(&self) -> Self {
        match self {
            Storage::Cpu(s) => Storage::Cpu(s.clone_box()),
            Storage::Cuda(s) => Storage::Cuda(s.clone_box()),
            Storage::Rocm(s) => Storage::Rocm(s.clone_box()),
            Storage::Metal(s) => Storage::Metal(s.clone_box()),
            Storage::Wgpu(s) => Storage::Wgpu(s.clone_box()),
            Storage::OneApi(s) => Storage::OneApi(s.clone_box()),
        }
    }
}

/// Helper to get the inner storage regardless of variant.
macro_rules! dispatch_storage {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Storage::Cpu(s) => s.$method($($arg),*),
            Storage::Cuda(s) => s.$method($($arg),*),
            Storage::Rocm(s) => s.$method($($arg),*),
            Storage::Metal(s) => s.$method($($arg),*),
            Storage::Wgpu(s) => s.$method($($arg),*),
            Storage::OneApi(s) => s.$method($($arg),*),
        }
    };
}

impl Storage {
    pub fn device(&self) -> Device {
        dispatch_storage!(self, device)
    }

    pub fn dtype(&self) -> DType {
        dispatch_storage!(self, dtype)
    }

    pub fn len(&self) -> usize {
        dispatch_storage!(self, len)
    }

    pub fn is_empty(&self) -> bool {
        dispatch_storage!(self, is_empty)
    }

    pub fn to_f64_vec(&self, shape: &[usize], strides: &[usize], offset: usize) -> Result<Vec<f64>> {
        dispatch_storage!(self, to_f64_vec, shape, strides, offset)
    }

    pub fn as_any(&self) -> &dyn std::any::Any {
        dispatch_storage!(self, as_any)
    }

    /// Wrap a boxed storage into the correct Storage variant for the given device.
    pub fn from_boxed(storage: Box<dyn BackendStorageBoxed>, device: &Device) -> Self {
        match device.device_type() {
            DeviceType::Cpu => Storage::Cpu(storage),
            DeviceType::Cuda => Storage::Cuda(storage),
            DeviceType::Rocm => Storage::Rocm(storage),
            DeviceType::Metal => Storage::Metal(storage),
            DeviceType::Wgpu => Storage::Wgpu(storage),
            DeviceType::OneApi => Storage::OneApi(storage),
        }
    }

    /// Transfer storage to a different device.
    ///
    /// Like `torch.Tensor.to(device)` at the storage level.
    pub fn to_device(&self, device: &Device, shape: &[usize], strides: &[usize], offset: usize) -> Result<Storage> {
        if self.device() == *device {
            return Ok(self.clone());
        }
        let new_storage = dispatch_storage!(self, to_device, device, shape, strides, offset);
        new_storage.map(|s| Storage::from_boxed(s, device))
    }
}
