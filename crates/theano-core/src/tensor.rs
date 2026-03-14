use std::sync::Arc;

use parking_lot::RwLock;
use theano_types::{DType, Device, Layout, Shape, Result, TheanoError};

use crate::storage::Storage;

/// Opaque handle for autograd graph linkage.
///
/// The autograd crate will define concrete implementations of this trait.
pub trait GradFn: Send + Sync {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>>;
    fn name(&self) -> &str;
}

/// The inner representation of a Tensor.
///
/// Users never interact with this directly — they hold `Tensor` which is `Arc<TensorInner>`.
pub(crate) struct TensorInner {
    /// The raw data buffer on the target device.
    pub storage: Storage,
    /// Dimensions of the tensor.
    pub shape: Vec<usize>,
    /// Strides for each dimension (in elements, not bytes).
    pub strides: Vec<usize>,
    /// Element offset into storage.
    pub offset: usize,
    /// Data type.
    pub dtype: DType,
    /// Which device the data lives on.
    pub device: Device,
    /// Memory layout.
    pub layout: Layout,
    /// Whether this tensor participates in autograd.
    pub requires_grad: bool,
    /// Accumulated gradient (populated by backward()).
    pub grad: RwLock<Option<Tensor>>,
    /// Link to the autograd graph node that produced this tensor.
    pub grad_fn: Option<Arc<dyn GradFn>>,
}

/// A multi-dimensional array of numbers on a compute device.
///
/// `Tensor` is the fundamental data structure in Theano, equivalent to `torch.Tensor`.
/// It is cheaply clonable (shared ownership via `Arc`).
///
/// # Examples
/// ```ignore
/// use theano_core::Tensor;
/// let a = Tensor::zeros(&[2, 3]);
/// let b = Tensor::ones(&[2, 3]);
/// let c = &a + &b;
/// ```
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorInner>,
}

// SAFETY: TensorInner fields are either immutable after construction or protected by RwLock.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    /// Create a tensor from its constituent parts.
    pub(crate) fn from_parts(
        storage: Storage,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                storage,
                shape,
                strides,
                offset,
                dtype,
                device,
                layout: Layout::Dense,
                requires_grad: false,
                grad: RwLock::new(None),
                grad_fn: None,
            }),
        }
    }

    /// Create a tensor with autograd metadata.
    pub fn from_parts_with_grad(
        storage: Storage,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
        dtype: DType,
        device: Device,
        requires_grad: bool,
        grad_fn: Option<Arc<dyn GradFn>>,
    ) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                storage,
                shape,
                strides,
                offset,
                dtype,
                device,
                layout: Layout::Dense,
                requires_grad,
                grad: RwLock::new(None),
                grad_fn,
            }),
        }
    }

    // ---- Metadata accessors ----

    /// The shape (dimensions) of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    /// Shape as a `Shape` object.
    pub fn size(&self) -> Shape {
        Shape::new(self.inner.shape.clone())
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.inner.shape.len()
    }

    /// Alias for ndim() — PyTorch compat.
    pub fn dim(&self) -> usize {
        self.ndim()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.inner.shape.is_empty() {
            1
        } else {
            self.inner.shape.iter().product()
        }
    }

    /// The strides of this tensor (in elements).
    pub fn strides(&self) -> &[usize] {
        &self.inner.strides
    }

    /// Element offset into the underlying storage.
    pub fn storage_offset(&self) -> usize {
        self.inner.offset
    }

    /// Data type of the tensor elements.
    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    /// The device this tensor lives on.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// The memory layout.
    pub fn layout(&self) -> Layout {
        self.inner.layout
    }

    /// Whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    /// Get the gradient of this tensor, if it exists.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.grad.read().clone()
    }

    /// Get the grad_fn (autograd node) of this tensor.
    pub fn grad_fn(&self) -> Option<Arc<dyn GradFn>> {
        self.inner.grad_fn.clone()
    }

    /// Whether this is a leaf tensor (created by user, not by an operation).
    pub fn is_leaf(&self) -> bool {
        self.inner.grad_fn.is_none()
    }

    /// Set the gradient on this tensor. Used by the autograd engine.
    pub fn set_grad(&self, grad: Tensor) {
        let mut g = self.inner.grad.write();
        *g = Some(grad);
    }

    /// Stable identity based on the Arc pointer. Two tensors sharing the same
    /// Arc<TensorInner> will have the same id. Used by the autograd engine.
    pub fn data_ptr_id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Whether this tensor is contiguous in memory (row-major / C-contiguous).
    pub fn is_contiguous(&self) -> bool {
        if self.inner.shape.is_empty() {
            return true;
        }
        let expected = Shape::new(self.inner.shape.clone()).contiguous_strides();
        self.inner.strides == expected
    }

    /// Whether this tensor is a scalar (0-dimensional).
    pub fn is_scalar(&self) -> bool {
        self.inner.shape.is_empty()
    }

    /// Get the reference to the underlying storage.
    pub fn storage(&self) -> &Storage {
        &self.inner.storage
    }

    /// Get the inner Arc (for creating views).
    pub(crate) fn inner(&self) -> &TensorInner {
        &self.inner
    }

    /// Set requires_grad on this tensor. Returns a new tensor with the flag set.
    pub fn requires_grad_(self, requires_grad: bool) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: self.inner.shape.clone(),
                strides: self.inner.strides.clone(),
                offset: self.inner.offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad,
                grad: RwLock::new(None),
                grad_fn: None,
            }),
        }
    }

    /// Detach from the autograd graph, returning a tensor that shares storage but has no grad_fn.
    pub fn detach(&self) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: self.inner.shape.clone(),
                strides: self.inner.strides.clone(),
                offset: self.inner.offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: false,
                grad: RwLock::new(None),
                grad_fn: None,
            }),
        }
    }

    // ---- Device transfer (like torch.Tensor.to / .cpu() / .cuda()) ----

    /// Move this tensor to a different device.
    ///
    /// Like `torch.Tensor.to(device)`. Returns a new tensor on the target device.
    /// If already on the target device, returns a cheap clone (shared storage).
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::ones(&[2, 3]);
    /// let x_gpu = x.to(&Device::Cuda(0))?;
    /// let x_cpu = x_gpu.to(&Device::Cpu)?;
    /// ```
    pub fn to(&self, device: &Device) -> Result<Tensor> {
        if self.device() == device {
            return Ok(self.clone());
        }
        let new_storage = self.inner.storage.to_device(
            device,
            &self.inner.shape,
            &self.inner.strides,
            self.inner.offset,
        )?;
        // After transfer, data is contiguous — reset strides and offset
        let strides = Shape::new(self.inner.shape.clone()).contiguous_strides();
        Ok(Tensor::from_parts_with_grad(
            new_storage,
            self.inner.shape.clone(),
            strides,
            0,
            self.inner.dtype,
            device.clone(),
            self.inner.requires_grad,
            None, // detach from graph on device transfer (like PyTorch)
        ))
    }

    /// Move this tensor to CPU. Shorthand for `.to(&Device::Cpu)`.
    ///
    /// Like `torch.Tensor.cpu()`.
    pub fn cpu(&self) -> Result<Tensor> {
        self.to(&Device::Cpu)
    }

    /// Move this tensor to CUDA device 0. Shorthand for `.to(&Device::Cuda(0))`.
    ///
    /// Like `torch.Tensor.cuda()`.
    pub fn cuda(&self) -> Result<Tensor> {
        self.to(&Device::Cuda(0))
    }

    /// Move this tensor to a specific CUDA device.
    ///
    /// Like `torch.Tensor.cuda(device=n)`.
    pub fn cuda_device(&self, ordinal: usize) -> Result<Tensor> {
        self.to(&Device::Cuda(ordinal))
    }

    /// Whether this tensor is on CPU.
    pub fn is_cpu(&self) -> bool {
        self.inner.device.is_cpu()
    }

    /// Whether this tensor is on a CUDA device.
    pub fn is_cuda(&self) -> bool {
        self.inner.device.is_cuda()
    }

    /// Convert to f64 Vec for debugging/testing (copies to CPU).
    pub fn to_vec_f64(&self) -> Result<Vec<f64>> {
        self.inner
            .storage
            .to_f64_vec(&self.inner.shape, &self.inner.strides, self.inner.offset)
    }

    /// Get a single scalar value from a 0-d or 1-element tensor.
    pub fn item(&self) -> Result<f64> {
        if self.numel() != 1 {
            return Err(TheanoError::runtime(format!(
                "a Tensor with {} elements cannot be converted to Scalar",
                self.numel()
            )));
        }
        let v = self.to_vec_f64()?;
        Ok(v[0])
    }

    /// Normalize a potentially negative dimension index.
    pub fn normalize_dim(&self, dim: i64) -> Result<usize> {
        let ndim = self.ndim() as i64;
        let d = if dim < 0 { dim + ndim } else { dim };
        if d < 0 || d >= ndim {
            return Err(TheanoError::DimensionOutOfRange {
                got: dim,
                min: -ndim,
                max: ndim,
            });
        }
        Ok(d as usize)
    }
}
