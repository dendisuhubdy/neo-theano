use theano_types::{Result, Shape, TheanoError};

use crate::tensor::Tensor;
use crate::tensor::TensorInner;
use std::sync::Arc;
use parking_lot::RwLock;

impl Tensor {
    /// Returns a tensor with the same data but a different shape. Like `torch.Tensor.reshape`.
    ///
    /// The total number of elements must remain the same. If the tensor is contiguous,
    /// this is a zero-copy view. Otherwise, a contiguous copy is made first.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let new_numel: usize = if new_shape.is_empty() {
            1
        } else {
            new_shape.iter().product()
        };
        if new_numel != self.numel() {
            return Err(TheanoError::InvalidShape {
                msg: format!(
                    "shape {:?} is invalid for input of size {}",
                    new_shape,
                    self.numel()
                ),
            });
        }

        // For contiguous tensors, just create a view with new shape/strides
        if self.is_contiguous() {
            let strides = Shape::new(new_shape.to_vec()).contiguous_strides();
            Ok(Tensor {
                inner: Arc::new(TensorInner {
                    storage: self.inner.storage.clone(),
                    shape: new_shape.to_vec(),
                    strides,
                    offset: self.inner.offset,
                    dtype: self.inner.dtype,
                    device: self.inner.device.clone(),
                    layout: self.inner.layout,
                    requires_grad: self.inner.requires_grad,
                    grad: RwLock::new(None),
                    grad_fn: self.inner.grad_fn.clone(),
                }),
            })
        } else {
            // TODO: make contiguous first, then reshape
            Err(TheanoError::not_implemented(
                "reshape on non-contiguous tensor",
            ))
        }
    }

    /// Alias for reshape. Like `torch.Tensor.view` with usize dimensions.
    pub fn view(&self, shape: &[usize]) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(TheanoError::runtime(
                "view size is not compatible with input tensor's size and stride \
                 (at least one dimension spans across two contiguous subspaces). \
                 Use .reshape(...) instead.",
            ));
        }
        self.reshape(shape)
    }

    /// Reshape with shared storage, supporting `-1` for one inferred dimension.
    /// Like `torch.Tensor.view(-1)` or `torch.Tensor.view(2, -1, 3)`.
    ///
    /// Requires the tensor to be contiguous. At most one dimension can be -1,
    /// which will be inferred from the total number of elements.
    pub fn view_i64(&self, shape: &[i64]) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(TheanoError::runtime(
                "view size is not compatible with input tensor's size and stride \
                 (at least one dimension spans across two contiguous subspaces). \
                 Use .reshape(...) instead.",
            ));
        }

        let numel = self.numel();
        let mut inferred_idx: Option<usize> = None;
        let mut known_product: usize = 1;

        for (i, &d) in shape.iter().enumerate() {
            if d == -1 {
                if inferred_idx.is_some() {
                    return Err(TheanoError::InvalidShape {
                        msg: "only one dimension can be inferred (-1)".to_string(),
                    });
                }
                inferred_idx = Some(i);
            } else if d < -1 {
                return Err(TheanoError::InvalidShape {
                    msg: format!("invalid shape dimension: {d}"),
                });
            } else {
                known_product *= d as usize;
            }
        }

        let resolved: Vec<usize> = if let Some(idx) = inferred_idx {
            if known_product == 0 {
                return Err(TheanoError::InvalidShape {
                    msg: "cannot infer dimension with zero-size known dimensions".to_string(),
                });
            }
            if numel % known_product != 0 {
                return Err(TheanoError::InvalidShape {
                    msg: format!(
                        "shape {:?} is invalid for input of size {}",
                        shape, numel
                    ),
                });
            }
            let inferred = numel / known_product;
            shape
                .iter()
                .enumerate()
                .map(|(i, &d)| if i == idx { inferred } else { d as usize })
                .collect()
        } else {
            shape.iter().map(|&d| d as usize).collect()
        };

        self.reshape(&resolved)
    }

    /// Transpose two dimensions. Like `torch.Tensor.transpose`.
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<Tensor> {
        let d0 = self.normalize_dim(dim0)?;
        let d1 = self.normalize_dim(dim1)?;

        let mut new_shape = self.inner.shape.clone();
        let mut new_strides = self.inner.strides.clone();
        new_shape.swap(d0, d1);
        new_strides.swap(d0, d1);

        Ok(Tensor {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: self.inner.offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: self.inner.requires_grad,
                grad: RwLock::new(None),
                grad_fn: self.inner.grad_fn.clone(),
            }),
        })
    }

    /// Transpose for 2-D tensors (matrix transpose). Like `torch.Tensor.t()`.
    pub fn t(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(TheanoError::runtime(format!(
                "t() expects a tensor with <= 2 dimensions, but self is {}D",
                self.ndim()
            )));
        }
        self.transpose(0, 1)
    }

    /// Permute the dimensions. Like `torch.Tensor.permute`.
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        if dims.len() != self.ndim() {
            return Err(TheanoError::InvalidShape {
                msg: format!(
                    "permute: number of dims {} doesn't match tensor ndim {}",
                    dims.len(),
                    self.ndim()
                ),
            });
        }

        let mut seen = vec![false; self.ndim()];
        for &d in dims {
            if d >= self.ndim() {
                return Err(TheanoError::DimensionOutOfRange {
                    got: d as i64,
                    min: 0,
                    max: self.ndim() as i64,
                });
            }
            if seen[d] {
                return Err(TheanoError::InvalidArgument {
                    msg: format!("permute: repeated dim {d}"),
                });
            }
            seen[d] = true;
        }

        let new_shape: Vec<usize> = dims.iter().map(|&d| self.inner.shape[d]).collect();
        let new_strides: Vec<usize> = dims.iter().map(|&d| self.inner.strides[d]).collect();

        Ok(Tensor {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: self.inner.offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: self.inner.requires_grad,
                grad: RwLock::new(None),
                grad_fn: self.inner.grad_fn.clone(),
            }),
        })
    }

    /// Return a contiguous tensor. If already contiguous, returns self (no copy).
    /// Like `torch.Tensor.contiguous`.
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        // Need to copy data in the correct order
        let data = self.to_vec_f64()?;
        let strides = Shape::new(self.inner.shape.clone()).contiguous_strides();
        let storage = crate::tensor_create::CpuF64Storage {
            data,
            dtype: self.inner.dtype,
        };
        Ok(Tensor::from_parts(
            crate::storage::Storage::Cpu(Box::new(storage)),
            self.inner.shape.clone(),
            strides,
            0,
            self.inner.dtype,
            self.inner.device.clone(),
        ))
    }

    /// Select a sub-tensor along a dimension. Like `torch.Tensor.select`.
    ///
    /// Reduces dimensionality by 1.
    pub fn select(&self, dim: i64, index: i64) -> Result<Tensor> {
        let d = self.normalize_dim(dim)?;
        let size = self.inner.shape[d];
        let idx = if index < 0 {
            (index + size as i64) as usize
        } else {
            index as usize
        };
        if idx >= size {
            return Err(TheanoError::IndexOutOfBounds {
                index,
                size,
            });
        }

        let new_offset = self.inner.offset + idx * self.inner.strides[d];
        let mut new_shape = self.inner.shape.clone();
        let mut new_strides = self.inner.strides.clone();
        new_shape.remove(d);
        new_strides.remove(d);

        Ok(Tensor {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: new_offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: self.inner.requires_grad,
                grad: RwLock::new(None),
                grad_fn: self.inner.grad_fn.clone(),
            }),
        })
    }

    /// Narrow (slice) along a dimension. Like `torch.Tensor.narrow`.
    pub fn narrow(&self, dim: i64, start: usize, length: usize) -> Result<Tensor> {
        let d = self.normalize_dim(dim)?;
        let size = self.inner.shape[d];
        if start + length > size {
            return Err(TheanoError::IndexOutOfBounds {
                index: (start + length) as i64,
                size,
            });
        }

        let new_offset = self.inner.offset + start * self.inner.strides[d];
        let mut new_shape = self.inner.shape.clone();
        new_shape[d] = length;

        Ok(Tensor {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: new_shape,
                strides: self.inner.strides.clone(),
                offset: new_offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: self.inner.requires_grad,
                grad: RwLock::new(None),
                grad_fn: self.inner.grad_fn.clone(),
            }),
        })
    }

    /// Add a dimension of size 1 at the given position. Like `torch.Tensor.unsqueeze`.
    pub fn unsqueeze(&self, dim: i64) -> Result<Tensor> {
        let ndim = self.ndim() as i64 + 1;
        let d = if dim < 0 { dim + ndim } else { dim };
        if d < 0 || d >= ndim {
            return Err(TheanoError::DimensionOutOfRange {
                got: dim,
                min: -ndim,
                max: ndim,
            });
        }
        let d = d as usize;

        let mut new_shape = self.inner.shape.clone();
        let mut new_strides = self.inner.strides.clone();

        let stride_val = if d < new_strides.len() {
            new_strides[d] * new_shape[d]
        } else if !new_strides.is_empty() {
            1
        } else {
            1
        };

        new_shape.insert(d, 1);
        new_strides.insert(d, stride_val);

        Ok(Tensor {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: self.inner.offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: self.inner.requires_grad,
                grad: RwLock::new(None),
                grad_fn: self.inner.grad_fn.clone(),
            }),
        })
    }

    /// Remove a dimension of size 1 at the given position. Like `torch.Tensor.squeeze`.
    pub fn squeeze(&self, dim: i64) -> Result<Tensor> {
        let d = self.normalize_dim(dim)?;
        if self.inner.shape[d] != 1 {
            // PyTorch just returns the same tensor if dim size != 1
            return Ok(self.clone());
        }

        let mut new_shape = self.inner.shape.clone();
        let mut new_strides = self.inner.strides.clone();
        new_shape.remove(d);
        new_strides.remove(d);

        Ok(Tensor {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: self.inner.offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: self.inner.requires_grad,
                grad: RwLock::new(None),
                grad_fn: self.inner.grad_fn.clone(),
            }),
        })
    }

    /// Squeeze all dimensions of size 1. Like `torch.Tensor.squeeze()` with no args.
    pub fn squeeze_all(&self) -> Tensor {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        for (i, &s) in self.inner.shape.iter().enumerate() {
            if s != 1 {
                new_shape.push(s);
                new_strides.push(self.inner.strides[i]);
            }
        }

        Tensor {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: self.inner.offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: self.inner.requires_grad,
                grad: RwLock::new(None),
                grad_fn: self.inner.grad_fn.clone(),
            }),
        }
    }

    /// Expand to a larger size (broadcast). Like `torch.Tensor.expand`.
    ///
    /// Expanded dimensions have stride 0 (no data copy).
    pub fn expand(&self, shape: &[usize]) -> Result<Tensor> {
        if shape.len() < self.ndim() {
            return Err(TheanoError::InvalidShape {
                msg: format!(
                    "expand: target shape {:?} has fewer dimensions than tensor shape {:?}",
                    shape,
                    self.shape()
                ),
            });
        }

        let extra_dims = shape.len() - self.ndim();
        let mut new_strides = vec![0usize; shape.len()];

        for i in 0..shape.len() {
            let src_dim = if i < extra_dims {
                1 // implicit leading dimension
            } else {
                self.inner.shape[i - extra_dims]
            };

            if shape[i] == src_dim {
                new_strides[i] = if i < extra_dims {
                    0
                } else {
                    self.inner.strides[i - extra_dims]
                };
            } else if src_dim == 1 {
                new_strides[i] = 0; // broadcast: stride = 0
            } else {
                return Err(TheanoError::InvalidShape {
                    msg: format!(
                        "expand: the expanded size {} must match the existing size {} at non-singleton dimension {}",
                        shape[i], src_dim, i
                    ),
                });
            }
        }

        Ok(Tensor {
            inner: Arc::new(TensorInner {
                storage: self.inner.storage.clone(),
                shape: shape.to_vec(),
                strides: new_strides,
                offset: self.inner.offset,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                layout: self.inner.layout,
                requires_grad: self.inner.requires_grad,
                grad: RwLock::new(None),
                grad_fn: self.inner.grad_fn.clone(),
            }),
        })
    }

    /// Flatten dimensions from start_dim to end_dim. Like `torch.Tensor.flatten`.
    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Result<Tensor> {
        let s = self.normalize_dim(start_dim)?;
        let e = self.normalize_dim(end_dim)?;
        if s > e {
            return Err(TheanoError::invalid_argument(format!(
                "flatten: start_dim {start_dim} must be <= end_dim {end_dim}"
            )));
        }

        let flat_size: usize = self.inner.shape[s..=e].iter().product();
        let mut new_shape = Vec::new();
        new_shape.extend_from_slice(&self.inner.shape[..s]);
        new_shape.push(flat_size);
        new_shape.extend_from_slice(&self.inner.shape[e + 1..]);

        self.reshape(&new_shape)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_reshape() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.to_vec_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tr = t.transpose(0, 1).unwrap();
        assert_eq!(tr.shape(), &[3, 2]);
        assert_eq!(tr.strides(), &[1, 3]);
        assert!(!tr.is_contiguous());
        // Data via strided access
        let data = tr.to_vec_f64().unwrap();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_permute() {
        let t = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[2, 3, 2],
        );
        let p = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[2, 2, 3]);
    }

    #[test]
    fn test_select() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = t.select(0, 1).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.to_vec_f64().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_narrow() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let n = t.narrow(1, 1, 2).unwrap();
        assert_eq!(n.shape(), &[2, 2]);
        assert_eq!(n.to_vec_f64().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unsqueeze_squeeze() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let u = t.unsqueeze(0).unwrap();
        assert_eq!(u.shape(), &[1, 3]);
        let s = u.squeeze(0).unwrap();
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn test_expand() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]);
        let e = t.expand(&[4, 3]).unwrap();
        assert_eq!(e.shape(), &[4, 3]);
        // All rows should be the same
        let data = e.to_vec_f64().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        );
        let f = t.flatten(0, 1).unwrap();
        assert_eq!(f.shape(), &[6]);
    }

    #[test]
    fn test_contiguous() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tr = t.transpose(0, 1).unwrap();
        assert!(!tr.is_contiguous());
        let c = tr.contiguous().unwrap();
        assert!(c.is_contiguous());
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.to_vec_f64().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_view_i64_infer_dim() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // view(3, -1) -> [3, 2]
        let v = t.view_i64(&[3, -1]).unwrap();
        assert_eq!(v.shape(), &[3, 2]);
        assert_eq!(v.to_vec_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_view_i64_flatten() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // view(-1) -> [6]
        let v = t.view_i64(&[-1]).unwrap();
        assert_eq!(v.shape(), &[6]);
        assert_eq!(v.to_vec_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_view_i64_no_infer() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let v = t.view_i64(&[2, 3]).unwrap();
        assert_eq!(v.shape(), &[2, 3]);
    }

    #[test]
    fn test_view_i64_multiple_inferred_fails() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let r = t.view_i64(&[-1, -1]);
        assert!(r.is_err());
    }

    #[test]
    fn test_view_i64_non_contiguous_fails() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tr = t.transpose(0, 1).unwrap();
        let r = tr.view_i64(&[-1]);
        assert!(r.is_err());
    }

    #[test]
    fn test_view_i64_3d() {
        let t = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[12],
        );
        let v = t.view_i64(&[2, -1, 3]).unwrap();
        assert_eq!(v.shape(), &[2, 2, 3]);
    }
}
