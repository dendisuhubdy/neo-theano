use theano_types::{Result, Shape, TheanoError};

use crate::tensor::Tensor;

impl Tensor {
    /// Select elements along a dimension using an index tensor. Like `torch.index_select`.
    ///
    /// Returns a new tensor with the same number of dimensions, where dimension `dim`
    /// has the same size as the `indices` tensor.
    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Result<Tensor> {
        if dim >= self.ndim() {
            return Err(TheanoError::DimensionOutOfRange {
                got: dim as i64,
                min: 0,
                max: self.ndim() as i64,
            });
        }
        if indices.ndim() != 1 {
            return Err(TheanoError::invalid_argument(
                "index_select: indices must be a 1-D tensor",
            ));
        }

        let idx_data = indices.to_vec_f64()?;
        let src_data = self.to_vec_f64()?;
        let src_shape = self.shape();
        let dim_size = src_shape[dim];

        // Compute output shape
        let mut out_shape = src_shape.to_vec();
        out_shape[dim] = idx_data.len();

        let out_numel: usize = out_shape.iter().product();
        let mut result = vec![0.0f64; out_numel];

        let src_strides = Shape::new(src_shape.to_vec()).contiguous_strides();
        let out_strides = Shape::new(out_shape.clone()).contiguous_strides();
        let ndim = src_shape.len();

        // Iterate over output multi-index
        let mut idx = vec![0usize; ndim];
        for flat_i in 0..out_numel {
            // Map the output index to source index
            let src_dim_idx = idx_data[idx[dim]] as usize;
            if src_dim_idx >= dim_size {
                return Err(TheanoError::IndexOutOfBounds {
                    index: src_dim_idx as i64,
                    size: dim_size,
                });
            }

            let mut src_flat = 0;
            for d in 0..ndim {
                let src_idx = if d == dim { src_dim_idx } else { idx[d] };
                src_flat += src_idx * src_strides[d];
            }

            result[flat_i] = src_data[src_flat];

            // Increment multi-index
            for k in (0..ndim).rev() {
                idx[k] += 1;
                if idx[k] < out_shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }

        Ok(Self::from_f64_result(&result, &out_shape, self.dtype()))
    }

    /// Python-style slicing along a dimension. Like `tensor[start:end:step]` in PyTorch.
    ///
    /// - `start`: inclusive start index (None = 0 for positive step, end for negative step)
    /// - `end`: exclusive end index (None = dim_size for positive step, -1 for negative step)
    /// - `step`: step size (must be non-zero)
    pub fn slice(&self, dim: i64, start: Option<i64>, end: Option<i64>, step: i64) -> Result<Tensor> {
        if step == 0 {
            return Err(TheanoError::invalid_argument(
                "slice: step cannot be zero",
            ));
        }

        let d = self.normalize_dim(dim)?;
        let dim_size = self.shape()[d] as i64;

        // Normalize start
        let start = match start {
            Some(s) => {
                let s = if s < 0 { s + dim_size } else { s };
                s.clamp(0, dim_size)
            }
            None => {
                if step > 0 { 0 } else { dim_size - 1 }
            }
        };

        // Normalize end
        let end = match end {
            Some(e) => {
                let e = if e < 0 { e + dim_size } else { e };
                e.clamp(0, dim_size)
            }
            None => {
                if step > 0 { dim_size } else { -1 }
            }
        };

        // Collect the indices
        let mut indices = Vec::new();
        let mut i = start;
        if step > 0 {
            while i < end {
                indices.push(i as usize);
                i += step;
            }
        } else {
            while i > end {
                indices.push(i as usize);
                i += step;
            }
        }

        if indices.is_empty() {
            // Return empty tensor along this dimension
            let mut out_shape = self.shape().to_vec();
            out_shape[d] = 0;
            return Ok(Self::from_f64_result(&[], &out_shape, self.dtype()));
        }

        // Build result by gathering slices
        let src_data = self.to_vec_f64()?;
        let src_shape = self.shape();
        let src_strides = Shape::new(src_shape.to_vec()).contiguous_strides();
        let ndim = src_shape.len();

        let mut out_shape = src_shape.to_vec();
        out_shape[d] = indices.len();

        let out_numel: usize = out_shape.iter().product();
        let mut result = vec![0.0f64; out_numel];

        let mut out_idx = vec![0usize; ndim];
        for flat_i in 0..out_numel {
            let mut src_flat = 0;
            for dd in 0..ndim {
                let src_idx = if dd == d { indices[out_idx[dd]] } else { out_idx[dd] };
                src_flat += src_idx * src_strides[dd];
            }

            result[flat_i] = src_data[src_flat];

            // Increment multi-index
            for k in (0..ndim).rev() {
                out_idx[k] += 1;
                if out_idx[k] < out_shape[k] {
                    break;
                }
                out_idx[k] = 0;
            }
        }

        Ok(Self::from_f64_result(&result, &out_shape, self.dtype()))
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_index_select() {
        let t = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        );
        let indices = Tensor::from_slice(&[0.0, 2.0], &[2]);
        let r = t.index_select(0, &indices).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.to_vec_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_index_select_dim1() {
        let t = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        );
        let indices = Tensor::from_slice(&[0.0, 2.0], &[2]);
        let r = t.index_select(1, &indices).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.to_vec_f64().unwrap(), vec![1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_slice_basic() {
        let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        // t[1:4]
        let r = t.slice(0, Some(1), Some(4), 1).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.to_vec_f64().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_slice_with_step() {
        let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6]);
        // t[0:6:2]
        let r = t.slice(0, Some(0), Some(6), 2).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.to_vec_f64().unwrap(), vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_slice_negative_index() {
        let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        // t[-3:]  ->  [2, 3, 4]
        let r = t.slice(0, Some(-3), None, 1).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.to_vec_f64().unwrap(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_negative_step() {
        let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        // t[4:1:-1]  ->  [4, 3, 2]
        let r = t.slice(0, Some(4), Some(1), -1).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.to_vec_f64().unwrap(), vec![4.0, 3.0, 2.0]);
    }

    #[test]
    fn test_slice_2d() {
        let t = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        );
        // t[:, 0:2]
        let r = t.slice(1, Some(0), Some(2), 1).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(
            r.to_vec_f64().unwrap(),
            vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_slice_none_bounds() {
        let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        // t[:] — full copy
        let r = t.slice(0, None, None, 1).unwrap();
        assert_eq!(r.shape(), &[5]);
        assert_eq!(r.to_vec_f64().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }
}
