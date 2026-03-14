use theano_types::{Result, Shape, TheanoError};

use crate::tensor::Tensor;
use crate::tensor_create::CpuF64Storage;
use crate::storage::Storage;

impl Tensor {
    // ---- Elementwise unary operations ----

    fn unary_op(&self, f: impl Fn(f64) -> f64) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let result: Vec<f64> = data.iter().map(|&x| f(x)).collect();
        Ok(Self::from_f64_result(&result, self.shape(), self.dtype()))
    }

    /// Negate all elements. Like `torch.neg`.
    pub fn neg(&self) -> Result<Tensor> {
        self.unary_op(|x| -x)
    }

    /// Absolute value. Like `torch.abs`.
    pub fn abs(&self) -> Result<Tensor> {
        self.unary_op(|x| x.abs())
    }

    /// Exponential. Like `torch.exp`.
    pub fn exp(&self) -> Result<Tensor> {
        self.unary_op(|x| x.exp())
    }

    /// Natural logarithm. Like `torch.log`.
    pub fn log(&self) -> Result<Tensor> {
        self.unary_op(|x| x.ln())
    }

    /// Square root. Like `torch.sqrt`.
    pub fn sqrt(&self) -> Result<Tensor> {
        self.unary_op(|x| x.sqrt())
    }

    /// Reciprocal square root.
    pub fn rsqrt(&self) -> Result<Tensor> {
        self.unary_op(|x| 1.0 / x.sqrt())
    }

    /// Sine. Like `torch.sin`.
    pub fn sin(&self) -> Result<Tensor> {
        self.unary_op(|x| x.sin())
    }

    /// Cosine. Like `torch.cos`.
    pub fn cos(&self) -> Result<Tensor> {
        self.unary_op(|x| x.cos())
    }

    /// Hyperbolic tangent. Like `torch.tanh`.
    pub fn tanh(&self) -> Result<Tensor> {
        self.unary_op(|x| x.tanh())
    }

    /// Sigmoid. Like `torch.sigmoid`.
    pub fn sigmoid(&self) -> Result<Tensor> {
        self.unary_op(|x| 1.0 / (1.0 + (-x).exp()))
    }

    /// ReLU. Like `torch.relu`.
    pub fn relu(&self) -> Result<Tensor> {
        self.unary_op(|x| x.max(0.0))
    }

    /// GELU (Gaussian Error Linear Unit). Like `torch.nn.functional.gelu`.
    pub fn gelu(&self) -> Result<Tensor> {
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        self.unary_op(|x| {
            let c = (2.0_f64 / std::f64::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
        })
    }

    /// SiLU (Swish). Like `torch.nn.functional.silu`.
    pub fn silu(&self) -> Result<Tensor> {
        self.unary_op(|x| x / (1.0 + (-x).exp()))
    }

    /// Floor. Like `torch.floor`.
    pub fn floor(&self) -> Result<Tensor> {
        self.unary_op(|x| x.floor())
    }

    /// Ceil. Like `torch.ceil`.
    pub fn ceil(&self) -> Result<Tensor> {
        self.unary_op(|x| x.ceil())
    }

    /// Round. Like `torch.round`.
    pub fn round(&self) -> Result<Tensor> {
        self.unary_op(|x| x.round())
    }

    /// Sign. Like `torch.sign`.
    pub fn sign(&self) -> Result<Tensor> {
        self.unary_op(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    /// Reciprocal. Like `torch.reciprocal`.
    pub fn reciprocal(&self) -> Result<Tensor> {
        self.unary_op(|x| 1.0 / x)
    }

    /// Square. Like `torch.square`.
    pub fn square(&self) -> Result<Tensor> {
        self.unary_op(|x| x * x)
    }

    // ---- Elementwise binary operations ----

    fn binary_op(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Result<Tensor> {
        let out_shape = Shape::broadcast_shape(&self.size(), &other.size())
            .ok_or_else(|| TheanoError::broadcast_error(self.shape(), other.shape()))?;

        let a = self.broadcast_to(out_shape.dims())?;
        let b = other.broadcast_to(out_shape.dims())?;

        let a_data = a.to_vec_f64()?;
        let b_data = b.to_vec_f64()?;

        let result: Vec<f64> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();

        Ok(Self::from_f64_result(&result, out_shape.dims(), self.dtype()))
    }

    /// Element-wise addition. Like `torch.add`.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| a + b)
    }

    /// Element-wise subtraction. Like `torch.sub`.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| a - b)
    }

    /// Element-wise multiplication. Like `torch.mul`.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| a * b)
    }

    /// Element-wise division. Like `torch.div`.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| a / b)
    }

    /// Element-wise power. Like `torch.pow`.
    pub fn pow(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| a.powf(b))
    }

    /// Scalar addition.
    pub fn add_scalar(&self, scalar: f64) -> Result<Tensor> {
        self.unary_op(|x| x + scalar)
    }

    /// Scalar multiplication.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Tensor> {
        self.unary_op(|x| x * scalar)
    }

    // ---- Comparison operations ----

    /// Element-wise equality. Like `torch.eq`.
    pub fn eq_tensor(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| if (a - b).abs() < f64::EPSILON { 1.0 } else { 0.0 })
    }

    /// Element-wise greater-than. Like `torch.gt`.
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    /// Element-wise less-than. Like `torch.lt`.
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    /// Element-wise greater-than-or-equal. Like `torch.ge`.
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }

    /// Element-wise less-than-or-equal. Like `torch.le`.
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| if a <= b { 1.0 } else { 0.0 })
    }

    /// Clamp values to [min, max]. Like `torch.clamp`.
    pub fn clamp(&self, min: f64, max: f64) -> Result<Tensor> {
        self.unary_op(|x| x.clamp(min, max))
    }

    /// Element-wise minimum of two tensors. Like `torch.minimum`.
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| a.min(b))
    }

    /// Element-wise maximum of two tensors. Like `torch.maximum`.
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, |a, b| a.max(b))
    }

    /// Element-wise conditional selection. Like `torch.where`.
    ///
    /// Returns elements from `self` where `condition` is true (non-zero),
    /// and from `other` where `condition` is false (zero).
    pub fn where_cond(&self, condition: &Tensor, other: &Tensor) -> Result<Tensor> {
        let cond_data = condition.to_vec_f64()?;
        let self_data = self.to_vec_f64()?;
        let other_data = other.to_vec_f64()?;
        if cond_data.len() != self_data.len() || cond_data.len() != other_data.len() {
            return Err(TheanoError::runtime(
                "where_cond: all tensors must have the same number of elements",
            ));
        }
        let result: Vec<f64> = cond_data
            .iter()
            .zip(self_data.iter().zip(other_data.iter()))
            .map(|(&c, (&s, &o))| if c != 0.0 { s } else { o })
            .collect();
        Ok(Self::from_f64_result(&result, self.shape(), self.dtype()))
    }

    /// Greater-than comparison with a scalar. Like `tensor > scalar`.
    pub fn gt_scalar(&self, scalar: f64) -> Result<Tensor> {
        self.unary_op(|x| if x > scalar { 1.0 } else { 0.0 })
    }

    // ---- Reduction operations ----

    /// Sum of all elements. Like `torch.sum`.
    pub fn sum(&self) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let s: f64 = data.iter().sum();
        Ok(Tensor::scalar(s))
    }

    /// Sum along a dimension. Like `torch.sum(dim=d)`.
    pub fn sum_dim(&self, dim: i64, keep_dim: bool) -> Result<Tensor> {
        self.reduce_dim(dim, keep_dim, |slice| slice.iter().sum())
    }

    /// Mean of all elements. Like `torch.mean`.
    pub fn mean(&self) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let s: f64 = data.iter().sum();
        Ok(Tensor::scalar(s / data.len() as f64))
    }

    /// Mean along a dimension. Like `torch.mean(dim=d)`.
    pub fn mean_dim(&self, dim: i64, keep_dim: bool) -> Result<Tensor> {
        let d = self.normalize_dim(dim)?;
        let size = self.shape()[d] as f64;
        self.reduce_dim(dim, keep_dim, |slice| slice.iter().sum::<f64>() / size)
    }

    /// Max of all elements. Like `torch.max`.
    pub fn max(&self) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let m = data
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        Ok(Tensor::scalar(m))
    }

    /// Min of all elements. Like `torch.min`.
    pub fn min(&self) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let m = data
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        Ok(Tensor::scalar(m))
    }

    /// Product of all elements. Like `torch.prod`.
    pub fn prod(&self) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let p: f64 = data.iter().product();
        Ok(Tensor::scalar(p))
    }

    /// Argmax of all elements. Like `torch.argmax`.
    pub fn argmax(&self) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let (idx, _) = data
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        Ok(Tensor::scalar(idx as f64))
    }

    /// Argmin of all elements. Like `torch.argmin`.
    pub fn argmin(&self) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let (idx, _) = data
            .iter()
            .enumerate()
            .fold((0, f64::INFINITY), |(bi, bv), (i, &v)| {
                if v < bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        Ok(Tensor::scalar(idx as f64))
    }

    // ---- Matrix operations ----

    /// Matrix multiplication. Like `torch.matmul`.
    ///
    /// Supports:
    /// - [M, K] x [K, N] -> [M, N]        (2D x 2D)
    /// - [K] x [K, N] -> [N]              (1D x 2D)
    /// - [M, K] x [K] -> [M]              (2D x 1D)
    /// - [..., M, K] x [..., K, N] -> [..., M, N]  (batched)
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let a_ndim = self.ndim();
        let b_ndim = other.ndim();

        match (a_ndim, b_ndim) {
            (2, 2) => self.mm(other),
            (1, 2) => {
                // vec @ matrix: treat as [1, K] @ [K, N] -> [N]
                let a = self.unsqueeze(0)?;
                let result = a.mm(other)?;
                result.squeeze(0)
            }
            (2, 1) => {
                // matrix @ vec: treat as [M, K] @ [K, 1] -> [M]
                let b = other.unsqueeze(1)?;
                let result = self.mm(&b)?;
                result.squeeze(1)
            }
            (1, 1) => {
                // dot product
                let a_data = self.to_vec_f64()?;
                let b_data = other.to_vec_f64()?;
                if a_data.len() != b_data.len() {
                    return Err(TheanoError::shape_mismatch(self.shape(), other.shape()));
                }
                let dot: f64 = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum();
                Ok(Tensor::scalar(dot))
            }
            _ => {
                // Batched matmul
                self.bmm(other)
            }
        }
    }

    /// Matrix-matrix multiplication (2D only). Like `torch.mm`.
    pub fn mm(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(TheanoError::runtime("mm: both tensors must be 2D"));
        }
        let m = self.shape()[0];
        let k = self.shape()[1];
        let k2 = other.shape()[0];
        let n = other.shape()[1];

        if k != k2 {
            return Err(TheanoError::shape_mismatch(
                &[m, k],
                &[k2, n],
            ));
        }

        let a_data = self.to_vec_f64()?;
        let b_data = other.to_vec_f64()?;
        let mut c_data = vec![0.0f64; m * n];

        // Naive but correct matmul: C[i,j] = sum_p A[i,p] * B[p,j]
        for i in 0..m {
            for p in 0..k {
                let a_val = a_data[i * k + p];
                for j in 0..n {
                    c_data[i * n + j] += a_val * b_data[p * n + j];
                }
            }
        }

        Ok(Self::from_f64_result(&c_data, &[m, n], self.dtype()))
    }

    /// Batched matrix-matrix multiplication. Like `torch.bmm`.
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor> {
        let a_ndim = self.ndim();
        let b_ndim = other.ndim();

        if a_ndim < 2 || b_ndim < 2 {
            return Err(TheanoError::runtime("bmm: both tensors must be at least 2D"));
        }

        let m = self.shape()[a_ndim - 2];
        let k = self.shape()[a_ndim - 1];
        let k2 = other.shape()[b_ndim - 2];
        let n = other.shape()[b_ndim - 1];

        if k != k2 {
            return Err(TheanoError::shape_mismatch(self.shape(), other.shape()));
        }

        // Compute batch shape via broadcasting
        let a_batch = &self.shape()[..a_ndim - 2];
        let b_batch = &other.shape()[..b_ndim - 2];
        let batch_shape = Shape::broadcast_shape(
            &Shape::new(a_batch.to_vec()),
            &Shape::new(b_batch.to_vec()),
        )
        .ok_or_else(|| TheanoError::broadcast_error(a_batch, b_batch))?;

        let batch_numel = batch_shape.numel();
        let a_flat = self.to_vec_f64()?;
        let b_flat = other.to_vec_f64()?;

        let mut result = vec![0.0f64; batch_numel * m * n];

        let a_mat_size = m * k;
        let b_mat_size = k * n;
        let c_mat_size = m * n;

        // Compute batch counts for broadcasting
        let a_batch_numel: usize = a_batch.iter().product::<usize>().max(1);
        let b_batch_numel: usize = b_batch.iter().product::<usize>().max(1);

        for batch in 0..batch_numel {
            // Handle broadcasting: if one side has fewer batches, wrap around
            let a_off = (batch % a_batch_numel) * a_mat_size;
            let b_off = (batch % b_batch_numel) * b_mat_size;
            let c_off = batch * c_mat_size;

            for i in 0..m {
                for p in 0..k {
                    let a_val = a_flat[a_off + i * k + p];
                    for j in 0..n {
                        result[c_off + i * n + j] += a_val * b_flat[b_off + p * n + j];
                    }
                }
            }
        }

        let mut out_shape = batch_shape.dims().to_vec();
        out_shape.push(m);
        out_shape.push(n);

        Ok(Self::from_f64_result(&result, &out_shape, self.dtype()))
    }

    // ---- Concatenation / stacking ----

    /// Concatenate tensors along a dimension. Like `torch.cat`.
    pub fn cat(tensors: &[Tensor], dim: i64) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TheanoError::invalid_argument("cat: empty tensor list"));
        }

        let first = &tensors[0];
        let d = first.normalize_dim(dim)?;

        // Validate all tensors have same ndim and matching shapes except along `dim`
        for t in &tensors[1..] {
            if t.ndim() != first.ndim() {
                return Err(TheanoError::runtime(
                    "cat: all tensors must have the same number of dimensions",
                ));
            }
            for (i, (&a, &b)) in first.shape().iter().zip(t.shape().iter()).enumerate() {
                if i != d && a != b {
                    return Err(TheanoError::shape_mismatch(first.shape(), t.shape()));
                }
            }
        }

        // Compute output shape
        let mut out_shape = first.shape().to_vec();
        out_shape[d] = tensors.iter().map(|t| t.shape()[d]).sum();

        // Gather all data
        let mut all_data: Vec<Vec<f64>> = Vec::new();
        for t in tensors {
            all_data.push(t.to_vec_f64()?);
        }

        let out_numel: usize = out_shape.iter().product();
        let mut result = vec![0.0f64; out_numel];

        // Compute output by iterating over the output multi-index
        let out_strides = Shape::new(out_shape.clone()).contiguous_strides();
        let ndim = out_shape.len();

        // For each element in the output, find which source tensor it comes from
        let mut idx = vec![0usize; ndim];
        for flat_i in 0..out_numel {
            // Determine which tensor this index falls into along dim d
            let mut cat_idx = idx[d];
            let mut src_tensor = 0;
            for (ti, t) in tensors.iter().enumerate() {
                let t_size = t.shape()[d];
                if cat_idx < t_size {
                    src_tensor = ti;
                    break;
                }
                cat_idx -= t_size;
            }

            // Compute the flat index in the source tensor
            let src_shape = tensors[src_tensor].shape();
            let src_strides = Shape::new(src_shape.to_vec()).contiguous_strides();
            let mut src_flat = 0;
            for dim_i in 0..ndim {
                let src_idx = if dim_i == d { cat_idx } else { idx[dim_i] };
                src_flat += src_idx * src_strides[dim_i];
            }

            result[flat_i] = all_data[src_tensor][src_flat];

            // Increment multi-index
            for k in (0..ndim).rev() {
                idx[k] += 1;
                if idx[k] < out_shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }

        Ok(Self::from_f64_result(&result, &out_shape, first.dtype()))
    }

    /// Stack tensors along a new dimension. Like `torch.stack`.
    pub fn stack(tensors: &[Tensor], dim: i64) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TheanoError::invalid_argument("stack: empty tensor list"));
        }

        let expanded: Vec<Tensor> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;

        Tensor::cat(&expanded, dim)
    }

    // ---- Softmax ----

    /// Softmax along a dimension. Like `torch.softmax`.
    pub fn softmax(&self, dim: i64) -> Result<Tensor> {
        let d = self.normalize_dim(dim)?;
        let data = self.to_vec_f64()?;
        let shape = self.shape();
        let strides = Shape::new(shape.to_vec()).contiguous_strides();
        let numel = self.numel();

        // Make contiguous first
        let data = if !self.is_contiguous() {
            self.contiguous()?.to_vec_f64()?
        } else {
            data
        };

        let mut result = vec![0.0f64; numel];

        // Process slices along the softmax dimension
        let outer_size: usize = shape[..d].iter().product();
        let dim_size = shape[d];
        let inner_size: usize = shape[d + 1..].iter().product();

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // Find max for numerical stability
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    max_val = max_val.max(data[idx]);
                }

                // Compute exp and sum
                let mut sum = 0.0f64;
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    let e = (data[idx] - max_val).exp();
                    result[idx] = e;
                    sum += e;
                }

                // Normalize
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    result[idx] /= sum;
                }
            }
        }

        Ok(Self::from_f64_result(&result, shape, self.dtype()))
    }

    /// Log-softmax along a dimension. Like `torch.log_softmax`.
    pub fn log_softmax(&self, dim: i64) -> Result<Tensor> {
        let sm = self.softmax(dim)?;
        sm.log()
    }

    // ---- Internal helpers ----

    /// Broadcast this tensor to the given shape.
    fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor> {
        if self.shape() == target_shape {
            return Ok(self.clone());
        }
        self.expand(target_shape)
    }

    /// Reduce along a single dimension.
    fn reduce_dim(
        &self,
        dim: i64,
        keep_dim: bool,
        f: impl Fn(&[f64]) -> f64,
    ) -> Result<Tensor> {
        let d = self.normalize_dim(dim)?;
        let data = self.to_vec_f64()?;
        let shape = self.shape();
        let ndim = shape.len();

        let outer_size: usize = shape[..d].iter().product();
        let dim_size = shape[d];
        let inner_size: usize = shape[d + 1..].iter().product();

        let mut result = Vec::with_capacity(outer_size * inner_size);

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut slice = Vec::with_capacity(dim_size);
                for i in 0..dim_size {
                    let idx = outer * dim_size * inner_size + i * inner_size + inner;
                    slice.push(data[idx]);
                }
                result.push(f(&slice));
            }
        }

        let mut out_shape: Vec<usize> = Vec::new();
        for i in 0..ndim {
            if i == d {
                if keep_dim {
                    out_shape.push(1);
                }
            } else {
                out_shape.push(shape[i]);
            }
        }

        Ok(Self::from_f64_result(&result, &out_shape, self.dtype()))
    }

    /// Internal helper to create a tensor from computed f64 results.
    pub(crate) fn from_f64_result(data: &[f64], shape: &[usize], dtype: theano_types::DType) -> Tensor {
        let strides = Shape::new(shape.to_vec()).contiguous_strides();
        let storage = CpuF64Storage {
            data: data.to_vec(),
            dtype,
        };
        Tensor::from_parts(
            Storage::Cpu(Box::new(storage)),
            shape.to_vec(),
            strides,
            0,
            dtype,
            theano_types::Device::Cpu,
        )
    }
}

// ---- Operator overloads ----

impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.add(rhs).expect("add failed")
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        Tensor::sub(self, rhs).expect("sub failed")
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        Tensor::mul(self, rhs).expect("mul failed")
    }
}

impl std::ops::Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        Tensor::div(self, rhs).expect("div failed")
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self).expect("neg failed")
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_add() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], &[3]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_broadcasting() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]);
        let b = Tensor::from_slice(&[10.0, 20.0], &[2, 1]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(
            c.to_vec_f64().unwrap(),
            vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
        );
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0], &[3]);
        let b = a.neg().unwrap();
        assert_eq!(b.to_vec_f64().unwrap(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);
        let b = a.relu().unwrap();
        assert_eq!(b.to_vec_f64().unwrap(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let a = Tensor::scalar(0.0);
        let b = a.sigmoid().unwrap();
        let v = b.item().unwrap();
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sum() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = a.sum().unwrap();
        assert_eq!(s.item().unwrap(), 21.0);
    }

    #[test]
    fn test_sum_dim() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let s = a.sum_dim(0, false).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.to_vec_f64().unwrap(), vec![5.0, 7.0, 9.0]);

        let s = a.sum_dim(1, true).unwrap();
        assert_eq!(s.shape(), &[2, 1]);
        assert_eq!(s.to_vec_f64().unwrap(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let m = a.mean().unwrap();
        assert_eq!(m.item().unwrap(), 2.5);
    }

    #[test]
    fn test_max_min() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);
        assert_eq!(a.max().unwrap().item().unwrap(), 5.0);
        assert_eq!(a.min().unwrap().item().unwrap(), 1.0);
    }

    #[test]
    fn test_argmax() {
        let a = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);
        assert_eq!(a.argmax().unwrap().item().unwrap(), 4.0);
        assert_eq!(a.argmin().unwrap().item().unwrap(), 1.0);
    }

    #[test]
    fn test_matmul_2d() {
        // [2,3] x [3,2] -> [2,2]
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c.to_vec_f64().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_1d_2d() {
        // [3] x [3,2] -> [2]
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.to_vec_f64().unwrap(), vec![58.0, 64.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.item().unwrap(), 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let s = a.softmax(0).unwrap();
        let data = s.to_vec_f64().unwrap();
        let sum: f64 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // softmax should be monotonically increasing for increasing inputs
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
    }

    #[test]
    fn test_cat() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[1, 3]);
        let c = Tensor::cat(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(
            c.to_vec_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let c = Tensor::stack(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(
            c.to_vec_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_operator_overloads() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let c = &a + &b;
        assert_eq!(c.to_vec_f64().unwrap(), vec![5.0, 7.0, 9.0]);
        let d = &a * &b;
        assert_eq!(d.to_vec_f64().unwrap(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_clamp() {
        let a = Tensor::from_slice(&[-1.0, 0.5, 2.0, 3.0], &[4]);
        let c = a.clamp(0.0, 1.0).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = a.exp().unwrap().log().unwrap();
        let data = b.to_vec_f64().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - (i as f64 + 1.0)).abs() < 1e-10,
                "exp(log(x)) != x at index {i}"
            );
        }
    }
}
