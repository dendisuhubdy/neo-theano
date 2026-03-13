use theano_backend::{BackendStorage, BinaryOp, ReduceOp, UnaryOp};
use theano_types::{DType, Device, Result, Shape, TheanoError};

/// CPU storage: a typed, aligned data buffer.
///
/// Internally stores data as a `Vec<f64>` for simplicity in this initial implementation.
/// A production version would use typed storage (`Vec<f32>`, `Vec<f16>`, etc.) with
/// SIMD-optimized kernels.
#[derive(Clone)]
pub struct CpuStorage {
    data: Vec<f64>,
    dtype: DType,
}

impl CpuStorage {
    /// Create a new CPU storage from f64 data.
    pub fn new(data: Vec<f64>, dtype: DType) -> Self {
        Self { data, dtype }
    }

    /// Get the raw f64 data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable access to the raw f64 data.
    pub fn data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.data
    }
}

impl BackendStorage for CpuStorage {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn unary_op(&self, op: UnaryOp, shape: &[usize], strides: &[usize]) -> Result<Self> {
        let data = gather_data(&self.data, shape, strides);
        let result = match op {
            UnaryOp::Neg => data.iter().map(|&x| -x).collect(),
            UnaryOp::Abs => data.iter().map(|&x| x.abs()).collect(),
            UnaryOp::Exp => data.iter().map(|&x| x.exp()).collect(),
            UnaryOp::Log => data.iter().map(|&x| x.ln()).collect(),
            UnaryOp::Log2 => data.iter().map(|&x| x.log2()).collect(),
            UnaryOp::Log10 => data.iter().map(|&x| x.log10()).collect(),
            UnaryOp::Sqrt => data.iter().map(|&x| x.sqrt()).collect(),
            UnaryOp::Rsqrt => data.iter().map(|&x| 1.0 / x.sqrt()).collect(),
            UnaryOp::Sin => data.iter().map(|&x| x.sin()).collect(),
            UnaryOp::Cos => data.iter().map(|&x| x.cos()).collect(),
            UnaryOp::Tan => data.iter().map(|&x| x.tan()).collect(),
            UnaryOp::Asin => data.iter().map(|&x| x.asin()).collect(),
            UnaryOp::Acos => data.iter().map(|&x| x.acos()).collect(),
            UnaryOp::Atan => data.iter().map(|&x| x.atan()).collect(),
            UnaryOp::Sinh => data.iter().map(|&x| x.sinh()).collect(),
            UnaryOp::Cosh => data.iter().map(|&x| x.cosh()).collect(),
            UnaryOp::Tanh => data.iter().map(|&x| x.tanh()).collect(),
            UnaryOp::Sigmoid => data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
            UnaryOp::Relu => data.iter().map(|&x| x.max(0.0)).collect(),
            UnaryOp::Gelu => data
                .iter()
                .map(|&x| {
                    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                    0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
                })
                .collect(),
            UnaryOp::Silu => data.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
            UnaryOp::Floor => data.iter().map(|&x| x.floor()).collect(),
            UnaryOp::Ceil => data.iter().map(|&x| x.ceil()).collect(),
            UnaryOp::Round => data.iter().map(|&x| x.round()).collect(),
            UnaryOp::Sign => data
                .iter()
                .map(|&x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            UnaryOp::Reciprocal => data.iter().map(|&x| 1.0 / x).collect(),
            UnaryOp::Square => data.iter().map(|&x| x * x).collect(),
            UnaryOp::Erf => data.iter().map(|&x| erf_approx(x)).collect(),
        };
        Ok(CpuStorage::new(result, self.dtype))
    }

    fn binary_op(
        &self,
        rhs: &Self,
        op: BinaryOp,
        lhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs_shape: &[usize],
        rhs_strides: &[usize],
        out_shape: &[usize],
    ) -> Result<Self> {
        let out_numel: usize = out_shape.iter().product();
        let lhs_data = gather_data(&self.data, lhs_shape, lhs_strides);
        let rhs_data = gather_data(&rhs.data, rhs_shape, rhs_strides);

        // Broadcast iteration
        let mut result = Vec::with_capacity(out_numel);
        let lhs_bstrides = broadcast_strides(lhs_shape, out_shape);
        let rhs_bstrides = broadcast_strides(rhs_shape, out_shape);

        for i in 0..out_numel {
            let li = broadcast_index(i, out_shape, &lhs_bstrides);
            let ri = broadcast_index(i, out_shape, &rhs_bstrides);
            let a = lhs_data[li];
            let b = rhs_data[ri];
            let v = match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => a / b,
                BinaryOp::Pow => a.powf(b),
                BinaryOp::Min => a.min(b),
                BinaryOp::Max => a.max(b),
                BinaryOp::Eq => {
                    if (a - b).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Ne => {
                    if (a - b).abs() >= f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Gt => {
                    if a > b {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Ge => {
                    if a >= b {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Lt => {
                    if a < b {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Le => {
                    if a <= b {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::And => {
                    if a != 0.0 && b != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Or => {
                    if a != 0.0 || b != 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                BinaryOp::Xor => {
                    if (a != 0.0) != (b != 0.0) {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            result.push(v);
        }
        Ok(CpuStorage::new(result, self.dtype))
    }

    fn reduce_op(
        &self,
        op: ReduceOp,
        shape: &[usize],
        strides: &[usize],
        reduce_dims: &[usize],
        _keep_dim: bool,
    ) -> Result<Self> {
        let data = gather_data(&self.data, shape, strides);
        let ndim = shape.len();

        // Compute output shape (removing reduced dims)
        let mut out_shape: Vec<usize> = Vec::new();
        for (i, &s) in shape.iter().enumerate() {
            if !reduce_dims.contains(&i) {
                out_shape.push(s);
            }
        }
        let out_numel = out_shape.iter().product::<usize>().max(1);

        // Group elements by their output index
        let total = shape.iter().product::<usize>();
        let mut groups: Vec<Vec<f64>> = vec![Vec::new(); out_numel];

        let in_strides = Shape::new(shape.to_vec()).contiguous_strides();

        for flat_idx in 0..total {
            // Compute multi-index
            let mut remaining = flat_idx;
            let mut multi_idx = vec![0usize; ndim];
            for d in 0..ndim {
                multi_idx[d] = remaining / in_strides[d];
                remaining %= in_strides[d];
            }

            // Compute output flat index (skip reduced dims)
            let mut out_flat = 0;
            let out_strides = Shape::new(out_shape.clone()).contiguous_strides();
            let mut out_dim = 0;
            for d in 0..ndim {
                if !reduce_dims.contains(&d) {
                    if !out_strides.is_empty() {
                        out_flat += multi_idx[d] * out_strides[out_dim];
                    }
                    out_dim += 1;
                }
            }

            groups[out_flat].push(data[flat_idx]);
        }

        let result: Vec<f64> = groups
            .iter()
            .map(|g| match op {
                ReduceOp::Sum => g.iter().sum(),
                ReduceOp::Mean => g.iter().sum::<f64>() / g.len() as f64,
                ReduceOp::Max => g.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ReduceOp::Min => g.iter().copied().fold(f64::INFINITY, f64::min),
                ReduceOp::Prod => g.iter().product(),
                ReduceOp::ArgMax => {
                    g.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as f64)
                        .unwrap_or(0.0)
                }
                ReduceOp::ArgMin => {
                    g.iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i as f64)
                        .unwrap_or(0.0)
                }
                ReduceOp::Any => {
                    if g.iter().any(|&x| x != 0.0) {
                        1.0
                    } else {
                        0.0
                    }
                }
                ReduceOp::All => {
                    if g.iter().all(|&x| x != 0.0) {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect();

        Ok(CpuStorage::new(result, self.dtype))
    }

    fn matmul(
        &self,
        rhs: &Self,
        lhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs_shape: &[usize],
        rhs_strides: &[usize],
    ) -> Result<Self> {
        let lhs_data = gather_data(&self.data, lhs_shape, lhs_strides);
        let rhs_data = gather_data(&rhs.data, rhs_shape, rhs_strides);

        let m = lhs_shape[lhs_shape.len() - 2];
        let k = lhs_shape[lhs_shape.len() - 1];
        let n = rhs_shape[rhs_shape.len() - 1];

        let mut result = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_val = lhs_data[i * k + p];
                for j in 0..n {
                    result[i * n + j] += a_val * rhs_data[p * n + j];
                }
            }
        }

        Ok(CpuStorage::new(result, self.dtype))
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        // For f64 internal storage, this is mostly a no-op marker
        Ok(CpuStorage::new(self.data.clone(), dtype))
    }

    fn contiguous(&self, shape: &[usize], strides: &[usize]) -> Result<Self> {
        let data = gather_data(&self.data, shape, strides);
        Ok(CpuStorage::new(data, self.dtype))
    }

    fn fill(value: f64, len: usize, dtype: DType, _device: &Device) -> Result<Self> {
        Ok(CpuStorage::new(vec![value; len], dtype))
    }

    fn from_f64_slice(data: &[f64], dtype: DType, _device: &Device) -> Result<Self> {
        Ok(CpuStorage::new(data.to_vec(), dtype))
    }

    fn to_f64_vec(&self, shape: &[usize], strides: &[usize]) -> Result<Vec<f64>> {
        Ok(gather_data(&self.data, shape, strides))
    }

    fn where_cond(
        &self,
        cond: &Self,
        other: &Self,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self> {
        let self_data = gather_data(&self.data, shape, strides);
        let cond_data = gather_data(&cond.data, shape, strides);
        let other_data = gather_data(&other.data, shape, strides);

        let result: Vec<f64> = cond_data
            .iter()
            .zip(self_data.iter().zip(other_data.iter()))
            .map(|(&c, (&s, &o))| if c != 0.0 { s } else { o })
            .collect();

        Ok(CpuStorage::new(result, self.dtype))
    }

    fn index_select(
        &self,
        _dim: usize,
        _indices: &Self,
        _shape: &[usize],
        _strides: &[usize],
    ) -> Result<Self> {
        Err(TheanoError::not_implemented("CpuStorage::index_select"))
    }
}

// ---- Helper functions ----

/// Gather data from strided storage into a contiguous Vec.
fn gather_data(data: &[f64], shape: &[usize], strides: &[usize]) -> Vec<f64> {
    let numel: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };

    if shape.is_empty() {
        return if data.is_empty() {
            vec![0.0]
        } else {
            vec![data[0]]
        };
    }

    // Check if contiguous
    let expected_strides = Shape::new(shape.to_vec()).contiguous_strides();
    if strides == expected_strides.as_slice() {
        return data[..numel].to_vec();
    }

    // Non-contiguous: gather
    let mut result = Vec::with_capacity(numel);
    gather_recursive(data, shape, strides, 0, 0, &mut result);
    result
}

fn gather_recursive(
    data: &[f64],
    shape: &[usize],
    strides: &[usize],
    dim: usize,
    offset: usize,
    result: &mut Vec<f64>,
) {
    if dim == shape.len() {
        result.push(if offset < data.len() { data[offset] } else { 0.0 });
        return;
    }
    for i in 0..shape[dim] {
        gather_recursive(data, shape, strides, dim + 1, offset + i * strides[dim], result);
    }
}

/// Compute broadcast strides for mapping output indices to source indices.
fn broadcast_strides(src_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let src_ndim = src_shape.len();
    let out_ndim = out_shape.len();
    let mut strides = vec![0usize; out_ndim];

    let src_contiguous = Shape::new(src_shape.to_vec()).contiguous_strides();
    let offset = out_ndim - src_ndim;

    for i in 0..out_ndim {
        if i < offset {
            strides[i] = 0; // broadcast new leading dims
        } else {
            let si = i - offset;
            if src_shape[si] == out_shape[i] {
                strides[i] = src_contiguous[si];
            } else {
                strides[i] = 0; // broadcast dimension (size 1 → size n)
            }
        }
    }
    strides
}

/// Compute flat index in source array from output flat index using broadcast strides.
fn broadcast_index(flat_idx: usize, out_shape: &[usize], bstrides: &[usize]) -> usize {
    let ndim = out_shape.len();
    let out_contiguous = Shape::new(out_shape.to_vec()).contiguous_strides();
    let mut remaining = flat_idx;
    let mut src_idx = 0;
    for d in 0..ndim {
        let coord = remaining / out_contiguous[d];
        remaining %= out_contiguous[d];
        src_idx += coord * bstrides[d];
    }
    src_idx
}

/// Abramowitz and Stegun approximation for erf.
fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_storage_basic() {
        let s = CpuStorage::new(vec![1.0, 2.0, 3.0], DType::F32);
        assert_eq!(s.len(), 3);
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.device(), Device::Cpu);
    }

    #[test]
    fn test_unary_ops() {
        let s = CpuStorage::new(vec![-1.0, 0.0, 1.0, 2.0], DType::F32);
        let shape = [4];
        let strides = [1];

        let neg = s.unary_op(UnaryOp::Neg, &shape, &strides).unwrap();
        assert_eq!(neg.data(), &[1.0, 0.0, -1.0, -2.0]);

        let abs = s.unary_op(UnaryOp::Abs, &shape, &strides).unwrap();
        assert_eq!(abs.data(), &[1.0, 0.0, 1.0, 2.0]);

        let relu = s.unary_op(UnaryOp::Relu, &shape, &strides).unwrap();
        assert_eq!(relu.data(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_binary_ops() {
        let a = CpuStorage::new(vec![1.0, 2.0, 3.0], DType::F32);
        let b = CpuStorage::new(vec![4.0, 5.0, 6.0], DType::F32);
        let shape = [3];
        let strides = [1];

        let add = a
            .binary_op(&b, BinaryOp::Add, &shape, &strides, &shape, &strides, &shape)
            .unwrap();
        assert_eq!(add.data(), &[5.0, 7.0, 9.0]);

        let mul = a
            .binary_op(&b, BinaryOp::Mul, &shape, &strides, &shape, &strides, &shape)
            .unwrap();
        assert_eq!(mul.data(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_matmul() {
        let a = CpuStorage::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F32);
        let b = CpuStorage::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], DType::F32);

        let c = a
            .matmul(&b, &[2, 3], &[3, 1], &[3, 2], &[2, 1])
            .unwrap();
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_reduce_sum() {
        let s = CpuStorage::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F32);
        let r = s
            .reduce_op(ReduceOp::Sum, &[2, 3], &[3, 1], &[0], false)
            .unwrap();
        assert_eq!(r.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_gather_strided() {
        // Transposed [2,3] tensor: shape=[3,2], strides=[1,3]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gathered = gather_data(&data, &[3, 2], &[1, 3]);
        assert_eq!(gathered, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
