//! ROCm storage -- GPU memory buffer on an AMD device.

use std::sync::Arc;

use theano_types::{DType, Device, Result, TheanoError};
use theano_backend::{BackendStorage, BinaryOp, ReduceOp, UnaryOp};

use crate::allocator::Block;
use crate::device::RocmDevice;

/// GPU memory buffer on a ROCm/HIP device.
///
/// Holds a reference to the device and a memory block from the caching allocator.
/// When dropped, the block is returned to the allocator pool (not freed).
#[derive(Clone)]
pub struct RocmStorage {
    /// The ROCm device this storage lives on.
    device: Arc<RocmDevice>,
    /// Memory block from the caching allocator.
    block: Arc<Block>,
    /// Data type of elements.
    dtype: DType,
    /// Number of elements.
    len: usize,
    /// For mock mode: CPU-side mirror of the data (for testing without real GPU).
    #[cfg(not(feature = "hip"))]
    mock_data: Arc<Vec<f64>>,
}

impl RocmStorage {
    /// Allocate a new ROCm storage for `len` elements of the given dtype.
    pub fn allocate(
        device: Arc<RocmDevice>,
        len: usize,
        dtype: DType,
    ) -> Result<Self> {
        let byte_size = len * dtype.size_in_bytes();
        let block = device
            .allocator()
            .allocate(byte_size)
            .map_err(TheanoError::from)?;

        Ok(Self {
            device,
            block: Arc::new(block),
            dtype,
            len,
            #[cfg(not(feature = "hip"))]
            mock_data: Arc::new(vec![0.0; len]),
        })
    }

    /// Create a ROCm storage from host f64 data (copies H2D).
    pub fn from_host_f64(
        device: Arc<RocmDevice>,
        data: &[f64],
        dtype: DType,
    ) -> Result<Self> {
        let len = data.len();
        let byte_size = len * dtype.size_in_bytes();
        let block = device
            .allocator()
            .allocate(byte_size)
            .map_err(TheanoError::from)?;

        #[cfg(feature = "hip")]
        {
            // Real H2D copy via HIP
            // hip_sys::hipMemcpy(block.ptr as *mut _, data.as_ptr() as *const _,
            //     byte_size, hip_sys::hipMemcpyHostToDevice);
        }

        Ok(Self {
            device,
            block: Arc::new(block),
            dtype,
            len,
            #[cfg(not(feature = "hip"))]
            mock_data: Arc::new(data.to_vec()),
        })
    }

    /// Copy data back to host as f64 (D2H transfer).
    pub fn to_host_f64(&self) -> Result<Vec<f64>> {
        #[cfg(feature = "hip")]
        {
            // Real D2H copy via HIP
            // let mut data = vec![0.0f64; self.len];
            // hip_sys::hipMemcpy(data.as_mut_ptr() as *mut _, self.block.ptr as *const _,
            //     self.len * self.dtype.size_in_bytes(), hip_sys::hipMemcpyDeviceToHost);
            // return Ok(data);
            todo!("HIP D2H copy")
        }

        #[cfg(not(feature = "hip"))]
        {
            Ok((*self.mock_data).clone())
        }
    }

    /// Get the ROCm device.
    pub fn rocm_device(&self) -> &RocmDevice {
        &self.device
    }

    /// Get the device pointer (or mock address).
    pub fn device_ptr(&self) -> usize {
        self.block.ptr
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        self.len * self.dtype.size_in_bytes()
    }

    /// Apply a unary op (mock implementation for testing).
    #[cfg(not(feature = "hip"))]
    fn mock_unary(&self, op: UnaryOp) -> Result<Self> {
        let data = &*self.mock_data;
        let result: Vec<f64> = data.iter().map(|&x| match op {
            UnaryOp::Neg => -x,
            UnaryOp::Abs => x.abs(),
            UnaryOp::Exp => x.exp(),
            UnaryOp::Log => x.ln(),
            UnaryOp::Log2 => x.log2(),
            UnaryOp::Log10 => x.log10(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Rsqrt => 1.0 / x.sqrt(),
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Tan => x.tan(),
            UnaryOp::Asin => x.asin(),
            UnaryOp::Acos => x.acos(),
            UnaryOp::Atan => x.atan(),
            UnaryOp::Sinh => x.sinh(),
            UnaryOp::Cosh => x.cosh(),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::Gelu => {
                let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
            }
            UnaryOp::Silu => x / (1.0 + (-x).exp()),
            UnaryOp::Floor => x.floor(),
            UnaryOp::Ceil => x.ceil(),
            UnaryOp::Round => x.round(),
            UnaryOp::Sign => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
            UnaryOp::Reciprocal => 1.0 / x,
            UnaryOp::Square => x * x,
            UnaryOp::Erf => erf_approx(x),
        }).collect();

        Ok(Self {
            device: self.device.clone(),
            block: self.block.clone(),
            dtype: self.dtype,
            len: self.len,
            mock_data: Arc::new(result),
        })
    }
}

#[cfg(not(feature = "hip"))]
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

impl BackendStorage for RocmStorage {
    fn device(&self) -> Device {
        Device::Rocm(self.device.ordinal())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn len(&self) -> usize {
        self.len
    }

    fn unary_op(&self, op: UnaryOp, _shape: &[usize], _strides: &[usize]) -> Result<Self> {
        #[cfg(feature = "hip")]
        {
            // Launch HIP kernel for the unary op
            todo!("HIP unary_op kernel launch")
        }

        #[cfg(not(feature = "hip"))]
        {
            self.mock_unary(op)
        }
    }

    fn binary_op(
        &self,
        rhs: &Self,
        op: BinaryOp,
        _lhs_shape: &[usize],
        _lhs_strides: &[usize],
        _rhs_shape: &[usize],
        _rhs_strides: &[usize],
        _out_shape: &[usize],
    ) -> Result<Self> {
        #[cfg(feature = "hip")]
        {
            todo!("HIP binary_op kernel launch")
        }

        #[cfg(not(feature = "hip"))]
        {
            let a = &*self.mock_data;
            let b = &*rhs.mock_data;
            let result: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| match op {
                BinaryOp::Add => x + y,
                BinaryOp::Sub => x - y,
                BinaryOp::Mul => x * y,
                BinaryOp::Div => x / y,
                BinaryOp::Pow => x.powf(y),
                BinaryOp::Min => x.min(y),
                BinaryOp::Max => x.max(y),
                BinaryOp::Eq => if (x - y).abs() < f64::EPSILON { 1.0 } else { 0.0 },
                BinaryOp::Ne => if (x - y).abs() >= f64::EPSILON { 1.0 } else { 0.0 },
                BinaryOp::Gt => if x > y { 1.0 } else { 0.0 },
                BinaryOp::Ge => if x >= y { 1.0 } else { 0.0 },
                BinaryOp::Lt => if x < y { 1.0 } else { 0.0 },
                BinaryOp::Le => if x <= y { 1.0 } else { 0.0 },
                BinaryOp::And => if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 },
                BinaryOp::Or => if x != 0.0 || y != 0.0 { 1.0 } else { 0.0 },
                BinaryOp::Xor => if (x != 0.0) != (y != 0.0) { 1.0 } else { 0.0 },
            }).collect();

            Ok(Self {
                device: self.device.clone(),
                block: self.block.clone(),
                dtype: self.dtype,
                len: result.len(),
                mock_data: Arc::new(result),
            })
        }
    }

    fn reduce_op(
        &self,
        op: ReduceOp,
        _shape: &[usize],
        _strides: &[usize],
        _reduce_dims: &[usize],
        _keep_dim: bool,
    ) -> Result<Self> {
        #[cfg(feature = "hip")]
        {
            todo!("HIP reduce_op kernel launch")
        }

        #[cfg(not(feature = "hip"))]
        {
            let data = &*self.mock_data;
            let result = match op {
                ReduceOp::Sum => vec![data.iter().sum()],
                ReduceOp::Mean => vec![data.iter().sum::<f64>() / data.len() as f64],
                ReduceOp::Max => vec![data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)],
                ReduceOp::Min => vec![data.iter().cloned().fold(f64::INFINITY, f64::min)],
                ReduceOp::Prod => vec![data.iter().product()],
                _ => return Err(TheanoError::not_implemented("ROCm reduce op")),
            };
            Ok(Self {
                device: self.device.clone(),
                block: self.block.clone(),
                dtype: self.dtype,
                len: result.len(),
                mock_data: Arc::new(result),
            })
        }
    }

    fn matmul(
        &self,
        rhs: &Self,
        lhs_shape: &[usize],
        _lhs_strides: &[usize],
        rhs_shape: &[usize],
        _rhs_strides: &[usize],
    ) -> Result<Self> {
        #[cfg(feature = "hip")]
        {
            // Use rocBLAS SGEMM
            todo!("rocBLAS matmul")
        }

        #[cfg(not(feature = "hip"))]
        {
            let a = &*self.mock_data;
            let b = &*rhs.mock_data;
            let m = lhs_shape[lhs_shape.len() - 2];
            let k = lhs_shape[lhs_shape.len() - 1];
            let n = rhs_shape[rhs_shape.len() - 1];

            let mut result = vec![0.0f64; m * n];
            for i in 0..m {
                for p in 0..k {
                    let a_val = a[i * k + p];
                    for j in 0..n {
                        result[i * n + j] += a_val * b[p * n + j];
                    }
                }
            }

            Ok(Self {
                device: self.device.clone(),
                block: self.block.clone(),
                dtype: self.dtype,
                len: result.len(),
                mock_data: Arc::new(result),
            })
        }
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self> {
        Ok(Self {
            device: self.device.clone(),
            block: self.block.clone(),
            dtype,
            len: self.len,
            #[cfg(not(feature = "hip"))]
            mock_data: self.mock_data.clone(),
        })
    }

    fn contiguous(&self, _shape: &[usize], _strides: &[usize]) -> Result<Self> {
        // For mock, data is always contiguous
        Ok(self.clone())
    }

    fn fill(value: f64, len: usize, dtype: DType, device: &Device) -> Result<Self> {
        let ordinal = match device {
            Device::Rocm(i) => *i,
            _ => return Err(TheanoError::runtime("RocmStorage::fill requires ROCm device")),
        };
        let rocm_dev = crate::device::get_device(ordinal).map_err(TheanoError::from)?;

        Ok(Self {
            device: rocm_dev,
            block: Arc::new(Block { ptr: 0, size: 0, allocated: false, stream: 0 }),
            dtype,
            len,
            #[cfg(not(feature = "hip"))]
            mock_data: Arc::new(vec![value; len]),
        })
    }

    fn from_f64_slice(data: &[f64], dtype: DType, device: &Device) -> Result<Self> {
        let ordinal = match device {
            Device::Rocm(i) => *i,
            _ => return Err(TheanoError::runtime("RocmStorage requires ROCm device")),
        };
        let rocm_dev = crate::device::get_device(ordinal).map_err(TheanoError::from)?;
        RocmStorage::from_host_f64(rocm_dev, data, dtype)
    }

    fn to_f64_vec(&self, _shape: &[usize], _strides: &[usize]) -> Result<Vec<f64>> {
        self.to_host_f64()
    }

    fn where_cond(
        &self,
        _cond: &Self,
        _other: &Self,
        _shape: &[usize],
        _strides: &[usize],
    ) -> Result<Self> {
        Err(TheanoError::not_implemented("RocmStorage::where_cond"))
    }

    fn index_select(
        &self,
        _dim: usize,
        _indices: &Self,
        _shape: &[usize],
        _strides: &[usize],
    ) -> Result<Self> {
        Err(TheanoError::not_implemented("RocmStorage::index_select"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocm_storage_allocate() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let storage = RocmStorage::allocate(dev, 100, DType::F32).unwrap();
        assert_eq!(storage.len(), 100);
        assert_eq!(storage.dtype(), DType::F32);
        assert_eq!(storage.device(), Device::Rocm(0));
    }

    #[test]
    fn test_rocm_storage_from_host() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let data = vec![1.0, 2.0, 3.0];
        let storage = RocmStorage::from_host_f64(dev, &data, DType::F32).unwrap();
        let back = storage.to_host_f64().unwrap();
        assert_eq!(back, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rocm_storage_unary() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let storage = RocmStorage::from_host_f64(dev, &data, DType::F32).unwrap();

        let result = storage.unary_op(UnaryOp::Relu, &[4], &[1]).unwrap();
        let back = result.to_host_f64().unwrap();
        assert_eq!(back, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rocm_storage_unary_neg() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let data = vec![1.0, -2.0, 3.0];
        let storage = RocmStorage::from_host_f64(dev, &data, DType::F32).unwrap();

        let result = storage.unary_op(UnaryOp::Neg, &[3], &[1]).unwrap();
        let back = result.to_host_f64().unwrap();
        assert_eq!(back, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_rocm_storage_binary() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let a = RocmStorage::from_host_f64(dev.clone(), &[1.0, 2.0, 3.0], DType::F32).unwrap();
        let b = RocmStorage::from_host_f64(dev, &[4.0, 5.0, 6.0], DType::F32).unwrap();

        let c = a.binary_op(&b, BinaryOp::Add, &[3], &[1], &[3], &[1], &[3]).unwrap();
        let back = c.to_host_f64().unwrap();
        assert_eq!(back, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_rocm_storage_binary_mul() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let a = RocmStorage::from_host_f64(dev.clone(), &[2.0, 3.0, 4.0], DType::F32).unwrap();
        let b = RocmStorage::from_host_f64(dev, &[5.0, 6.0, 7.0], DType::F32).unwrap();

        let c = a.binary_op(&b, BinaryOp::Mul, &[3], &[1], &[3], &[1], &[3]).unwrap();
        let back = c.to_host_f64().unwrap();
        assert_eq!(back, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_rocm_storage_reduce_sum() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let storage = RocmStorage::from_host_f64(dev, &data, DType::F32).unwrap();

        let result = storage.reduce_op(ReduceOp::Sum, &[4], &[1], &[0], false).unwrap();
        let back = result.to_host_f64().unwrap();
        assert_eq!(back, vec![10.0]);
    }

    #[test]
    fn test_rocm_storage_matmul() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let a = RocmStorage::from_host_f64(
            dev.clone(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            DType::F32,
        ).unwrap();
        let b = RocmStorage::from_host_f64(
            dev,
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            DType::F32,
        ).unwrap();

        let c = a.matmul(&b, &[2, 3], &[3, 1], &[3, 2], &[2, 1]).unwrap();
        let back = c.to_host_f64().unwrap();
        assert_eq!(back, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_rocm_storage_fill() {
        let storage = RocmStorage::fill(42.0, 5, DType::F32, &Device::Rocm(0)).unwrap();
        let back = storage.to_host_f64().unwrap();
        assert_eq!(back, vec![42.0; 5]);
    }

    #[test]
    fn test_rocm_storage_from_f64_slice() {
        let data = vec![1.0, 2.0, 3.0];
        let storage = RocmStorage::from_f64_slice(&data, DType::F32, &Device::Rocm(0)).unwrap();
        let back = storage.to_f64_vec(&[3], &[1]).unwrap();
        assert_eq!(back, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rocm_storage_to_dtype() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let storage = RocmStorage::from_host_f64(dev, &[1.0, 2.0], DType::F32).unwrap();
        let converted = storage.to_dtype(DType::F64).unwrap();
        assert_eq!(converted.dtype(), DType::F64);
        assert_eq!(converted.len(), 2);
    }

    #[test]
    fn test_rocm_storage_contiguous() {
        let dev = Arc::new(crate::device::RocmDevice::new(0).unwrap());
        let storage = RocmStorage::from_host_f64(dev, &[1.0, 2.0, 3.0], DType::F32).unwrap();
        let contiguous = storage.contiguous(&[3], &[1]).unwrap();
        let back = contiguous.to_host_f64().unwrap();
        assert_eq!(back, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rocm_storage_wrong_device() {
        let result = RocmStorage::fill(1.0, 5, DType::F32, &Device::Cpu);
        assert!(result.is_err());
    }
}
