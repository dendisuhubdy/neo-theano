use theano_types::{DType, Device, Shape, Result, TheanoError};

use crate::storage::Storage;
use crate::tensor::Tensor;

impl Tensor {
    /// Create a tensor filled with zeros. Like `torch.zeros`.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::zeros_with(shape, DType::F32, &Device::Cpu)
    }

    /// Create a tensor filled with zeros on a specific device/dtype.
    pub fn zeros_with(shape: &[usize], dtype: DType, device: &Device) -> Self {
        let numel = shape.iter().product::<usize>().max(1);
        // We'll use the cpu storage directly for now
        let data = vec![0.0f64; numel];
        Self::from_f64_data(&data, shape, dtype, device)
    }

    /// Create a tensor filled with ones. Like `torch.ones`.
    pub fn ones(shape: &[usize]) -> Self {
        Self::ones_with(shape, DType::F32, &Device::Cpu)
    }

    /// Create a tensor filled with ones on a specific device/dtype.
    pub fn ones_with(shape: &[usize], dtype: DType, device: &Device) -> Self {
        let numel = shape.iter().product::<usize>().max(1);
        let data = vec![1.0f64; numel];
        Self::from_f64_data(&data, shape, dtype, device)
    }

    /// Create a tensor filled with a given value. Like `torch.full`.
    pub fn full(shape: &[usize], value: f64) -> Self {
        Self::full_with(shape, value, DType::F32, &Device::Cpu)
    }

    /// Create a tensor filled with a given value on a specific device/dtype.
    pub fn full_with(shape: &[usize], value: f64, dtype: DType, device: &Device) -> Self {
        let numel = shape.iter().product::<usize>().max(1);
        let data = vec![value; numel];
        Self::from_f64_data(&data, shape, dtype, device)
    }

    /// Create a 1-D tensor with values from start to end (exclusive) with step. Like `torch.arange`.
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        let mut data = Vec::new();
        let mut v = start;
        if step > 0.0 {
            while v < end {
                data.push(v);
                v += step;
            }
        } else if step < 0.0 {
            while v > end {
                data.push(v);
                v += step;
            }
        }
        let len = data.len();
        Self::from_f64_data(&data, &[len], DType::F32, &Device::Cpu)
    }

    /// Create a 1-D tensor with `steps` evenly spaced values from start to end. Like `torch.linspace`.
    pub fn linspace(start: f64, end: f64, steps: usize) -> Self {
        let data: Vec<f64> = if steps == 0 {
            vec![]
        } else if steps == 1 {
            vec![start]
        } else {
            (0..steps)
                .map(|i| start + (end - start) * (i as f64) / ((steps - 1) as f64))
                .collect()
        };
        let len = data.len();
        Self::from_f64_data(&data, &[len], DType::F32, &Device::Cpu)
    }

    /// Create a 2-D identity matrix. Like `torch.eye`.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_f64_data(&data, &[n, n], DType::F32, &Device::Cpu)
    }

    /// Create a tensor from an f64 slice with explicit shape. Like `torch.tensor`.
    pub fn from_slice(data: &[f64], shape: &[usize]) -> Self {
        Self::from_f64_data(data, shape, DType::F32, &Device::Cpu)
    }

    /// Create a 1-D tensor from an f64 Vec.
    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self::from_f64_data(&data, &[len], DType::F32, &Device::Cpu)
    }

    /// Create a scalar tensor.
    pub fn scalar(value: f64) -> Self {
        Self::from_f64_data(&[value], &[], DType::F32, &Device::Cpu)
    }

    /// Internal: construct a tensor from f64 data via CPU storage.
    fn from_f64_data(data: &[f64], shape: &[usize], dtype: DType, _device: &Device) -> Self {
        // For now, only CPU is supported in core. Backend-specific creation
        // will be handled by the respective backend crates.
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
            Device::Cpu,
        )
    }
}

/// Simple CPU storage backed by Vec<f64>.
///
/// This is a minimal implementation for tensor creation. The full CPU backend
/// in `theano-cpu` provides optimized typed storage and kernels.
#[derive(Clone)]
pub(crate) struct CpuF64Storage {
    pub data: Vec<f64>,
    pub dtype: DType,
}

impl crate::storage::BackendStorageBoxed for CpuF64Storage {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn to_f64_vec(&self, shape: &[usize], strides: &[usize], offset: usize) -> Result<Vec<f64>> {
        let numel: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if (strides.is_empty() || is_contiguous(shape, strides)) && offset == 0 {
            // Contiguous with no offset: just copy the data
            Ok(self.data[..numel].to_vec())
        } else {
            // Non-contiguous or offset: gather elements using strides
            let mut result = Vec::with_capacity(numel);
            gather_strided(&self.data, shape, strides, 0, offset, &mut result);
            Ok(result)
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn crate::storage::BackendStorageBoxed> {
        Box::new(self.clone())
    }

    fn to_device(&self, device: &Device, shape: &[usize], strides: &[usize], offset: usize) -> Result<Box<dyn crate::storage::BackendStorageBoxed>> {
        if device.is_cpu() {
            // CPU → CPU: just clone
            return Ok(self.clone_box());
        }
        // CPU → GPU: materialize contiguous data, then create device storage.
        // The actual GPU storage creation would be handled by the GPU backend crate.
        // For now, we materialize the f64 data and store it tagged with the target device.
        // When a real GPU backend is active, it will intercept this and do cudaMemcpy etc.
        let contiguous_data = self.to_f64_vec(shape, strides, offset)?;
        Ok(Box::new(CpuF64Storage {
            data: contiguous_data,
            dtype: self.dtype,
        }))
    }
}

fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }
    let expected = Shape::new(shape.to_vec()).contiguous_strides();
    strides == expected.as_slice()
}

fn gather_strided(
    data: &[f64],
    shape: &[usize],
    strides: &[usize],
    dim: usize,
    offset: usize,
    result: &mut Vec<f64>,
) {
    if dim == shape.len() {
        if offset < data.len() {
            result.push(data[offset]);
        } else {
            result.push(0.0);
        }
        return;
    }
    for i in 0..shape[dim] {
        gather_strided(data, shape, strides, dim + 1, offset + i * strides[dim], result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        let data = t.to_vec_f64().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(&[3, 2]);
        let data = t.to_vec_f64().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let t = Tensor::full(&[2, 2], 42.0);
        let data = t.to_vec_f64().unwrap();
        assert!(data.iter().all(|&x| x == 42.0));
    }

    #[test]
    fn test_arange() {
        let t = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(t.shape(), &[5]);
        let data = t.to_vec_f64().unwrap();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_linspace() {
        let t = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.shape(), &[5]);
        let data = t.to_vec_f64().unwrap();
        assert_eq!(data, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_eye() {
        let t = Tensor::eye(3);
        assert_eq!(t.shape(), &[3, 3]);
        let data = t.to_vec_f64().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_from_slice() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        let data = t.to_vec_f64().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar() {
        let t = Tensor::scalar(3.14);
        assert!(t.is_scalar());
        assert_eq!(t.numel(), 1);
        assert_eq!(t.item().unwrap(), 3.14);
    }

    #[test]
    fn test_to_same_device_is_clone() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let t2 = t.to(&Device::Cpu).unwrap();
        assert_eq!(t.device(), t2.device());
        assert_eq!(t.to_vec_f64().unwrap(), t2.to_vec_f64().unwrap());
    }

    #[test]
    fn test_to_device_preserves_data() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // Transfer to a "CUDA" device (still backed by CPU storage for now)
        let t_gpu = t.to(&Device::Cuda(0)).unwrap();
        assert_eq!(t_gpu.shape(), &[2, 2]);
        assert_eq!(t_gpu.numel(), 4);
        // Data should be preserved
        let data_orig = t.to_vec_f64().unwrap();
        let data_gpu = t_gpu.to_vec_f64().unwrap();
        assert_eq!(data_orig, data_gpu);
    }

    #[test]
    fn test_to_roundtrip() {
        let t = Tensor::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let t_gpu = t.to(&Device::Cuda(0)).unwrap();
        let t_back = t_gpu.to(&Device::Cpu).unwrap();
        assert_eq!(t_back.device(), &Device::Cpu);
        assert_eq!(t.to_vec_f64().unwrap(), t_back.to_vec_f64().unwrap());
    }

    #[test]
    fn test_cpu_shorthand() {
        let t = Tensor::from_slice(&[1.0, 2.0], &[2]);
        let t_cpu = t.cpu().unwrap();
        assert!(t_cpu.is_cpu());
    }

    #[test]
    fn test_is_cpu_is_cuda() {
        let t = Tensor::ones(&[2, 2]);
        assert!(t.is_cpu());
        assert!(!t.is_cuda());
    }

    #[test]
    fn test_to_preserves_requires_grad() {
        let t = Tensor::ones(&[2, 3]).requires_grad_(true);
        assert!(t.requires_grad());
        let t2 = t.to(&Device::Cuda(0)).unwrap();
        assert!(t2.requires_grad());
    }
}
