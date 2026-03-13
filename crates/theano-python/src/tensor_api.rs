//! Python API surface for Tensor operations.
//!
//! This module defines the functions and methods that will be exposed
//! to Python via PyO3. Each function maps to a PyTorch Python API call.
//!
//! When PyO3 is enabled, `TensorAPI` will be annotated with `#[pyclass]`
//! and each public method with `#[pymethods]`, so that Python code like
//! `torch.tensor(...)`, `x.shape`, `x + y` all work transparently.

use theano_core::Tensor;

/// Python-facing Tensor wrapper.
/// Will be annotated with #[pyclass] when PyO3 is enabled.
pub struct TensorAPI {
    inner: Tensor,
}

impl TensorAPI {
    // ---- Creation functions (torch.*) ----

    /// torch.tensor(data)
    pub fn tensor_from_list(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self { inner: Tensor::from_slice(&data, &shape) }
    }

    /// torch.zeros(*size)
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self { inner: Tensor::zeros(&shape) }
    }

    /// torch.ones(*size)
    pub fn ones(shape: Vec<usize>) -> Self {
        Self { inner: Tensor::ones(&shape) }
    }

    /// torch.full(size, fill_value)
    pub fn full(shape: Vec<usize>, value: f64) -> Self {
        Self { inner: Tensor::full(&shape, value) }
    }

    /// torch.arange(start, end, step)
    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        Self { inner: Tensor::arange(start, end, step) }
    }

    /// torch.linspace(start, end, steps)
    pub fn linspace(start: f64, end: f64, steps: usize) -> Self {
        Self { inner: Tensor::linspace(start, end, steps) }
    }

    /// torch.eye(n)
    pub fn eye(n: usize) -> Self {
        Self { inner: Tensor::eye(n) }
    }

    // ---- Tensor properties ----

    pub fn shape(&self) -> Vec<usize> { self.inner.shape().to_vec() }
    pub fn ndim(&self) -> usize { self.inner.ndim() }
    pub fn numel(&self) -> usize { self.inner.numel() }
    pub fn dtype(&self) -> String { format!("{}", self.inner.dtype()) }
    pub fn device(&self) -> String { format!("{}", self.inner.device()) }
    pub fn is_contiguous(&self) -> bool { self.inner.is_contiguous() }

    // ---- Tensor methods ----

    pub fn item(&self) -> Result<f64, String> {
        self.inner.item().map_err(|e| e.to_string())
    }

    pub fn tolist(&self) -> Result<Vec<f64>, String> {
        self.inner.to_vec_f64().map_err(|e| e.to_string())
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, String> {
        self.inner.reshape(&shape)
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<Self, String> {
        self.inner.transpose(dim0, dim1)
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn t(&self) -> Result<Self, String> {
        self.inner.t()
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn contiguous(&self) -> Result<Self, String> {
        self.inner.contiguous()
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    // ---- Operations ----

    pub fn add(&self, other: &TensorAPI) -> Result<Self, String> {
        self.inner.add(&other.inner)
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn sub(&self, other: &TensorAPI) -> Result<Self, String> {
        self.inner.sub(&other.inner)
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn mul(&self, other: &TensorAPI) -> Result<Self, String> {
        self.inner.mul(&other.inner)
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn div(&self, other: &TensorAPI) -> Result<Self, String> {
        self.inner.div(&other.inner)
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn matmul(&self, other: &TensorAPI) -> Result<Self, String> {
        self.inner.matmul(&other.inner)
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn neg(&self) -> Result<Self, String> {
        self.inner.neg()
            .map(|t| Self { inner: t })
            .map_err(|e| e.to_string())
    }

    pub fn exp(&self) -> Result<Self, String> {
        self.inner.exp().map(|t| Self { inner: t }).map_err(|e| e.to_string())
    }

    pub fn log(&self) -> Result<Self, String> {
        self.inner.log().map(|t| Self { inner: t }).map_err(|e| e.to_string())
    }

    pub fn relu(&self) -> Result<Self, String> {
        self.inner.relu().map(|t| Self { inner: t }).map_err(|e| e.to_string())
    }

    pub fn sigmoid(&self) -> Result<Self, String> {
        self.inner.sigmoid().map(|t| Self { inner: t }).map_err(|e| e.to_string())
    }

    pub fn tanh_(&self) -> Result<Self, String> {
        self.inner.tanh().map(|t| Self { inner: t }).map_err(|e| e.to_string())
    }

    pub fn sum(&self) -> Result<Self, String> {
        self.inner.sum().map(|t| Self { inner: t }).map_err(|e| e.to_string())
    }

    pub fn mean(&self) -> Result<Self, String> {
        self.inner.mean().map(|t| Self { inner: t }).map_err(|e| e.to_string())
    }

    pub fn softmax(&self, dim: i64) -> Result<Self, String> {
        self.inner.softmax(dim).map(|t| Self { inner: t }).map_err(|e| e.to_string())
    }

    /// Get the inner Tensor (for Rust-side use).
    pub fn inner(&self) -> &Tensor { &self.inner }
}

impl From<Tensor> for TensorAPI {
    fn from(t: Tensor) -> Self { Self { inner: t } }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Creation tests ----

    #[test]
    fn test_tensor_from_list() {
        let t = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape(), vec![2, 2]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.tolist().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_zeros() {
        let t = TensorAPI::zeros(vec![3, 4]);
        assert_eq!(t.shape(), vec![3, 4]);
        assert_eq!(t.numel(), 12);
        let data = t.tolist().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = TensorAPI::ones(vec![2, 3]);
        assert_eq!(t.shape(), vec![2, 3]);
        let data = t.tolist().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let t = TensorAPI::full(vec![2, 2], 7.0);
        let data = t.tolist().unwrap();
        assert!(data.iter().all(|&x| x == 7.0));
    }

    #[test]
    fn test_arange() {
        let t = TensorAPI::arange(0.0, 5.0, 1.0);
        assert_eq!(t.shape(), vec![5]);
        assert_eq!(t.tolist().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_linspace() {
        let t = TensorAPI::linspace(0.0, 1.0, 5);
        assert_eq!(t.shape(), vec![5]);
        let data = t.tolist().unwrap();
        assert_eq!(data, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_eye() {
        let t = TensorAPI::eye(3);
        assert_eq!(t.shape(), vec![3, 3]);
        let data = t.tolist().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    // ---- Property tests ----

    #[test]
    fn test_properties() {
        let t = TensorAPI::ones(vec![2, 3, 4]);
        assert_eq!(t.shape(), vec![2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.dtype(), "float32");
        assert_eq!(t.device(), "cpu");
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_item() {
        let t = TensorAPI::tensor_from_list(vec![42.0], vec![1]);
        assert_eq!(t.item().unwrap(), 42.0);
    }

    #[test]
    fn test_item_fails_for_multi_element() {
        let t = TensorAPI::ones(vec![2, 3]);
        assert!(t.item().is_err());
    }

    // ---- View / reshape tests ----

    #[test]
    fn test_reshape() {
        let t = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(r.shape(), vec![3, 2]);
        assert_eq!(r.tolist().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_transpose() {
        let t = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tr = t.transpose(0, 1).unwrap();
        assert_eq!(tr.shape(), vec![3, 2]);
    }

    #[test]
    fn test_t() {
        let t = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tr = t.t().unwrap();
        assert_eq!(tr.shape(), vec![3, 2]);
    }

    #[test]
    fn test_contiguous() {
        let t = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tr = t.transpose(0, 1).unwrap();
        assert!(!tr.is_contiguous());
        let c = tr.contiguous().unwrap();
        assert!(c.is_contiguous());
        assert_eq!(c.shape(), vec![3, 2]);
    }

    // ---- Arithmetic tests ----

    #[test]
    fn test_add() {
        let a = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0], vec![3]);
        let b = TensorAPI::tensor_from_list(vec![4.0, 5.0, 6.0], vec![3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.tolist().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = TensorAPI::tensor_from_list(vec![10.0, 20.0], vec![2]);
        let b = TensorAPI::tensor_from_list(vec![3.0, 7.0], vec![2]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c.tolist().unwrap(), vec![7.0, 13.0]);
    }

    #[test]
    fn test_mul() {
        let a = TensorAPI::tensor_from_list(vec![2.0, 3.0], vec![2]);
        let b = TensorAPI::tensor_from_list(vec![4.0, 5.0], vec![2]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.tolist().unwrap(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_div() {
        let a = TensorAPI::tensor_from_list(vec![10.0, 20.0], vec![2]);
        let b = TensorAPI::tensor_from_list(vec![2.0, 5.0], vec![2]);
        let c = a.div(&b).unwrap();
        assert_eq!(c.tolist().unwrap(), vec![5.0, 4.0]);
    }

    #[test]
    fn test_matmul() {
        let a = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = TensorAPI::tensor_from_list(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), vec![2, 2]);
        assert_eq!(c.tolist().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    // ---- Unary ops tests ----

    #[test]
    fn test_neg() {
        let a = TensorAPI::tensor_from_list(vec![1.0, -2.0, 3.0], vec![3]);
        let b = a.neg().unwrap();
        assert_eq!(b.tolist().unwrap(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let a = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0], vec![3]);
        let b = a.exp().unwrap();
        let c = b.log().unwrap();
        let data = c.tolist().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - (i as f64 + 1.0)).abs() < 1e-10,
                "exp(log(x)) roundtrip failed at index {i}: got {v}"
            );
        }
    }

    #[test]
    fn test_relu() {
        let a = TensorAPI::tensor_from_list(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let b = a.relu().unwrap();
        assert_eq!(b.tolist().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let a = TensorAPI::tensor_from_list(vec![0.0], vec![1]);
        let b = a.sigmoid().unwrap();
        assert!((b.item().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tanh() {
        let a = TensorAPI::tensor_from_list(vec![0.0], vec![1]);
        let b = a.tanh_().unwrap();
        assert!((b.item().unwrap()).abs() < 1e-10);
    }

    // ---- Reduction tests ----

    #[test]
    fn test_sum() {
        let a = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let s = a.sum().unwrap();
        assert_eq!(s.item().unwrap(), 10.0);
    }

    #[test]
    fn test_mean() {
        let a = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let m = a.mean().unwrap();
        assert_eq!(m.item().unwrap(), 2.5);
    }

    #[test]
    fn test_softmax() {
        let a = TensorAPI::tensor_from_list(vec![1.0, 2.0, 3.0], vec![3]);
        let s = a.softmax(0).unwrap();
        let data = s.tolist().unwrap();
        let total: f64 = data.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
        // Monotonically increasing for increasing inputs
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
    }

    // ---- From<Tensor> test ----

    #[test]
    fn test_from_tensor() {
        let inner = Tensor::ones(&[2, 2]);
        let api: TensorAPI = inner.into();
        assert_eq!(api.shape(), vec![2, 2]);
        assert_eq!(api.numel(), 4);
    }

    #[test]
    fn test_inner_accessor() {
        let t = TensorAPI::zeros(vec![3]);
        let inner = t.inner();
        assert_eq!(inner.shape(), &[3]);
    }
}
