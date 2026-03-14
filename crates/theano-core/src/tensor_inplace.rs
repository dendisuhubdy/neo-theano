use theano_types::Result;

use crate::tensor::Tensor;

impl Tensor {
    /// Elementwise addition (out-of-place). Like `torch.Tensor.add`.
    ///
    /// Note: Despite PyTorch's `add_` convention, Rust's Arc-based storage
    /// prevents true in-place mutation. This returns a new tensor.
    /// Use the standard `.add()` method for the same result.
    #[deprecated(note = "use .add() instead — this is not truly in-place due to Arc-based storage")]
    pub fn add_(&self, other: &Tensor) -> Result<Tensor> {
        self.add(other)
    }

    /// Elementwise subtraction (out-of-place).
    ///
    /// Note: Not truly in-place due to Arc-based storage. Use `.sub()` instead.
    #[deprecated(note = "use .sub() instead — this is not truly in-place due to Arc-based storage")]
    pub fn sub_(&self, other: &Tensor) -> Result<Tensor> {
        self.sub(other)
    }

    /// Elementwise multiplication (out-of-place).
    ///
    /// Note: Not truly in-place due to Arc-based storage. Use `.mul()` instead.
    #[deprecated(note = "use .mul() instead — this is not truly in-place due to Arc-based storage")]
    pub fn mul_(&self, other: &Tensor) -> Result<Tensor> {
        self.mul(other)
    }

    /// Elementwise division (out-of-place).
    ///
    /// Note: Not truly in-place due to Arc-based storage. Use `.div()` instead.
    #[deprecated(note = "use .div() instead — this is not truly in-place due to Arc-based storage")]
    pub fn div_(&self, other: &Tensor) -> Result<Tensor> {
        self.div(other)
    }

    /// Fill tensor with zeros. Like `torch.Tensor.zero_`.
    ///
    /// Returns a new tensor filled with zeros (due to Arc-based storage).
    pub fn zero_(&self) -> Result<Tensor> {
        let data = vec![0.0f64; self.numel()];
        Ok(Self::from_f64_result(&data, self.shape(), self.dtype()))
    }

    /// Fill tensor with a given value. Like `torch.Tensor.fill_`.
    ///
    /// Returns a new tensor filled with the value (due to Arc-based storage).
    pub fn fill_(&self, value: f64) -> Result<Tensor> {
        let data = vec![value; self.numel()];
        Ok(Self::from_f64_result(&data, self.shape(), self.dtype()))
    }

    /// Clamp values to [min, max]. Like `torch.Tensor.clamp_`.
    ///
    /// Returns a new tensor with clamped values (due to Arc-based storage).
    pub fn clamp_(&self, min: f64, max: f64) -> Result<Tensor> {
        let data = self.to_vec_f64()?;
        let result: Vec<f64> = data.iter().map(|&x| x.clamp(min, max)).collect();
        Ok(Self::from_f64_result(&result, self.shape(), self.dtype()))
    }
}

#[cfg(test)]
mod tests {
    #[allow(deprecated)]
    use crate::Tensor;

    #[test]
    fn test_add_inplace() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        // Use the non-deprecated .add() method
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_inplace() {
        let a = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_mul_inplace() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], &[3]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_inplace() {
        let a = Tensor::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let b = Tensor::from_slice(&[2.0, 4.0, 5.0], &[3]);
        let c = a.div(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_zero_inplace() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let z = a.zero_().unwrap();
        assert_eq!(z.to_vec_f64().unwrap(), vec![0.0, 0.0, 0.0]);
        assert_eq!(z.shape(), &[3]);
    }

    #[test]
    fn test_fill_inplace() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let f = a.fill_(42.0).unwrap();
        assert_eq!(f.to_vec_f64().unwrap(), vec![42.0, 42.0, 42.0, 42.0]);
        assert_eq!(f.shape(), &[2, 2]);
    }

    #[test]
    fn test_clamp_inplace() {
        let a = Tensor::from_slice(&[-1.0, 0.5, 2.0, 3.0], &[4]);
        let c = a.clamp_(0.0, 1.0).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0], &[1, 3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(
            c.to_vec_f64().unwrap(),
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }
}
