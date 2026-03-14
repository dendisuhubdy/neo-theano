use theano_types::{Result, Shape, TheanoError};

use crate::tensor::Tensor;

impl Tensor {
    /// In-place addition. Like `torch.Tensor.add_`.
    ///
    /// Returns a new tensor with the result (due to Arc-based storage).
    pub fn add_(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.to_vec_f64()?;
        let out_shape = Shape::broadcast_shape(&self.size(), &other.size())
            .ok_or_else(|| TheanoError::broadcast_error(self.shape(), other.shape()))?;

        if self.shape() == out_shape.dims() {
            let b = if other.shape() == out_shape.dims() {
                other.clone()
            } else {
                other.expand(out_shape.dims())?
            };
            let b_data = b.to_vec_f64()?;
            let data: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(&a, &b)| a + b).collect();
            Ok(Self::from_f64_result(&data, self.shape(), self.dtype()))
        } else {
            self.add(other)
        }
    }

    /// In-place subtraction. Like `torch.Tensor.sub_`.
    ///
    /// Returns a new tensor with the result (due to Arc-based storage).
    pub fn sub_(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.to_vec_f64()?;
        let out_shape = Shape::broadcast_shape(&self.size(), &other.size())
            .ok_or_else(|| TheanoError::broadcast_error(self.shape(), other.shape()))?;

        if self.shape() == out_shape.dims() {
            let b = if other.shape() == out_shape.dims() {
                other.clone()
            } else {
                other.expand(out_shape.dims())?
            };
            let b_data = b.to_vec_f64()?;
            let data: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(&a, &b)| a - b).collect();
            Ok(Self::from_f64_result(&data, self.shape(), self.dtype()))
        } else {
            self.sub(other)
        }
    }

    /// In-place multiplication. Like `torch.Tensor.mul_`.
    ///
    /// Returns a new tensor with the result (due to Arc-based storage).
    pub fn mul_(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.to_vec_f64()?;
        let out_shape = Shape::broadcast_shape(&self.size(), &other.size())
            .ok_or_else(|| TheanoError::broadcast_error(self.shape(), other.shape()))?;

        if self.shape() == out_shape.dims() {
            let b = if other.shape() == out_shape.dims() {
                other.clone()
            } else {
                other.expand(out_shape.dims())?
            };
            let b_data = b.to_vec_f64()?;
            let data: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(&a, &b)| a * b).collect();
            Ok(Self::from_f64_result(&data, self.shape(), self.dtype()))
        } else {
            self.mul(other)
        }
    }

    /// In-place division. Like `torch.Tensor.div_`.
    ///
    /// Returns a new tensor with the result (due to Arc-based storage).
    pub fn div_(&self, other: &Tensor) -> Result<Tensor> {
        let a_data = self.to_vec_f64()?;
        let out_shape = Shape::broadcast_shape(&self.size(), &other.size())
            .ok_or_else(|| TheanoError::broadcast_error(self.shape(), other.shape()))?;

        if self.shape() == out_shape.dims() {
            let b = if other.shape() == out_shape.dims() {
                other.clone()
            } else {
                other.expand(out_shape.dims())?
            };
            let b_data = b.to_vec_f64()?;
            let data: Vec<f64> = a_data.iter().zip(b_data.iter()).map(|(&a, &b)| a / b).collect();
            Ok(Self::from_f64_result(&data, self.shape(), self.dtype()))
        } else {
            self.div(other)
        }
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

    /// Clamp values in-place to [min, max]. Like `torch.Tensor.clamp_`.
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
    use crate::Tensor;

    #[test]
    fn test_add_inplace() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let c = a.add_(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_inplace() {
        let a = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let c = a.sub_(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_mul_inplace() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], &[3]);
        let c = a.mul_(&b).unwrap();
        assert_eq!(c.to_vec_f64().unwrap(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_inplace() {
        let a = Tensor::from_slice(&[10.0, 20.0, 30.0], &[3]);
        let b = Tensor::from_slice(&[2.0, 4.0, 5.0], &[3]);
        let c = a.div_(&b).unwrap();
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
    fn test_add_inplace_broadcast() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0], &[1, 3]);
        let c = a.add_(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(
            c.to_vec_f64().unwrap(),
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }
}
