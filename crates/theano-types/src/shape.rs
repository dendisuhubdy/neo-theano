use std::fmt;

/// Dynamic tensor shape, mirroring PyTorch's `torch.Size`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Create a scalar shape (0 dimensions).
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Number of dimensions (ndim).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1 // scalar
        } else {
            self.dims.iter().product()
        }
    }

    /// Get the dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get a specific dimension.
    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    /// Compute contiguous (row-major) strides for this shape.
    pub fn contiguous_strides(&self) -> Vec<usize> {
        if self.dims.is_empty() {
            return vec![];
        }
        let mut strides = vec![1usize; self.dims.len()];
        for i in (0..self.dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Check if two shapes are broadcastable (PyTorch broadcasting rules).
    pub fn broadcast_shape(a: &Shape, b: &Shape) -> Option<Shape> {
        let ndim = a.ndim().max(b.ndim());
        let mut result = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let da = if i < ndim - a.ndim() {
                1
            } else {
                a.dims[i - (ndim - a.ndim())]
            };
            let db = if i < ndim - b.ndim() {
                1
            } else {
                b.dims[i - (ndim - b.ndim())]
            };
            if da == db {
                result.push(da);
            } else if da == 1 {
                result.push(db);
            } else if db == 1 {
                result.push(da);
            } else {
                return None; // not broadcastable
            }
        }
        Some(Shape::new(result))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self { dims }
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "theano.Size([")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "])")
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &usize {
        &self.dims[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_basics() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dim(1), 3);
    }

    #[test]
    fn test_scalar_shape() {
        let s = Shape::scalar();
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
    }

    #[test]
    fn test_contiguous_strides() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.contiguous_strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_broadcast_shape() {
        let a = Shape::new(vec![1, 3]);
        let b = Shape::new(vec![4, 1]);
        let c = Shape::broadcast_shape(&a, &b).unwrap();
        assert_eq!(c.dims(), &[4, 3]);

        let a = Shape::new(vec![3]);
        let b = Shape::new(vec![2, 3]);
        let c = Shape::broadcast_shape(&a, &b).unwrap();
        assert_eq!(c.dims(), &[2, 3]);

        // Not broadcastable
        let a = Shape::new(vec![2, 3]);
        let b = Shape::new(vec![4, 3]);
        assert!(Shape::broadcast_shape(&a, &b).is_none());
    }
}
