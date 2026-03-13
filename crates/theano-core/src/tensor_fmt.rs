use std::fmt;

use crate::tensor::Tensor;

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, device={})",
            self.shape(),
            self.dtype(),
            self.device(),
        )?;
        if self.numel() <= 20 {
            if let Ok(data) = self.to_vec_f64() {
                write!(f, "\n{}", format_data(&data, self.shape(), 0, 0))?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor(")?;
        if let Ok(data) = self.to_vec_f64() {
            write!(f, "{}", format_data(&data, self.shape(), 0, 0))?;
        } else {
            write!(f, "...")?;
        }
        if self.dtype() != theano_types::DType::F32 {
            write!(f, ", dtype={}", self.dtype())?;
        }
        if !self.device().is_cpu() {
            write!(f, ", device={}", self.device())?;
        }
        write!(f, ")")
    }
}

fn format_data(data: &[f64], shape: &[usize], dim: usize, offset: usize) -> String {
    if shape.is_empty() {
        // Scalar
        return format_number(data[offset]);
    }
    if dim == shape.len() - 1 {
        // Last dimension: format as a row
        let mut s = String::from("[");
        for i in 0..shape[dim] {
            if i > 0 {
                s.push_str(", ");
            }
            if i >= 6 && shape[dim] > 8 {
                s.push_str("..., ");
                let last = shape[dim] - 1;
                s.push_str(&format_number(data[offset + last]));
                break;
            }
            s.push_str(&format_number(data[offset + i]));
        }
        s.push(']');
        return s;
    }

    let inner_size: usize = shape[dim + 1..].iter().product();
    let mut s = String::from("[");
    for i in 0..shape[dim] {
        if i > 0 {
            s.push_str(",\n");
            for _ in 0..=dim {
                s.push(' ');
            }
        }
        s.push_str(&format_data(data, shape, dim + 1, offset + i * inner_size));
    }
    s.push(']');
    s
}

fn format_number(x: f64) -> String {
    if x == x.floor() && x.abs() < 1e15 {
        format!("{x:.1}")
    } else {
        format!("{x:.4}")
    }
}
