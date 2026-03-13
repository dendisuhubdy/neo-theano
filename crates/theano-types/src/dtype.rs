use std::fmt;

/// Represents all supported data types for tensors, mirroring PyTorch's dtype system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    // Floating point
    F16,
    BF16,
    F32,
    F64,
    // Unsigned integer
    U8,
    // Signed integer
    I8,
    I16,
    I32,
    I64,
    // Boolean
    Bool,
    // Complex
    Complex32,
    Complex64,
    // Quantized
    QInt8,
    QUInt8,
    QInt32,
}

impl DType {
    /// Size of the data type in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::Bool | DType::U8 | DType::I8 | DType::QInt8 | DType::QUInt8 => 1,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::F32 | DType::I32 | DType::QInt32 | DType::Complex32 => 4,
            DType::F64 | DType::I64 | DType::Complex64 => 8,
        }
    }

    /// Whether this dtype is a floating point type.
    pub fn is_floating_point(self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Whether this dtype is a complex type.
    pub fn is_complex(self) -> bool {
        matches!(self, DType::Complex32 | DType::Complex64)
    }

    /// Whether this dtype is an integer type.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::U8 | DType::I8 | DType::I16 | DType::I32 | DType::I64
        )
    }

    /// Whether this dtype is a quantized type.
    pub fn is_quantized(self) -> bool {
        matches!(self, DType::QInt8 | DType::QUInt8 | DType::QInt32)
    }

    /// Whether this dtype is a signed type.
    pub fn is_signed(self) -> bool {
        !matches!(self, DType::U8 | DType::QUInt8 | DType::Bool)
    }

    /// The default dtype for floating point operations.
    pub fn default_float() -> Self {
        DType::F32
    }

    /// The default dtype for integer operations.
    pub fn default_int() -> Self {
        DType::I64
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::U8 => "uint8",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::Bool => "bool",
            DType::Complex32 => "complex32",
            DType::Complex64 => "complex64",
            DType::QInt8 => "qint8",
            DType::QUInt8 => "quint8",
            DType::QInt32 => "qint32",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_in_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F64.size_in_bytes(), 8);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
        assert_eq!(DType::I64.size_in_bytes(), 8);
        assert_eq!(DType::Bool.size_in_bytes(), 1);
    }

    #[test]
    fn test_type_categories() {
        assert!(DType::F32.is_floating_point());
        assert!(!DType::I32.is_floating_point());
        assert!(DType::Complex64.is_complex());
        assert!(DType::I32.is_integer());
        assert!(DType::QInt8.is_quantized());
    }
}
