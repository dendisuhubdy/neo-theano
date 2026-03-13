//! Python API surface for dtype.
//!
//! Maps PyTorch dtype names (`torch.float32`, `torch.int64`, etc.) to the
//! internal `DType` enum. When PyO3 is enabled these will be registered as
//! module-level constants so that `theano.float32` works just like
//! `torch.float32`.

use theano_types::DType;

/// Python-facing dtype helper.
pub struct DTypeAPI;

impl DTypeAPI {
    pub fn float16() -> DType { DType::F16 }
    pub fn bfloat16() -> DType { DType::BF16 }
    pub fn float32() -> DType { DType::F32 }
    pub fn float64() -> DType { DType::F64 }
    pub fn int8() -> DType { DType::I8 }
    pub fn int16() -> DType { DType::I16 }
    pub fn int32() -> DType { DType::I32 }
    pub fn int64() -> DType { DType::I64 }
    pub fn bool() -> DType { DType::Bool }

    /// Parse a PyTorch-style dtype string into a `DType`.
    ///
    /// Accepts both canonical names (`float32`) and common aliases (`float`,
    /// `half`, `long`, etc.).
    pub fn from_string(s: &str) -> Option<DType> {
        match s {
            "float16" | "half" => Some(DType::F16),
            "bfloat16" => Some(DType::BF16),
            "float32" | "float" => Some(DType::F32),
            "float64" | "double" => Some(DType::F64),
            "int8" => Some(DType::I8),
            "int16" | "short" => Some(DType::I16),
            "int32" | "int" => Some(DType::I32),
            "int64" | "long" => Some(DType::I64),
            "bool" => Some(DType::Bool),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theano_types::DType;

    #[test]
    fn test_direct_accessors() {
        assert_eq!(DTypeAPI::float16(), DType::F16);
        assert_eq!(DTypeAPI::bfloat16(), DType::BF16);
        assert_eq!(DTypeAPI::float32(), DType::F32);
        assert_eq!(DTypeAPI::float64(), DType::F64);
        assert_eq!(DTypeAPI::int8(), DType::I8);
        assert_eq!(DTypeAPI::int16(), DType::I16);
        assert_eq!(DTypeAPI::int32(), DType::I32);
        assert_eq!(DTypeAPI::int64(), DType::I64);
        assert_eq!(DTypeAPI::bool(), DType::Bool);
    }

    #[test]
    fn test_from_string_canonical() {
        assert_eq!(DTypeAPI::from_string("float16"), Some(DType::F16));
        assert_eq!(DTypeAPI::from_string("bfloat16"), Some(DType::BF16));
        assert_eq!(DTypeAPI::from_string("float32"), Some(DType::F32));
        assert_eq!(DTypeAPI::from_string("float64"), Some(DType::F64));
        assert_eq!(DTypeAPI::from_string("int8"), Some(DType::I8));
        assert_eq!(DTypeAPI::from_string("int16"), Some(DType::I16));
        assert_eq!(DTypeAPI::from_string("int32"), Some(DType::I32));
        assert_eq!(DTypeAPI::from_string("int64"), Some(DType::I64));
        assert_eq!(DTypeAPI::from_string("bool"), Some(DType::Bool));
    }

    #[test]
    fn test_from_string_aliases() {
        assert_eq!(DTypeAPI::from_string("half"), Some(DType::F16));
        assert_eq!(DTypeAPI::from_string("float"), Some(DType::F32));
        assert_eq!(DTypeAPI::from_string("double"), Some(DType::F64));
        assert_eq!(DTypeAPI::from_string("short"), Some(DType::I16));
        assert_eq!(DTypeAPI::from_string("int"), Some(DType::I32));
        assert_eq!(DTypeAPI::from_string("long"), Some(DType::I64));
    }

    #[test]
    fn test_from_string_unknown() {
        assert_eq!(DTypeAPI::from_string("complex128"), None);
        assert_eq!(DTypeAPI::from_string(""), None);
        assert_eq!(DTypeAPI::from_string("Float32"), None);
    }
}
