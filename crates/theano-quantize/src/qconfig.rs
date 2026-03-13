//! Quantization configuration.

/// Quantized data types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantDType {
    Int8,
    UInt8,
    Int4,
    FP8,
}

impl QuantDType {
    pub fn bits(&self) -> usize {
        match self {
            QuantDType::Int8 | QuantDType::UInt8 | QuantDType::FP8 => 8,
            QuantDType::Int4 => 4,
        }
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, QuantDType::Int8 | QuantDType::Int4)
    }

    pub fn qmin(&self) -> f64 {
        match self {
            QuantDType::Int8 => -128.0,
            QuantDType::UInt8 => 0.0,
            QuantDType::Int4 => -8.0,
            QuantDType::FP8 => -448.0,
        }
    }

    pub fn qmax(&self) -> f64 {
        match self {
            QuantDType::Int8 => 127.0,
            QuantDType::UInt8 => 255.0,
            QuantDType::Int4 => 7.0,
            QuantDType::FP8 => 448.0,
        }
    }
}

/// Quantization configuration for a layer.
#[derive(Clone, Debug)]
pub struct QConfig {
    pub weight_dtype: QuantDType,
    pub activation_dtype: QuantDType,
    pub symmetric: bool,
    pub per_channel: bool,
}

impl QConfig {
    pub fn default_int8() -> Self {
        Self {
            weight_dtype: QuantDType::Int8,
            activation_dtype: QuantDType::UInt8,
            symmetric: true,
            per_channel: false,
        }
    }

    pub fn default_int4() -> Self {
        Self {
            weight_dtype: QuantDType::Int4,
            activation_dtype: QuantDType::Int8,
            symmetric: true,
            per_channel: true,
        }
    }
}

impl Default for QConfig {
    fn default() -> Self {
        Self::default_int8()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_dtype_bits() {
        assert_eq!(QuantDType::Int8.bits(), 8);
        assert_eq!(QuantDType::UInt8.bits(), 8);
        assert_eq!(QuantDType::Int4.bits(), 4);
        assert_eq!(QuantDType::FP8.bits(), 8);
    }

    #[test]
    fn test_quant_dtype_signed() {
        assert!(QuantDType::Int8.is_signed());
        assert!(!QuantDType::UInt8.is_signed());
        assert!(QuantDType::Int4.is_signed());
        assert!(!QuantDType::FP8.is_signed());
    }

    #[test]
    fn test_quant_dtype_range() {
        assert_eq!(QuantDType::Int8.qmin(), -128.0);
        assert_eq!(QuantDType::Int8.qmax(), 127.0);
        assert_eq!(QuantDType::UInt8.qmin(), 0.0);
        assert_eq!(QuantDType::UInt8.qmax(), 255.0);
        assert_eq!(QuantDType::Int4.qmin(), -8.0);
        assert_eq!(QuantDType::Int4.qmax(), 7.0);
        assert_eq!(QuantDType::FP8.qmin(), -448.0);
        assert_eq!(QuantDType::FP8.qmax(), 448.0);
    }

    #[test]
    fn test_qconfig_default_int8() {
        let cfg = QConfig::default_int8();
        assert_eq!(cfg.weight_dtype, QuantDType::Int8);
        assert_eq!(cfg.activation_dtype, QuantDType::UInt8);
        assert!(cfg.symmetric);
        assert!(!cfg.per_channel);
    }

    #[test]
    fn test_qconfig_default_int4() {
        let cfg = QConfig::default_int4();
        assert_eq!(cfg.weight_dtype, QuantDType::Int4);
        assert_eq!(cfg.activation_dtype, QuantDType::Int8);
        assert!(cfg.symmetric);
        assert!(cfg.per_channel);
    }

    #[test]
    fn test_qconfig_default_trait() {
        let cfg = QConfig::default();
        assert_eq!(cfg.weight_dtype, QuantDType::Int8);
    }

    #[test]
    fn test_quant_dtype_equality() {
        assert_eq!(QuantDType::Int8, QuantDType::Int8);
        assert_ne!(QuantDType::Int8, QuantDType::UInt8);
    }

    #[test]
    fn test_qconfig_clone() {
        let cfg = QConfig::default_int8();
        let cloned = cfg.clone();
        assert_eq!(cloned.weight_dtype, cfg.weight_dtype);
        assert_eq!(cloned.activation_dtype, cfg.activation_dtype);
        assert_eq!(cloned.symmetric, cfg.symmetric);
        assert_eq!(cloned.per_channel, cfg.per_channel);
    }
}
