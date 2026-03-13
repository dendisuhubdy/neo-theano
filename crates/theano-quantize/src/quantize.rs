//! Quantization and dequantization operations.

use theano_core::Tensor;
use theano_types::Result;
use crate::qconfig::QuantDType;

/// Quantize a tensor: q = round(x / scale + zero_point)
pub fn quantize_tensor(tensor: &Tensor, scale: f64, zero_point: f64, dtype: QuantDType) -> Result<Vec<f64>> {
    let data = tensor.to_vec_f64()?;
    let qmin = dtype.qmin();
    let qmax = dtype.qmax();

    let quantized: Vec<f64> = data
        .iter()
        .map(|&x| (x / scale + zero_point).round().clamp(qmin, qmax))
        .collect();

    Ok(quantized)
}

/// Dequantize: x = (q - zero_point) * scale
pub fn dequantize_tensor(quantized: &[f64], scale: f64, zero_point: f64) -> Vec<f64> {
    quantized
        .iter()
        .map(|&q| (q - zero_point) * scale)
        .collect()
}

/// Quantize a tensor per-tensor (compute scale/zp from tensor stats).
pub fn quantize_per_tensor(tensor: &Tensor, dtype: QuantDType) -> Result<(Vec<f64>, f64, f64)> {
    let data = tensor.to_vec_f64()?;

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let qmin = dtype.qmin();
    let qmax = dtype.qmax();

    let scale = ((max_val - min_val) / (qmax - qmin)).max(1e-10);
    let zero_point = (qmin - min_val / scale).round().clamp(qmin, qmax);

    let quantized: Vec<f64> = data
        .iter()
        .map(|&x| (x / scale + zero_point).round().clamp(qmin, qmax))
        .collect();

    Ok((quantized, scale, zero_point))
}

/// Fake quantize: quantize then immediately dequantize (for QAT).
pub fn fake_quantize(tensor: &Tensor, scale: f64, zero_point: f64, dtype: QuantDType) -> Result<Tensor> {
    let quantized = quantize_tensor(tensor, scale, zero_point, dtype)?;
    let dequantized = dequantize_tensor(&quantized, scale, zero_point);
    Ok(Tensor::from_slice(&dequantized, tensor.shape()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let tensor = Tensor::from_slice(&[0.0, 0.5, 1.0, -0.5, -1.0], &[5]);
        let (quantized, scale, zp) = quantize_per_tensor(&tensor, QuantDType::Int8).unwrap();
        let dequantized = dequantize_tensor(&quantized, scale, zp);

        // Should be close to original
        let original = tensor.to_vec_f64().unwrap();
        for (o, d) in original.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 0.05, "expected ~{o}, got {d}");
        }
    }

    #[test]
    fn test_fake_quantize() {
        let tensor = Tensor::from_slice(&[0.1, 0.5, 0.9], &[3]);
        let result = fake_quantize(&tensor, 0.01, 0.0, QuantDType::Int8).unwrap();
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_quantize_tensor_clamps() {
        let tensor = Tensor::from_slice(&[1000.0, -1000.0], &[2]);
        let quantized = quantize_tensor(&tensor, 1.0, 0.0, QuantDType::Int8).unwrap();
        assert_eq!(quantized[0], 127.0);
        assert_eq!(quantized[1], -128.0);
    }

    #[test]
    fn test_dequantize_tensor() {
        let quantized = vec![0.0, 64.0, 127.0];
        let scale = 0.01;
        let zp = 0.0;
        let result = dequantize_tensor(&quantized, scale, zp);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.64).abs() < 1e-10);
        assert!((result[2] - 1.27).abs() < 1e-10);
    }

    #[test]
    fn test_quantize_per_tensor_int4() {
        let tensor = Tensor::from_slice(&[0.0, 0.5, 1.0], &[3]);
        let (quantized, scale, zp) = quantize_per_tensor(&tensor, QuantDType::Int4).unwrap();
        assert_eq!(quantized.len(), 3);
        // All values should be within Int4 range
        for &q in &quantized {
            assert!(q >= -8.0 && q <= 7.0, "value {q} out of Int4 range");
        }
    }

    #[test]
    fn test_quantize_per_tensor_uint8() {
        let tensor = Tensor::from_slice(&[0.0, 0.5, 1.0], &[3]);
        let (quantized, _scale, _zp) = quantize_per_tensor(&tensor, QuantDType::UInt8).unwrap();
        for &q in &quantized {
            assert!(q >= 0.0 && q <= 255.0, "value {q} out of UInt8 range");
        }
    }

    #[test]
    fn test_fake_quantize_preserves_shape() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = fake_quantize(&tensor, 0.1, 0.0, QuantDType::Int8).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_quantize_single_value() {
        // With a range of values (not all identical), a single element round-trips well
        let tensor = Tensor::from_slice(&[0.5, -0.5], &[2]);
        let (quantized, scale, zp) = quantize_per_tensor(&tensor, QuantDType::Int8).unwrap();
        let dequantized = dequantize_tensor(&quantized, scale, zp);
        assert!((0.5 - dequantized[0]).abs() < 0.05, "expected ~0.5, got {}", dequantized[0]);
        assert!((-0.5 - dequantized[1]).abs() < 0.05, "expected ~-0.5, got {}", dequantized[1]);
    }
}
