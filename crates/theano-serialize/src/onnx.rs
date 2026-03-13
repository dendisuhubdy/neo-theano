//! ONNX export stubs.
//! Full implementation will use onnx-protobuf crate.

use theano_types::TheanoError;

/// ONNX opset version.
pub const DEFAULT_OPSET_VERSION: i64 = 17;

/// ONNX model representation (stub).
pub struct OnnxModel {
    pub opset_version: i64,
    pub producer_name: String,
    pub graph_name: String,
}

impl OnnxModel {
    pub fn new(graph_name: &str) -> Self {
        Self {
            opset_version: DEFAULT_OPSET_VERSION,
            producer_name: "theano".to_string(),
            graph_name: graph_name.to_string(),
        }
    }
}

/// Export a model to ONNX format (stub -- returns not-implemented error).
pub fn export_onnx(_model: &OnnxModel) -> Result<Vec<u8>, TheanoError> {
    Err(TheanoError::not_implemented("ONNX export requires onnx-protobuf crate"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_model_creation() {
        let model = OnnxModel::new("my_model");
        assert_eq!(model.opset_version, DEFAULT_OPSET_VERSION);
        assert_eq!(model.producer_name, "theano");
        assert_eq!(model.graph_name, "my_model");
    }

    #[test]
    fn test_export_onnx_not_implemented() {
        let model = OnnxModel::new("test");
        let result = export_onnx(&model);
        assert!(result.is_err());
    }
}
