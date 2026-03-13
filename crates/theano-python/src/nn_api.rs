//! Python API surface for nn module.
//!
//! This module enumerates the layers and optimizers that the Python `torch.nn`
//! and `torch.optim` namespaces will expose. The actual implementations live in
//! `theano-nn` and `theano-optim`; this crate provides the thin Python facade.

/// Python-facing nn module.
pub struct NNAPI;

impl NNAPI {
    /// List of supported layer types.
    pub fn supported_layers() -> Vec<&'static str> {
        vec![
            "Linear", "Conv1d", "Conv2d",
            "BatchNorm1d", "LayerNorm", "GroupNorm",
            "ReLU", "Sigmoid", "Tanh", "GELU", "SiLU",
            "Softmax", "LogSoftmax",
            "Dropout",
            "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
            "Embedding",
            "RNNCell", "LSTMCell",
            "MultiheadAttention",
            "Sequential",
            "Flatten",
            "MSELoss", "CrossEntropyLoss", "L1Loss", "BCELoss",
        ]
    }

    /// List of supported optimizers.
    pub fn supported_optimizers() -> Vec<&'static str> {
        vec!["SGD", "Adam", "AdamW", "RMSprop", "Adagrad"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_layers_not_empty() {
        let layers = NNAPI::supported_layers();
        assert!(!layers.is_empty());
        assert!(layers.contains(&"Linear"));
        assert!(layers.contains(&"Conv2d"));
        assert!(layers.contains(&"MultiheadAttention"));
        assert!(layers.contains(&"Sequential"));
    }

    #[test]
    fn test_supported_layers_contains_all_expected() {
        let layers = NNAPI::supported_layers();
        let expected = vec![
            "Linear", "Conv1d", "Conv2d",
            "BatchNorm1d", "LayerNorm", "GroupNorm",
            "ReLU", "Sigmoid", "Tanh", "GELU", "SiLU",
            "Softmax", "LogSoftmax",
            "Dropout",
            "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
            "Embedding",
            "RNNCell", "LSTMCell",
            "MultiheadAttention",
            "Sequential",
            "Flatten",
            "MSELoss", "CrossEntropyLoss", "L1Loss", "BCELoss",
        ];
        for name in expected {
            assert!(layers.contains(&name), "missing layer: {name}");
        }
    }

    #[test]
    fn test_supported_optimizers() {
        let opts = NNAPI::supported_optimizers();
        assert_eq!(opts.len(), 5);
        assert!(opts.contains(&"SGD"));
        assert!(opts.contains(&"Adam"));
        assert!(opts.contains(&"AdamW"));
        assert!(opts.contains(&"RMSprop"));
        assert!(opts.contains(&"Adagrad"));
    }
}
