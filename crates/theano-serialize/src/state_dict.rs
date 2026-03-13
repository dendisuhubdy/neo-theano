//! State dictionary serialization.
//! Like PyTorch's model.state_dict() / model.load_state_dict().

use std::collections::HashMap;
use theano_core::Tensor;
use theano_types::{Result, TheanoError};
use crate::safetensors;

/// A state dictionary — named tensor collection for model parameters.
pub type StateDict = HashMap<String, Tensor>;

/// Save a state dict to bytes (SafeTensors format).
pub fn save_state_dict(state_dict: &StateDict) -> Vec<u8> {
    safetensors::save_safetensors(state_dict)
}

/// Load a state dict from bytes.
pub fn load_state_dict(bytes: &[u8]) -> Result<StateDict> {
    safetensors::load_safetensors(bytes)
}

/// Validate that a loaded state dict matches expected parameter names.
pub fn validate_state_dict(
    loaded: &StateDict,
    expected_keys: &[&str],
    strict: bool,
) -> Result<()> {
    if strict {
        for key in expected_keys {
            if !loaded.contains_key(*key) {
                return Err(TheanoError::runtime(format!(
                    "missing key in state_dict: '{}'", key
                )));
            }
        }
        for key in loaded.keys() {
            if !expected_keys.contains(&key.as_str()) {
                return Err(TheanoError::runtime(format!(
                    "unexpected key in state_dict: '{}'", key
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_dict_roundtrip() {
        let mut sd = StateDict::new();
        sd.insert("layer.weight".to_string(), Tensor::ones(&[4, 3]));
        sd.insert("layer.bias".to_string(), Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], &[4]));

        let bytes = save_state_dict(&sd);
        let loaded = load_state_dict(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["layer.weight"].shape(), &[4, 3]);
        assert_eq!(loaded["layer.bias"].shape(), &[4]);
    }

    #[test]
    fn test_validate_state_dict_strict() {
        let mut sd = StateDict::new();
        sd.insert("w".to_string(), Tensor::ones(&[2]));
        sd.insert("b".to_string(), Tensor::ones(&[2]));

        // Matching keys should succeed
        assert!(validate_state_dict(&sd, &["w", "b"], true).is_ok());

        // Missing key should fail
        assert!(validate_state_dict(&sd, &["w", "b", "c"], true).is_err());

        // Extra key should fail
        assert!(validate_state_dict(&sd, &["w"], true).is_err());
    }

    #[test]
    fn test_validate_state_dict_non_strict() {
        let mut sd = StateDict::new();
        sd.insert("w".to_string(), Tensor::ones(&[2]));
        sd.insert("extra".to_string(), Tensor::ones(&[2]));

        // Non-strict should pass even with extra/missing keys
        assert!(validate_state_dict(&sd, &["w"], false).is_ok());
        assert!(validate_state_dict(&sd, &["w", "b", "c"], false).is_ok());
    }
}
