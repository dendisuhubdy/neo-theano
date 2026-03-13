//! Serialization support for Theano.
//! Primary format: SafeTensors. Also supports state_dict serialization,
//! ONNX export stubs, and PyTorch .pt format loading stubs.

pub mod safetensors;
pub mod state_dict;
pub mod onnx;

pub use safetensors::{save_safetensors, load_safetensors};
pub use state_dict::{StateDict, save_state_dict, load_state_dict};
