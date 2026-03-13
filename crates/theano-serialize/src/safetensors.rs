//! SafeTensors format support.
//! SafeTensors is a simple, safe format for storing tensors.
//! Format: 8-byte header length (LE) + JSON header + raw tensor data.

use std::collections::HashMap;
use theano_core::Tensor;
use theano_types::{DType, Result, TheanoError};

/// Metadata for a single tensor in SafeTensors format.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize), // (start, end) in data section
}

/// A SafeTensors file (in-memory representation).
#[derive(Clone, Debug)]
pub struct SafeTensorsFile {
    pub tensors: HashMap<String, TensorInfo>,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

impl SafeTensorsFile {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            data: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a tensor to the file.
    pub fn add_tensor(&mut self, name: &str, tensor: &Tensor) {
        let values = tensor.to_vec_f64().unwrap();

        let start = self.data.len();

        // Write f64 values as little-endian bytes
        for v in &values {
            self.data.extend_from_slice(&v.to_le_bytes());
        }

        let end = self.data.len();

        self.tensors.insert(name.to_string(), TensorInfo {
            name: name.to_string(),
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            data_offsets: (start, end),
        });
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self.tensors.get(name)
            .ok_or_else(|| TheanoError::runtime(format!("tensor '{}' not found", name)))?;

        let bytes = &self.data[info.data_offsets.0..info.data_offsets.1];
        let num_elements = bytes.len() / 8;
        let values: Vec<f64> = (0..num_elements)
            .map(|i| {
                let slice = &bytes[i*8..(i+1)*8];
                f64::from_le_bytes(slice.try_into().unwrap())
            })
            .collect();

        Ok(Tensor::from_slice(&values, &info.shape))
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Build JSON header
        let header = self.build_header();
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(TheanoError::runtime("invalid safetensors: too short"));
        }

        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err(TheanoError::runtime("invalid safetensors: truncated header"));
        }

        let header_str = std::str::from_utf8(&bytes[8..8+header_len])
            .map_err(|e| TheanoError::runtime(format!("invalid header UTF-8: {}", e)))?;

        let data = bytes[8+header_len..].to_vec();

        // Parse simple JSON header (minimal parser)
        let tensors = parse_header(header_str)?;

        Ok(Self {
            tensors,
            data,
            metadata: HashMap::new(),
        })
    }

    fn build_header(&self) -> String {
        // Build a simple JSON header
        let mut parts = Vec::new();
        // Sort keys for deterministic output
        let mut keys: Vec<&String> = self.tensors.keys().collect();
        keys.sort();
        for name in keys {
            let info = &self.tensors[name];
            let dtype_str = format!("{}", info.dtype);
            let shape_str: Vec<String> = info.shape.iter().map(|s| s.to_string()).collect();
            parts.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype_str}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                shape_str.join(","),
                info.data_offsets.0,
                info.data_offsets.1
            ));
        }
        format!("{{{}}}", parts.join(","))
    }
}

impl Default for SafeTensorsFile {
    fn default() -> Self { Self::new() }
}

/// Save tensors to SafeTensors format bytes.
pub fn save_safetensors(tensors: &HashMap<String, Tensor>) -> Vec<u8> {
    let mut file = SafeTensorsFile::new();
    for (name, tensor) in tensors {
        file.add_tensor(name, tensor);
    }
    file.to_bytes()
}

/// Load tensors from SafeTensors format bytes.
pub fn load_safetensors(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let file = SafeTensorsFile::from_bytes(bytes)?;
    let mut result = HashMap::new();
    for name in file.tensor_names() {
        result.insert(name.to_string(), file.get_tensor(name)?);
    }
    Ok(result)
}

/// Minimal JSON header parser for SafeTensors.
fn parse_header(header: &str) -> Result<HashMap<String, TensorInfo>> {
    let mut tensors = HashMap::new();

    // Strip outer braces
    let inner = header.trim().trim_start_matches('{').trim_end_matches('}');
    if inner.is_empty() {
        return Ok(tensors);
    }

    // Split by top-level commas (very simplified)
    // For a proper implementation, use serde_json
    // This handles the format we generate in build_header
    let mut depth = 0;
    let mut current = String::new();
    let mut entries = Vec::new();

    for ch in inner.chars() {
        match ch {
            '{' => { depth += 1; current.push(ch); }
            '}' => { depth -= 1; current.push(ch); }
            ',' if depth == 0 => {
                entries.push(current.clone());
                current.clear();
            }
            _ => { current.push(ch); }
        }
    }
    if !current.is_empty() {
        entries.push(current);
    }

    for entry in entries {
        // entry format: "name":{"dtype":"float32","shape":[2,3],"data_offsets":[0,48]}
        let parts: Vec<&str> = entry.splitn(2, ':').collect();
        if parts.len() != 2 { continue; }

        let name = parts[0].trim().trim_matches('"').to_string();
        let value = parts[1].trim();

        // Extract dtype
        let dtype = if value.contains("float32") { DType::F32 }
                   else if value.contains("float64") { DType::F64 }
                   else if value.contains("float16") { DType::F16 }
                   else if value.contains("int64") { DType::I64 }
                   else if value.contains("int32") { DType::I32 }
                   else { DType::F32 };

        // Extract shape - find [...]
        let shape = extract_array(value, "shape");
        let offsets = extract_array(value, "data_offsets");

        let data_offsets = if offsets.len() == 2 {
            (offsets[0], offsets[1])
        } else {
            (0, 0)
        };

        tensors.insert(name.clone(), TensorInfo {
            name,
            dtype,
            shape,
            data_offsets,
        });
    }

    Ok(tensors)
}

fn extract_array(s: &str, key: &str) -> Vec<usize> {
    if let Some(pos) = s.find(key) {
        let rest = &s[pos..];
        if let Some(start) = rest.find('[') {
            if let Some(end) = rest[start..].find(']') {
                let arr = &rest[start+1..start+end];
                return arr.split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect();
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_roundtrip() {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let t2 = Tensor::from_slice(&[5.0, 6.0], &[2]);

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), t1);
        tensors.insert("bias".to_string(), t2);

        let bytes = save_safetensors(&tensors);
        let loaded = load_safetensors(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["weight"].shape(), &[2, 2]);
        assert_eq!(loaded["weight"].to_vec_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded["bias"].to_vec_f64().unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_safetensors_file_api() {
        let mut file = SafeTensorsFile::new();
        file.add_tensor("x", &Tensor::ones(&[3, 3]));

        let names = file.tensor_names();
        assert_eq!(names.len(), 1);

        let t = file.get_tensor("x").unwrap();
        assert_eq!(t.shape(), &[3, 3]);
    }

    #[test]
    fn test_safetensors_empty() {
        let tensors = HashMap::new();
        let bytes = save_safetensors(&tensors);
        let loaded = load_safetensors(&bytes).unwrap();
        assert_eq!(loaded.len(), 0);
    }
}
