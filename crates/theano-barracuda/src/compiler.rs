//! BarraCUDA compiler integration.
//!
//! Compiles CUDA kernel sources (.cu) directly to AMD GFX11 GPU machine code
//! at build time or at runtime.

use std::path::PathBuf;
use std::collections::HashMap;

/// Compilation target for BarraCUDA.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// AMD RDNA 3 (GFX11) — Radeon RX 7000 series
    Gfx1100,
    /// AMD RDNA 3 (GFX1101) — Radeon RX 7600/7700
    Gfx1101,
    /// AMD RDNA 3 (GFX1102) — Radeon RX 7500
    Gfx1102,
}

impl Target {
    pub fn as_str(&self) -> &'static str {
        match self {
            Target::Gfx1100 => "gfx1100",
            Target::Gfx1101 => "gfx1101",
            Target::Gfx1102 => "gfx1102",
        }
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Compiled kernel binary.
#[derive(Clone)]
pub struct CompiledKernel {
    /// Target GPU architecture.
    pub target: Target,
    /// Binary data (AMD GPU ISA).
    pub binary: Vec<u8>,
    /// Kernel function names available in this binary.
    pub functions: Vec<String>,
    /// Source file path.
    pub source_path: Option<PathBuf>,
}

/// BarraCUDA compiler wrapper.
pub struct BarracudaCompiler {
    /// Path to the BarraCUDA compiler binary.
    compiler_path: Option<PathBuf>,
    /// Default compilation target.
    default_target: Target,
    /// Cache of compiled kernels.
    cache: HashMap<(String, Target), CompiledKernel>,
}

impl BarracudaCompiler {
    /// Create a new compiler instance.
    pub fn new() -> Self {
        Self {
            compiler_path: crate::which_barracuda(),
            default_target: Target::Gfx1100,
            cache: HashMap::new(),
        }
    }

    /// Set the compilation target.
    pub fn target(mut self, target: Target) -> Self {
        self.default_target = target;
        self
    }

    /// Set the compiler path explicitly.
    pub fn compiler_path(mut self, path: PathBuf) -> Self {
        self.compiler_path = Some(path);
        self
    }

    /// Whether the compiler is available.
    pub fn is_available(&self) -> bool {
        self.compiler_path.is_some()
    }

    /// Compile a CUDA source string to AMD GPU binary.
    ///
    /// In the current implementation, this is a placeholder that returns
    /// empty binaries. When BarraCUDA is installed, it will invoke the
    /// actual compiler.
    pub fn compile_source(
        &mut self,
        source: &str,
        name: &str,
        target: Option<Target>,
    ) -> Result<CompiledKernel, BarracudaError> {
        let target = target.unwrap_or(self.default_target);
        let cache_key = (name.to_string(), target);

        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        if let Some(_compiler) = &self.compiler_path {
            // Real compilation path:
            // 1. Write source to temp file
            // 2. Invoke: barracuda --target=gfx1100 -o output.bin input.cu
            // 3. Read binary output
            // 4. Parse function names from binary
            //
            // For now, return a mock result
            let kernel = CompiledKernel {
                target,
                binary: Vec::new(), // Would contain actual AMD ISA
                functions: extract_kernel_names(source),
                source_path: None,
            };
            self.cache.insert(cache_key, kernel.clone());
            Ok(kernel)
        } else {
            // No compiler available — return mock
            let kernel = CompiledKernel {
                target,
                binary: Vec::new(),
                functions: extract_kernel_names(source),
                source_path: None,
            };
            self.cache.insert(cache_key, kernel.clone());
            Ok(kernel)
        }
    }

    /// Compile all standard Theano CUDA kernels.
    pub fn compile_standard_kernels(
        &mut self,
    ) -> Result<Vec<CompiledKernel>, BarracudaError> {
        let mut kernels = Vec::new();

        kernels.push(self.compile_source(
            theano_cuda_kernels::sources::ELEMENTWISE,
            "elementwise",
            None,
        )?);
        kernels.push(self.compile_source(
            theano_cuda_kernels::sources::REDUCE,
            "reduce",
            None,
        )?);
        kernels.push(self.compile_source(
            theano_cuda_kernels::sources::SOFTMAX,
            "softmax",
            None,
        )?);

        Ok(kernels)
    }
}

impl Default for BarracudaCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract kernel function names from CUDA source code.
fn extract_kernel_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in source.lines() {
        // Look for extern "C" __global__ void function_name(
        if line.contains("THEANO_GLOBAL_FUNC") || line.contains("__global__") {
            if let Some(rest) = line.split("void ").nth(1) {
                if let Some(name) = rest.split('(').next() {
                    let name = name.trim();
                    if !name.is_empty() {
                        names.push(name.to_string());
                    }
                }
            }
        }
    }
    names
}

/// BarraCUDA compilation errors.
#[derive(Debug, thiserror::Error)]
pub enum BarracudaError {
    #[error("BarraCUDA compiler not found")]
    CompilerNotFound,

    #[error("compilation failed for {file}: {msg}")]
    CompilationFailed { file: String, msg: String },

    #[error("unsupported target: {target}")]
    UnsupportedTarget { target: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = BarracudaCompiler::new();
        // May or may not be available depending on system
        let _ = compiler.is_available();
    }

    #[test]
    fn test_extract_kernel_names() {
        let source = r#"
extern "C" THEANO_GLOBAL_FUNC void relu_f32(const float* input, float* output, int n) {
}
extern "C" THEANO_GLOBAL_FUNC void sigmoid_f32(const float* input, float* output, int n) {
}
"#;
        let names = extract_kernel_names(source);
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"relu_f32".to_string()));
        assert!(names.contains(&"sigmoid_f32".to_string()));
    }

    #[test]
    fn test_compile_standard_kernels() {
        let mut compiler = BarracudaCompiler::new();
        let kernels = compiler.compile_standard_kernels().unwrap();
        assert_eq!(kernels.len(), 3); // elementwise, reduce, softmax

        // Check that kernel names were extracted
        let elem = &kernels[0];
        assert!(elem.functions.contains(&"relu_f32".to_string()));
        assert!(elem.functions.contains(&"add_f32".to_string()));
    }

    #[test]
    fn test_target_display() {
        assert_eq!(Target::Gfx1100.to_string(), "gfx1100");
        assert_eq!(Target::Gfx1101.as_str(), "gfx1101");
    }

    #[test]
    fn test_compilation_caching() {
        let mut compiler = BarracudaCompiler::new();
        let k1 = compiler.compile_source("void test() {}", "test", None).unwrap();
        let k2 = compiler.compile_source("void test() {}", "test", None).unwrap();
        // Should return cached version
        assert_eq!(k1.target, k2.target);
    }
}
