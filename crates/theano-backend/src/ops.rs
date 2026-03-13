use theano_types::{DType, Device, Layout, Result};

/// Enumeration of unary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Log2,
    Log10,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Sigmoid,
    Relu,
    Gelu,
    Silu,
    Floor,
    Ceil,
    Round,
    Sign,
    Reciprocal,
    Square,
    Erf,
}

/// Enumeration of binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    And,
    Or,
    Xor,
}

/// Enumeration of reduction operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    ArgMax,
    ArgMin,
    Any,
    All,
}

/// Trait for backend storage — the raw data buffer on a device.
///
/// Each backend provides its own storage type (CpuStorage, CudaStorage, etc.)
/// that implements this trait.
pub trait BackendStorage: Sized + Send + Sync {
    /// The device type associated with this storage.
    fn device(&self) -> Device;

    /// Data type of the elements.
    fn dtype(&self) -> DType;

    /// Number of elements in the storage.
    fn len(&self) -> usize;

    /// Whether the storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Apply a unary operation elementwise.
    fn unary_op(&self, op: UnaryOp, shape: &[usize], strides: &[usize]) -> Result<Self>;

    /// Apply a binary operation elementwise with broadcasting.
    fn binary_op(
        &self,
        rhs: &Self,
        op: BinaryOp,
        lhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs_shape: &[usize],
        rhs_strides: &[usize],
        out_shape: &[usize],
    ) -> Result<Self>;

    /// Reduce along a set of dimensions.
    fn reduce_op(
        &self,
        op: ReduceOp,
        shape: &[usize],
        strides: &[usize],
        reduce_dims: &[usize],
        keep_dim: bool,
    ) -> Result<Self>;

    /// Matrix multiplication: [..., M, K] x [..., K, N] -> [..., M, N]
    fn matmul(
        &self,
        rhs: &Self,
        lhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs_shape: &[usize],
        rhs_strides: &[usize],
    ) -> Result<Self>;

    /// Cast to a different data type.
    fn to_dtype(&self, dtype: DType) -> Result<Self>;

    /// Make a contiguous copy with the given shape/strides.
    fn contiguous(&self, shape: &[usize], strides: &[usize]) -> Result<Self>;

    /// Create a storage filled with a scalar value.
    fn fill(value: f64, len: usize, dtype: DType, device: &Device) -> Result<Self>;

    /// Create a storage from an f64 slice (for initialization).
    fn from_f64_slice(data: &[f64], dtype: DType, device: &Device) -> Result<Self>;

    /// Copy data to a Vec<f64> on the host (for debugging/testing).
    fn to_f64_vec(&self, shape: &[usize], strides: &[usize]) -> Result<Vec<f64>>;

    /// Apply where/conditional: result[i] = if cond[i] then self[i] else other[i]
    fn where_cond(
        &self,
        cond: &Self,
        other: &Self,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self>;

    /// Gather along a dimension.
    fn index_select(&self, dim: usize, indices: &Self, shape: &[usize], strides: &[usize]) -> Result<Self>;
}

/// Top-level backend trait — marker for a complete backend implementation.
pub trait Backend: Sized + Clone + Send + Sync + 'static {
    type Storage: BackendStorage;

    /// The name of this backend.
    fn name() -> &'static str;

    /// The device type this backend targets.
    fn device_type() -> theano_types::DeviceType;
}
