use std::sync::Arc;

use theano_core::Tensor;
use theano_core::tensor::GradFn;
use theano_types::Result;

use crate::grad_fns::*;
use crate::no_grad::is_grad_enabled;

/// A variable that wraps a Tensor and optionally tracks gradient computation.
///
/// This is the primary autograd interface. When `requires_grad` is true,
/// operations on Variables build a computational graph that enables
/// `backward()` to compute gradients.
///
/// Like `torch.Tensor` with `requires_grad=True`.
#[derive(Clone)]
pub struct Variable {
    tensor: Tensor,
    /// Inputs to the operation that produced this variable.
    /// Stored here because GradFn needs to know which variables
    /// to propagate gradients to.
    pub(crate) inputs: Vec<Variable>,
}

impl Variable {
    /// Create a leaf variable (no grad_fn).
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            inputs: vec![],
        }
    }

    /// Create a leaf variable that requires gradient computation.
    pub fn requires_grad(tensor: Tensor) -> Self {
        Self {
            tensor: tensor.requires_grad_(true),
            inputs: vec![],
        }
    }

    /// Get the underlying tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Consume and return the underlying tensor.
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    /// Whether this variable requires gradient.
    pub fn requires_grad_flag(&self) -> bool {
        self.tensor.requires_grad()
    }

    /// Get the gradient, if accumulated.
    pub fn grad(&self) -> Option<Tensor> {
        self.tensor.grad()
    }

    /// Get the grad_fn.
    pub fn grad_fn(&self) -> Option<Arc<dyn GradFn>> {
        self.tensor.grad_fn()
    }

    /// Detach from the graph.
    pub fn detach(&self) -> Variable {
        Variable::new(self.tensor.detach())
    }

    /// Create a new variable as the output of an operation.
    fn from_op(
        tensor: Tensor,
        grad_fn: Arc<dyn GradFn>,
        inputs: Vec<Variable>,
    ) -> Self {
        let needs_grad = inputs.iter().any(|v| v.requires_grad_flag());
        if needs_grad && is_grad_enabled() {
            let tensor = Tensor::from_parts_with_grad(
                tensor.storage().clone(),
                tensor.shape().to_vec(),
                tensor.strides().to_vec(),
                tensor.storage_offset(),
                tensor.dtype(),
                tensor.device().clone(),
                true,
                Some(grad_fn),
            );
            Self { tensor, inputs }
        } else {
            Self {
                tensor,
                inputs: vec![],
            }
        }
    }

    // ---- Elementwise operations ----

    pub fn add(&self, other: &Variable) -> Result<Variable> {
        let result = self.tensor.add(&other.tensor)?;
        let grad_fn = Arc::new(AddBackward {
            lhs_shape: self.tensor.shape().to_vec(),
            rhs_shape: other.tensor.shape().to_vec(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone(), other.clone()]))
    }

    pub fn sub(&self, other: &Variable) -> Result<Variable> {
        let result = self.tensor.sub(&other.tensor)?;
        let grad_fn = Arc::new(SubBackward {
            lhs_shape: self.tensor.shape().to_vec(),
            rhs_shape: other.tensor.shape().to_vec(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone(), other.clone()]))
    }

    pub fn mul(&self, other: &Variable) -> Result<Variable> {
        let result = self.tensor.mul(&other.tensor)?;
        let grad_fn = Arc::new(MulBackward {
            lhs: SavedTensor(self.tensor.detach()),
            rhs: SavedTensor(other.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone(), other.clone()]))
    }

    pub fn div(&self, other: &Variable) -> Result<Variable> {
        let result = self.tensor.div(&other.tensor)?;
        let grad_fn = Arc::new(DivBackward {
            lhs: SavedTensor(self.tensor.detach()),
            rhs: SavedTensor(other.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone(), other.clone()]))
    }

    pub fn neg(&self) -> Result<Variable> {
        let result = self.tensor.neg()?;
        let grad_fn = Arc::new(NegBackward);
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn mul_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.tensor.mul_scalar(scalar)?;
        let grad_fn = Arc::new(MulScalarBackward { scalar });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.tensor.add_scalar(scalar)?;
        let grad_fn = Arc::new(AddScalarBackward);
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn pow(&self, other: &Variable) -> Result<Variable> {
        let result = self.tensor.pow(&other.tensor)?;
        let grad_fn = Arc::new(PowBackward {
            base: SavedTensor(self.tensor.detach()),
            exponent: SavedTensor(other.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone(), other.clone()]))
    }

    // ---- Unary math operations ----

    pub fn exp(&self) -> Result<Variable> {
        let result = self.tensor.exp()?;
        let grad_fn = Arc::new(ExpBackward {
            output: SavedTensor(result.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn log(&self) -> Result<Variable> {
        let result = self.tensor.log()?;
        let grad_fn = Arc::new(LogBackward {
            input: SavedTensor(self.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn sqrt(&self) -> Result<Variable> {
        let result = self.tensor.sqrt()?;
        let grad_fn = Arc::new(SqrtBackward {
            output: SavedTensor(result.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn tanh(&self) -> Result<Variable> {
        let result = self.tensor.tanh()?;
        let grad_fn = Arc::new(TanhBackward {
            output: SavedTensor(result.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn sigmoid(&self) -> Result<Variable> {
        let result = self.tensor.sigmoid()?;
        let grad_fn = Arc::new(SigmoidBackward {
            output: SavedTensor(result.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn relu(&self) -> Result<Variable> {
        let result = self.tensor.relu()?;
        let grad_fn = Arc::new(ReluBackward {
            input: SavedTensor(self.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn abs(&self) -> Result<Variable> {
        let result = self.tensor.abs()?;
        let grad_fn = Arc::new(AbsBackward {
            input: SavedTensor(self.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn sin(&self) -> Result<Variable> {
        let result = self.tensor.sin()?;
        let grad_fn = Arc::new(SinBackward {
            input: SavedTensor(self.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn cos(&self) -> Result<Variable> {
        let result = self.tensor.cos()?;
        let grad_fn = Arc::new(CosBackward {
            input: SavedTensor(self.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn clamp(&self, min: f64, max: f64) -> Result<Variable> {
        let result = self.tensor.clamp(min, max)?;
        let grad_fn = Arc::new(ClampBackward {
            input: SavedTensor(self.tensor.detach()),
            min,
            max,
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    // ---- Reductions ----

    pub fn sum(&self) -> Result<Variable> {
        let result = self.tensor.sum()?;
        let grad_fn = Arc::new(SumBackward {
            input_shape: self.tensor.shape().to_vec(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn mean(&self) -> Result<Variable> {
        let result = self.tensor.mean()?;
        let grad_fn = Arc::new(MeanBackward {
            input_shape: self.tensor.shape().to_vec(),
            numel: self.tensor.numel(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn sum_dim(&self, dim: i64, keep_dim: bool) -> Result<Variable> {
        let d = self.tensor.normalize_dim(dim)?;
        let result = self.tensor.sum_dim(dim, keep_dim)?;
        let grad_fn = Arc::new(SumDimBackward {
            input_shape: self.tensor.shape().to_vec(),
            dim: d,
            keep_dim,
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn mean_dim(&self, dim: i64, keep_dim: bool) -> Result<Variable> {
        let d = self.tensor.normalize_dim(dim)?;
        let result = self.tensor.mean_dim(dim, keep_dim)?;
        let grad_fn = Arc::new(MeanDimBackward {
            input_shape: self.tensor.shape().to_vec(),
            dim: d,
            keep_dim,
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    // ---- Matrix operations ----

    pub fn matmul(&self, other: &Variable) -> Result<Variable> {
        let result = self.tensor.matmul(&other.tensor)?;
        let grad_fn = Arc::new(MatmulBackward {
            lhs: SavedTensor(self.tensor.detach()),
            rhs: SavedTensor(other.tensor.detach()),
            lhs_ndim: self.tensor.ndim(),
            rhs_ndim: other.tensor.ndim(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone(), other.clone()]))
    }

    // ---- View operations ----

    pub fn reshape(&self, shape: &[usize]) -> Result<Variable> {
        let result = self.tensor.reshape(shape)?;
        let grad_fn = Arc::new(ReshapeBackward {
            input_shape: self.tensor.shape().to_vec(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<Variable> {
        let d0 = self.tensor.normalize_dim(dim0)?;
        let d1 = self.tensor.normalize_dim(dim1)?;
        let result = self.tensor.transpose(dim0, dim1)?;
        let grad_fn = Arc::new(TransposeBackward { dim0: d0, dim1: d1 });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    pub fn t(&self) -> Result<Variable> {
        self.transpose(0, 1)
    }

    // ---- Softmax ----

    pub fn softmax(&self, dim: i64) -> Result<Variable> {
        let d = self.tensor.normalize_dim(dim)?;
        let result = self.tensor.softmax(dim)?;
        let grad_fn = Arc::new(SoftmaxBackward {
            output: SavedTensor(result.detach()),
            dim: d,
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }
}

// ---- Backward engine ----

/// Run backward pass from a scalar loss variable.
/// Like `loss.backward()` in PyTorch.
pub fn backward_var(loss: &Variable) {
    if loss.tensor.numel() != 1 {
        panic!(
            "backward can only be called on scalar (1-element) tensors, got shape {:?}",
            loss.tensor.shape()
        );
    }

    // Use the engine's backward function
    crate::engine::backward_from_variable(loss);
}

impl Variable {
    /// Run backward pass from this variable (must be scalar).
    pub fn backward(&self) {
        backward_var(self);
    }
}

// Operator overloads for Variable

impl std::ops::Add for &Variable {
    type Output = Variable;
    fn add(self, rhs: &Variable) -> Variable {
        self.add(rhs).expect("Variable add failed")
    }
}

impl std::ops::Sub for &Variable {
    type Output = Variable;
    fn sub(self, rhs: &Variable) -> Variable {
        Variable::sub(self, rhs).expect("Variable sub failed")
    }
}

impl std::ops::Mul for &Variable {
    type Output = Variable;
    fn mul(self, rhs: &Variable) -> Variable {
        Variable::mul(self, rhs).expect("Variable mul failed")
    }
}

impl std::ops::Div for &Variable {
    type Output = Variable;
    fn div(self, rhs: &Variable) -> Variable {
        Variable::div(self, rhs).expect("Variable div failed")
    }
}

impl std::ops::Neg for &Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        Variable::neg(self).expect("Variable neg failed")
    }
}
