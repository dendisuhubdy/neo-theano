use std::sync::Arc;

use theano_core::Tensor;
use theano_core::tensor::GradFn;
use theano_types::{Device, Result};

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

    /// Replace the underlying tensor data (used by optimizers).
    /// The new tensor will be a leaf with requires_grad=true and no grad_fn.
    pub fn update_param(&mut self, new_tensor: Tensor) {
        self.tensor = new_tensor.requires_grad_(true);
        self.inputs = vec![];
    }

    // ---- Device transfer ----

    /// Move this variable to a different device. Like `tensor.to(device)` in PyTorch.
    pub fn to(&self, device: &Device) -> Result<Variable> {
        let new_tensor = self.tensor.to(device)?;
        if self.requires_grad_flag() {
            Ok(Variable::requires_grad(new_tensor))
        } else {
            Ok(Variable::new(new_tensor))
        }
    }

    /// Move to CPU.
    pub fn cpu(&self) -> Result<Variable> {
        self.to(&Device::Cpu)
    }

    /// Move to CUDA device 0.
    pub fn cuda(&self) -> Result<Variable> {
        self.to(&Device::Cuda(0))
    }

    /// The device this variable's tensor lives on.
    pub fn device(&self) -> &Device {
        self.tensor.device()
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

    /// Public version of from_op for use by custom autograd functions
    /// and checkpoint.
    pub fn from_op_public(
        tensor: Tensor,
        grad_fn: Arc<dyn GradFn>,
        inputs: Vec<Variable>,
    ) -> Self {
        Self::from_op(tensor, grad_fn, inputs)
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

    // ---- Concatenation / Stacking ----

    /// Concatenate variables along an existing dimension. Like `torch.cat`.
    pub fn cat(vars: &[&Variable], dim: i64) -> Result<Variable> {
        if vars.is_empty() {
            return Err(theano_types::TheanoError::invalid_argument("cat: empty variable list"));
        }
        let d = vars[0].tensor.normalize_dim(dim)?;
        let tensors: Vec<Tensor> = vars.iter().map(|v| v.tensor.clone()).collect();
        let result = Tensor::cat(&tensors, dim)?;
        let sizes: Vec<usize> = vars.iter().map(|v| v.tensor.shape()[d]).collect();
        let grad_fn = Arc::new(CatBackward {
            dim: d,
            sizes,
        });
        let input_vars: Vec<Variable> = vars.iter().map(|v| (*v).clone()).collect();
        Ok(Variable::from_op(result, grad_fn, input_vars))
    }

    /// Stack variables along a new dimension. Like `torch.stack`.
    pub fn stack(vars: &[&Variable], dim: i64) -> Result<Variable> {
        if vars.is_empty() {
            return Err(theano_types::TheanoError::invalid_argument("stack: empty variable list"));
        }
        let tensors: Vec<Tensor> = vars.iter().map(|v| v.tensor.clone()).collect();
        let result = Tensor::stack(&tensors, dim)?;
        let d = if dim < 0 {
            (result.ndim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        let num_tensors = vars.len();
        let grad_fn = Arc::new(StackBackward {
            dim: d,
            num_tensors,
        });
        let input_vars: Vec<Variable> = vars.iter().map(|v| (*v).clone()).collect();
        Ok(Variable::from_op(result, grad_fn, input_vars))
    }

    // ---- Conditional / Selection operations ----

    /// Conditional selection: where condition is true (non-zero), use self;
    /// otherwise use other. Like `torch.where`.
    pub fn where_cond(&self, condition: &Variable, other: &Variable) -> Result<Variable> {
        let result = self.tensor.where_cond(&condition.tensor, &other.tensor)?;
        let grad_fn = Arc::new(WhereBackward {
            condition: SavedTensor(condition.tensor.detach()),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone(), other.clone()]))
    }

    /// Select a sub-tensor along a dimension, reducing it by 1. Like `torch.Tensor.select`.
    pub fn select(&self, dim: i64, index: i64) -> Result<Variable> {
        let d = self.tensor.normalize_dim(dim)?;
        let result = self.tensor.select(dim, index)?;
        let grad_fn = Arc::new(SelectBackward {
            dim: d,
            index: if index < 0 {
                (index + self.tensor.shape()[d] as i64) as usize
            } else {
                index as usize
            },
            input_shape: self.tensor.shape().to_vec(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    /// Narrow (slice) along a dimension. Like `torch.Tensor.narrow`.
    pub fn narrow(&self, dim: i64, start: i64, length: i64) -> Result<Variable> {
        let d = self.tensor.normalize_dim(dim)?;
        let start_usize = if start < 0 {
            (start + self.tensor.shape()[d] as i64) as usize
        } else {
            start as usize
        };
        let length_usize = length as usize;
        let result = self.tensor.narrow(dim, start_usize, length_usize)?;
        let grad_fn = Arc::new(NarrowBackward {
            dim: d,
            start: start_usize,
            input_shape: self.tensor.shape().to_vec(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    /// View with a new shape (must be contiguous). Like `torch.Tensor.view`.
    pub fn view(&self, shape: &[i64]) -> Result<Variable> {
        // Convert i64 shape to usize, handling -1 for inference
        let mut new_shape: Vec<usize> = Vec::new();
        let mut infer_idx: Option<usize> = None;
        let numel = self.tensor.numel();
        let mut known_product: usize = 1;

        for (i, &s) in shape.iter().enumerate() {
            if s == -1 {
                if infer_idx.is_some() {
                    return Err(theano_types::TheanoError::invalid_argument(
                        "view: only one dimension can be inferred (-1)",
                    ));
                }
                infer_idx = Some(i);
                new_shape.push(0); // placeholder
            } else if s < 0 {
                return Err(theano_types::TheanoError::invalid_argument(
                    "view: invalid shape dimension",
                ));
            } else {
                new_shape.push(s as usize);
                known_product *= s as usize;
            }
        }

        if let Some(idx) = infer_idx {
            if known_product == 0 {
                return Err(theano_types::TheanoError::invalid_argument(
                    "view: cannot infer dimension with zero-size dimensions",
                ));
            }
            new_shape[idx] = numel / known_product;
        }

        let result = self.tensor.view(&new_shape)?;
        let grad_fn = Arc::new(ReshapeBackward {
            input_shape: self.tensor.shape().to_vec(),
        });
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    /// Return a contiguous variable. Like `torch.Tensor.contiguous`.
    pub fn contiguous(&self) -> Result<Variable> {
        let result = self.tensor.contiguous()?;
        // Identity backward — gradient just passes through
        let grad_fn = Arc::new(ContiguousBackward);
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
    }

    /// Select elements along a dimension using an index tensor. Like `torch.index_select`.
    pub fn index_select(&self, dim: usize, indices: &Variable) -> Result<Variable> {
        let result = self.tensor.index_select(dim, &indices.tensor)?;
        let grad_fn = Arc::new(IndexSelectBackward {
            dim,
            indices: SavedTensor(indices.tensor.detach()),
            input_shape: self.tensor.shape().to_vec(),
        });
        // Note: indices variable is not differentiated (integer indices)
        Ok(Variable::from_op(result, grad_fn, vec![self.clone()]))
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
