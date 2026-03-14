use std::sync::Arc;

use theano_core::Tensor;
use theano_core::tensor::GradFn;

use crate::no_grad::is_grad_enabled;
use crate::variable::Variable;

/// Context for saving tensors between forward and backward passes.
///
/// Like `ctx` in PyTorch's `torch.autograd.Function`.
pub struct FunctionCtx {
    saved_tensors: Vec<Tensor>,
    needs_input_grad: Vec<bool>,
}

impl FunctionCtx {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            saved_tensors: Vec::new(),
            needs_input_grad: Vec::new(),
        }
    }

    /// Save tensors for use during backward.
    ///
    /// Like `ctx.save_for_backward(...)` in PyTorch.
    pub fn save_for_backward(&mut self, tensors: Vec<Tensor>) {
        self.saved_tensors = tensors;
    }

    /// Retrieve the tensors saved during forward.
    ///
    /// Like `ctx.saved_tensors` in PyTorch.
    pub fn saved_tensors(&self) -> &[Tensor] {
        &self.saved_tensors
    }

    /// Whether each input requires gradient.
    pub fn needs_input_grad(&self) -> &[bool] {
        &self.needs_input_grad
    }

    /// Set which inputs require gradient. Called internally before forward.
    pub(crate) fn set_needs_input_grad(&mut self, needs: Vec<bool>) {
        self.needs_input_grad = needs;
    }
}

impl Default for FunctionCtx {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for custom differentiable operations.
///
/// Like `torch.autograd.Function` in PyTorch. Users implement `forward` and
/// `backward` to define custom autograd operations.
///
/// # Example
/// ```ignore
/// struct MyReLU;
/// impl AutogradFunction for MyReLU {
///     fn forward(ctx: &mut FunctionCtx, inputs: &[&Variable]) -> Variable {
///         let input = inputs[0];
///         ctx.save_for_backward(vec![input.tensor().detach()]);
///         input.relu().unwrap()
///     }
///     fn backward(ctx: &FunctionCtx, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
///         let saved = ctx.saved_tensors();
///         let input = &saved[0];
///         let mask = input.gt_scalar(0.0).unwrap();
///         let grad_input = grad_output[0].mul(&mask).unwrap();
///         vec![Some(grad_input)]
///     }
/// }
/// let output = MyReLU::apply(&[&input]);
/// ```
pub trait AutogradFunction: Send + Sync + 'static {
    /// Forward pass: compute outputs from inputs.
    ///
    /// Use `ctx.save_for_backward(...)` to save tensors needed during backward.
    fn forward(ctx: &mut FunctionCtx, inputs: &[&Variable]) -> Variable;

    /// Backward pass: compute input gradients from output gradients.
    ///
    /// `grad_output` contains the gradient of the loss w.r.t. the output.
    /// Returns a `Vec<Option<Tensor>>`, one per input. `None` means the input
    /// doesn't need a gradient.
    fn backward(ctx: &FunctionCtx, grad_output: &[Tensor]) -> Vec<Option<Tensor>>;

    /// Apply this function to the given inputs, building the autograd graph.
    ///
    /// This is the entry point users call instead of `forward` directly.
    fn apply(inputs: &[&Variable]) -> Variable
    where
        Self: Sized,
    {
        let mut ctx = FunctionCtx::new();
        let needs_grad: Vec<bool> = inputs.iter().map(|v| v.requires_grad_flag()).collect();
        ctx.set_needs_input_grad(needs_grad.clone());

        let any_needs_grad = needs_grad.iter().any(|&g| g);

        // Run forward
        let output = Self::forward(&mut ctx, inputs);

        // If any input requires grad and grad is enabled, attach the backward node
        if any_needs_grad && is_grad_enabled() {
            let grad_fn = Arc::new(FunctionBackward::<Self> {
                ctx: parking_lot::RwLock::new(ctx),
                _marker: std::marker::PhantomData,
            });
            let input_vars: Vec<Variable> = inputs.iter().map(|v| (*v).clone()).collect();
            Variable::from_op_public(output.into_tensor(), grad_fn, input_vars)
        } else {
            output
        }
    }
}

/// GradFn implementation that bridges custom AutogradFunction backward
/// into the autograd engine.
pub struct FunctionBackward<F: AutogradFunction> {
    ctx: parking_lot::RwLock<FunctionCtx>,
    _marker: std::marker::PhantomData<F>,
}

// Safety: FunctionCtx contains Tensors which are Send+Sync, and the
// RwLock provides thread-safe access.
unsafe impl<F: AutogradFunction> Send for FunctionBackward<F> {}
unsafe impl<F: AutogradFunction> Sync for FunctionBackward<F> {}

impl<F: AutogradFunction> GradFn for FunctionBackward<F> {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let ctx = self.ctx.read();
        F::backward(&ctx, grad_output)
    }

    fn name(&self) -> &str {
        "FunctionBackward"
    }
}

/// The original `Function` trait from the first implementation, retained
/// for backward compatibility.
///
/// Users can implement this trait to define custom forward/backward passes,
/// similar to `torch.autograd.Function` in PyTorch.
///
/// # Example
/// ```ignore
/// struct MyReLU;
/// impl Function for MyReLU {
///     fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
///         let x = &inputs[0];
///         vec![x.relu().unwrap()]
///     }
///     fn backward(&self, grad_outputs: &[Tensor], saved: &[Tensor]) -> Vec<Option<Tensor>> {
///         let x = &saved[0];
///         let mask = x.gt(&Tensor::zeros(x.shape())).unwrap();
///         vec![Some(grad_outputs[0].mul(&mask).unwrap())]
///     }
/// }
/// ```
pub trait Function: Send + Sync {
    /// Forward pass: compute outputs from inputs.
    fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor>;

    /// Backward pass: compute input gradients from output gradients.
    ///
    /// `grad_outputs` contains the gradient of the loss w.r.t. each output.
    /// `saved` contains tensors saved during forward for use in backward.
    ///
    /// Returns a Vec of Option<Tensor>, one per input. None means the input
    /// doesn't need a gradient.
    fn backward(&self, grad_outputs: &[Tensor], saved: &[Tensor]) -> Vec<Option<Tensor>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DoubleFunc;

    impl AutogradFunction for DoubleFunc {
        fn forward(ctx: &mut FunctionCtx, inputs: &[&Variable]) -> Variable {
            let input = inputs[0];
            ctx.save_for_backward(vec![input.tensor().detach()]);
            input.mul_scalar(2.0).unwrap()
        }

        fn backward(_ctx: &FunctionCtx, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
            // d/dx (2x) = 2
            vec![Some(grad_output[0].mul_scalar(2.0).unwrap())]
        }
    }

    #[test]
    fn test_custom_function_forward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let y = DoubleFunc::apply(&[&x]);
        let data = y.tensor().to_vec_f64().unwrap();
        assert_eq!(data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_custom_function_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let y = DoubleFunc::apply(&[&x]);
        let loss = y.sum().unwrap();
        loss.backward();
        let g = x.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(g, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_function_ctx_save_for_backward() {
        let mut ctx = FunctionCtx::new();
        let t = Tensor::from_slice(&[1.0, 2.0], &[2]);
        ctx.save_for_backward(vec![t.clone()]);
        assert_eq!(ctx.saved_tensors().len(), 1);
        assert_eq!(
            ctx.saved_tensors()[0].to_vec_f64().unwrap(),
            vec![1.0, 2.0]
        );
    }

    struct CustomReLU;

    impl AutogradFunction for CustomReLU {
        fn forward(ctx: &mut FunctionCtx, inputs: &[&Variable]) -> Variable {
            let input = inputs[0];
            ctx.save_for_backward(vec![input.tensor().detach()]);
            input.relu().unwrap()
        }

        fn backward(ctx: &FunctionCtx, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
            let saved = ctx.saved_tensors();
            let input = &saved[0];
            let mask = input.gt_scalar(0.0).unwrap();
            let grad_input = grad_output[0].mul(&mask).unwrap();
            vec![Some(grad_input)]
        }
    }

    #[test]
    fn test_custom_relu_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]));
        let y = CustomReLU::apply(&[&x]);
        let loss = y.sum().unwrap();
        loss.backward();
        let g = x.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(g, vec![0.0, 0.0, 1.0, 1.0]);
    }
}
