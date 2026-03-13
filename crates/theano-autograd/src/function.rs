use theano_core::Tensor;

/// Trait for custom autograd functions.
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
