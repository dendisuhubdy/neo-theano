use theano_core::Tensor;
use theano_core::tensor::GradFn;

/// Saved tensor reference for backward computation.
/// Uses Arc to avoid cloning tensor data.
#[derive(Clone)]
pub struct SavedTensor(pub Tensor);

// ---- Arithmetic grad functions ----

pub struct AddBackward {
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let g_lhs = reduce_grad_to_shape(grad, &self.lhs_shape);
        let g_rhs = reduce_grad_to_shape(grad, &self.rhs_shape);
        vec![Some(g_lhs), Some(g_rhs)]
    }

    fn name(&self) -> &str {
        "AddBackward"
    }
}

pub struct SubBackward {
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let g_lhs = reduce_grad_to_shape(grad, &self.lhs_shape);
        let g_rhs = reduce_grad_to_shape(&grad.neg().unwrap(), &self.rhs_shape);
        vec![Some(g_lhs), Some(g_rhs)]
    }

    fn name(&self) -> &str {
        "SubBackward"
    }
}

pub struct MulBackward {
    pub lhs: SavedTensor,
    pub rhs: SavedTensor,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        // d/d(lhs) = grad * rhs, d/d(rhs) = grad * lhs
        let g_lhs_full = grad.mul(&self.rhs.0).unwrap();
        let g_rhs_full = grad.mul(&self.lhs.0).unwrap();
        let g_lhs = reduce_grad_to_shape(&g_lhs_full, self.lhs.0.shape());
        let g_rhs = reduce_grad_to_shape(&g_rhs_full, self.rhs.0.shape());
        vec![Some(g_lhs), Some(g_rhs)]
    }

    fn name(&self) -> &str {
        "MulBackward"
    }
}

pub struct DivBackward {
    pub lhs: SavedTensor,
    pub rhs: SavedTensor,
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        // d/d(lhs) = grad / rhs
        // d/d(rhs) = -grad * lhs / rhs^2
        let g_lhs_full = grad.div(&self.rhs.0).unwrap();
        let rhs_sq = self.rhs.0.square().unwrap();
        let g_rhs_full = grad.mul(&self.lhs.0).unwrap().neg().unwrap().div(&rhs_sq).unwrap();
        let g_lhs = reduce_grad_to_shape(&g_lhs_full, self.lhs.0.shape());
        let g_rhs = reduce_grad_to_shape(&g_rhs_full, self.rhs.0.shape());
        vec![Some(g_lhs), Some(g_rhs)]
    }

    fn name(&self) -> &str {
        "DivBackward"
    }
}

pub struct NegBackward;

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        vec![Some(grad_output[0].neg().unwrap())]
    }

    fn name(&self) -> &str {
        "NegBackward"
    }
}

// ---- Unary math grad functions ----

pub struct ExpBackward {
    pub output: SavedTensor, // save exp(x) to avoid recomputing
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx exp(x) = exp(x)
        vec![Some(grad_output[0].mul(&self.output.0).unwrap())]
    }

    fn name(&self) -> &str {
        "ExpBackward"
    }
}

pub struct LogBackward {
    pub input: SavedTensor,
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx log(x) = 1/x
        vec![Some(grad_output[0].div(&self.input.0).unwrap())]
    }

    fn name(&self) -> &str {
        "LogBackward"
    }
}

pub struct SqrtBackward {
    pub output: SavedTensor,
}

impl GradFn for SqrtBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx sqrt(x) = 0.5 / sqrt(x)
        let two_sqrt = self.output.0.mul_scalar(2.0).unwrap();
        vec![Some(grad_output[0].div(&two_sqrt).unwrap())]
    }

    fn name(&self) -> &str {
        "SqrtBackward"
    }
}

pub struct TanhBackward {
    pub output: SavedTensor,
}

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx tanh(x) = 1 - tanh(x)^2
        let tanh_sq = self.output.0.square().unwrap();
        let one_minus = Tensor::ones(tanh_sq.shape()).sub(&tanh_sq).unwrap();
        vec![Some(grad_output[0].mul(&one_minus).unwrap())]
    }

    fn name(&self) -> &str {
        "TanhBackward"
    }
}

pub struct SigmoidBackward {
    pub output: SavedTensor,
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        let one_minus = Tensor::ones(self.output.0.shape()).sub(&self.output.0).unwrap();
        let grad_input = self.output.0.mul(&one_minus).unwrap();
        vec![Some(grad_output[0].mul(&grad_input).unwrap())]
    }

    fn name(&self) -> &str {
        "SigmoidBackward"
    }
}

pub struct ReluBackward {
    pub input: SavedTensor,
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx relu(x) = 1 if x > 0, else 0
        let zero = Tensor::zeros(self.input.0.shape());
        let mask = self.input.0.gt(&zero).unwrap();
        vec![Some(grad_output[0].mul(&mask).unwrap())]
    }

    fn name(&self) -> &str {
        "ReluBackward"
    }
}

pub struct AbsBackward {
    pub input: SavedTensor,
}

impl GradFn for AbsBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx |x| = sign(x)
        let sign = self.input.0.sign().unwrap();
        vec![Some(grad_output[0].mul(&sign).unwrap())]
    }

    fn name(&self) -> &str {
        "AbsBackward"
    }
}

pub struct SinBackward {
    pub input: SavedTensor,
}

impl GradFn for SinBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx sin(x) = cos(x)
        let cos_x = self.input.0.cos().unwrap();
        vec![Some(grad_output[0].mul(&cos_x).unwrap())]
    }

    fn name(&self) -> &str {
        "SinBackward"
    }
}

pub struct CosBackward {
    pub input: SavedTensor,
}

impl GradFn for CosBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx cos(x) = -sin(x)
        let neg_sin_x = self.input.0.sin().unwrap().neg().unwrap();
        vec![Some(grad_output[0].mul(&neg_sin_x).unwrap())]
    }

    fn name(&self) -> &str {
        "CosBackward"
    }
}

pub struct PowBackward {
    pub base: SavedTensor,
    pub exponent: SavedTensor,
}

impl GradFn for PowBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        // d/d(base) = exponent * base^(exponent - 1) * grad
        let exp_minus_one = self.exponent.0.add_scalar(-1.0).unwrap();
        let base_pow = self.base.0.pow(&exp_minus_one).unwrap();
        let g_base_full = grad.mul(&self.exponent.0).unwrap().mul(&base_pow).unwrap();
        let g_base = reduce_grad_to_shape(&g_base_full, self.base.0.shape());

        // d/d(exponent) = base^exponent * log(base) * grad
        let output = self.base.0.pow(&self.exponent.0).unwrap();
        let log_base = self.base.0.log().unwrap();
        let g_exp_full = grad.mul(&output).unwrap().mul(&log_base).unwrap();
        let g_exp = reduce_grad_to_shape(&g_exp_full, self.exponent.0.shape());

        vec![Some(g_base), Some(g_exp)]
    }

    fn name(&self) -> &str {
        "PowBackward"
    }
}

pub struct MulScalarBackward {
    pub scalar: f64,
}

impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        vec![Some(grad_output[0].mul_scalar(self.scalar).unwrap())]
    }

    fn name(&self) -> &str {
        "MulScalarBackward"
    }
}

pub struct AddScalarBackward;

impl GradFn for AddScalarBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx (x + c) = 1
        vec![Some(grad_output[0].clone())]
    }

    fn name(&self) -> &str {
        "AddScalarBackward"
    }
}

// ---- Reduction grad functions ----

pub struct SumBackward {
    pub input_shape: Vec<usize>,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d/dx sum(x) = ones_like(x) * grad_output (scalar)
        let grad = &grad_output[0];
        // Expand the scalar grad to the input shape
        let expanded = grad.expand(&self.input_shape).unwrap();
        vec![Some(expanded)]
    }

    fn name(&self) -> &str {
        "SumBackward"
    }
}

pub struct MeanBackward {
    pub input_shape: Vec<usize>,
    pub numel: usize,
}

impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let scaled = grad.mul_scalar(1.0 / self.numel as f64).unwrap();
        let expanded = scaled.expand(&self.input_shape).unwrap();
        vec![Some(expanded)]
    }

    fn name(&self) -> &str {
        "MeanBackward"
    }
}

pub struct SumDimBackward {
    pub input_shape: Vec<usize>,
    pub dim: usize,
    pub keep_dim: bool,
}

impl GradFn for SumDimBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        // If keep_dim was false, we need to unsqueeze the grad
        let grad = if !self.keep_dim {
            grad.unsqueeze(self.dim as i64).unwrap()
        } else {
            grad.clone()
        };
        let expanded = grad.expand(&self.input_shape).unwrap();
        vec![Some(expanded)]
    }

    fn name(&self) -> &str {
        "SumDimBackward"
    }
}

pub struct MeanDimBackward {
    pub input_shape: Vec<usize>,
    pub dim: usize,
    pub keep_dim: bool,
}

impl GradFn for MeanDimBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let dim_size = self.input_shape[self.dim] as f64;
        let grad = if !self.keep_dim {
            grad.unsqueeze(self.dim as i64).unwrap()
        } else {
            grad.clone()
        };
        let scaled = grad.mul_scalar(1.0 / dim_size).unwrap();
        let expanded = scaled.expand(&self.input_shape).unwrap();
        vec![Some(expanded)]
    }

    fn name(&self) -> &str {
        "MeanDimBackward"
    }
}

// ---- Matrix operation grad functions ----

pub struct MatmulBackward {
    pub lhs: SavedTensor,
    pub rhs: SavedTensor,
    pub lhs_ndim: usize,
    pub rhs_ndim: usize,
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];

        match (self.lhs_ndim, self.rhs_ndim) {
            (2, 2) => {
                // C = A @ B
                // dA = dC @ B^T, dB = A^T @ dC
                let b_t = self.rhs.0.t().unwrap();
                let a_t = self.lhs.0.t().unwrap();
                let g_lhs = grad.matmul(&b_t).unwrap();
                let g_rhs = a_t.matmul(grad).unwrap();
                vec![Some(g_lhs), Some(g_rhs)]
            }
            (1, 2) => {
                // c = a @ B (a is vector)
                // da = dC @ B^T, dB = a^T @ dC (outer product)
                let b_t = self.rhs.0.t().unwrap();
                let g_lhs = grad.matmul(&b_t).unwrap();
                // a: [K], grad: [N] -> dB: [K, N]
                let a_col = self.lhs.0.unsqueeze(1).unwrap(); // [K, 1]
                let grad_row = grad.unsqueeze(0).unwrap(); // [1, N]
                let g_rhs = a_col.matmul(&grad_row).unwrap();
                vec![Some(g_lhs), Some(g_rhs)]
            }
            (2, 1) => {
                // c = A @ b (b is vector)
                // dA = dC (outer) b^T, db = A^T @ dC
                let a_t = self.lhs.0.t().unwrap();
                let g_rhs = a_t.matmul(grad).unwrap();
                // grad: [M], b: [K] -> dA: [M, K]
                let grad_col = grad.unsqueeze(1).unwrap(); // [M, 1]
                let b_row = self.rhs.0.unsqueeze(0).unwrap(); // [1, K]
                let g_lhs = grad_col.matmul(&b_row).unwrap();
                vec![Some(g_lhs), Some(g_rhs)]
            }
            (1, 1) => {
                // dot product: c = a . b
                // da = dC * b, db = dC * a
                let g_lhs = grad.mul(&self.rhs.0).unwrap();
                let g_rhs = grad.mul(&self.lhs.0).unwrap();
                // grad is scalar, need to broadcast
                let g_lhs = if g_lhs.shape() != self.lhs.0.shape() {
                    reduce_grad_to_shape(&g_lhs, self.lhs.0.shape())
                } else {
                    g_lhs
                };
                let g_rhs = if g_rhs.shape() != self.rhs.0.shape() {
                    reduce_grad_to_shape(&g_rhs, self.rhs.0.shape())
                } else {
                    g_rhs
                };
                vec![Some(g_lhs), Some(g_rhs)]
            }
            _ => {
                // Batched matmul — simplified: assume contiguous batched case
                // C = A @ B => dA = dC @ B^T, dB = A^T @ dC
                let b_t = self.rhs.0.transpose(-2, -1).unwrap();
                let a_t = self.lhs.0.transpose(-2, -1).unwrap();
                let g_lhs = grad.matmul(&b_t).unwrap();
                let g_rhs = a_t.matmul(grad).unwrap();
                vec![Some(g_lhs), Some(g_rhs)]
            }
        }
    }

    fn name(&self) -> &str {
        "MatmulBackward"
    }
}

// ---- View grad functions ----

pub struct ReshapeBackward {
    pub input_shape: Vec<usize>,
}

impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = grad_output[0].reshape(&self.input_shape).unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &str {
        "ReshapeBackward"
    }
}

pub struct TransposeBackward {
    pub dim0: usize,
    pub dim1: usize,
}

impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // Transpose is its own inverse
        let grad = grad_output[0]
            .transpose(self.dim0 as i64, self.dim1 as i64)
            .unwrap();
        vec![Some(grad)]
    }

    fn name(&self) -> &str {
        "TransposeBackward"
    }
}

pub struct ExpandBackward {
    pub input_shape: Vec<usize>,
}

impl GradFn for ExpandBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = reduce_grad_to_shape(&grad_output[0], &self.input_shape);
        vec![Some(grad)]
    }

    fn name(&self) -> &str {
        "ExpandBackward"
    }
}

// ---- Softmax backward ----

pub struct SoftmaxBackward {
    pub output: SavedTensor,
    pub dim: usize,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // d softmax / dx_i = softmax_i * (delta_ij - softmax_j)
        // In practice: grad_input = softmax * (grad_output - sum(grad_output * softmax, dim))
        let s = &self.output.0;
        let gs = grad_output[0].mul(s).unwrap();
        let sum_gs = gs.sum_dim(self.dim as i64, true).unwrap();
        let grad_input = s.mul(&grad_output[0].sub(&sum_gs).unwrap()).unwrap();
        vec![Some(grad_input)]
    }

    fn name(&self) -> &str {
        "SoftmaxBackward"
    }
}

// ---- Clamp backward ----

pub struct ClampBackward {
    pub input: SavedTensor,
    pub min: f64,
    pub max: f64,
}

impl GradFn for ClampBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // Gradient passes through where input is in [min, max]
        let data = self.input.0.to_vec_f64().unwrap();
        let mask_data: Vec<f64> = data.iter().map(|&x| {
            if x >= self.min && x <= self.max { 1.0 } else { 0.0 }
        }).collect();
        let mask = Tensor::from_slice(&mask_data, self.input.0.shape());
        vec![Some(grad_output[0].mul(&mask).unwrap())]
    }

    fn name(&self) -> &str {
        "ClampBackward"
    }
}

// ---- Concatenation / Stacking grad functions ----

pub struct CatBackward {
    pub dim: usize,
    pub sizes: Vec<usize>,
}

impl GradFn for CatBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        // Split the gradient along the cat dimension according to saved sizes
        let mut grads = Vec::new();
        let mut offset = 0;
        for &size in &self.sizes {
            let g = grad.narrow(self.dim as i64, offset, size).unwrap();
            // Make contiguous since narrow returns a view
            let g = g.contiguous().unwrap();
            grads.push(Some(g));
            offset += size;
        }
        grads
    }

    fn name(&self) -> &str {
        "CatBackward"
    }
}

pub struct StackBackward {
    pub dim: usize,
    pub num_tensors: usize,
}

impl GradFn for StackBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        // Unstack: select along the stacked dimension for each input
        let mut grads = Vec::new();
        for i in 0..self.num_tensors {
            let g = grad.select(self.dim as i64, i as i64).unwrap();
            let g = g.contiguous().unwrap();
            grads.push(Some(g));
        }
        grads
    }

    fn name(&self) -> &str {
        "StackBackward"
    }
}

// ---- Conditional / Selection grad functions ----

pub struct WhereBackward {
    pub condition: SavedTensor,
}

impl GradFn for WhereBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let cond = &self.condition.0;
        // For self (input 0): gradient where condition is true, zero otherwise
        let zeros = Tensor::zeros(grad.shape());
        let grad_self = grad.where_cond(cond, &zeros).unwrap();
        // For other (input 1): gradient where condition is false, zero otherwise
        let grad_other = zeros.where_cond(cond, grad).unwrap();
        vec![Some(grad_self), Some(grad_other)]
    }

    fn name(&self) -> &str {
        "WhereBackward"
    }
}

pub struct SelectBackward {
    pub dim: usize,
    pub index: usize,
    pub input_shape: Vec<usize>,
}

impl GradFn for SelectBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        // Create a zero tensor of the input shape and scatter the gradient
        // into the correct position
        let grad_data = grad.contiguous().unwrap().to_vec_f64().unwrap();

        let dim = self.dim;
        let shape = &self.input_shape;

        let outer_size: usize = shape[..dim].iter().product();
        let dim_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();

        let total: usize = shape.iter().product();
        let mut new_data = vec![0.0f64; total];
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let src_flat = outer * inner_size + inner;
                let dst_flat = outer * dim_size * inner_size + self.index * inner_size + inner;
                new_data[dst_flat] = grad_data[src_flat];
            }
        }

        let result = Tensor::from_slice(&new_data, &self.input_shape);
        vec![Some(result)]
    }

    fn name(&self) -> &str {
        "SelectBackward"
    }
}

pub struct NarrowBackward {
    pub dim: usize,
    pub start: usize,
    pub input_shape: Vec<usize>,
}

impl GradFn for NarrowBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let grad = grad.contiguous().unwrap();
        let grad_data = grad.to_vec_f64().unwrap();

        let dim = self.dim;
        let shape = &self.input_shape;

        let outer_size: usize = shape[..dim].iter().product();
        let dim_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();
        let narrow_len = grad.shape()[dim];

        let total: usize = shape.iter().product();
        let mut new_data = vec![0.0f64; total];

        for outer in 0..outer_size {
            for i in 0..narrow_len {
                for inner in 0..inner_size {
                    let src_flat = outer * narrow_len * inner_size + i * inner_size + inner;
                    let dst_flat = outer * dim_size * inner_size + (self.start + i) * inner_size + inner;
                    new_data[dst_flat] = grad_data[src_flat];
                }
            }
        }

        let result = Tensor::from_slice(&new_data, &self.input_shape);
        vec![Some(result)]
    }

    fn name(&self) -> &str {
        "NarrowBackward"
    }
}

pub struct ContiguousBackward;

impl GradFn for ContiguousBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        // Identity: gradient passes through unchanged
        vec![Some(grad_output[0].clone())]
    }

    fn name(&self) -> &str {
        "ContiguousBackward"
    }
}

pub struct IndexSelectBackward {
    pub dim: usize,
    pub indices: SavedTensor,
    pub input_shape: Vec<usize>,
}

impl GradFn for IndexSelectBackward {
    fn backward(&self, grad_output: &[Tensor]) -> Vec<Option<Tensor>> {
        let grad = &grad_output[0];
        let grad = grad.contiguous().unwrap();
        let grad_data = grad.to_vec_f64().unwrap();
        let idx_data = self.indices.0.to_vec_f64().unwrap();
        let indices: Vec<usize> = idx_data.iter().map(|&v| v as usize).collect();

        let dim = self.dim;
        let shape = &self.input_shape;

        let outer_size: usize = shape[..dim].iter().product();
        let dim_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();
        let num_indices = indices.len();

        let total: usize = shape.iter().product();
        let mut new_data = vec![0.0f64; total];

        // Scatter-add the gradients back
        for outer in 0..outer_size {
            for (ii, &idx) in indices.iter().enumerate() {
                for inner in 0..inner_size {
                    let src_flat = outer * num_indices * inner_size + ii * inner_size + inner;
                    let dst_flat = outer * dim_size * inner_size + idx * inner_size + inner;
                    new_data[dst_flat] += grad_data[src_flat];
                }
            }
        }

        let result = Tensor::from_slice(&new_data, &self.input_shape);
        vec![Some(result)]
    }

    fn name(&self) -> &str {
        "IndexSelectBackward"
    }
}

// ---- Helper: reduce gradient to match input shape after broadcasting ----

/// When a binary operation broadcasts, the gradient of the output has the broadcast
/// shape. We need to sum-reduce it back to the original input shape.
fn reduce_grad_to_shape(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    if grad.shape() == target_shape {
        return grad.clone();
    }

    let grad_shape = grad.shape();
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();

    let mut result = grad.clone();

    // Sum over leading dimensions that were broadcast-added
    if grad_ndim > target_ndim {
        for _ in 0..(grad_ndim - target_ndim) {
            result = result.sum_dim(0, false).unwrap();
        }
    }

    // Sum over dimensions where target has size 1 but grad doesn't
    let result_shape = result.shape().to_vec();
    for (i, (&gs, &ts)) in result_shape.iter().zip(target_shape.iter()).enumerate() {
        if ts == 1 && gs != 1 {
            result = result.sum_dim(i as i64, true).unwrap();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_grad_to_shape_same() {
        let grad = Tensor::ones(&[2, 3]);
        let result = reduce_grad_to_shape(&grad, &[2, 3]);
        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_reduce_grad_to_shape_broadcast_dim() {
        // grad: [2, 3], target: [1, 3] — sum over dim 0
        let grad = Tensor::ones(&[2, 3]);
        let result = reduce_grad_to_shape(&grad, &[1, 3]);
        assert_eq!(result.shape(), &[1, 3]);
        let data = result.to_vec_f64().unwrap();
        assert_eq!(data, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_reduce_grad_to_shape_leading_dims() {
        // grad: [4, 3], target: [3] — sum over leading dim
        let grad = Tensor::ones(&[4, 3]);
        let result = reduce_grad_to_shape(&grad, &[3]);
        assert_eq!(result.shape(), &[3]);
        let data = result.to_vec_f64().unwrap();
        assert_eq!(data, vec![4.0, 4.0, 4.0]);
    }
}
