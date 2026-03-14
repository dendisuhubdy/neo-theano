use std::collections::HashMap;

use theano_core::Tensor;

use crate::variable::Variable;

fn var_id(var: &Variable) -> usize {
    // Use the stable Arc pointer identity from the tensor
    var.tensor().data_ptr_id()
}

/// Run backward pass from a scalar Variable.
///
/// This is the core backward engine, implementing reverse-mode autodiff
/// with topological sort (like PyTorch's engine).
pub fn backward_from_variable(root: &Variable) {
    // Start with gradient of 1.0 for the root (scalar loss)
    let root_grad = Tensor::scalar(1.0);

    // Build the graph: collect all nodes and their relationships
    // We'll do a DFS to find all nodes, then process in reverse topological order
    let mut topo_order: Vec<Variable> = Vec::new();
    let mut visited: HashMap<usize, bool> = HashMap::new();

    // Topological sort via DFS
    topo_sort(root, &mut topo_order, &mut visited);

    // Accumulate gradients in reverse topological order
    let mut grad_map: HashMap<usize, Tensor> = HashMap::new();
    grad_map.insert(var_id(root), root_grad);

    for var in topo_order.iter().rev() {
        let vid = var_id(var);
        let grad = match grad_map.get(&vid) {
            Some(g) => g.clone(),
            None => continue,
        };

        if let Some(grad_fn) = var.grad_fn() {
            // Compute gradients for inputs
            let input_grads = grad_fn.backward(&[grad]);

            // Get the inputs to this operation
            let inputs = var_inputs(var);

            for (input_var, grad_opt) in inputs.iter().zip(input_grads.iter()) {
                if let Some(g) = grad_opt {
                    let input_vid = var_id(input_var);
                    let entry = grad_map
                        .entry(input_vid)
                        .or_insert_with(|| Tensor::zeros(input_var.tensor().shape()));

                    // Accumulate gradient
                    let accumulated = entry.add(g).unwrap_or_else(|_| g.clone());
                    grad_map.insert(input_vid, accumulated);
                }
            }
        }

        // If this is a leaf that requires grad, store the gradient on the tensor
        if var.tensor().is_leaf() && var.requires_grad_flag() {
            if let Some(g) = grad_map.get(&vid) {
                // Store grad on the tensor via its RwLock
                // We need to access the inner grad field — we'll use a helper
                set_tensor_grad(var.tensor(), g.clone());
            }
        }
    }
}

/// Public backward function that takes a tensor (for compatibility).
///
/// **Note**: The tensor-path backward is not implemented. Use `Variable::backward()`
/// instead, which is the primary autograd API.
pub fn backward(loss: &Tensor) {
    if loss.numel() != 1 {
        panic!(
            "backward can only be called on scalar (1-element) tensors, got shape {:?}",
            loss.shape()
        );
    }

    panic!(
        "Tensor::backward() is not implemented. Use Variable::backward() instead. \
         Wrap your tensors in Variable::requires_grad() to use the autograd engine."
    );
}

/// DFS topological sort on Variables.
fn topo_sort(
    var: &Variable,
    order: &mut Vec<Variable>,
    visited: &mut HashMap<usize, bool>,
) {
    let vid = var_id(var);
    if visited.contains_key(&vid) {
        return;
    }
    visited.insert(vid, true);

    // Visit inputs first
    for input in var_inputs(var) {
        topo_sort(&input, order, visited);
    }

    order.push(var.clone());
}

/// Get the inputs of a variable (stored alongside it).
fn var_inputs(var: &Variable) -> Vec<Variable> {
    // Access the inputs field
    var.inputs.clone()
}

/// Set the grad on a tensor's inner grad field.
fn set_tensor_grad(tensor: &Tensor, grad: Tensor) {
    // Access the grad RwLock through the tensor's public API
    // We need to use unsafe or a helper. For now, we'll use the public
    // grad() accessor to check, and store via a method we add.
    //
    // Since Tensor wraps Arc<TensorInner> and TensorInner.grad is RwLock<Option<Tensor>>,
    // we need a way to write to it. Let's use a method on Tensor.
    tensor.set_grad(grad);
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;

    #[test]
    fn test_simple_backward() {
        // f(x) = x^2, df/dx = 2x
        let x = Variable::requires_grad(Tensor::from_slice(&[3.0], &[1]));
        let y = x.mul(&x).unwrap(); // x^2
        let loss = y.sum().unwrap(); // scalar
        loss.backward();

        let grad = x.grad().unwrap();
        let g = grad.to_vec_f64().unwrap();
        assert!(
            (g[0] - 6.0).abs() < 1e-10,
            "Expected grad=6.0, got {}",
            g[0]
        );
    }

    #[test]
    fn test_add_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[2.0, 3.0], &[2]));
        let y = Variable::requires_grad(Tensor::from_slice(&[4.0, 5.0], &[2]));
        let z = x.add(&y).unwrap();
        let loss = z.sum().unwrap();
        loss.backward();

        let gx = x.grad().unwrap().to_vec_f64().unwrap();
        let gy = y.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(gx, vec![1.0, 1.0]); // d(x+y)/dx = 1
        assert_eq!(gy, vec![1.0, 1.0]); // d(x+y)/dy = 1
    }

    #[test]
    fn test_mul_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[2.0, 3.0], &[2]));
        let y = Variable::requires_grad(Tensor::from_slice(&[4.0, 5.0], &[2]));
        let z = x.mul(&y).unwrap();
        let loss = z.sum().unwrap();
        loss.backward();

        let gx = x.grad().unwrap().to_vec_f64().unwrap();
        let gy = y.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(gx, vec![4.0, 5.0]); // d(x*y)/dx = y
        assert_eq!(gy, vec![2.0, 3.0]); // d(x*y)/dy = x
    }

    #[test]
    fn test_chain_backward() {
        // f(x) = (x * 2 + 1)^2, df/dx = 2 * (x*2+1) * 2 = 4*(2x+1)
        // At x=1: f = 9, df/dx = 4*3 = 12
        let x = Variable::requires_grad(Tensor::scalar(1.0));
        let two = Variable::new(Tensor::scalar(2.0));
        let one = Variable::new(Tensor::scalar(1.0));

        let x2 = x.mul(&two).unwrap();
        let x2p1 = x2.add(&one).unwrap();
        let loss = x2p1.mul(&x2p1).unwrap();
        loss.backward();

        let g = x.grad().unwrap().item().unwrap();
        assert!(
            (g - 12.0).abs() < 1e-10,
            "Expected 12.0, got {g}"
        );
    }

    #[test]
    fn test_relu_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]));
        let y = x.relu().unwrap();
        let loss = y.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(g, vec![0.0, 0.0, 1.0, 1.0]); // relu grad: 0 if x<=0, 1 if x>0
    }

    #[test]
    fn test_matmul_backward() {
        // f(X) = sum(X @ W), df/dX = ones @ W^T, df/dW = X^T @ ones
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let w = Variable::requires_grad(Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));
        let y = x.matmul(&w).unwrap();
        let loss = y.sum().unwrap();
        loss.backward();

        // dL/dX = ones @ W^T
        // W^T = [[5,7],[6,8]]
        // ones = [[1,1],[1,1]]
        // dL/dX = [[5+7, 6+8], [5+7, 6+8]] = [[12, 14], [12, 14]]
        let gx = x.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(gx, vec![11.0, 15.0, 11.0, 15.0]);

        // dL/dW = X^T @ ones
        // X^T = [[1,3],[2,4]]
        // dL/dW = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]]
        let gw = w.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(gw, vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_exp_backward() {
        let x = Variable::requires_grad(Tensor::scalar(1.0));
        let y = x.exp().unwrap();
        let loss = y.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().item().unwrap();
        let expected = 1.0_f64.exp(); // d/dx exp(x) = exp(x)
        assert!(
            (g - expected).abs() < 1e-10,
            "Expected {expected}, got {g}"
        );
    }

    #[test]
    fn test_sigmoid_backward() {
        let x = Variable::requires_grad(Tensor::scalar(0.0));
        let y = x.sigmoid().unwrap();
        let loss = y.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().item().unwrap();
        // sigmoid(0) = 0.5, sigmoid'(0) = 0.5 * 0.5 = 0.25
        assert!(
            (g - 0.25).abs() < 1e-10,
            "Expected 0.25, got {g}"
        );
    }

    #[test]
    fn test_no_grad_prevents_tracking() {
        use crate::no_grad::no_grad;

        let x = Variable::requires_grad(Tensor::scalar(2.0));
        let y = no_grad(|| x.mul_scalar(3.0).unwrap());

        // y should not have a grad_fn
        assert!(y.grad_fn().is_none());
    }

    #[test]
    fn test_log_backward() {
        let x = Variable::requires_grad(Tensor::scalar(2.0));
        let y = x.log().unwrap();
        let loss = y.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().item().unwrap();
        // d/dx log(x) = 1/x = 0.5
        assert!(
            (g - 0.5).abs() < 1e-10,
            "Expected 0.5, got {g}"
        );
    }

    #[test]
    fn test_broadcast_backward() {
        // x: [1, 3], y: [2, 1] -> z = x + y: [2, 3]
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]));
        let y = Variable::requires_grad(Tensor::from_slice(&[10.0, 20.0], &[2, 1]));
        let z = x.add(&y).unwrap();
        let loss = z.sum().unwrap();
        loss.backward();

        // dL/dx: sum over broadcast dim 0 -> [1, 3], each = 2 (summed over 2 rows)
        let gx = x.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(gx, vec![2.0, 2.0, 2.0]);

        // dL/dy: sum over broadcast dim 1 -> [2, 1], each = 3 (summed over 3 cols)
        let gy = y.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(gy, vec![3.0, 3.0]);
    }

    #[test]
    fn test_mean_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]));
        let loss = x.mean().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(g, vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_sub_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[5.0, 6.0], &[2]));
        let y = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0], &[2]));
        let z = x.sub(&y).unwrap();
        let loss = z.sum().unwrap();
        loss.backward();

        let gx = x.grad().unwrap().to_vec_f64().unwrap();
        let gy = y.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(gx, vec![1.0, 1.0]);
        assert_eq!(gy, vec![-1.0, -1.0]);
    }

    #[test]
    fn test_div_backward() {
        // f(x, y) = x / y, df/dx = 1/y, df/dy = -x/y^2
        let x = Variable::requires_grad(Tensor::scalar(6.0));
        let y = Variable::requires_grad(Tensor::scalar(3.0));
        let z = x.div(&y).unwrap();
        z.backward();

        let gx = x.grad().unwrap().item().unwrap();
        let gy = y.grad().unwrap().item().unwrap();
        assert!((gx - 1.0 / 3.0).abs() < 1e-10);
        assert!((gy - (-6.0 / 9.0)).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_backward() {
        let x = Variable::requires_grad(Tensor::scalar(0.0));
        let y = x.tanh().unwrap();
        y.backward();

        let g = x.grad().unwrap().item().unwrap();
        // tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
        assert!(
            (g - 1.0).abs() < 1e-10,
            "Expected 1.0, got {g}"
        );
    }

    // ---- Tests for new Variable operations ----

    #[test]
    fn test_cat_forward() {
        let a = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]));
        let b = Variable::requires_grad(Tensor::from_slice(&[4.0, 5.0, 6.0], &[1, 3]));
        let c = Variable::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.tensor().shape(), &[2, 3]);
        assert_eq!(
            c.tensor().to_vec_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_cat_backward() {
        let a = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]));
        let b = Variable::requires_grad(Tensor::from_slice(&[4.0, 5.0, 6.0], &[1, 3]));
        let c = Variable::cat(&[&a, &b], 0).unwrap();
        let loss = c.sum().unwrap();
        loss.backward();

        let ga = a.grad().unwrap().to_vec_f64().unwrap();
        let gb = b.grad().unwrap().to_vec_f64().unwrap();
        // d/da sum(cat(a, b)) = [1, 1, 1]
        assert_eq!(ga, vec![1.0, 1.0, 1.0]);
        assert_eq!(gb, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_cat_backward_dim1() {
        let a = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0], &[2, 1]));
        let b = Variable::requires_grad(Tensor::from_slice(&[3.0, 4.0, 5.0, 6.0], &[2, 2]));
        let c = Variable::cat(&[&a, &b], 1).unwrap();
        assert_eq!(c.tensor().shape(), &[2, 3]);
        let loss = c.sum().unwrap();
        loss.backward();

        let ga = a.grad().unwrap().to_vec_f64().unwrap();
        let gb = b.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(ga, vec![1.0, 1.0]);
        assert_eq!(gb, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_stack_forward() {
        let a = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let b = Variable::requires_grad(Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]));
        let c = Variable::stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.tensor().shape(), &[2, 3]);
        assert_eq!(
            c.tensor().to_vec_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_stack_backward() {
        let a = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let b = Variable::requires_grad(Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]));
        let c = Variable::stack(&[&a, &b], 0).unwrap();
        let loss = c.sum().unwrap();
        loss.backward();

        let ga = a.grad().unwrap().to_vec_f64().unwrap();
        let gb = b.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(ga, vec![1.0, 1.0, 1.0]);
        assert_eq!(gb, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_select_forward() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let s = x.select(0, 1).unwrap();
        assert_eq!(s.tensor().shape(), &[3]);
        assert_eq!(s.tensor().to_vec_f64().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_select_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let s = x.select(0, 1).unwrap();
        let loss = s.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        // Gradient should be zero for row 0, ones for row 1
        assert_eq!(g, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_narrow_forward() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let n = x.narrow(1, 1, 2).unwrap();
        assert_eq!(n.tensor().shape(), &[2, 2]);
        assert_eq!(n.tensor().to_vec_f64().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_narrow_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let n = x.narrow(1, 1, 2).unwrap();
        let loss = n.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        // First column should be 0, second and third columns should be 1
        assert_eq!(g, vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_view_forward() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let v = x.view(&[3, 2]).unwrap();
        assert_eq!(v.tensor().shape(), &[3, 2]);
        assert_eq!(
            v.tensor().to_vec_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_view_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let v = x.view(&[6]).unwrap();
        let loss = v.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(g, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(x.grad().unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn test_view_with_infer() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let v = x.view(&[-1, 2]).unwrap();
        assert_eq!(v.tensor().shape(), &[3, 2]);
    }

    #[test]
    fn test_contiguous_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let c = x.contiguous().unwrap();
        let loss = c.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        assert_eq!(g, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_index_select_forward() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let indices = Variable::new(Tensor::from_slice(&[0.0, 2.0], &[2]));
        let s = x.index_select(1, &indices).unwrap();
        assert_eq!(s.tensor().shape(), &[2, 2]);
        assert_eq!(s.tensor().to_vec_f64().unwrap(), vec![1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_index_select_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
        ));
        let indices = Variable::new(Tensor::from_slice(&[0.0, 2.0], &[2]));
        let s = x.index_select(1, &indices).unwrap();
        let loss = s.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        // Gradient: column 0 and 2 get 1.0, column 1 gets 0.0
        assert_eq!(g, vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_index_select_backward_duplicate_indices() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let indices = Variable::new(Tensor::from_slice(&[0.0, 0.0, 2.0], &[3]));
        let s = x.index_select(0, &indices).unwrap();
        let loss = s.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        // Index 0 selected twice: gradient accumulates to 2.0
        assert_eq!(g, vec![2.0, 0.0, 1.0]);
    }

    #[test]
    fn test_where_cond_forward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]));
        let cond = Variable::new(Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0], &[4]));
        let other = Variable::requires_grad(Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4]));
        let result = x.where_cond(&cond, &other).unwrap();
        assert_eq!(
            result.tensor().to_vec_f64().unwrap(),
            vec![1.0, 20.0, 3.0, 40.0]
        );
    }

    #[test]
    fn test_where_cond_backward() {
        let x = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]));
        let cond = Variable::new(Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0], &[4]));
        let other = Variable::requires_grad(Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4]));
        let result = x.where_cond(&cond, &other).unwrap();
        let loss = result.sum().unwrap();
        loss.backward();

        let gx = x.grad().unwrap().to_vec_f64().unwrap();
        let go = other.grad().unwrap().to_vec_f64().unwrap();
        // x gets gradient where cond is true
        assert_eq!(gx, vec![1.0, 0.0, 1.0, 0.0]);
        // other gets gradient where cond is false
        assert_eq!(go, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cat_with_mul_backward() {
        // Test cat backward with a non-trivial upstream gradient
        let a = Variable::requires_grad(Tensor::from_slice(&[1.0, 2.0], &[2]));
        let b = Variable::requires_grad(Tensor::from_slice(&[3.0, 4.0], &[2]));
        let c = Variable::cat(&[&a, &b], 0).unwrap();
        // Multiply by a constant to get non-uniform gradients
        let scale = Variable::new(Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]));
        let scaled = c.mul(&scale).unwrap();
        let loss = scaled.sum().unwrap();
        loss.backward();

        let ga = a.grad().unwrap().to_vec_f64().unwrap();
        let gb = b.grad().unwrap().to_vec_f64().unwrap();
        // gradient = scale values
        assert_eq!(ga, vec![1.0, 2.0]);
        assert_eq!(gb, vec![3.0, 4.0]);
    }

    #[test]
    fn test_narrow_on_dim0() {
        let x = Variable::requires_grad(Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
        ));
        let n = x.narrow(0, 1, 2).unwrap();
        assert_eq!(n.tensor().shape(), &[2, 3]);
        let loss = n.sum().unwrap();
        loss.backward();

        let g = x.grad().unwrap().to_vec_f64().unwrap();
        // Row 0 gets 0, rows 1-2 get 1
        assert_eq!(g, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }
}
