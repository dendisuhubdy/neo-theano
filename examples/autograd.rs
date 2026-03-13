//! Autograd example -- computing gradients with reverse-mode automatic differentiation.
//!
//! Run with: cargo run -p theano --example autograd

use theano::prelude::*;

fn main() {
    println!("=== Neo Theano: Automatic Differentiation ===\n");

    // f(x) = x^2, df/dx = 2x
    let x = Variable::requires_grad(Tensor::scalar(3.0));
    let y = x.mul(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward();

    println!("x = 3.0");
    println!("f(x) = x^2 = {}", y.tensor().item().unwrap());
    println!("df/dx = 2x = {}", x.grad().unwrap().item().unwrap());

    println!();

    // f(a, b) = a*b + a^2
    // df/da = b + 2a = 3 + 4 = 7
    // df/db = a = 2
    let a = Variable::requires_grad(Tensor::scalar(2.0));
    let b = Variable::requires_grad(Tensor::scalar(3.0));
    let ab = a.mul(&b).unwrap();
    let a2 = a.mul(&a).unwrap();
    let result = ab.add(&a2).unwrap();
    result.backward();

    println!("a = 2.0, b = 3.0");
    println!("f(a,b) = a*b + a^2 = {}", result.tensor().item().unwrap());
    println!("df/da = b + 2a = {}", a.grad().unwrap().item().unwrap());
    println!("df/db = a = {}", b.grad().unwrap().item().unwrap());

    println!();

    // Demonstrate no_grad context
    {
        let _guard = theano::NoGradGuard::new();
        let p = Variable::requires_grad(Tensor::scalar(5.0));
        let q = p.mul(&p).unwrap();
        // q has no grad_fn because we are inside a no_grad block
        println!("Inside no_grad: q.grad_fn is None = {}", q.grad_fn().is_none());
    }

    println!("\nDone!");
}
