//! Basic tensor operations example.
//!
//! Run with: cargo run -p theano --example basic_tensor

use theano::prelude::*;

fn main() {
    println!("=== Neo Theano: Basic Tensor Operations ===\n");

    // Tensor creation
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("a = {a}");

    let b = Tensor::ones(&[2, 3]);
    println!("b = {b}");

    // Arithmetic
    let c = &a + &b;
    println!("a + b = {c}");

    let d = &a * &b;
    println!("a * b = {d}");

    // Reductions
    let sum = a.sum().unwrap();
    println!("sum(a) = {sum}");

    let mean = a.mean().unwrap();
    println!("mean(a) = {mean}");

    // Matrix multiply
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = Tensor::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let z = x.matmul(&y).unwrap();
    println!("x @ y = {z}");

    // Views
    let reshaped = a.reshape(&[3, 2]).unwrap();
    println!("a.reshape([3, 2]) = {reshaped}");

    let transposed = a.transpose(0, 1).unwrap();
    println!("a.transpose(0, 1) shape = {:?}", transposed.shape());

    // Activations
    let v = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let relu_v = v.relu().unwrap();
    println!("relu({v}) = {relu_v}");

    let sigmoid_v = v.sigmoid().unwrap();
    println!("sigmoid({v}) = {sigmoid_v}");

    println!("\nDone!");
}
