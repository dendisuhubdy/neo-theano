//! Simple neural network training example.
//!
//! Trains a small MLP on an XOR-like regression problem
//! using SGD with momentum and MSE loss.
//!
//! Run with: cargo run -p theano --example neural_network

use theano::prelude::*;
use theano::nn::{Linear, ReLU, Sequential, MSELoss, Module};
use theano::optim::{SGD, Optimizer};

fn main() {
    println!("=== Neo Theano: Neural Network Training ===\n");

    // Create a simple MLP: 2 -> 16 -> 1
    let model = Sequential::new(vec![])
        .add(Linear::new(2, 16))
        .add(ReLU)
        .add(Linear::new(16, 1));

    println!("Model parameters: {}", model.num_parameters());

    // Collect parameters for the optimizer
    let params = model.parameters();
    let mut optimizer = SGD::new(params, 0.01).momentum(0.9);

    let loss_fn = MSELoss::new();

    // Training data: XOR-like problem
    let inputs: Vec<([f64; 2], f64)> = vec![
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    // Training loop
    println!("Training...");
    for epoch in 0..200 {
        let mut total_loss = 0.0;

        for (input_data, target_val) in &inputs {
            optimizer.zero_grad();

            let input = Variable::new(Tensor::from_slice(input_data.as_slice(), &[1, 2]));
            let target = Variable::new(Tensor::from_slice(&[*target_val], &[1, 1]));

            let output = model.forward(&input);
            let loss = loss_fn.forward(&output, &target);

            loss.backward();
            optimizer.step();

            total_loss += loss.tensor().item().unwrap();
        }

        if epoch % 40 == 0 || epoch == 199 {
            println!("  Epoch {:3}: loss = {:.6}", epoch, total_loss / 4.0);
        }
    }

    // Inference
    println!("\nInference:");
    for (input_data, expected) in &inputs {
        let input = Variable::new(Tensor::from_slice(input_data.as_slice(), &[1, 2]));
        let output = model.forward(&input);
        println!(
            "  [{:.0}, {:.0}] -> {:.4} (expected {:.1})",
            input_data[0],
            input_data[1],
            output.tensor().item().unwrap(),
            expected,
        );
    }

    println!("\nDone!");
}
