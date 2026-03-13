//! MNIST CNN training example (API demonstration).
//!
//! This example shows what the MNIST training API looks like.
//! Full training requires downloading the MNIST dataset.
//!
//! Run with: cargo run -p theano --example mnist_cnn

use theano::prelude::*;
use theano::nn::*;
// Adam and Optimizer would be used in a full training loop.
#[allow(unused_imports)]
use theano::optim::{Adam, Optimizer};

fn main() {
    println!("=== Neo Theano: MNIST CNN (API Demo) ===\n");

    // Define a CNN model for MNIST
    let conv1 = Conv2d::with_options(1, 32, (3, 3), (1, 1), (1, 1), true);
    let conv2 = Conv2d::with_options(32, 64, (3, 3), (1, 1), (1, 1), true);
    let pool = MaxPool2d::new(2);
    let fc1 = Linear::new(64 * 7 * 7, 128);
    let fc2 = Linear::new(128, 10);
    let relu = ReLU;
    let flatten = Flatten::new();

    // Count parameters
    let mut total_params = 0;
    for p in conv1.parameters().iter()
        .chain(conv2.parameters().iter())
        .chain(fc1.parameters().iter())
        .chain(fc2.parameters().iter())
    {
        total_params += p.tensor().numel();
    }
    println!("Total parameters: {total_params}");

    // Forward pass with dummy data
    let dummy_input = Variable::new(Tensor::ones(&[1, 1, 28, 28]));

    let x = conv1.forward(&dummy_input);
    println!("After conv1: {:?}", x.tensor().shape());

    let x = relu.forward(&x);
    let x = pool.forward(&x);
    println!("After pool1: {:?}", x.tensor().shape());

    let x = conv2.forward(&x);
    let x = relu.forward(&x);
    let x = pool.forward(&x);
    println!("After pool2: {:?}", x.tensor().shape());

    let x = flatten.forward(&x);
    println!("After flatten: {:?}", x.tensor().shape());

    let x = fc1.forward(&x);
    let x = relu.forward(&x);
    let x = fc2.forward(&x);
    println!("Output logits: {:?}", x.tensor().shape());

    // Softmax for probabilities
    let probs = x.softmax(-1).unwrap();
    println!("Predicted class probabilities shape: {:?}", probs.tensor().shape());

    // Show what the training loop would look like
    println!("\nModel architecture:");
    println!("  Conv2d(1, 32, 3x3, padding=1)");
    println!("  ReLU + MaxPool2d(2)");
    println!("  Conv2d(32, 64, 3x3, padding=1)");
    println!("  ReLU + MaxPool2d(2)");
    println!("  Flatten");
    println!("  Linear(3136, 128) + ReLU");
    println!("  Linear(128, 10)");

    println!("\n--- Training loop sketch ---");
    println!("let params = collect_all_parameters();");
    println!("let mut optimizer = Adam::new(params, 0.001);");
    println!("let loss_fn = CrossEntropyLoss::new();");
    println!("for (images, labels) in dataloader {{");
    println!("    optimizer.zero_grad();");
    println!("    let logits = model_forward(images);");
    println!("    let loss = loss_fn.forward(&logits, &labels);");
    println!("    loss.backward();");
    println!("    optimizer.step();");
    println!("}}");

    println!("\nDone! (Full training requires MNIST dataset)");
}
