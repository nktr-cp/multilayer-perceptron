//! # Multilayer Perceptron
//!
//! A from-scratch implementation of a multilayer perceptron with automatic differentiation
//! in Rust, designed for both native and WebAssembly targets.
//!
//! ## Features
//!
//! - **Automatic Differentiation**: Computation graph with reverse-mode autodiff
//! - **Neural Network Layers**: Dense layers with various activation functions
//! - **WebAssembly Support**: Deploy to browsers for interactive demos
//! - **Pure Rust**: No external ML libraries, everything built from scratch
//!
//! ## Quick Start
//!
//! ```rust
//! use multilayer_perceptron::prelude::*;
//!
//! // Create a tensor
//! let tensor = Tensor::ones(2, 3);
//! println!("Tensor shape: {:?}", tensor.shape());
//!
//! // Create a simple neural network (when implemented)
//! // let mut model = Sequential::new();
//! // model.add(Dense::new(784, 128, Activation::ReLU));
//! // model.add(Dense::new(128, 10, Activation::Softmax));
//! ```

#![warn(clippy::all)]

// Module declarations will be added as we implement them
pub mod ops;
pub mod tensor;
// pub mod graph;
// pub mod layers;
// pub mod dataset;
// pub mod train;

/// Prelude module for convenient imports
///
/// This module re-exports the most commonly used types and functions
/// for convenient access when using the multilayer perceptron library.
pub mod prelude {
  pub use crate::ops::{OpBuilder, OpNode};
  pub use crate::tensor::Tensor;
}

// Temporary placeholder function for CI setup
/// Add two numbers together.
///
/// This is a temporary function for CI/CD setup and will be removed
/// once the actual neural network implementation is in place.
///
/// # Examples
///
/// ```
/// use multilayer_perceptron::add;
/// assert_eq!(add(2, 2), 4);
/// ```
pub fn add(left: u64, right: u64) -> u64 {
  left + right
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_add() {
    assert_eq!(add(2, 2), 4);
    assert_eq!(add(0, 5), 5);
    assert_eq!(add(10, 15), 25);
  }

  #[test]
  fn test_add_edge_cases() {
    assert_eq!(add(0, 0), 0);
    assert_eq!(add(u64::MAX - 1, 1), u64::MAX);
  }
}
