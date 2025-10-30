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
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! // Create a tensor
//! let tensor = Tensor::ones(2, 3);
//! println!("Tensor shape: {:?}", tensor.shape());
//!
//! // Create a neural network with PyTorch-like API
//! let graph = Rc::new(RefCell::new(ComputationGraph::new()));
//! let mut model = Sequential::new()
//!     .relu_layer(784, 128)
//!     .relu_layer(128, 64)
//!     .softmax_layer(64, 10)
//!     .with_graph(graph);
//!
//! // Forward pass
//! let input = Tensor::random(1, 784);
//! let output = model.forward(input).unwrap();
//! println!("Output shape: {:?}", output.shape());
//! ```

#![warn(clippy::all)]

// Module declarations will be added as we implement them
pub mod dataset;
pub mod error;
pub mod generic_dataset;
pub mod graph;
pub mod layers;
pub mod loss;
pub mod metrics;
pub mod ops;
pub mod optimizer;
pub mod tensor;
pub mod trainer;

// WebAssembly bindings (only included when targeting wasm32)
#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// Prelude module for convenient imports
///
/// This module re-exports the most commonly used types and functions
/// for convenient access when using the multilayer perceptron library.
pub mod prelude {
  pub use crate::dataset::{DataLoader, Dataset, Diagnosis, FeatureStats, PreprocessConfig};
  pub use crate::error::{Result, TensorError};
  pub use crate::generic_dataset::{
    load_csv, parse_field, CsvConfig, DataValue, DatasetLike, GenericDataFrame,
  };
  pub use crate::graph::{ComputationGraph, EdgeId, GraphEdge, GraphNode, NodeId};
  pub use crate::layers::{
    Activation, DenseLayer, Layer, LayerInfo, ModelSummary, Sequential, WeightInit,
  };
  pub use crate::loss::{BinaryCrossEntropy, Loss, MeanSquaredError};
  pub use crate::metrics::{
    Accuracy, BinaryClassificationMetrics, F1Score, Metric, Precision, Recall,
  };
  pub use crate::ops::{OpBuilder, OpNode};
  pub use crate::optimizer::{GradientDescent, Optimizer, SGDMomentum, SGD};
  pub use crate::tensor::Tensor;
  pub use crate::trainer::{EpochHistory, Trainer, TrainingConfig, TrainingHistory};
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
