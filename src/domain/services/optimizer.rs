//! Optimization algorithms for training neural networks
//!
//! This module provides different optimizers for updating model parameters
//! during training. Currently supports SGD (Stochastic Gradient Descent).

use crate::core::{Result, Tensor};
use ndarray::Zip;
use std::collections::HashMap;

/// Trait for optimizers that update model parameters
pub trait Optimizer {
  /// Update parameters using computed gradients
  ///
  /// # Arguments
  /// * `parameters` - Mutable reference to parameter tensors
  ///
  /// # Returns
  /// Result indicating success or failure
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()>;

  /// Zero out all gradients in the parameters
  ///
  /// # Arguments
  /// * `parameters` - Mutable reference to parameter tensors
  fn zero_grad(&self, parameters: &mut [&mut Tensor]);

  /// Get the learning rate
  fn learning_rate(&self) -> f64;

  /// Set the learning rate
  fn set_learning_rate(&mut self, lr: f64);

  /// Whether this optimizer expects gradients computed on the full dataset.
  fn requires_full_batch(&self) -> bool {
    false
  }

  /// Identifier for display/debugging.
  fn name(&self) -> &'static str;
}

/// Stochastic Gradient Descent optimizer
///
/// SGD updates parameters using the following rule:
/// θ = θ - lr * ∇θ
///
/// Where:
/// - θ are the parameters
/// - lr is the learning rate
/// - ∇θ are the gradients
#[derive(Debug, Clone)]
pub struct SGD {
  /// Learning rate
  learning_rate: f64,
}

impl SGD {
  /// Create a new SGD optimizer
  ///
  /// # Arguments
  /// * `learning_rate` - The learning rate for parameter updates
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let optimizer = SGD::new(0.01);
  /// assert_eq!(optimizer.learning_rate(), 0.01);
  /// ```
  pub fn new(learning_rate: f64) -> Self {
    assert!(learning_rate > 0.0, "Learning rate must be positive");
    Self { learning_rate }
  }
}

impl Optimizer for SGD {
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
    for param in parameters.iter_mut() {
      if let Some(gradient) = param.grad() {
        Zip::from(&mut param.data)
          .and(&gradient)
          .for_each(|param_elem, &grad_elem| {
            *param_elem -= self.learning_rate * grad_elem;
          });
      }
    }
    Ok(())
  }

  fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
    for param in parameters.iter_mut() {
      param.zero_grad();
    }
  }

  fn learning_rate(&self) -> f64 {
    self.learning_rate
  }

  fn set_learning_rate(&mut self, lr: f64) {
    assert!(lr > 0.0, "Learning rate must be positive");
    self.learning_rate = lr;
  }

  fn name(&self) -> &'static str {
    "SGD"
  }
}

/// SGD with momentum optimizer
///
/// SGD with momentum uses exponentially weighted averages of gradients
/// to accelerate learning and dampen oscillations:
///
/// v_t = β * v_{t-1} + (1-β) * ∇θ_t
/// θ_t = θ_{t-1} - lr * v_t
///
/// Where:
/// - v_t is the velocity (exponentially weighted average)
/// - β is the momentum coefficient (typically 0.9)
/// - lr is the learning rate
/// - ∇θ_t are the current gradients
#[derive(Debug)]
pub struct SGDMomentum {
  /// Learning rate
  learning_rate: f64,
  /// Momentum coefficient (typically 0.9)
  momentum: f64,
  /// Velocity terms for each parameter
  velocities: HashMap<usize, Tensor>,
}

impl SGDMomentum {
  /// Create a new SGD with momentum optimizer
  ///
  /// # Arguments
  /// * `learning_rate` - The learning rate for parameter updates
  /// * `momentum` - The momentum coefficient (typically 0.9)
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let optimizer = SGDMomentum::new(0.01, 0.9);
  /// assert_eq!(optimizer.learning_rate(), 0.01);
  /// ```
  pub fn new(learning_rate: f64, momentum: f64) -> Self {
    assert!(learning_rate > 0.0, "Learning rate must be positive");
    assert!(
      (0.0..1.0).contains(&momentum),
      "Momentum must be between 0 and 1"
    );

    Self {
      learning_rate,
      momentum,
      velocities: HashMap::new(),
    }
  }

  /// Get the momentum coefficient
  pub fn momentum(&self) -> f64 {
    self.momentum
  }

  /// Set the momentum coefficient
  pub fn set_momentum(&mut self, momentum: f64) {
    assert!(
      (0.0..1.0).contains(&momentum),
      "Momentum must be between 0 and 1"
    );
    self.momentum = momentum;
  }
}

impl Clone for SGDMomentum {
  fn clone(&self) -> Self {
    Self {
      learning_rate: self.learning_rate,
      momentum: self.momentum,
      velocities: HashMap::new(), // Reset velocities on clone
    }
  }
}

impl Optimizer for SGDMomentum {
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
    for (i, param) in parameters.iter_mut().enumerate() {
      if let Some(gradient_data) = param.grad() {
        // Get or create velocity for this parameter
        let velocity = self
          .velocities
          .entry(i)
          .or_insert_with(|| Tensor::zeros(param.shape().0, param.shape().1));

        // Update velocity: v_t = β * v_{t-1} + (1-β) * ∇θ_t
        Zip::from(&mut velocity.data)
          .and(&gradient_data)
          .for_each(|vel_elem, &grad_elem| {
            *vel_elem = self.momentum * *vel_elem + (1.0 - self.momentum) * grad_elem;
          });

        // Update parameters: θ_t = θ_{t-1} - lr * v_t
        Zip::from(&mut param.data)
          .and(&velocity.data)
          .for_each(|param_elem, &vel_elem| {
            *param_elem -= self.learning_rate * vel_elem;
          });
      }
    }
    Ok(())
  }

  fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
    for param in parameters.iter_mut() {
      param.zero_grad();
    }
  }

  fn learning_rate(&self) -> f64 {
    self.learning_rate
  }

  fn set_learning_rate(&mut self, lr: f64) {
    assert!(lr > 0.0, "Learning rate must be positive");
    self.learning_rate = lr;
  }

  fn name(&self) -> &'static str {
    "SGD with Momentum"
  }
}

/// Full-batch Gradient Descent optimizer
#[derive(Debug, Clone)]
pub struct GradientDescent {
  learning_rate: f64,
}

impl GradientDescent {
  pub fn new(learning_rate: f64) -> Self {
    assert!(learning_rate > 0.0, "Learning rate must be positive");
    Self { learning_rate }
  }
}

impl Optimizer for GradientDescent {
  fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
    for param in parameters.iter_mut() {
      if let Some(gradient) = param.grad() {
        Zip::from(&mut param.data)
          .and(&gradient)
          .for_each(|param_elem, &grad_elem| {
            *param_elem -= self.learning_rate * grad_elem;
          });
      }
    }
    Ok(())
  }

  fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
    for param in parameters.iter_mut() {
      param.zero_grad();
    }
  }

  fn learning_rate(&self) -> f64 {
    self.learning_rate
  }

  fn set_learning_rate(&mut self, lr: f64) {
    assert!(lr > 0.0, "Learning rate must be positive");
    self.learning_rate = lr;
  }

  fn requires_full_batch(&self) -> bool {
    true
  }

  fn name(&self) -> &'static str {
    "Gradient Descent"
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::Tensor;
  use approx::assert_abs_diff_eq;

  #[test]
  fn test_sgd_creation() {
    let optimizer = SGD::new(0.01);
    assert_eq!(optimizer.learning_rate(), 0.01);
    assert_eq!(optimizer.name(), "SGD");
  }

  #[test]
  #[should_panic(expected = "Learning rate must be positive")]
  fn test_sgd_negative_learning_rate() {
    SGD::new(-0.01);
  }

  #[test]
  fn test_sgd_set_learning_rate() {
    let mut optimizer = SGD::new(0.01);
    optimizer.set_learning_rate(0.1);
    assert_eq!(optimizer.learning_rate(), 0.1);
  }

  #[test]
  fn test_sgd_zero_grad() {
    let optimizer = SGD::new(0.01);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    // Set some fake gradient
    param.set_gradient(Some(Tensor::ones(2, 2)));
    assert!(param.gradient().is_some());

    optimizer.zero_grad(&mut [&mut param]);
    // After zero_grad, gradient should still exist but be zero
    if let Some(grad) = param.grad() {
      for elem in grad.iter() {
        assert_eq!(*elem, 0.0);
      }
    } else {
      panic!("Gradient should still exist after zero_grad");
    }
  }

  #[test]
  fn test_sgd_step() {
    let mut optimizer = SGD::new(0.1);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    // Set gradient to ones, so update should be -0.1 * 1.0 = -0.1
    param.set_gradient(Some(Tensor::ones(2, 2)));

    let original_value = param.data[[0, 0]];
    optimizer.step(&mut [&mut param]).unwrap();

    // θ = θ - lr * ∇θ = 1.0 - 0.1 * 1.0 = 0.9
    assert_abs_diff_eq!(param.data[[0, 0]], original_value - 0.1, epsilon = 1e-6);
  }

  #[test]
  fn test_sgd_step_no_gradient() {
    let mut optimizer = SGD::new(0.1);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    let original_value = param.data[[0, 0]];
    optimizer.step(&mut [&mut param]).unwrap();

    // No gradient, so parameter should not change
    assert_abs_diff_eq!(param.data[[0, 0]], original_value, epsilon = 1e-6);
  }

  #[test]
  fn test_sgd_momentum_creation() {
    let optimizer = SGDMomentum::new(0.01, 0.9);
    assert_eq!(optimizer.learning_rate(), 0.01);
    assert_eq!(optimizer.name(), "SGD with Momentum");
    assert_eq!(optimizer.momentum(), 0.9);
  }

  #[test]
  #[should_panic(expected = "Momentum must be between 0 and 1")]
  fn test_sgd_momentum_invalid_momentum() {
    SGDMomentum::new(0.01, 1.5);
  }

  #[test]
  fn test_sgd_momentum_set_momentum() {
    let mut optimizer = SGDMomentum::new(0.01, 0.9);
    optimizer.set_momentum(0.95);
    assert_eq!(optimizer.momentum(), 0.95);
  }

  #[test]
  fn test_sgd_momentum_step() {
    let mut optimizer = SGDMomentum::new(0.1, 0.9);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    // Set gradient to ones
    param.set_gradient(Some(Tensor::ones(2, 2)));

    let original_value = param.data[[0, 0]];
    optimizer.step(&mut [&mut param]).unwrap();

    // First step: v_0 = 0.9 * 0 + 0.1 * 1.0 = 0.1
    // θ_1 = 1.0 - 0.1 * 0.1 = 0.99
    assert_abs_diff_eq!(param.data[[0, 0]], original_value - 0.01, epsilon = 1e-6);
  }

  #[test]
  fn test_sgd_momentum_multiple_steps() {
    let mut optimizer = SGDMomentum::new(0.1, 0.9);
    let mut param = Tensor::ones(2, 2);
    param.set_requires_grad(true);

    // First step
    param.set_gradient(Some(Tensor::ones(2, 2)));
    let value_after_first = {
      optimizer.step(&mut [&mut param]).unwrap();
      param.data[[0, 0]]
    };

    // Second step with same gradient
    param.set_gradient(Some(Tensor::ones(2, 2)));
    optimizer.step(&mut [&mut param]).unwrap();
    let value_after_second = param.data[[0, 0]];

    // With momentum, second step should have larger update than first
    let first_update = 1.0 - value_after_first;
    let second_update = value_after_first - value_after_second;
    assert!(second_update > first_update);
  }

  #[test]
  fn test_sgd_momentum_clone() {
    let optimizer = SGDMomentum::new(0.01, 0.9);
    let cloned = optimizer.clone();

    assert_eq!(optimizer.learning_rate(), cloned.learning_rate());
    assert_eq!(optimizer.momentum(), cloned.momentum());
    // Velocities should be reset in clone
    assert!(cloned.velocities.is_empty());
  }

  #[test]
  fn test_gradient_descent_name() {
    let optimizer = GradientDescent::new(0.1);
    assert_eq!(optimizer.name(), "Gradient Descent");
    assert!(optimizer.requires_full_batch());
  }
}
