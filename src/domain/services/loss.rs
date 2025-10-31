//! Loss functions for training neural networks
//!
//! This module provides different loss functions for computing the error
//! between model predictions and true labels during training.

use crate::core::{Result, Tensor};

/// Trait for loss functions
pub trait Loss {
  /// Compute the loss between predictions and targets
  ///
  /// # Arguments
  /// * `predictions` - Model predictions tensor
  /// * `targets` - True target labels tensor
  ///
  /// # Returns
  /// Scalar loss value
  fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;

  /// Compute the gradient of the loss with respect to predictions
  ///
  /// # Arguments
  /// * `predictions` - Model predictions tensor
  /// * `targets` - True target labels tensor
  ///
  /// # Returns
  /// Gradient tensor with same shape as predictions
  fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;

  /// Get the name of the loss function
  fn name(&self) -> &'static str;
}

/// Binary Cross Entropy Loss
///
/// Binary Cross Entropy (BCE) loss is used for binary classification tasks.
/// The loss is computed as:
///
/// BCE(y, ŷ) = -(1/N) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
///
/// Where:
/// - y is the true binary label (0 or 1)
/// - ŷ is the predicted probability (between 0 and 1)
/// - N is the number of samples
///
/// The gradient with respect to predictions is:
/// ∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ)) / N
#[derive(Debug, Clone)]
pub struct BinaryCrossEntropy {
  /// Small epsilon value to prevent log(0)
  epsilon: f64,
}

impl BinaryCrossEntropy {
  /// Create a new Binary Cross Entropy loss
  ///
  /// # Arguments
  /// * `epsilon` - Small value to add for numerical stability (default: 1e-8)
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let loss_fn = BinaryCrossEntropy::new();
  /// ```
  pub fn new() -> Self {
    Self { epsilon: 1e-8 }
  }

  /// Create a new Binary Cross Entropy loss with custom epsilon
  ///
  /// # Arguments
  /// * `epsilon` - Small value to add for numerical stability
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let loss_fn = BinaryCrossEntropy::with_epsilon(1e-10);
  /// ```
  pub fn with_epsilon(epsilon: f64) -> Self {
    assert!(epsilon > 0.0, "Epsilon must be positive");
    Self { epsilon }
  }

  /// Get the epsilon value used for numerical stability
  pub fn epsilon(&self) -> f64 {
    self.epsilon
  }

  /// Clamp predictions to avoid log(0) and log(1-0) issues
  fn clamp_predictions(&self, predictions: &Tensor) -> Result<Tensor> {
    let min_val = self.epsilon;
    let max_val = 1.0 - self.epsilon;

    let mut clamped_data = predictions.data.clone();
    for i in 0..clamped_data.nrows() {
      for j in 0..clamped_data.ncols() {
        let val = clamped_data[[i, j]];
        if val < min_val {
          clamped_data[[i, j]] = min_val;
        } else if val > max_val {
          clamped_data[[i, j]] = max_val;
        }
      }
    }

    Tensor::from_data(clamped_data)
  }
}

impl Default for BinaryCrossEntropy {
  fn default() -> Self {
    Self::new()
  }
}

impl Loss for BinaryCrossEntropy {
  fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // Clamp predictions to prevent log(0) and log(1)
    let clamped_preds = self.clamp_predictions(predictions)?;

    // Compute 1 - predictions
    let (rows, cols) = predictions.shape();
    let one_tensor = Tensor::ones(rows, cols);
    let one_minus_preds = one_tensor.sub(&clamped_preds)?;

    // Compute 1 - targets
    let one_minus_targets = one_tensor.sub(targets)?;

    // Log terms: targets * log(predictions) + (1-targets) * log(1-predictions)
    let log_preds = clamped_preds.log()?;
    let log_one_minus_preds = one_minus_preds.log()?;

    let term1 = targets.mul(&log_preds)?;
    let term2 = one_minus_targets.mul(&log_one_minus_preds)?;

    let log_likelihood = term1.add(&term2)?;

    // Take negative mean: -(1/N) * sum(log_likelihood)
    let mean_log_likelihood = log_likelihood.mean()?;
    mean_log_likelihood.mul_scalar(-1.0)
  }

  fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // Clamp predictions to prevent division by 0
    let clamped_preds = self.clamp_predictions(predictions)?;

    // Compute 1 - predictions and 1 - targets
    let (rows, cols) = predictions.shape();
    let one_tensor = Tensor::ones(rows, cols);
    let one_minus_preds = one_tensor.sub(&clamped_preds)?;
    let one_minus_targets = one_tensor.sub(targets)?;

    // Gradient: -(targets/predictions - (1-targets)/(1-predictions)) / batch_size
    let term1 = targets.div(&clamped_preds)?;
    let term2 = one_minus_targets.div(&one_minus_preds)?;
    let gradient = term1.sub(&term2)?;

    let batch_size = predictions.shape().0 as f64;
    gradient.mul_scalar(-1.0 / batch_size)
  }

  fn name(&self) -> &'static str {
    "BinaryCrossEntropy"
  }
}

/// Mean Squared Error Loss
///
/// MSE loss is computed as:
/// MSE(y, ŷ) = (1/N) * Σ(y - ŷ)²
///
/// The gradient with respect to predictions is:
/// ∂MSE/∂ŷ = -2(y - ŷ) / N
#[derive(Debug, Clone, Default)]
pub struct MeanSquaredError;

impl MeanSquaredError {
  /// Create a new Mean Squared Error loss
  ///
  /// # Examples
  /// ```
  /// use multilayer_perceptron::prelude::*;
  ///
  /// let loss_fn = MeanSquaredError::new();
  /// ```
  pub fn new() -> Self {
    Self
  }
}

impl Loss for MeanSquaredError {
  fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // (y - ŷ)²
    let diff = targets.sub(predictions)?;
    let squared_diff = diff.mul(&diff)?;

    // Mean of squared differences
    squared_diff.mean()
  }

  fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    assert_eq!(
      predictions.shape(),
      targets.shape(),
      "Predictions and targets must have the same shape"
    );

    // -2(y - ŷ) / N
    let total_elements = (predictions.shape().0 * predictions.shape().1) as f64;
    let diff = targets.sub(predictions)?;
    diff.mul_scalar(-2.0 / total_elements)
  }

  fn name(&self) -> &'static str {
    "MeanSquaredError"
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;

  #[test]
  fn test_binary_cross_entropy_creation() {
    let loss_fn = BinaryCrossEntropy::new();
    assert_eq!(loss_fn.epsilon(), 1e-8);
    assert_eq!(loss_fn.name(), "BinaryCrossEntropy");

    let loss_fn_custom = BinaryCrossEntropy::with_epsilon(1e-10);
    assert_eq!(loss_fn_custom.epsilon(), 1e-10);
  }

  #[test]
  #[should_panic(expected = "Epsilon must be positive")]
  fn test_binary_cross_entropy_negative_epsilon() {
    BinaryCrossEntropy::with_epsilon(-1e-8);
  }

  #[test]
  fn test_binary_cross_entropy_perfect_predictions() {
    let loss_fn = BinaryCrossEntropy::new();

    // Perfect predictions (with epsilon to avoid log(0))
    let predictions = Tensor::new(vec![vec![0.99999999, 0.00000001]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // Loss should be very close to 0 for perfect predictions
    assert!(loss.data[[0, 0]] < 1e-6);
  }

  #[test]
  fn test_binary_cross_entropy_worst_predictions() {
    let loss_fn = BinaryCrossEntropy::with_epsilon(1e-7);

    // Worst predictions (opposite of targets)
    let predictions = Tensor::new(vec![vec![0.0000001, 0.9999999]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // Loss should be large for bad predictions
    assert!(loss.data[[0, 0]] > 10.0);
  }

  #[test]
  fn test_binary_cross_entropy_forward() {
    let loss_fn = BinaryCrossEntropy::new();

    let predictions = Tensor::new(vec![vec![0.8, 0.3], vec![0.9, 0.1]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0], vec![1.0, 0.0]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // Verify loss is scalar and positive
    assert_eq!(loss.shape(), (1, 1));
    assert!(loss.data[[0, 0]] >= 0.0);
  }

  #[test]
  fn test_binary_cross_entropy_backward() {
    let loss_fn = BinaryCrossEntropy::new();

    let predictions = Tensor::new(vec![vec![0.8, 0.3], vec![0.9, 0.1]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 0.0], vec![1.0, 0.0]]).unwrap();

    let gradient = loss_fn.backward(&predictions, &targets).unwrap();

    // Gradient should have same shape as predictions
    assert_eq!(gradient.shape(), predictions.shape());

    // Check gradient signs: for target=1, pred=0.8, gradient should be negative
    // (encouraging higher prediction)
    assert!(gradient.data[[0, 0]] < 0.0);

    // For target=0, pred=0.3, gradient should be positive
    // (encouraging lower prediction)
    assert!(gradient.data[[0, 1]] > 0.0);
  }

  #[test]
  fn test_binary_cross_entropy_numerical_stability() {
    let loss_fn = BinaryCrossEntropy::new();

    // Test with extreme values that could cause log(0)
    let predictions = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();
    let targets = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();

    // Should not panic due to epsilon clamping
    let loss = loss_fn.forward(&predictions, &targets).unwrap();
    let gradient = loss_fn.backward(&predictions, &targets).unwrap();

    assert!(loss.data[[0, 0]].is_finite());
    assert!(gradient.data[[0, 0]].is_finite());
    assert!(gradient.data[[0, 1]].is_finite());
  }

  #[test]
  #[should_panic(expected = "Predictions and targets must have the same shape")]
  fn test_binary_cross_entropy_shape_mismatch() {
    let loss_fn = BinaryCrossEntropy::new();

    let predictions = Tensor::new(vec![vec![0.8, 0.3]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0]]).unwrap();

    loss_fn.forward(&predictions, &targets).unwrap();
  }

  #[test]
  fn test_mean_squared_error_creation() {
    let loss_fn = MeanSquaredError::new();
    assert_eq!(loss_fn.name(), "MeanSquaredError");

    let loss_fn_default = MeanSquaredError::new();
    assert_eq!(loss_fn_default.name(), "MeanSquaredError");
  }

  #[test]
  fn test_mean_squared_error_forward() {
    let loss_fn = MeanSquaredError::new();

    let predictions = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.5, 2.5], vec![2.5, 3.5]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();

    // MSE = mean((targets - predictions)²) = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
    assert_eq!(loss.shape(), (1, 1));
    assert_abs_diff_eq!(loss.data[[0, 0]], 0.25, epsilon = 1e-6);
  }

  #[test]
  fn test_mean_squared_error_backward() {
    let loss_fn = MeanSquaredError::new();

    let predictions = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.5, 2.5], vec![2.5, 3.5]]).unwrap();

    let gradient = loss_fn.backward(&predictions, &targets).unwrap();

    // Gradient = -2(y - ŷ) / N = -2 * [0.5, 0.5, -0.5, -0.5] / 4 = [-0.25, -0.25, 0.25, 0.25]
    assert_eq!(gradient.shape(), predictions.shape());
    assert_abs_diff_eq!(gradient.data[[0, 0]], -0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[0, 1]], -0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[1, 0]], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[1, 1]], 0.25, epsilon = 1e-6);
  }

  #[test]
  fn test_mean_squared_error_perfect_predictions() {
    let loss_fn = MeanSquaredError::new();

    let predictions = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();

    let loss = loss_fn.forward(&predictions, &targets).unwrap();
    let gradient = loss_fn.backward(&predictions, &targets).unwrap();

    // Perfect predictions should have zero loss and gradient
    assert_abs_diff_eq!(loss.data[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[0, 0]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(gradient.data[[0, 1]], 0.0, epsilon = 1e-6);
  }

  #[test]
  #[should_panic(expected = "Predictions and targets must have the same shape")]
  fn test_mean_squared_error_shape_mismatch() {
    let loss_fn = MeanSquaredError::new();

    let predictions = Tensor::new(vec![vec![1.0, 2.0]]).unwrap();
    let targets = Tensor::new(vec![vec![1.0]]).unwrap();

    loss_fn.forward(&predictions, &targets).unwrap();
  }
}
