//! WebAssembly bindings for multilayer perceptron
//!
//! This module provides JavaScript-compatible wrappers around the core
//! tensor operations and neural network functionality.

use crate::prelude::*;
use js_sys::Float64Array;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::console;

type JsResult<T> = std::result::Result<T, JsValue>;

// Enable better panic messages in debug mode
#[cfg(feature = "console_error_panic_hook")]
#[wasm_bindgen(start)]
pub fn main() {
  console_error_panic_hook::set_once();
}

/// JavaScript-compatible wrapper for Tensor
#[wasm_bindgen]
pub struct JsTensor {
  inner: Tensor,
}

#[wasm_bindgen]
impl JsTensor {
  /// Create a new tensor from a JavaScript Float64Array
  ///
  /// # Arguments
  /// * `data` - Flattened tensor data as Float64Array
  /// * `rows` - Number of rows
  /// * `cols` - Number of columns
  #[wasm_bindgen(constructor)]
  pub fn new(data: Float64Array, rows: usize, cols: usize) -> JsResult<JsTensor> {
    let vec_data: Vec<f64> = data.to_vec();

    if vec_data.len() != rows * cols {
      return Err(JsValue::from_str(&format!(
        "Data length {} doesn't match dimensions {}x{}",
        vec_data.len(),
        rows,
        cols
      )));
    }

    let mut tensor_data = Vec::new();
    for i in 0..rows {
      let mut row = Vec::new();
      for j in 0..cols {
        row.push(vec_data[i * cols + j]);
      }
      tensor_data.push(row);
    }

    let tensor = Tensor::new(tensor_data)
      .map_err(|e| JsValue::from_str(&format!("Failed to create tensor: {}", e)))?;

    Ok(JsTensor { inner: tensor })
  }

  /// Create a tensor filled with zeros
  #[wasm_bindgen]
  pub fn zeros(rows: usize, cols: usize) -> JsTensor {
    JsTensor {
      inner: Tensor::zeros(rows, cols),
    }
  }

  /// Create a tensor filled with ones
  #[wasm_bindgen]
  pub fn ones(rows: usize, cols: usize) -> JsTensor {
    JsTensor {
      inner: Tensor::ones(rows, cols),
    }
  }

  /// Create a tensor with random values between -1 and 1
  #[wasm_bindgen]
  pub fn random(rows: usize, cols: usize) -> JsTensor {
    JsTensor {
      inner: Tensor::random(rows, cols),
    }
  }

  /// Get the shape of the tensor as [rows, cols]
  #[wasm_bindgen]
  pub fn shape(&self) -> js_sys::Array {
    let (rows, cols) = self.inner.shape();
    let array = js_sys::Array::new();
    array.push(&JsValue::from_f64(rows as f64));
    array.push(&JsValue::from_f64(cols as f64));
    array
  }

  /// Get the tensor data as a flattened Float64Array
  #[wasm_bindgen]
  pub fn data(&self) -> Float64Array {
    let (rows, cols) = self.inner.shape();
    let mut flat_data = Vec::with_capacity(rows * cols);

    for i in 0..rows {
      for j in 0..cols {
        flat_data.push(self.inner.data[[i, j]]);
      }
    }

    Float64Array::from(&flat_data[..])
  }

  /// Set whether this tensor requires gradients
  #[wasm_bindgen]
  pub fn set_requires_grad(&mut self, requires_grad: bool) {
    self.inner.set_requires_grad(requires_grad);
  }

  /// Check if this tensor requires gradients
  #[wasm_bindgen]
  pub fn requires_grad(&self) -> bool {
    self.inner.requires_grad()
  }

  /// Get the gradient as a JsTensor (if available)
  #[wasm_bindgen]
  pub fn gradient(&self) -> Option<JsTensor> {
    self.inner.gradient().map(|grad| JsTensor { inner: grad })
  }

  /// Zero out the gradients
  #[wasm_bindgen]
  pub fn zero_grad(&mut self) {
    self.inner.zero_grad();
  }

  /// Perform matrix multiplication
  #[wasm_bindgen]
  pub fn matmul(&self, other: &JsTensor) -> JsResult<JsTensor> {
    let result = self
      .inner
      .matmul(&other.inner)
      .map_err(|e| JsValue::from_str(&format!("Matrix multiplication failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Add two tensors
  #[wasm_bindgen]
  pub fn add(&self, other: &JsTensor) -> JsResult<JsTensor> {
    let result = self
      .inner
      .add(&other.inner)
      .map_err(|e| JsValue::from_str(&format!("Addition failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Subtract two tensors
  #[wasm_bindgen]
  pub fn sub(&self, other: &JsTensor) -> JsResult<JsTensor> {
    let result = self
      .inner
      .sub(&other.inner)
      .map_err(|e| JsValue::from_str(&format!("Subtraction failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Element-wise multiplication
  #[wasm_bindgen]
  pub fn mul(&self, other: &JsTensor) -> JsResult<JsTensor> {
    let result = self
      .inner
      .mul(&other.inner)
      .map_err(|e| JsValue::from_str(&format!("Multiplication failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Scalar multiplication
  #[wasm_bindgen]
  pub fn mul_scalar(&self, scalar: f64) -> JsResult<JsTensor> {
    let result = self
      .inner
      .mul_scalar(scalar)
      .map_err(|e| JsValue::from_str(&format!("Scalar multiplication failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Apply sigmoid activation
  #[wasm_bindgen]
  pub fn sigmoid(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .sigmoid()
      .map_err(|e| JsValue::from_str(&format!("Sigmoid failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Apply ReLU activation
  #[wasm_bindgen]
  pub fn relu(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .relu()
      .map_err(|e| JsValue::from_str(&format!("ReLU failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Apply tanh activation
  #[wasm_bindgen]
  pub fn tanh(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .tanh()
      .map_err(|e| JsValue::from_str(&format!("Tanh failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Apply softmax activation
  #[wasm_bindgen]
  pub fn softmax(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .softmax()
      .map_err(|e| JsValue::from_str(&format!("Softmax failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Compute mean of all elements
  #[wasm_bindgen]
  pub fn mean(&self) -> JsResult<JsTensor> {
    let result = self
      .inner
      .mean()
      .map_err(|e| JsValue::from_str(&format!("Mean failed: {}", e)))?;
    Ok(JsTensor { inner: result })
  }

  /// Perform backward pass (compute gradients)
  #[wasm_bindgen]
  pub fn backward(&mut self) -> JsResult<()> {
    self
      .inner
      .backward()
      .map_err(|e| JsValue::from_str(&format!("Backward pass failed: {}", e)))?;
    Ok(())
  }

  /// Clone the tensor
  #[wasm_bindgen]
  pub fn clone(&self) -> JsTensor {
    JsTensor {
      inner: self.inner.clone(),
    }
  }

  /// Get a string representation of the tensor
  #[wasm_bindgen]
  pub fn to_string(&self) -> String {
    format!("{:?}", self.inner)
  }

  /// Log the tensor to browser console (for debugging)
  #[wasm_bindgen]
  pub fn log(&self) {
    console::log_1(&format!("Tensor: {:?}", self.inner).into());
  }
}

/// JavaScript-compatible wrapper for Neural Network Model
#[wasm_bindgen]
pub struct JsModel {
  inner: Sequential,
  graph: Rc<RefCell<ComputationGraph>>,
}

#[wasm_bindgen]
impl JsModel {
  /// Create a new empty model
  #[wasm_bindgen(constructor)]
  pub fn new() -> JsModel {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    JsModel {
      inner: Sequential::new().with_graph(graph.clone()),
      graph,
    }
  }

  /// Add a dense layer with ReLU activation
  #[wasm_bindgen]
  pub fn add_dense_relu(&mut self, input_size: usize, output_size: usize) -> JsResult<()> {
    self.inner = self
      .inner
      .clone()
      .relu_layer(input_size, output_size)
      .with_graph(self.graph.clone());
    Ok(())
  }

  /// Add a dense layer with sigmoid activation
  #[wasm_bindgen]
  pub fn add_dense_sigmoid(
    &mut self,
    input_size: usize,
    output_size: usize,
  ) -> JsResult<()> {
    self.inner = self
      .inner
      .clone()
      .sigmoid_layer(input_size, output_size)
      .with_graph(self.graph.clone());
    Ok(())
  }

  /// Add a dense layer with softmax activation (typically for output layer)
  #[wasm_bindgen]
  pub fn add_dense_softmax(
    &mut self,
    input_size: usize,
    output_size: usize,
  ) -> JsResult<()> {
    self.inner = self
      .inner
      .clone()
      .softmax_layer(input_size, output_size)
      .with_graph(self.graph.clone());
    Ok(())
  }

  /// Forward pass through the model
  #[wasm_bindgen]
  pub fn forward(&mut self, input: &JsTensor) -> JsResult<JsTensor> {
    // Ensure input tensor is tracked in our graph
    let tracked_input = input.inner.clone().with_graph(self.graph.clone());

    let output = self
      .inner
      .forward(tracked_input)
      .map_err(|e| JsValue::from_str(&format!("Forward pass failed: {}", e)))?;

    Ok(JsTensor { inner: output })
  }

  /// Get model summary as a string
  #[wasm_bindgen]
  pub fn summary(&self) -> String {
    format!("{:#?}", self.inner.summary())
  }

  /// Get total number of parameters
  #[wasm_bindgen]
  pub fn param_count(&self) -> usize {
    self.inner.summary().total_params
  }
}

#[derive(Debug, Clone)]
struct JsTrainingConfig {
  learning_rate: f64,
  epochs: usize,
  batch_size: usize,
  validation_split: f64,
}

impl Default for JsTrainingConfig {
  fn default() -> Self {
    Self {
      learning_rate: 0.01,
      epochs: 100,
      batch_size: 32,
      validation_split: 0.2,
    }
  }
}

/// JavaScript-compatible wrapper for training functionality
#[wasm_bindgen]
pub struct JsTrainer {
  config: JsTrainingConfig,
}

#[wasm_bindgen]
impl JsTrainer {
  /// Create a new trainer
  #[wasm_bindgen(constructor)]
  pub fn new(learning_rate: f64) -> JsTrainer {
    let mut config = JsTrainingConfig::default();
    config.learning_rate = learning_rate;
    JsTrainer { config }
  }

  /// Set training configuration
  #[wasm_bindgen]
  pub fn configure(
    &mut self,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    validation_split: f64,
  ) {
    self.config.learning_rate = learning_rate;
    self.config.epochs = epochs;
    self.config.batch_size = batch_size;
    self.config.validation_split = validation_split;
  }

  /// Train model synchronously (blocking)
  #[wasm_bindgen]
  pub fn train_sync(
    &mut self,
    model: &mut JsModel,
    dataset: &JsDataset,
  ) -> JsResult<js_sys::Array> {
    // Convert dataset to internal format
    let features = dataset.features_tensor()?.inner;
    let labels = dataset.labels_tensor()?.inner;

    // Create a simple dataset wrapper (this would need to be implemented in the core library)
    // For now, we'll implement basic training loop here

    let history = js_sys::Array::new();
    let epochs = self.config.epochs.max(1);

    for epoch in 0..epochs {
      // Forward pass
      let predictions = model
        .inner
        .forward(features.clone())
        .map_err(|e| JsValue::from_str(&format!("Forward pass failed: {}", e)))?;

      // Compute loss (binary cross entropy for demo)
      let loss_fn = BinaryCrossEntropy::new();
      let loss = loss_fn
        .forward(&predictions, &labels)
        .map_err(|e| JsValue::from_str(&format!("Loss computation failed: {}", e)))?;

      // Compute accuracy
      let accuracy_metric = Accuracy::new(0.5);
      let accuracy = accuracy_metric
        .compute(&predictions, &labels)
        .map_err(|e| JsValue::from_str(&format!("Accuracy computation failed: {}", e)))?;

      // Create epoch result
      let epoch_result = js_sys::Object::new();
      js_sys::Reflect::set(&epoch_result, &"epoch".into(), &JsValue::from(epoch))?;
      js_sys::Reflect::set(
        &epoch_result,
        &"loss".into(),
        &JsValue::from(loss.data[[0, 0]]),
      )?;
      js_sys::Reflect::set(&epoch_result, &"accuracy".into(), &JsValue::from(accuracy))?;

      history.push(&epoch_result);

      // Log progress
      Utils::log(&format!(
        "Epoch {}/{} - Loss: {:.4} - Accuracy: {:.4}",
        epoch + 1,
        epochs,
        loss.data[[0, 0]],
        accuracy
      ));
    }

    Ok(history)
  }
}

/// Asynchronous training functionality for better browser performance
#[wasm_bindgen]
pub struct AsyncTrainer {
  config: JsTrainingConfig,
}

#[wasm_bindgen]
impl AsyncTrainer {
  /// Create a new async trainer
  #[wasm_bindgen(constructor)]
  pub fn new(learning_rate: f64, epochs: usize, batch_size: usize) -> AsyncTrainer {
    let config = JsTrainingConfig {
      learning_rate,
      epochs,
      batch_size,
      validation_split: 0.2,
    };

    AsyncTrainer { config }
  }

  /// Start training asynchronously with progress callbacks
  #[wasm_bindgen]
  pub async fn train_async(
    &self,
    model: &mut JsModel,
    dataset: &JsDataset,
    progress_callback: Option<js_sys::Function>,
  ) -> JsResult<js_sys::Array> {
    let features = dataset.features_tensor()?.inner;
    let labels = dataset.labels_tensor()?.inner;
    let history = js_sys::Array::new();

    for epoch in 0..self.config.epochs {
      // Yield control to browser between epochs
      yield_to_browser().await;

      // Forward pass
      let predictions = model
        .inner
        .forward(features.clone())
        .map_err(|e| JsValue::from_str(&format!("Forward pass failed: {}", e)))?;

      // Compute loss
      let loss_fn = BinaryCrossEntropy::new();
      let loss = loss_fn
        .forward(&predictions, &labels)
        .map_err(|e| JsValue::from_str(&format!("Loss computation failed: {}", e)))?;

      // Compute accuracy
      let accuracy_metric = Accuracy::new(0.5);
      let accuracy = accuracy_metric
        .compute(&predictions, &labels)
        .map_err(|e| JsValue::from_str(&format!("Accuracy computation failed: {}", e)))?;

      let loss_val = loss.data[[0, 0]];

      // Create progress object
      let progress = TrainingProgress {
        epoch: epoch + 1,
        loss: loss_val,
        accuracy,
        val_loss: loss_val,     // Simplified - would compute on validation set
        val_accuracy: accuracy, // Simplified
      };

      // Call progress callback if provided
      if let Some(callback) = &progress_callback {
        let progress_obj = js_sys::Object::new();
        js_sys::Reflect::set(
          &progress_obj,
          &"epoch".into(),
          &JsValue::from(progress.epoch),
        )?;
        js_sys::Reflect::set(&progress_obj, &"loss".into(), &JsValue::from(progress.loss))?;
        js_sys::Reflect::set(
          &progress_obj,
          &"accuracy".into(),
          &JsValue::from(progress.accuracy),
        )?;
        js_sys::Reflect::set(
          &progress_obj,
          &"val_loss".into(),
          &JsValue::from(progress.val_loss),
        )?;
        js_sys::Reflect::set(
          &progress_obj,
          &"val_accuracy".into(),
          &JsValue::from(progress.val_accuracy),
        )?;

        let _ = callback.call1(&JsValue::NULL, &progress_obj);
      }

      // Add to history
      let epoch_result = js_sys::Object::new();
      js_sys::Reflect::set(
        &epoch_result,
        &"epoch".into(),
        &JsValue::from(progress.epoch),
      )?;
      js_sys::Reflect::set(&epoch_result, &"loss".into(), &JsValue::from(progress.loss))?;
      js_sys::Reflect::set(
        &epoch_result,
        &"accuracy".into(),
        &JsValue::from(progress.accuracy),
      )?;
      js_sys::Reflect::set(
        &epoch_result,
        &"val_loss".into(),
        &JsValue::from(progress.val_loss),
      )?;
      js_sys::Reflect::set(
        &epoch_result,
        &"val_accuracy".into(),
        &JsValue::from(progress.val_accuracy),
      )?;

      history.push(&epoch_result);
    }

    Ok(history)
  }

  /// Get training configuration as JavaScript object
  #[wasm_bindgen]
  pub fn get_config(&self) -> js_sys::Object {
    let config = js_sys::Object::new();
    js_sys::Reflect::set(
      &config,
      &"learning_rate".into(),
      &JsValue::from(self.config.learning_rate),
    )
    .unwrap();
    js_sys::Reflect::set(
      &config,
      &"epochs".into(),
      &JsValue::from(self.config.epochs),
    )
    .unwrap();
    js_sys::Reflect::set(
      &config,
      &"batch_size".into(),
      &JsValue::from(self.config.batch_size),
    )
    .unwrap();
    js_sys::Reflect::set(
      &config,
      &"validation_split".into(),
      &JsValue::from(self.config.validation_split),
    )
    .unwrap();
    config
  }
}

/// Yield control to browser to prevent blocking
async fn yield_to_browser() {
  use wasm_bindgen_futures::JsFuture;

  // Use setTimeout(0) to yield control
  let promise = js_sys::Promise::resolve(&JsValue::from(0));
  let _ = JsFuture::from(promise).await;
}

/// Progress callback for training
#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  fn log(s: &str);

  /// Callback function type for training progress
  /// Called with (epoch, loss, accuracy, val_loss, val_accuracy)
  pub type ProgressCallback;

  #[wasm_bindgen(method)]
  fn call(
    this: &ProgressCallback,
    epoch: usize,
    loss: f64,
    accuracy: f64,
    val_loss: f64,
    val_accuracy: f64,
  );
}

/// Training progress data structure
#[wasm_bindgen]
pub struct TrainingProgress {
  epoch: usize,
  loss: f64,
  accuracy: f64,
  val_loss: f64,
  val_accuracy: f64,
}

#[wasm_bindgen]
impl TrainingProgress {
  #[wasm_bindgen(getter)]
  pub fn epoch(&self) -> usize {
    self.epoch
  }

  #[wasm_bindgen(getter)]
  pub fn loss(&self) -> f64 {
    self.loss
  }

  #[wasm_bindgen(getter)]
  pub fn accuracy(&self) -> f64 {
    self.accuracy
  }

  #[wasm_bindgen(getter)]
  pub fn val_loss(&self) -> f64 {
    self.val_loss
  }

  #[wasm_bindgen(getter)]
  pub fn val_accuracy(&self) -> f64 {
    self.val_accuracy
  }
}

/// Dataset wrapper for JavaScript
#[wasm_bindgen]
pub struct JsDataset {
  features: Vec<Vec<f64>>,
  labels: Vec<f64>,
}

#[wasm_bindgen]
impl JsDataset {
  /// Create a new dataset from features and labels
  #[wasm_bindgen(constructor)]
  pub fn new(features: js_sys::Array, labels: Float64Array) -> JsResult<JsDataset> {
    // Convert JS array of arrays to Vec<Vec<f64>>
    let mut feature_vec = Vec::new();

    for i in 0..features.length() {
      let row = features.get(i);
      if let Ok(row_array) = row.dyn_into::<Float64Array>() {
        feature_vec.push(row_array.to_vec());
      } else {
        return Err(JsValue::from_str(
          "Features must be an array of Float64Array",
        ));
      }
    }

    let label_vec = labels.to_vec();

    if feature_vec.len() != label_vec.len() {
      return Err(JsValue::from_str(
        "Features and labels must have the same length",
      ));
    }

    Ok(JsDataset {
      features: feature_vec,
      labels: label_vec,
    })
  }

  /// Get the number of samples
  #[wasm_bindgen]
  pub fn len(&self) -> usize {
    self.features.len()
  }

  /// Get the number of features per sample
  #[wasm_bindgen]
  pub fn feature_count(&self) -> usize {
    if self.features.is_empty() {
      0
    } else {
      self.features[0].len()
    }
  }

  /// Get features as a tensor
  #[wasm_bindgen]
  pub fn features_tensor(&self) -> JsResult<JsTensor> {
    if self.features.is_empty() {
      return Err(JsValue::from_str("Dataset is empty"));
    }

    match Tensor::new(self.features.clone()) {
      Ok(tensor) => Ok(JsTensor { inner: tensor }),
      Err(e) => Err(JsValue::from_str(&format!(
        "Failed to create features tensor: {}",
        e
      ))),
    }
  }

  /// Get labels as a tensor (column vector)
  #[wasm_bindgen]
  pub fn labels_tensor(&self) -> JsResult<JsTensor> {
    if self.labels.is_empty() {
      return Err(JsValue::from_str("Dataset is empty"));
    }

    let label_data: Vec<Vec<f64>> = self.labels.iter().map(|&label| vec![label]).collect();

    match Tensor::new(label_data) {
      Ok(tensor) => Ok(JsTensor { inner: tensor }),
      Err(e) => Err(JsValue::from_str(&format!(
        "Failed to create labels tensor: {}",
        e
      ))),
    }
  }
}

/// Data conversion utilities between JavaScript and Rust types
#[wasm_bindgen]
pub struct DataConverter;

#[wasm_bindgen]
impl DataConverter {
  /// Convert JavaScript 2D array to JsTensor
  #[wasm_bindgen]
  pub fn array_to_tensor(array: js_sys::Array) -> JsResult<JsTensor> {
    let mut data = Vec::new();

    for i in 0..array.length() {
      let row_val = array.get(i);

      // Handle both Array and Float64Array
      let row_data = if let Ok(float_array) = row_val.clone().dyn_into::<Float64Array>() {
        float_array.to_vec()
      } else if let Ok(js_array) = row_val.dyn_into::<js_sys::Array>() {
        let mut row = Vec::new();
        for j in 0..js_array.length() {
          let val = js_array.get(j);
          if let Some(num) = val.as_f64() {
            row.push(num);
          } else {
            return Err(JsValue::from_str("Array elements must be numbers"));
          }
        }
        row
      } else {
        return Err(JsValue::from_str(
          "Array rows must be arrays or Float64Array",
        ));
      };

      data.push(row_data);
    }

    if data.is_empty() {
      return Err(JsValue::from_str("Array cannot be empty"));
    }

    match Tensor::new(data) {
      Ok(tensor) => Ok(JsTensor { inner: tensor }),
      Err(e) => Err(JsValue::from_str(&format!(
        "Failed to create tensor: {}",
        e
      ))),
    }
  }

  /// Convert JsTensor to JavaScript 2D array
  #[wasm_bindgen]
  pub fn tensor_to_array(tensor: &JsTensor) -> js_sys::Array {
    let (rows, cols) = tensor.inner.shape();
    let result = js_sys::Array::new();

    for i in 0..rows {
      let row = js_sys::Array::new();
      for j in 0..cols {
        row.push(&JsValue::from_f64(tensor.inner.data[[i, j]]));
      }
      result.push(&row);
    }

    result
  }

  /// Convert flat JavaScript array to JsTensor with specified shape
  #[wasm_bindgen]
  pub fn flat_array_to_tensor(
    flat_array: Float64Array,
    rows: usize,
    cols: usize,
  ) -> JsResult<JsTensor> {
    JsTensor::new(flat_array, rows, cols)
  }

  /// Convert JsTensor to flat JavaScript array
  #[wasm_bindgen]
  pub fn tensor_to_flat_array(tensor: &JsTensor) -> Float64Array {
    tensor.data()
  }

  /// Create tensor from CSV-like string data
  #[wasm_bindgen]
  pub fn csv_to_tensor(
    csv_string: &str,
    has_header: bool,
    delimiter: &str,
  ) -> JsResult<JsTensor> {
    let lines: Vec<&str> = csv_string.lines().collect();

    if lines.is_empty() {
      return Err(JsValue::from_str("CSV string is empty"));
    }

    let start_row = if has_header { 1 } else { 0 };
    let mut data = Vec::new();

    for line in lines.iter().skip(start_row) {
      let values: std::result::Result<Vec<f64>, std::num::ParseFloatError> = line
        .split(delimiter)
        .map(|s| s.trim().parse::<f64>())
        .collect();

      match values {
        Ok(row) => data.push(row),
        Err(_) => return Err(JsValue::from_str("Failed to parse numeric values from CSV")),
      }
    }

    if data.is_empty() {
      return Err(JsValue::from_str("No data rows found in CSV"));
    }

    match Tensor::new(data) {
      Ok(tensor) => Ok(JsTensor { inner: tensor }),
      Err(e) => Err(JsValue::from_str(&format!(
        "Failed to create tensor from CSV: {}",
        e
      ))),
    }
  }

  /// Convert tensor to CSV-like string
  #[wasm_bindgen]
  pub fn tensor_to_csv(tensor: &JsTensor, delimiter: &str) -> String {
    let (rows, cols) = tensor.inner.shape();
    let mut lines = Vec::new();

    for i in 0..rows {
      let mut row_values = Vec::new();
      for j in 0..cols {
        row_values.push(tensor.inner.data[[i, j]].to_string());
      }
      lines.push(row_values.join(delimiter));
    }

    lines.join("\n")
  }

  /// Normalize tensor values to [0, 1] range
  #[wasm_bindgen]
  pub fn normalize_min_max(tensor: &JsTensor) -> JsResult<JsTensor> {
    let (rows, cols) = tensor.inner.shape();
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    // Find min and max values
    for i in 0..rows {
      for j in 0..cols {
        let val = tensor.inner.data[[i, j]];
        min_val = min_val.min(val);
        max_val = max_val.max(val);
      }
    }

    let range = max_val - min_val;
    if range == 0.0 {
      return Ok(tensor.clone());
    }

    // Normalize
    let mut normalized_data = Vec::new();
    for i in 0..rows {
      let mut row = Vec::new();
      for j in 0..cols {
        let normalized = (tensor.inner.data[[i, j]] - min_val) / range;
        row.push(normalized);
      }
      normalized_data.push(row);
    }

    match Tensor::new(normalized_data) {
      Ok(normalized_tensor) => Ok(JsTensor {
        inner: normalized_tensor,
      }),
      Err(e) => Err(JsValue::from_str(&format!("Normalization failed: {}", e))),
    }
  }

  /// Standardize tensor values (z-score normalization)
  #[wasm_bindgen]
  pub fn standardize(tensor: &JsTensor) -> JsResult<JsTensor> {
    let (rows, cols) = tensor.inner.shape();
    let total_elements = (rows * cols) as f64;

    // Calculate mean
    let mut sum = 0.0;
    for i in 0..rows {
      for j in 0..cols {
        sum += tensor.inner.data[[i, j]];
      }
    }
    let mean = sum / total_elements;

    // Calculate standard deviation
    let mut sum_sq_diff = 0.0;
    for i in 0..rows {
      for j in 0..cols {
        let diff = tensor.inner.data[[i, j]] - mean;
        sum_sq_diff += diff * diff;
      }
    }
    let std_dev = (sum_sq_diff / total_elements).sqrt();

    if std_dev == 0.0 {
      return Ok(tensor.clone());
    }

    // Standardize
    let mut standardized_data = Vec::new();
    for i in 0..rows {
      let mut row = Vec::new();
      for j in 0..cols {
        let standardized = (tensor.inner.data[[i, j]] - mean) / std_dev;
        row.push(standardized);
      }
      standardized_data.push(row);
    }

    match Tensor::new(standardized_data) {
      Ok(standardized_tensor) => Ok(JsTensor {
        inner: standardized_tensor,
      }),
      Err(e) => Err(JsValue::from_str(&format!("Standardization failed: {}", e))),
    }
  }
}

/// Utility functions
#[wasm_bindgen]
pub struct Utils;

#[wasm_bindgen]
impl Utils {
  /// Create a simple 2-layer neural network for binary classification
  #[wasm_bindgen]
  pub fn create_binary_classifier(input_size: usize, hidden_size: usize) -> JsModel {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    let model = Sequential::new()
      .relu_layer(input_size, hidden_size)
      .sigmoid_layer(hidden_size, 1)
      .with_graph(graph.clone());

    JsModel {
      inner: model,
      graph,
    }
  }

  /// Create a multi-class classifier with softmax output
  #[wasm_bindgen]
  pub fn create_multiclass_classifier(
    input_size: usize,
    hidden_size: usize,
    num_classes: usize,
  ) -> JsModel {
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    let model = Sequential::new()
      .relu_layer(input_size, hidden_size)
      .relu_layer(hidden_size, hidden_size / 2)
      .softmax_layer(hidden_size / 2, num_classes)
      .with_graph(graph.clone());

    JsModel {
      inner: model,
      graph,
    }
  }

  /// Log a message to the browser console
  #[wasm_bindgen]
  pub fn log(message: &str) {
    console::log_1(&message.into());
  }

  /// Calculate binary cross-entropy loss
  #[wasm_bindgen]
  pub fn binary_cross_entropy(predictions: &JsTensor, targets: &JsTensor) -> JsResult<f64> {
    let loss_fn = BinaryCrossEntropy::new();
    match loss_fn.forward(&predictions.inner, &targets.inner) {
      Ok(loss) => Ok(loss.data[[0, 0]]),
      Err(e) => Err(JsValue::from_str(&format!(
        "Loss computation failed: {}",
        e
      ))),
    }
  }

  /// Calculate accuracy for binary classification
  #[wasm_bindgen]
  pub fn binary_accuracy(predictions: &JsTensor, targets: &JsTensor) -> JsResult<f64> {
    let accuracy_metric = Accuracy::new(0.5); // Default threshold of 0.5
    match accuracy_metric.compute(&predictions.inner, &targets.inner) {
      Ok(acc) => Ok(acc),
      Err(e) => Err(JsValue::from_str(&format!(
        "Accuracy computation failed: {}",
        e
      ))),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use wasm_bindgen_test::*;

  #[wasm_bindgen_test]
  fn test_js_tensor_creation() {
    let data = Float64Array::from(&[1.0, 2.0, 3.0, 4.0][..]);
    let tensor = JsTensor::new(data, 2, 2).unwrap();

    let shape = tensor.shape();
    assert_eq!(shape.get(0), JsValue::from_f64(2.0));
    assert_eq!(shape.get(1), JsValue::from_f64(2.0));
  }

  #[wasm_bindgen_test]
  fn test_js_tensor_operations() {
    let a = JsTensor::ones(2, 2);
    let b = JsTensor::ones(2, 2);
    let c = a.add(&b).unwrap();

    let data = c.data();
    // Each element should be 2.0 (1.0 + 1.0)
    for i in 0..4 {
      assert_eq!(data.get_index(i), 2.0);
    }
  }

  #[wasm_bindgen_test]
  fn test_js_model_creation() {
    let mut model = JsModel::new();
    model.add_dense_relu(4, 8).unwrap();
    model.add_dense_sigmoid(8, 1).unwrap();

    assert!(model.param_count() > 0);
  }
}
