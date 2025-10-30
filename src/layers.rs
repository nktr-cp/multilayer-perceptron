use crate::error::Result;
use crate::tensor::Tensor;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
  None,
  Sigmoid,
  ReLU,
  Tanh,
  Softmax,
}

impl Activation {
  pub fn apply(&self, input: &Tensor) -> Result<Tensor> {
    match self {
      Activation::None => Ok(input.clone()),
      Activation::Sigmoid => input.sigmoid(),
      Activation::ReLU => input.relu(),
      Activation::Tanh => input.tanh(),
      Activation::Softmax => input.softmax(),
    }
  }

  pub fn name(&self) -> &'static str {
    match self {
      Activation::None => "none",
      Activation::Sigmoid => "sigmoid",
      Activation::ReLU => "relu",
      Activation::Tanh => "tanh",
      Activation::Softmax => "softmax",
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightInit {
  Zeros,
  Ones,
  Random,
  XavierUniform,
  HeUniform,
}

pub trait Layer: Debug {
  fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
  fn output_shape(&self, input_shape: (usize, usize)) -> (usize, usize);
  fn param_count(&self) -> usize;
  fn name(&self) -> &'static str;
  fn clone_layer(&self) -> Box<dyn Layer>;
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

  /// Get mutable references to parameters (for optimizer)
  fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
    Vec::new()
  }

  /// Zero out gradients for this layer
  fn zero_gradients(&mut self) {
    for param in self.parameters_mut() {
      param.zero_grad();
    }
  }

  /// Sync gradients from computation graph
  fn sync_gradients(&mut self) -> Result<()> {
    for param in self.parameters_mut() {
      param.sync_gradient_from_graph()?;
    }
    Ok(())
  }
}

impl Clone for Box<dyn Layer> {
  fn clone(&self) -> Self {
    self.clone_layer()
  }
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
  weights: Tensor,
  bias: Tensor,
  activation: Activation,
  input_size: usize,
  output_size: usize,
}

impl DenseLayer {
  pub fn new(
    input_size: usize,
    output_size: usize,
    activation: Activation,
    weight_init: WeightInit,
  ) -> Self {
    let (weights, bias) = Self::initialize_params(input_size, output_size, weight_init);

    Self {
      weights,
      bias,
      activation,
      input_size,
      output_size,
    }
  }

  pub fn with_activation(input_size: usize, output_size: usize, activation: Activation) -> Self {
    Self::new(
      input_size,
      output_size,
      activation,
      WeightInit::XavierUniform,
    )
  }

  pub fn relu(input_size: usize, output_size: usize) -> Self {
    Self::new(
      input_size,
      output_size,
      Activation::ReLU,
      WeightInit::HeUniform,
    )
  }

  pub fn sigmoid(input_size: usize, output_size: usize) -> Self {
    Self::new(
      input_size,
      output_size,
      Activation::Sigmoid,
      WeightInit::XavierUniform,
    )
  }

  pub fn softmax(input_size: usize, output_size: usize) -> Self {
    Self::new(
      input_size,
      output_size,
      Activation::Softmax,
      WeightInit::XavierUniform,
    )
  }

  fn initialize_params(
    input_size: usize,
    output_size: usize,
    init_method: WeightInit,
  ) -> (Tensor, Tensor) {
    use rand::Rng;

    let weights = match init_method {
      WeightInit::Zeros => Tensor::zeros(input_size, output_size),
      WeightInit::Ones => Tensor::ones(input_size, output_size),
      WeightInit::Random => Tensor::random(input_size, output_size),
      WeightInit::XavierUniform => {
        // Xavier/Glorot initialization: uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..input_size)
          .map(|_| {
            (0..output_size)
              .map(|_| rng.gen_range(-limit..limit))
              .collect()
          })
          .collect();
        Tensor::new(data).expect("Failed to create Xavier initialized weights")
      }
      WeightInit::HeUniform => {
        // He initialization: uniform(-√(6/fan_in), √(6/fan_in))
        let limit = (6.0 / input_size as f64).sqrt();
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..input_size)
          .map(|_| {
            (0..output_size)
              .map(|_| rng.gen_range(-limit..limit))
              .collect()
          })
          .collect();
        Tensor::new(data).expect("Failed to create He initialized weights")
      }
    };

    // Bias is typically initialized to zeros
    let bias = Tensor::zeros(1, output_size);

    (weights, bias)
  }

  pub fn enable_gradients(&mut self) {
    self.weights.set_requires_grad(true);
    self.bias.set_requires_grad(true);
  }

  /// Connect layer parameters to computation graph
  pub fn connect_to_graph(
    &mut self,
    graph: std::rc::Rc<std::cell::RefCell<crate::graph::ComputationGraph>>,
  ) {
    // Ensure tensors require gradients before registering them with the graph so the
    // graph copies also retain the proper requires_grad flag.
    self.weights.set_requires_grad(true);
    self.bias.set_requires_grad(true);

    self.weights = self.weights.clone().with_graph(graph.clone());
    self.bias = self.bias.clone().with_graph(graph.clone());
  }
}

impl Layer for DenseLayer {
  fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
    if !self.weights.requires_grad() {
      self.enable_gradients();
    }

    let linear_output = input.matmul(&self.weights)?;
    let with_bias = linear_output.add(&self.bias)?;

    self.activation.apply(&with_bias)
  }

  fn output_shape(&self, input_shape: (usize, usize)) -> (usize, usize) {
    (input_shape.0, self.output_size)
  }

  fn param_count(&self) -> usize {
    (self.input_size * self.output_size) + self.output_size
  }

  fn name(&self) -> &'static str {
    "DenseLayer"
  }

  fn clone_layer(&self) -> Box<dyn Layer> {
    Box::new(self.clone())
  }

  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }

  fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
    vec![&mut self.weights, &mut self.bias]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;

  #[test]
  fn test_activation_enum() {
    assert_eq!(Activation::Sigmoid.name(), "sigmoid");
    assert_eq!(Activation::ReLU.name(), "relu");
    assert_eq!(Activation::Softmax.name(), "softmax");
    assert_eq!(Activation::None.name(), "none");
    assert_eq!(Activation::Tanh.name(), "tanh");
  }

  #[test]
  fn test_dense_layer_creation() {
    let layer = DenseLayer::new(10, 5, Activation::ReLU, WeightInit::XavierUniform);

    assert_eq!(layer.input_size, 10);
    assert_eq!(layer.output_size, 5);
    assert_eq!(layer.activation, Activation::ReLU);
    assert_eq!(layer.weights.shape(), (10, 5));
    assert_eq!(layer.bias.shape(), (1, 5));
  }

  #[test]
  fn test_dense_layer_convenience_constructors() {
    let relu_layer = DenseLayer::relu(784, 128);
    assert_eq!(relu_layer.activation, Activation::ReLU);
    assert_eq!(relu_layer.input_size, 784);
    assert_eq!(relu_layer.output_size, 128);

    let sigmoid_layer = DenseLayer::sigmoid(128, 64);
    assert_eq!(sigmoid_layer.activation, Activation::Sigmoid);

    let softmax_layer = DenseLayer::softmax(64, 10);
    assert_eq!(softmax_layer.activation, Activation::Softmax);
  }

  #[test]
  fn test_dense_layer_forward() {
    use crate::graph::ComputationGraph;
    use std::cell::RefCell;
    use std::rc::Rc;

    let mut layer = DenseLayer::new(3, 2, Activation::None, WeightInit::Ones);

    // Create input tensor
    let input_data = vec![vec![1.0, 2.0, 3.0]]; // batch size 1, 3 features
    let mut input = Tensor::new(input_data).unwrap();
    input.set_requires_grad(true);

    // Set up computation graph
    let graph = Rc::new(RefCell::new(ComputationGraph::new()));
    let input = input.with_graph(graph);

    // Forward pass
    let output = layer.forward(&input).unwrap();

    // With ones initialization: output = [1,2,3] * [[1,1],[1,1],[1,1]] + [0,0] = [6,6]
    assert_eq!(output.shape(), (1, 2));
    assert_abs_diff_eq!(output.data[[0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output.data[[0, 1]], 6.0, epsilon = 1e-6);
  }

  #[test]
  fn test_layer_trait_methods() {
    let layer = DenseLayer::new(784, 128, Activation::ReLU, WeightInit::XavierUniform);

    assert_eq!(layer.output_shape((32, 784)), (32, 128));
    assert_eq!(layer.param_count(), 784 * 128 + 128);
    assert_eq!(layer.name(), "DenseLayer");
  }

  #[test]
  fn test_weight_initialization_shapes() {
    let layer = DenseLayer::new(100, 50, Activation::Sigmoid, WeightInit::Random);

    assert_eq!(layer.weights.shape(), (100, 50));
    assert_eq!(layer.bias.shape(), (1, 50));

    // Check that weights are not all the same (should be random)
    let first_weight = layer.weights.data[[0, 0]];
    let mut all_same = true;
    for i in 0..layer.weights.shape().0 {
      for j in 0..layer.weights.shape().1 {
        if layer.weights.data[[i, j]] != first_weight {
          all_same = false;
          break;
        }
      }
      if !all_same {
        break;
      }
    }
    assert!(
      !all_same,
      "Weights should not all be the same with random initialization"
    );
  }

  #[test]
  fn test_xavier_uniform_initialization_range() {
    let input_size = 100;
    let output_size = 50;
    let layer = DenseLayer::new(
      input_size,
      output_size,
      Activation::Sigmoid,
      WeightInit::XavierUniform,
    );

    let expected_limit = (6.0 / (input_size + output_size) as f64).sqrt();

    // Check that all weights are within the expected range
    for i in 0..layer.weights.shape().0 {
      for j in 0..layer.weights.shape().1 {
        let weight = layer.weights.data[[i, j]];
        assert!(
          weight >= -expected_limit && weight <= expected_limit,
          "Weight {} is outside expected range [{}, {}]",
          weight,
          -expected_limit,
          expected_limit
        );
      }
    }
  }

  #[test]
  fn test_he_uniform_initialization_range() {
    let input_size = 784;
    let output_size = 128;
    let layer = DenseLayer::new(
      input_size,
      output_size,
      Activation::ReLU,
      WeightInit::HeUniform,
    );

    let expected_limit = (6.0 / input_size as f64).sqrt();

    // Check that all weights are within the expected range
    for i in 0..layer.weights.shape().0 {
      for j in 0..layer.weights.shape().1 {
        let weight = layer.weights.data[[i, j]];
        assert!(
          weight >= -expected_limit && weight <= expected_limit,
          "Weight {} is outside expected range [{}, {}]",
          weight,
          -expected_limit,
          expected_limit
        );
      }
    }
  }

  #[test]
  fn test_layer_cloning() {
    let layer = DenseLayer::new(10, 5, Activation::ReLU, WeightInit::Random);
    let boxed: Box<dyn Layer> = Box::new(layer);
    let cloned_boxed = boxed.clone();

    assert_eq!(boxed.name(), cloned_boxed.name());
    assert_eq!(boxed.param_count(), cloned_boxed.param_count());
    assert_eq!(
      boxed.output_shape((1, 10)),
      cloned_boxed.output_shape((1, 10))
    );
  }
}

/// Sequential model for building neural networks layer by layer
///
/// A Sequential model is a linear stack of layers where data flows from input to output
/// through each layer in sequence. This is the most common type of neural network architecture.
#[derive(Debug, Clone)]
pub struct Sequential {
  /// List of layers in the model
  layers: Vec<Box<dyn Layer>>,
  /// Flag indicating whether model is in training mode
  training: bool,
  /// Computation graph for automatic differentiation
  graph: Option<std::rc::Rc<std::cell::RefCell<crate::graph::ComputationGraph>>>,
}

impl Sequential {
  /// Create a new empty Sequential model
  pub fn new() -> Self {
    Self {
      layers: Vec::new(),
      training: true,
      graph: None,
    }
  }

  /// Add a layer to the model
  ///
  /// # Arguments
  /// * `layer` - The layer to add to the model
  ///
  /// # Returns
  /// Returns self for method chaining
  #[allow(clippy::should_implement_trait)]
  pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
    self.layers.push(Box::new(layer));
    self
  }

  /// Add a dense layer with specified parameters
  ///
  /// # Arguments
  /// * `input_size` - Number of input features
  /// * `output_size` - Number of output features  
  /// * `activation` - Activation function to use
  /// * `weight_init` - Weight initialization method
  ///
  /// # Returns
  /// Returns self for method chaining
  pub fn dense(
    self,
    input_size: usize,
    output_size: usize,
    activation: Activation,
    weight_init: WeightInit,
  ) -> Self {
    self.add(DenseLayer::new(
      input_size,
      output_size,
      activation,
      weight_init,
    ))
  }

  /// Add a dense layer with ReLU activation and He initialization
  pub fn relu_layer(self, input_size: usize, output_size: usize) -> Self {
    self.add(DenseLayer::relu(input_size, output_size))
  }

  /// Add a dense layer with sigmoid activation
  pub fn sigmoid_layer(self, input_size: usize, output_size: usize) -> Self {
    self.add(DenseLayer::sigmoid(input_size, output_size))
  }

  /// Add a dense layer with softmax activation (typically for output layer)
  pub fn softmax_layer(self, input_size: usize, output_size: usize) -> Self {
    self.add(DenseLayer::softmax(input_size, output_size))
  }

  /// Set up computation graph for the model
  pub fn with_graph(
    mut self,
    graph: std::rc::Rc<std::cell::RefCell<crate::graph::ComputationGraph>>,
  ) -> Self {
    self.graph = Some(graph.clone());

    // Connect all layer parameters to the computation graph
    for layer in &mut self.layers {
      if let Some(dense_layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
        dense_layer.connect_to_graph(graph.clone());
      }
    }

    self
  }

  /// Get the number of layers in the model
  pub fn len(&self) -> usize {
    self.layers.len()
  }

  /// Check if the model is empty
  pub fn is_empty(&self) -> bool {
    self.layers.is_empty()
  }

  /// Set training mode
  pub fn train(&mut self) {
    self.training = true;
  }

  /// Set evaluation mode
  pub fn eval(&mut self) {
    self.training = false;
  }

  /// Check if model is in training mode
  pub fn is_training(&self) -> bool {
    self.training
  }

  /// Perform forward propagation through all layers
  ///
  /// # Arguments
  /// * `input` - Input tensor
  ///
  /// # Returns
  /// Output tensor after passing through all layers
  pub fn forward(&mut self, mut input: Tensor) -> Result<Tensor> {
    // Set up computation graph if provided and input is not already tracked
    if let (Some(graph), false) = (&self.graph, input.is_tracked()) {
      input.set_requires_grad(true);
      input = input.with_graph(graph.clone());
    }

    // Pass input through each layer sequentially
    for layer in &mut self.layers {
      input = layer.forward(&input)?;
    }

    Ok(input)
  }

  /// Perform backward propagation through all layers
  ///
  /// This method assumes that forward pass has been called and
  /// gradients have been computed via tensor.backward().
  /// For non-scalar outputs, you need to provide gradient of loss w.r.t. output.
  pub fn backward(&mut self, output: &mut Tensor, grad_output: Option<&Tensor>) -> Result<()> {
    if self.training {
      match grad_output {
        Some(grad) => {
          // Use provided gradient output
          if let (Some(graph_weak), Some(node_id)) = (&output.graph, output.graph_id) {
            if let Some(graph) = graph_weak.upgrade() {
              graph
                .borrow_mut()
                .backward(node_id, Some(grad.data.clone()))?;
            }
          }
        }
        None => {
          // For scalar outputs, no grad_output needed
          output.backward()?;
        }
      }
    }
    Ok(())
  }

  /// Simple backward for scalar loss (most common case)
  pub fn backward_scalar(&mut self, output: &mut Tensor) -> Result<()> {
    self.backward(output, None)
  }

  /// Get model summary information
  pub fn summary(&self) -> ModelSummary {
    let mut total_params = 0;
    let mut layer_info = Vec::new();

    let mut current_shape = (0, 0); // Will be set by first layer

    for (i, layer) in self.layers.iter().enumerate() {
      let layer_params = layer.param_count();
      total_params += layer_params;

      if i == 0 {
        // For first layer, we need input shape to be provided separately
        // This is a limitation of the current design
        layer_info.push(LayerInfo {
          name: layer.name().to_string(),
          input_shape: (0, 0),  // Unknown without input
          output_shape: (0, 0), // Unknown without input
          param_count: layer_params,
        });
      } else {
        let output_shape = layer.output_shape(current_shape);
        layer_info.push(LayerInfo {
          name: layer.name().to_string(),
          input_shape: current_shape,
          output_shape,
          param_count: layer_params,
        });
        current_shape = output_shape;
      }
    }

    ModelSummary {
      layers: layer_info,
      total_params,
    }
  }

  /// Get model summary with known input shape
  pub fn summary_with_input_shape(&self, input_shape: (usize, usize)) -> ModelSummary {
    let mut total_params = 0;
    let mut layer_info = Vec::new();
    let mut current_shape = input_shape;

    for layer in &self.layers {
      let layer_params = layer.param_count();
      total_params += layer_params;

      let output_shape = layer.output_shape(current_shape);
      layer_info.push(LayerInfo {
        name: layer.name().to_string(),
        input_shape: current_shape,
        output_shape,
        param_count: layer_params,
      });
      current_shape = output_shape;
    }

    ModelSummary {
      layers: layer_info,
      total_params,
    }
  }

  /// Zero gradients for all parameters
  pub fn zero_grad(&mut self) {
    for layer in &mut self.layers {
      layer.zero_gradients();
    }
  }

  /// Get all trainable parameters in the model
  pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
    let mut params = Vec::new();
    for layer in &mut self.layers {
      params.extend(layer.parameters_mut());
    }
    params
  }

  /// Sync gradients from computation graph for all layers
  pub fn sync_gradients(&mut self) -> Result<()> {
    for layer in &mut self.layers {
      layer.sync_gradients()?;
    }
    Ok(())
  }
}

impl Default for Sequential {
  fn default() -> Self {
    Self::new()
  }
}

/// Information about a single layer in the model
#[derive(Debug, Clone)]
pub struct LayerInfo {
  pub name: String,
  pub input_shape: (usize, usize),
  pub output_shape: (usize, usize),
  pub param_count: usize,
}

/// Summary information about the model
#[derive(Debug, Clone)]
pub struct ModelSummary {
  pub layers: Vec<LayerInfo>,
  pub total_params: usize,
}

impl std::fmt::Display for ModelSummary {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "Model Summary")?;
    writeln!(f, "=============")?;
    writeln!(
      f,
      "{:<15} {:<20} {:<20} {:<15}",
      "Layer", "Input Shape", "Output Shape", "Params"
    )?;
    writeln!(
      f,
      "-----------------------------------------------------------------------------"
    )?;

    for layer in &self.layers {
      writeln!(
        f,
        "{:<15} {:<20} {:<20} {:<15}",
        layer.name,
        format!("({}, {})", layer.input_shape.0, layer.input_shape.1),
        format!("({}, {})", layer.output_shape.0, layer.output_shape.1),
        layer.param_count
      )?;
    }

    writeln!(
      f,
      "============================================================================="
    )?;
    writeln!(f, "Total params: {}", self.total_params)?;

    Ok(())
  }
}

#[cfg(test)]
mod sequential_tests {
  use super::*;
  use approx::assert_abs_diff_eq;

  #[test]
  fn test_sequential_creation() {
    let model = Sequential::new();

    assert_eq!(model.len(), 0);
    assert!(model.is_empty());
    assert!(model.is_training());
  }

  #[test]
  fn test_sequential_add_layers() {
    let model = Sequential::new()
      .dense(784, 128, Activation::ReLU, WeightInit::HeUniform)
      .dense(128, 64, Activation::ReLU, WeightInit::HeUniform)
      .dense(64, 10, Activation::Softmax, WeightInit::XavierUniform);

    assert_eq!(model.len(), 3);
    assert!(!model.is_empty());
  }

  #[test]
  fn test_sequential_convenience_methods() {
    let model = Sequential::new()
      .relu_layer(784, 128)
      .sigmoid_layer(128, 64)
      .softmax_layer(64, 10);

    assert_eq!(model.len(), 3);
  }

  #[test]
  fn test_sequential_forward() {
    use crate::graph::ComputationGraph;
    use std::cell::RefCell;
    use std::rc::Rc;

    let graph = Rc::new(RefCell::new(ComputationGraph::new()));

    let mut model = Sequential::new()
      .dense(3, 2, Activation::None, WeightInit::Ones) // Linear layer for predictable output
      .with_graph(graph.clone());

    // Create input: batch_size=1, features=3
    let input_data = vec![vec![1.0, 2.0, 3.0]];
    let input = Tensor::new(input_data).unwrap();

    let output = model.forward(input).unwrap();

    // With ones initialization: output = [1,2,3] * [[1,1],[1,1],[1,1]] + [0,0] = [6,6]
    assert_eq!(output.shape(), (1, 2));
    assert_abs_diff_eq!(output.data[[0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output.data[[0, 1]], 6.0, epsilon = 1e-6);
  }

  #[test]
  fn test_sequential_training_mode() {
    let mut model = Sequential::new();

    assert!(model.is_training());

    model.eval();
    assert!(!model.is_training());

    model.train();
    assert!(model.is_training());
  }

  #[test]
  fn test_model_summary() {
    let model = Sequential::new()
      .dense(784, 128, Activation::ReLU, WeightInit::HeUniform)
      .dense(128, 10, Activation::Softmax, WeightInit::XavierUniform);

    let summary = model.summary_with_input_shape((32, 784)); // batch_size=32, features=784

    assert_eq!(summary.layers.len(), 2);
    assert_eq!(summary.layers[0].input_shape, (32, 784));
    assert_eq!(summary.layers[0].output_shape, (32, 128));
    assert_eq!(summary.layers[0].param_count, 784 * 128 + 128); // weights + bias

    assert_eq!(summary.layers[1].input_shape, (32, 128));
    assert_eq!(summary.layers[1].output_shape, (32, 10));
    assert_eq!(summary.layers[1].param_count, 128 * 10 + 10); // weights + bias

    let expected_total = (784 * 128 + 128) + (128 * 10 + 10);
    assert_eq!(summary.total_params, expected_total);
  }

  #[test]
  fn test_model_summary_display() {
    let model = Sequential::new().relu_layer(10, 5).softmax_layer(5, 2);

    let summary = model.summary_with_input_shape((1, 10));
    let summary_str = format!("{}", summary);

    assert!(summary_str.contains("Model Summary"));
    assert!(summary_str.contains("DenseLayer"));
    assert!(summary_str.contains("Total params:"));
  }

  #[test]
  fn test_multi_layer_forward() {
    use crate::graph::ComputationGraph;
    use std::cell::RefCell;
    use std::rc::Rc;

    let graph = Rc::new(RefCell::new(ComputationGraph::new()));

    let mut model = Sequential::new()
      .dense(2, 3, Activation::ReLU, WeightInit::Ones)
      .dense(3, 1, Activation::Sigmoid, WeightInit::Ones)
      .with_graph(graph.clone());

    let input_data = vec![vec![1.0, -1.0]]; // Test with positive and negative values
    let input = Tensor::new(input_data).unwrap();

    let output = model.forward(input).unwrap();

    // Verify output shape
    assert_eq!(output.shape(), (1, 1));

    // Output should be a valid sigmoid output (between 0 and 1)
    let output_val = output.data[[0, 0]];
    assert!((0.0..1.0).contains(&output_val));
  }
}
