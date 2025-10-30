//! Training Demo
//!
//! This example demonstrates how to train a neural network using the multilayer perceptron library.
//! It creates synthetic data for binary classification and trains a simple neural network.

use multilayer_perceptron::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<()> {
  println!("🚀 Multilayer Perceptron Training Demo");
  println!("======================================");

  // Load breast cancer dataset
  println!("\n📊 Loading breast cancer dataset...");
  let (train_x, train_y, val_x, val_y) = load_breast_cancer_dataset()?;

  println!("Training set shape: {:?}", train_x.shape());
  println!("Validation set shape: {:?}", val_x.shape());

  // Create computation graph
  let graph = Rc::new(RefCell::new(ComputationGraph::new()));

  // Connect training data to computation graph
  let train_x = train_x.with_graph(graph.clone());
  let train_y = train_y.with_graph(graph.clone());
  let val_x = val_x.with_graph(graph.clone());
  let val_y = val_y.with_graph(graph.clone());

  // Build neural network model
  println!("\n🧠 Building neural network model...");
  let mut model = Sequential::new()
    .relu_layer(30, 24) // Input layer: 30 features -> 24 hidden units
    .relu_layer(24, 16) // Hidden layer: 24 -> 16 units
    .relu_layer(16, 8) // Hidden layer: 16 -> 8 units
    .sigmoid_layer(8, 1) // Output layer: 8 -> 1 unit (binary classification)
    .with_graph(graph);

  // Print model summary
  let summary = model.summary_with_input_shape(train_x.shape());
  println!("Model Architecture:");
  println!("{}", summary);

  // Setup training components
  println!("\n⚙️  Setting up training components...");
  let full_batch_size = train_x.shape().0;
  let config = TrainingConfig {
    epochs: 500,
    batch_size: full_batch_size,
    shuffle: false,
    validation_frequency: 5,
    verbose: true,
    early_stopping_patience: 10,
    early_stopping_min_delta: 0.001,
    enable_early_stopping: true,
    learning_rate: 0.001,
  };
  let loss_fn = BinaryCrossEntropy::new();
  let optimizer = SGD::new(config.learning_rate);
  println!(
    "Using optimizer: {} (learning rate = {:.4})",
    optimizer.name(),
    config.learning_rate
  );

  let mut trainer = Trainer::new(&mut model, loss_fn, optimizer)
    .with_config(config)
    .with_train_metric(Accuracy::default())
    .with_train_metric(Precision::default())
    .with_train_metric(Recall::default())
    .with_val_metric(Accuracy::default())
    .with_val_metric(F1Score::default());

  // Start training
  println!("\n🎯 Starting training...");
  println!("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val F1");
  println!("------|------------|----------|-----------|---------|-------");

  let history = trainer.fit(&train_x, &train_y, Some(&val_x), Some(&val_y))?;

  // Training summary
  println!("\n📈 Training Summary");
  println!("==================");
  println!("Total training time: {:.2?}", history.total_time);
  println!("Total epochs: {}", history.epochs.len());

  if let Some(best_epoch) = history.best_epoch {
    let best = &history.epochs[best_epoch];
    println!("Best epoch: {} (0-indexed)", best_epoch);
    println!(
      "Best validation loss: {:.4}",
      best.val_loss.unwrap_or(best.train_loss)
    );
  }

  if history.stopped_early {
    println!("Training stopped early due to no improvement.");
  }

  // Final evaluation
  println!("\n🔍 Final Evaluation");
  println!("===================");
  model.eval();
  let final_predictions = model.forward(val_x.clone())?;

  let final_loss = BinaryCrossEntropy::new().forward(&final_predictions, &val_y)?;
  let accuracy_metric = Accuracy::default();
  let final_accuracy = accuracy_metric.compute(&final_predictions, &val_y)?;

  println!("Final validation loss: {:.4}", final_loss.data[[0, 0]]);
  println!("Final validation accuracy: {:.2}%", final_accuracy * 100.0);

  // Show some example predictions
  println!("\n🔮 Sample Predictions");
  println!("====================");
  for i in 0..5.min(val_x.shape().0) {
    let prediction = final_predictions.data[[i, 0]];
    let true_label = val_y.data[[i, 0]];
    let predicted_class = if prediction >= 0.5 { 1 } else { 0 };
    let true_class = if true_label >= 0.5 {
      "Malignant"
    } else {
      "Benign"
    };
    let predicted_diagnosis = if predicted_class == 1 {
      "Malignant"
    } else {
      "Benign"
    };

    println!(
      "Sample {}: Prediction: {:.3} ({}) | True: {} | Correct: {}",
      i + 1,
      prediction,
      predicted_diagnosis,
      true_class,
      predicted_class == (true_label as i32)
    );
  }

  println!("\n✅ Training demo completed successfully!");

  Ok(())
}

/// Load breast cancer dataset from CSV file
///
/// This function loads the Wisconsin breast cancer dataset from the CSV file,
/// preprocesses it, and splits it into training and validation sets.
fn load_breast_cancer_dataset() -> Result<(Tensor, Tensor, Tensor, Tensor)> {
  // Configure preprocessing
  let config = PreprocessConfig {
    standardize: true,     // Apply z-score normalization
    normalize: false,      // Don't apply min-max normalization (already standardizing)
    random_seed: Some(42), // For reproducible splits
  };

  // Load dataset from CSV
  let dataset = Dataset::from_csv("./data/data.csv", config)?;

  println!(
    "Loaded {} samples with {} features",
    dataset.len(),
    dataset.n_features()
  );

  // Split into train and validation sets (80/20 split)
  let (train_dataset, val_dataset) = dataset.train_test_split(0.2)?;

  println!("Train set: {} samples", train_dataset.len());
  println!("Validation set: {} samples", val_dataset.len());

  // Convert to tensors
  let (train_x, train_y) = train_dataset.to_tensors()?;
  let (val_x, val_y) = val_dataset.to_tensors()?;

  Ok((train_x, train_y, val_x, val_y))
}
