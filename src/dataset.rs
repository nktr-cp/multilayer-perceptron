//! Dataset module for handling breast cancer data loading and preprocessing
//!
//! This module provides functionality to load, preprocess, and manage the
//! Wisconsin Breast Cancer dataset for training multilayer perceptron models.

use crate::error::{Result, TensorError};
use crate::tensor::Tensor;
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;
use std::path::Path;

/// Represents the diagnosis label for breast cancer classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagnosis {
  /// Malignant (cancerous)
  M,
  /// Benign (non-cancerous)
  B,
}

impl Diagnosis {
  pub fn to_f64(self) -> f64 {
    match self {
      Diagnosis::M => 1.0,
      Diagnosis::B => 0.0,
    }
  }

  /// Parse diagnosis from string representation
  pub fn parse(s: &str) -> Result<Self> {
    match s.trim() {
      "M" => Ok(Diagnosis::M),
      "B" => Ok(Diagnosis::B),
      _ => Err(TensorError::InvalidValue(format!(
        "Invalid diagnosis value: {}. Expected 'M' or 'B'",
        s
      ))),
    }
  }
}

#[derive(Debug, Clone)]
pub struct PreprocessConfig {
  /// Whether to apply standardization (z-score normalization)
  pub standardize: bool,
  /// Whether to apply min-max normalization
  pub normalize: bool,
  /// Random seed for reproducible train/test splits
  pub random_seed: Option<u64>,
}

impl Default for PreprocessConfig {
  fn default() -> Self {
    Self {
      standardize: true,
      normalize: false,
      random_seed: Some(42),
    }
  }
}

#[derive(Debug, Clone)]
pub struct FeatureStats {
  pub means: Array1<f64>,
  pub stds: Array1<f64>,
  pub mins: Array1<f64>,
  pub maxs: Array1<f64>,
}

impl FeatureStats {
  /// Compute statistics from feature matrix
  pub fn from_features(features: &Array2<f64>) -> Self {
    let means = features.mean_axis(Axis(0)).unwrap();
    let stds = features.std_axis(Axis(0), 0.0);
    let mins = features.fold_axis(Axis(0), f64::INFINITY, |&acc, &x| acc.min(x));
    let maxs = features.fold_axis(Axis(0), f64::NEG_INFINITY, |&acc, &x| acc.max(x));

    Self {
      means,
      stds,
      mins,
      maxs,
    }
  }

  /// Apply standardization to features using computed statistics
  pub fn standardize(&self, features: &mut Array2<f64>) -> Result<()> {
    if features.shape()[1] != self.means.len() {
      return Err(TensorError::DimensionMismatch(
        "Feature dimension mismatch".to_string(),
      ));
    }

    for mut row in features.rows_mut() {
      for (i, value) in row.iter_mut().enumerate() {
        if self.stds[i] != 0.0 {
          *value = (*value - self.means[i]) / self.stds[i];
        }
      }
    }
    Ok(())
  }

  /// Apply min-max normalization to features using computed statistics
  pub fn normalize(&self, features: &mut Array2<f64>) -> Result<()> {
    if features.shape()[1] != self.mins.len() {
      return Err(TensorError::DimensionMismatch(
        "Feature dimension mismatch".to_string(),
      ));
    }

    for mut row in features.rows_mut() {
      for (i, value) in row.iter_mut().enumerate() {
        let range = self.maxs[i] - self.mins[i];
        if range != 0.0 {
          *value = (*value - self.mins[i]) / range;
        }
      }
    }
    Ok(())
  }
}

/// Raw breast cancer data record
#[derive(Debug, Clone)]
pub struct BreastCancerRecord {
  pub id: u32,
  pub diagnosis: Diagnosis,
  pub features: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
  pub features: Array2<f64>,
  pub labels: Array1<f64>,
  pub ids: Vec<u32>,
  pub stats: Option<FeatureStats>,
  pub config: PreprocessConfig,
}

impl Dataset {
  pub fn new(config: PreprocessConfig) -> Self {
    Self {
      features: Array2::zeros((0, 30)),
      labels: Array1::zeros(0),
      ids: Vec::new(),
      stats: None,
      config,
    }
  }

  pub fn len(&self) -> usize {
    self.features.shape()[0]
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn n_features(&self) -> usize {
    self.features.shape()[1]
  }

  pub fn from_csv<P: AsRef<Path>>(path: P, config: PreprocessConfig) -> Result<Self> {
    let file = std::fs::File::open(path)?;
    let mut reader = csv::ReaderBuilder::new()
      .has_headers(false)
      .from_reader(file);

    let mut records = Vec::new();

    for result in reader.records() {
      let record = result?;
      if record.len() < 32 {
        return Err(TensorError::InvalidValue(format!(
          "Expected at least 32 columns, got {}",
          record.len()
        )));
      }

      let id: u32 = record[0]
        .parse()
        .map_err(|e| TensorError::InvalidValue(format!("Invalid ID: {}", e)))?;

      // Parse diagnosis
      let diagnosis = Diagnosis::parse(&record[1])?;

      let mut features = Array1::zeros(30);
      for i in 0..30 {
        features[i] = record[i + 2].parse().map_err(|e| {
          TensorError::InvalidValue(format!("Invalid feature value at column {}: {}", i + 2, e))
        })?;
      }

      records.push(BreastCancerRecord {
        id,
        diagnosis,
        features,
      });
    }

    if records.is_empty() {
      return Err(TensorError::InvalidValue(
        "No data found in CSV file".to_string(),
      ));
    }

    let n_samples = records.len();
    let mut features_matrix = Array2::zeros((n_samples, 30));
    let mut labels = Array1::zeros(n_samples);
    let mut ids = Vec::with_capacity(n_samples);

    for (i, record) in records.into_iter().enumerate() {
      features_matrix.row_mut(i).assign(&record.features);
      labels[i] = record.diagnosis.to_f64();
      ids.push(record.id);
    }

    let mut dataset = Self {
      features: features_matrix,
      labels,
      ids,
      stats: None,
      config,
    };

    dataset.preprocess()?;

    Ok(dataset)
  }

  pub fn preprocess(&mut self) -> Result<()> {
    if self.stats.is_none() {
      self.stats = Some(FeatureStats::from_features(&self.features));
    }

    let stats = self.stats.as_ref().unwrap();

    if self.config.standardize {
      stats.standardize(&mut self.features)?;
    }

    if self.config.normalize {
      stats.normalize(&mut self.features)?;
    }

    Ok(())
  }

  pub fn preprocess_with_stats(&mut self, stats: &FeatureStats) -> Result<()> {
    if self.config.standardize {
      stats.standardize(&mut self.features)?;
    }

    if self.config.normalize {
      stats.normalize(&mut self.features)?;
    }

    Ok(())
  }

  pub fn train_test_split(&self, test_size: f64) -> Result<(Dataset, Dataset)> {
    if test_size <= 0.0 || test_size >= 1.0 {
      return Err(TensorError::InvalidValue(
        "Test size must be between 0 and 1".to_string(),
      ));
    }

    let n_samples = self.len();
    let n_test = (n_samples as f64 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    let mut indices: Vec<usize> = (0..n_samples).collect();

    if let Some(seed) = self.config.random_seed {
      let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
      indices.shuffle(&mut rng);
    } else {
      let mut rng = rand::thread_rng();
      indices.shuffle(&mut rng);
    }

    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];

    let mut train_features = Array2::zeros((n_train, self.n_features()));
    let mut train_labels = Array1::zeros(n_train);
    let mut train_ids = Vec::with_capacity(n_train);

    for (i, &idx) in train_indices.iter().enumerate() {
      train_features.row_mut(i).assign(&self.features.row(idx));
      train_labels[i] = self.labels[idx];
      train_ids.push(self.ids[idx]);
    }

    let mut test_features = Array2::zeros((n_test, self.n_features()));
    let mut test_labels = Array1::zeros(n_test);
    let mut test_ids = Vec::with_capacity(n_test);

    for (i, &idx) in test_indices.iter().enumerate() {
      test_features.row_mut(i).assign(&self.features.row(idx));
      test_labels[i] = self.labels[idx];
      test_ids.push(self.ids[idx]);
    }

    let train_dataset = Dataset {
      features: train_features,
      labels: train_labels,
      ids: train_ids,
      stats: self.stats.clone(),
      config: self.config.clone(),
    };

    let test_dataset = Dataset {
      features: test_features,
      labels: test_labels,
      ids: test_ids,
      stats: self.stats.clone(),
      config: self.config.clone(),
    };

    Ok((train_dataset, test_dataset))
  }

  pub fn get_stats(&self) -> Option<&FeatureStats> {
    self.stats.as_ref()
  }

  pub fn to_tensors(&self) -> Result<(Tensor, Tensor)> {
    let features_tensor = Tensor::from_array2(self.features.clone())?;
    let labels_tensor = Tensor::from_array1(self.labels.clone())?;
    Ok((features_tensor, labels_tensor))
  }

  pub fn subset(&self, indices: &[usize]) -> Result<Dataset> {
    if indices.iter().any(|&i| i >= self.len()) {
      return Err(TensorError::InvalidValue("Index out of bounds".to_string()));
    }

    let n_subset = indices.len();
    let mut features = Array2::zeros((n_subset, self.n_features()));
    let mut labels = Array1::zeros(n_subset);
    let mut ids = Vec::with_capacity(n_subset);

    for (i, &idx) in indices.iter().enumerate() {
      features.row_mut(i).assign(&self.features.row(idx));
      labels[i] = self.labels[idx];
      ids.push(self.ids[idx]);
    }

    Ok(Dataset {
      features,
      labels,
      ids,
      stats: self.stats.clone(),
      config: self.config.clone(),
    })
  }

  pub fn class_distribution(&self) -> HashMap<String, usize> {
    let mut distribution = HashMap::new();
    for &label in &self.labels {
      let class = if label > 0.5 { "M" } else { "B" };
      *distribution.entry(class.to_string()).or_insert(0) += 1;
    }
    distribution
  }
}

/// Data loader for mini-batch training
#[derive(Debug)]
pub struct DataLoader {
  dataset: Dataset,
  batch_size: usize,
  shuffle: bool,
  rng: rand::rngs::StdRng,
  indices: Vec<usize>,
  current_pos: usize,
}

impl DataLoader {
  pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool, seed: Option<u64>) -> Self {
    let n_samples = dataset.len();
    let indices: Vec<usize> = (0..n_samples).collect();

    let rng = if let Some(seed) = seed {
      rand::rngs::StdRng::seed_from_u64(seed)
    } else {
      rand::rngs::StdRng::from_entropy()
    };

    Self {
      dataset,
      batch_size,
      shuffle,
      rng,
      indices,
      current_pos: 0,
    }
  }

  pub fn num_batches(&self) -> usize {
    self.dataset.len().div_ceil(self.batch_size)
  }

  pub fn reset(&mut self) {
    self.current_pos = 0;
    if self.shuffle {
      self.indices.shuffle(&mut self.rng);
    }
  }

  pub fn next_batch(&mut self) -> Option<Result<(Tensor, Tensor)>> {
    if self.current_pos >= self.dataset.len() {
      return None;
    }

    let end_pos = (self.current_pos + self.batch_size).min(self.dataset.len());
    let batch_indices = &self.indices[self.current_pos..end_pos];

    let batch_dataset = match self.dataset.subset(batch_indices) {
      Ok(dataset) => dataset,
      Err(e) => return Some(Err(e)),
    };

    self.current_pos = end_pos;

    Some(batch_dataset.to_tensors())
  }

  pub fn batches(&mut self) -> DataLoaderIterator<'_> {
    self.reset();
    DataLoaderIterator { loader: self }
  }

  pub fn dataset(&self) -> &Dataset {
    &self.dataset
  }

  pub fn batch_size(&self) -> usize {
    self.batch_size
  }
}

pub struct DataLoaderIterator<'a> {
  loader: &'a mut DataLoader,
}

impl Iterator for DataLoaderIterator<'_> {
  type Item = Result<(Tensor, Tensor)>;

  fn next(&mut self) -> Option<Self::Item> {
    self.loader.next_batch()
  }
}

pub mod utils {
  use super::*;

  pub fn stratified_split(
    dataset: &Dataset,
    test_size: f64,
    random_seed: Option<u64>,
  ) -> Result<(Dataset, Dataset)> {
    if test_size <= 0.0 || test_size >= 1.0 {
      return Err(TensorError::InvalidValue(
        "Test size must be between 0 and 1".to_string(),
      ));
    }

    let mut class_0_indices = Vec::new(); // Benign (B)
    let mut class_1_indices = Vec::new(); // Malignant (M)

    for (i, &label) in dataset.labels.iter().enumerate() {
      if label > 0.5 {
        class_1_indices.push(i);
      } else {
        class_0_indices.push(i);
      }
    }

    if let Some(seed) = random_seed {
      let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
      class_0_indices.shuffle(&mut rng);
      class_1_indices.shuffle(&mut rng);
    } else {
      let mut rng = rand::thread_rng();
      class_0_indices.shuffle(&mut rng);
      class_1_indices.shuffle(&mut rng);
    }

    let n_test_0 = (class_0_indices.len() as f64 * test_size).round() as usize;
    let n_test_1 = (class_1_indices.len() as f64 * test_size).round() as usize;

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    train_indices.extend_from_slice(&class_0_indices[n_test_0..]);
    train_indices.extend_from_slice(&class_1_indices[n_test_1..]);

    test_indices.extend_from_slice(&class_0_indices[..n_test_0]);
    test_indices.extend_from_slice(&class_1_indices[..n_test_1]);

    if let Some(seed) = random_seed {
      let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 1); // Different seed for final shuffle
      train_indices.shuffle(&mut rng);
      test_indices.shuffle(&mut rng);
    } else {
      let mut rng = rand::thread_rng();
      train_indices.shuffle(&mut rng);
      test_indices.shuffle(&mut rng);
    }

    let train_dataset = dataset.subset(&train_indices)?;
    let test_dataset = dataset.subset(&test_indices)?;

    Ok((train_dataset, test_dataset))
  }

  pub fn print_dataset_info(dataset: &Dataset, name: &str) {
    println!("=== {} Dataset Info ===", name);
    println!("Number of samples: {}", dataset.len());
    println!("Number of features: {}", dataset.n_features());

    let distribution = dataset.class_distribution();
    println!("Class distribution:");
    for (class, count) in &distribution {
      let percentage = (*count as f64 / dataset.len() as f64) * 100.0;
      println!("  {}: {} ({:.1}%)", class, count, percentage);
    }

    if let Some(stats) = dataset.get_stats() {
      println!("Feature statistics (first 5 features):");
      for i in 0..5.min(stats.means.len()) {
        println!(
          "  Feature {}: mean={:.3}, std={:.3}, min={:.3}, max={:.3}",
          i, stats.means[i], stats.stds[i], stats.mins[i], stats.maxs[i]
        );
      }
    }
    println!();
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::Array1;

  #[test]
  fn test_diagnosis_conversion() {
    assert_eq!(Diagnosis::M.to_f64(), 1.0);
    assert_eq!(Diagnosis::B.to_f64(), 0.0);

    assert_eq!(Diagnosis::parse("M").unwrap(), Diagnosis::M);
    assert_eq!(Diagnosis::parse("B").unwrap(), Diagnosis::B);
    assert!(Diagnosis::parse("X").is_err());
  }

  #[test]
  fn test_feature_stats() {
    let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let stats = FeatureStats::from_features(&features);

    assert!((stats.means[0] - 3.0).abs() < 1e-10);
    assert!((stats.means[1] - 4.0).abs() < 1e-10);
  }

  #[test]
  fn test_dataset_creation() {
    let config = PreprocessConfig::default();
    let dataset = Dataset::new(config);
    assert_eq!(dataset.len(), 0);
    assert!(dataset.is_empty());
    assert_eq!(dataset.n_features(), 30);
  }

  #[test]
  fn test_train_test_split() {
    let features = Array2::zeros((10, 30));
    let labels = Array1::from_vec((0..10).map(|x| x as f64).collect());
    let ids = (0..10).map(|x| x as u32).collect();

    let dataset = Dataset {
      features,
      labels,
      ids,
      stats: None,
      config: PreprocessConfig::default(),
    };

    let (train, test) = dataset.train_test_split(0.3).unwrap();
    assert_eq!(train.len(), 7);
    assert_eq!(test.len(), 3);
  }
}
