use crate::core::{Result, TensorError};
use crate::domain::ports::DataRepository;
use crate::domain::types::{BreastCancerRecord, DataConfig, Dataset, Diagnosis};
use ndarray::{Array1, Array2};
use std::fs::File;
use std::path::PathBuf;

pub struct CsvDataRepository {
  path: PathBuf,
}

impl CsvDataRepository {
  pub fn new<P: Into<PathBuf>>(path: P) -> Self {
    Self { path: path.into() }
  }

  fn read_records(&self) -> Result<Vec<BreastCancerRecord>> {
    let file = File::open(&self.path)?;
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

    Ok(records)
  }
}

impl DataRepository for CsvDataRepository {
  fn load_dataset(&self, config: &DataConfig) -> Result<Dataset> {
    let records = self.read_records()?;

    let n_samples = records.len();
    let mut features_matrix = Array2::zeros((n_samples, 30));
    let mut labels = Array1::zeros(n_samples);
    let mut ids = Vec::with_capacity(n_samples);

    for (i, record) in records.into_iter().enumerate() {
      features_matrix.row_mut(i).assign(&record.features);
      labels[i] = record.diagnosis.to_f64();
      ids.push(record.id);
    }

    let mut dataset = Dataset::from_parts(features_matrix, labels, ids, None, config.clone());

    dataset.preprocess()?;

    Ok(dataset)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::domain::types::PreprocessConfig;
  use std::fs::{remove_file, File};
  use std::io::Write;
  use std::time::{SystemTime, UNIX_EPOCH};

  #[test]
  fn test_load_dataset() {
    let timestamp = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos();
    let csv_path = std::env::temp_dir().join(format!("mlp_test_{timestamp}.csv"));

    let mut file = File::create(&csv_path).unwrap();
    let row_one = vec!["1.0"; 30].join(",");
    let row_zero = vec!["0.0"; 30].join(",");
    writeln!(file, "1001,M,{row_one}").unwrap();
    writeln!(file, "1002,B,{row_zero}").unwrap();

    let repo = CsvDataRepository::new(&csv_path);
    let dataset = repo
      .load_dataset(&PreprocessConfig::default())
      .expect("Failed to load dataset");

    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.class_distribution().len(), 2);
    remove_file(csv_path).unwrap();
  }
}
