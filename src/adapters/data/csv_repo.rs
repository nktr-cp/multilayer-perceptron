use crate::core::{Result, TensorError};
use crate::domain::ports::DataRepository;
use crate::domain::types::{DataConfig, Dataset, TaskKind};
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

      let id = record[0].to_string();
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
    let mut targets = Array2::zeros((n_samples, 1));
    let mut ids = Vec::with_capacity(n_samples);

    for (i, record) in records.into_iter().enumerate() {
      features_matrix.row_mut(i).assign(&record.features);
      targets[[i, 0]] = record.diagnosis.to_f64();
      ids.push(record.id);
    }

    let mut dataset = Dataset::new(
      TaskKind::BinaryClassification,
      features_matrix,
      targets,
      config.clone(),
    )
    .with_sample_ids(ids);

    dataset.preprocess()?;

    Ok(dataset)
  }
}

#[derive(Debug, Clone)]
struct BreastCancerRecord {
  id: String,
  diagnosis: Diagnosis,
  features: Array1<f64>,
}

#[derive(Debug, Clone, Copy)]
enum Diagnosis {
  Malignant,
  Benign,
}

impl Diagnosis {
  pub fn to_f64(self) -> f64 {
    match self {
      Diagnosis::Malignant => 1.0,
      Diagnosis::Benign => 0.0,
    }
  }

  pub fn parse(value: &str) -> Result<Self> {
    match value.trim() {
      "M" => Ok(Diagnosis::Malignant),
      "B" => Ok(Diagnosis::Benign),
      other => Err(TensorError::InvalidValue(format!(
        "Invalid diagnosis value: {}. Expected 'M' or 'B'",
        other
      ))),
    }
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
    let distribution = dataset
      .class_distribution()
      .expect("classification distribution");
    assert_eq!(distribution.len(), 2);
    remove_file(csv_path).unwrap();
  }
}
