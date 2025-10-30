//! Generic dataset traits and implementations
//!
//! This module provides a trait-based approach for handling various data sources
//! with a unified interface similar to pandas DataFrame functionality.

use crate::error::{Result, TensorError};
use crate::tensor::Tensor;
use ndarray::Array2;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

/// Generic value type that can hold different data types
#[derive(Debug, Clone, PartialEq)]
pub enum DataValue {
  Int(i64),
  Float(f64),
  String(String),
  Bool(bool),
  Null,
}

impl DataValue {
  pub fn to_f64(&self) -> Option<f64> {
    match self {
      DataValue::Int(i) => Some(*i as f64),
      DataValue::Float(f) => Some(*f),
      DataValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
      _ => None,
    }
  }



  pub fn is_null(&self) -> bool {
    matches!(self, DataValue::Null)
  }

  pub fn is_numeric(&self) -> bool {
    matches!(self, DataValue::Int(_) | DataValue::Float(_))
  }
}

impl std::fmt::Display for DataValue {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      DataValue::Int(i) => write!(f, "{}", i),
      DataValue::Float(fl) => write!(f, "{}", fl),
      DataValue::String(s) => write!(f, "{}", s),
      DataValue::Bool(b) => write!(f, "{}", b),
      DataValue::Null => write!(f, "null"),
    }
  }
}

#[derive(Debug, Clone)]
pub struct ColumnInfo {
  pub name: String,
  pub dtype: DataType,
  pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
  Int64,
  Float64,
  String,
  Bool,
  Mixed,
}

#[derive(Debug, Clone)]
pub struct GenericDataFrame {
  pub columns: Vec<ColumnInfo>,
  pub data: Vec<Vec<DataValue>>,
  pub column_map: HashMap<String, usize>,
}

impl GenericDataFrame {
  pub fn new(columns: Vec<ColumnInfo>) -> Self {
    let column_map = columns
      .iter()
      .enumerate()
      .map(|(i, col)| (col.name.clone(), i))
      .collect();

    Self {
      columns,
      data: Vec::new(),
      column_map,
    }
  }

  pub fn len(&self) -> usize {
    self.data.len()
  }

  pub fn is_empty(&self) -> bool {
    self.data.is_empty()
  }

  pub fn n_columns(&self) -> usize {
    self.columns.len()
  }

  pub fn add_row(&mut self, row: Vec<DataValue>) -> Result<()> {
    if row.len() != self.columns.len() {
      return Err(TensorError::DimensionMismatch(format!(
        "Row has {} values, expected {}",
        row.len(),
        self.columns.len()
      )));
    }
    self.data.push(row);
    Ok(())
  }

  pub fn column(&self, name: &str) -> Result<Vec<DataValue>> {
    if let Some(&col_idx) = self.column_map.get(name) {
      Ok(self.data.iter().map(|row| row[col_idx].clone()).collect())
    } else {
      Err(TensorError::InvalidValue(format!(
        "Column '{}' not found",
        name
      )))
    }
  }

  pub fn column_at(&self, index: usize) -> Result<Vec<DataValue>> {
    if index >= self.columns.len() {
      return Err(TensorError::InvalidValue(format!(
        "Column index {} out of bounds",
        index
      )));
    }
    Ok(self.data.iter().map(|row| row[index].clone()).collect())
  }

  pub fn select(&self, column_names: &[&str]) -> Result<GenericDataFrame> {
    let mut selected_columns = Vec::new();
    let mut selected_indices = Vec::new();

    for &name in column_names {
      if let Some(&idx) = self.column_map.get(name) {
        selected_columns.push(self.columns[idx].clone());
        selected_indices.push(idx);
      } else {
        return Err(TensorError::InvalidValue(format!(
          "Column '{}' not found",
          name
        )));
      }
    }

    let mut new_df = GenericDataFrame::new(selected_columns);
    for row in &self.data {
      let selected_row: Vec<DataValue> = selected_indices
        .iter()
        .map(|&idx| row[idx].clone())
        .collect();
      new_df.add_row(selected_row)?;
    }

    Ok(new_df)
  }

  pub fn to_numeric_array(&self, column_names: &[&str]) -> Result<Array2<f64>> {
    let selected = self.select(column_names)?;
    let n_rows = selected.len();
    let n_cols = selected.n_columns();

    let mut array = Array2::zeros((n_rows, n_cols));

    for (row_idx, row) in selected.data.iter().enumerate() {
      for (col_idx, value) in row.iter().enumerate() {
        if let Some(numeric_val) = value.to_f64() {
          array[[row_idx, col_idx]] = numeric_val;
        } else {
          return Err(TensorError::InvalidValue(format!(
            "Non-numeric value at row {}, col {}: {:?}",
            row_idx, col_idx, value
          )));
        }
      }
    }

    Ok(array)
  }

  pub fn describe(&self) -> HashMap<String, HashMap<String, f64>> {
    let mut stats = HashMap::new();

    for (col_idx, col_info) in self.columns.iter().enumerate() {
      if matches!(col_info.dtype, DataType::Int64 | DataType::Float64) {
        let values: Vec<f64> = self
          .data
          .iter()
          .filter_map(|row| row[col_idx].to_f64())
          .collect();

        if !values.is_empty() {
          let mut col_stats = HashMap::new();
          let count = values.len() as f64;
          let mean = values.iter().sum::<f64>() / count;
          let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
          let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

          let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
          let std = variance.sqrt();

          col_stats.insert("count".to_string(), count);
          col_stats.insert("mean".to_string(), mean);
          col_stats.insert("std".to_string(), std);
          col_stats.insert("min".to_string(), min);
          col_stats.insert("max".to_string(), max);

          stats.insert(col_info.name.clone(), col_stats);
        }
      }
    }

    stats
  }

  pub fn head(&self, n: usize) -> GenericDataFrame {
    let mut result = GenericDataFrame::new(self.columns.clone());
    let n_rows = n.min(self.data.len());

    for i in 0..n_rows {
      result.data.push(self.data[i].clone());
    }

    result
  }

  pub fn info(&self) {
    println!("DataFrame Info:");
    println!("Shape: ({}, {})", self.len(), self.n_columns());
    println!("Columns:");
    for (i, col) in self.columns.iter().enumerate() {
      println!("  {}: {} ({:?})", i, col.name, col.dtype);
    }

    let null_counts: Vec<usize> = (0..self.n_columns())
      .map(|col_idx| {
        self
          .data
          .iter()
          .filter(|row| row[col_idx].is_null())
          .count()
      })
      .collect();

    println!("Null values per column:");
    for (i, &count) in null_counts.iter().enumerate() {
      if count > 0 {
        println!("  {}: {}", self.columns[i].name, count);
      }
    }
  }
}

pub trait DatasetLike: Debug {
  fn load<P: AsRef<Path>>(path: P) -> Result<Self>
  where
    Self: Sized;

  fn len(&self) -> usize;

  fn is_empty(&self) -> bool {
    self.len() == 0
  }

  fn to_tensors(&self) -> Result<(Tensor, Tensor)>;

  fn preprocess(&mut self) -> Result<()>;

  fn train_test_split(&self, test_size: f64) -> Result<(Self, Self)>
  where
    Self: Sized + Clone;
}

#[derive(Debug, Clone)]
pub struct CsvConfig {
  pub has_headers: bool,
  pub delimiter: u8,
  pub target_column: Option<String>,
  pub feature_columns: Option<Vec<String>>,
  pub skip_columns: Option<Vec<String>>,
}

impl Default for CsvConfig {
  fn default() -> Self {
    Self {
      has_headers: false,
      delimiter: b',',
      target_column: None,
      feature_columns: None,
      skip_columns: None,
    }
  }
}

pub fn load_csv<P: AsRef<Path>>(path: P, config: CsvConfig) -> Result<GenericDataFrame> {
  use csv::ReaderBuilder;
  use std::fs::File;

  let file = File::open(path)?;
  let mut reader = ReaderBuilder::new()
    .has_headers(config.has_headers)
    .delimiter(config.delimiter)
    .from_reader(file);

  let mut dataframe = if config.has_headers {
    // If headers are present, use them to create columns
    let headers = reader.headers()?.clone();
    let columns: Vec<ColumnInfo> = headers
      .iter()
      .map(|name| ColumnInfo {
        name: name.to_string(),
        dtype: DataType::Mixed, // Will be inferred
        nullable: true,
      })
      .collect();
    GenericDataFrame::new(columns)
  } else {
    // Create generic column names
    let first_record = reader.records().next();
    if let Some(record) = first_record {
      let record = record?;
      let columns: Vec<ColumnInfo> = (0..record.len())
        .map(|i| ColumnInfo {
          name: format!("col_{}", i),
          dtype: DataType::Mixed,
          nullable: true,
        })
        .collect();
      let mut df = GenericDataFrame::new(columns);

      let row: Vec<DataValue> = record.iter().map(parse_field).collect();
      df.add_row(row)?;
      df
    } else {
      return Err(TensorError::InvalidValue("Empty CSV file".to_string()));
    }
  };

  for result in reader.records() {
    let record = result?;
    let row: Vec<DataValue> = record.iter().map(parse_field).collect();
    dataframe.add_row(row)?;
  }

  Ok(dataframe)
}

pub fn parse_field(field: &str) -> DataValue {
  let trimmed = field.trim();

  if trimmed.is_empty() {
    return DataValue::Null;
  }

  if let Ok(int_val) = trimmed.parse::<i64>() {
    return DataValue::Int(int_val);
  }

  if let Ok(float_val) = trimmed.parse::<f64>() {
    return DataValue::Float(float_val);
  }

  // Try parsing as boolean
  match trimmed.to_lowercase().as_str() {
    "true" | "t" | "yes" | "y" | "1" => return DataValue::Bool(true),
    "false" | "f" | "no" | "n" | "0" => return DataValue::Bool(false),
    _ => {}
  }

  DataValue::String(trimmed.to_string())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_data_value_conversions() {
    assert_eq!(DataValue::Int(42).to_f64(), Some(42.0));
    assert_eq!(DataValue::Float(2.5).to_f64(), Some(2.5));
    assert_eq!(DataValue::Bool(true).to_f64(), Some(1.0));
    assert_eq!(DataValue::Bool(false).to_f64(), Some(0.0));
    assert_eq!(DataValue::String("hello".to_string()).to_f64(), None);
  }

  #[test]
  fn test_generic_dataframe() {
    let columns = vec![
      ColumnInfo {
        name: "id".to_string(),
        dtype: DataType::Int64,
        nullable: false,
      },
      ColumnInfo {
        name: "value".to_string(),
        dtype: DataType::Float64,
        nullable: true,
      },
    ];

    let mut df = GenericDataFrame::new(columns);

    df.add_row(vec![DataValue::Int(1), DataValue::Float(1.5)])
      .unwrap();
    df.add_row(vec![DataValue::Int(2), DataValue::Float(2.5)])
      .unwrap();

    assert_eq!(df.len(), 2);
    assert_eq!(df.n_columns(), 2);

    let id_column = df.column("id").unwrap();
    assert_eq!(id_column.len(), 2);
    assert_eq!(id_column[0], DataValue::Int(1));
  }

  #[test]
  fn test_field_parsing() {
    assert_eq!(parse_field("42"), DataValue::Int(42));
    assert_eq!(parse_field("2.5"), DataValue::Float(2.5));
    assert_eq!(parse_field("true"), DataValue::Bool(true));
    assert_eq!(parse_field("hello"), DataValue::String("hello".to_string()));
    assert_eq!(parse_field(""), DataValue::Null);
  }
}
