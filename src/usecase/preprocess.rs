use crate::core::Result;
use crate::domain::types::{Dataset, FeatureStats};

pub fn preprocess_dataset(dataset: &mut Dataset) -> Result<Option<FeatureStats>> {
  dataset.preprocess()?;
  Ok(dataset.get_stats().cloned())
}
