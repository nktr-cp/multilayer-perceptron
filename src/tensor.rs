use ndarray::Array2;
use rand::{thread_rng, Rng};
use std::fmt;

#[derive(Clone)]
pub struct Tensor {
    pub data: Array2<f64>,
    pub grad: Option<Array2<f64>>,
    requires_grad: bool,
}

impl Tensor {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };

        let mut array = Array2::zeros((rows, cols));
        for (i, row) in data.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                array[[i, j]] = value;
            }
        }

        Self {
            data: array,
            grad: None,
            requires_grad: false,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: Array2::zeros((rows, cols)),
            grad: None,
            requires_grad: false,
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: Array2::ones((rows, cols)),
            grad: None,
            requires_grad: false,
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-1.0..1.0));

        Self {
            data,
            grad: None,
            requires_grad: false,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        let shape = self.data.dim();
        (shape.0, shape.1)
    }

    pub fn dim(&self) -> (usize, usize) {
        self.shape()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn zero_grad(&mut self) {
        if let Some(ref mut grad) = self.grad {
            grad.fill(0.0);
        }
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if requires_grad && self.grad.is_none() {
            self.grad = Some(Array2::zeros(self.data.dim()));
        } else if !requires_grad {
            self.grad = None;
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("data", &self.data)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tensor = Tensor::new(data);

        assert_eq!(tensor.shape(), (2, 2));
        assert_eq!(tensor.data[[0, 0]], 1.0);
        assert_eq!(tensor.data[[0, 1]], 2.0);
        assert_eq!(tensor.data[[1, 0]], 3.0);
        assert_eq!(tensor.data[[1, 1]], 4.0);
        assert!(!tensor.requires_grad());
        assert!(tensor.grad.is_none());
    }

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros(3, 4);

        assert_eq!(tensor.shape(), (3, 4));
        assert_eq!(tensor.len(), 12);

        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(tensor.data[[i, j]], 0.0);
            }
        }
    }

    #[test]
    fn test_ones() {
        let tensor = Tensor::ones(2, 3);

        assert_eq!(tensor.shape(), (2, 3));
        assert_eq!(tensor.len(), 6);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(tensor.data[[i, j]], 1.0);
            }
        }
    }

    #[test]
    fn test_random() {
        let tensor = Tensor::random(2, 3);

        assert_eq!(tensor.shape(), (2, 3));
        assert_eq!(tensor.len(), 6);

        // Check that values are in the expected range
        for i in 0..2 {
            for j in 0..3 {
                let value = tensor.data[[i, j]];
                assert!((-1.0..1.0).contains(&value));
            }
        }
    }

    #[test]
    fn test_shape_and_dim() {
        let tensor = Tensor::zeros(5, 7);
        assert_eq!(tensor.shape(), (5, 7));
        assert_eq!(tensor.dim(), (5, 7));
        assert_eq!(tensor.len(), 35);
    }

    #[test]
    fn test_is_empty() {
        let empty_tensor = Tensor::zeros(0, 0);
        assert!(empty_tensor.is_empty());

        let non_empty_tensor = Tensor::zeros(1, 1);
        assert!(!non_empty_tensor.is_empty());
    }

    #[test]
    fn test_gradient_management() {
        let mut tensor = Tensor::ones(2, 2);

        // Initially no gradient required
        assert!(!tensor.requires_grad());
        assert!(tensor.grad.is_none());

        // Enable gradient computation
        tensor.set_requires_grad(true);
        assert!(tensor.requires_grad());
        assert!(tensor.grad.is_some());

        // Check gradient is initialized to zeros
        if let Some(ref grad) = tensor.grad {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(grad[[i, j]], 0.0);
                }
            }
        }

        // Disable gradient computation
        tensor.set_requires_grad(false);
        assert!(!tensor.requires_grad());
        assert!(tensor.grad.is_none());
    }

    #[test]
    fn test_zero_grad() {
        let mut tensor = Tensor::ones(2, 2);
        tensor.set_requires_grad(true);

        // Modify gradient manually to test zero_grad
        if let Some(ref mut grad) = tensor.grad {
            grad[[0, 0]] = 5.0;
            grad[[1, 1]] = 10.0;
        }

        tensor.zero_grad();

        if let Some(ref grad) = tensor.grad {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(grad[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_clone() {
        let tensor = Tensor::random(2, 3);
        let cloned = tensor.clone();

        assert_eq!(tensor.shape(), cloned.shape());
        assert_eq!(tensor.requires_grad(), cloned.requires_grad());

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(tensor.data[[i, j]], cloned.data[[i, j]]);
            }
        }
    }

    #[test]
    fn test_debug_format() {
        let tensor = Tensor::ones(2, 2);
        let debug_str = format!("{:?}", tensor);

        assert!(debug_str.contains("Tensor"));
        assert!(debug_str.contains("shape"));
        assert!(debug_str.contains("data"));
        assert!(debug_str.contains("requires_grad"));
        assert!(debug_str.contains("has_grad"));
    }
}
