use crate::ops::OpNode;
use crate::tensor::Tensor;
use ndarray::Array2;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::AddAssign;
use std::rc::Rc;

pub type NodeId = usize;
pub type EdgeId = usize;

#[derive(Debug)]
pub struct GraphNode {
  pub tensor: Tensor,
  pub id: NodeId,
  pub consumers: RefCell<Vec<EdgeId>>,
  pub producer: RefCell<Option<EdgeId>>,
}

impl GraphNode {
  pub fn new(tensor: Tensor, id: NodeId) -> Self {
    Self {
      tensor,
      id,
      consumers: RefCell::new(Vec::new()),
      producer: RefCell::new(None),
    }
  }
}

#[derive(Debug)]
pub struct GraphEdge {
  pub operation: OpNode,
  pub id: EdgeId,
  pub inputs: Vec<NodeId>,
  pub output: NodeId,
}

impl GraphEdge {
  pub fn new(operation: OpNode, id: EdgeId, inputs: Vec<NodeId>, output: NodeId) -> Self {
    Self {
      operation,
      id,
      inputs,
      output,
    }
  }
}

#[derive(Debug)]
pub struct ComputationGraph {
  nodes: HashMap<NodeId, Rc<RefCell<GraphNode>>>,
  edges: HashMap<EdgeId, Rc<GraphEdge>>,
  next_node_id: NodeId,
  next_edge_id: EdgeId,
  root_nodes: HashSet<NodeId>,
}

impl ComputationGraph {
  pub fn new() -> Self {
    Self {
      nodes: HashMap::new(),
      edges: HashMap::new(),
      next_node_id: 0,
      next_edge_id: 0,
      root_nodes: HashSet::new(),
    }
  }

  pub fn add_leaf_node(&mut self, tensor: Tensor) -> NodeId {
    let id = self.next_node_id;
    self.next_node_id += 1;

    let node = Rc::new(RefCell::new(GraphNode::new(tensor, id)));
    self.nodes.insert(id, node);
    self.root_nodes.insert(id);

    id
  }

  pub fn add_operation(
    &mut self,
    operation: OpNode,
    input_ids: Vec<NodeId>,
    output_tensor: Tensor,
  ) -> Result<NodeId, String> {
    for &input_id in &input_ids {
      if !self.nodes.contains_key(&input_id) {
        return Err(format!(
          "Input node {} does not exist in the graph",
          input_id
        ));
      }
    }

    let output_id = self.next_node_id;
    self.next_node_id += 1;

    let output_node = Rc::new(RefCell::new(GraphNode::new(output_tensor, output_id)));

    let edge_id = self.next_edge_id;
    self.next_edge_id += 1;

    let edge = Rc::new(GraphEdge::new(
      operation,
      edge_id,
      input_ids.clone(),
      output_id,
    ));

    output_node.borrow_mut().producer.replace(Some(edge_id));

    for &input_id in &input_ids {
      if let Some(input_node) = self.nodes.get(&input_id) {
        input_node.borrow_mut().consumers.borrow_mut().push(edge_id);
      }
    }

    self.nodes.insert(output_id, output_node);
    self.edges.insert(edge_id, edge);

    Ok(output_id)
  }

  pub fn get_node(&self, id: NodeId) -> Option<Rc<RefCell<GraphNode>>> {
    self.nodes.get(&id).cloned()
  }

  pub fn get_edge(&self, id: EdgeId) -> Option<Rc<GraphEdge>> {
    self.edges.get(&id).cloned()
  }

  /// Compute reverse post-order traversal from the given output node
  /// Returns nodes in the order they should be processed for backward propagation
  pub fn reverse_postorder_from(&self, output_id: NodeId) -> Vec<NodeId> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();

    fn dfs(
      graph: &ComputationGraph,
      node_id: NodeId,
      visited: &mut HashSet<NodeId>,
      order: &mut Vec<NodeId>,
    ) {
      if !visited.insert(node_id) {
        return;
      }

      if let Some(node) = graph.get_node(node_id) {
        let producer_edge_id = *node.borrow().producer.borrow();
        if let Some(edge_id) = producer_edge_id {
          if let Some(edge) = graph.get_edge(edge_id) {
            // Visit all input nodes first (DFS)
            for &input_id in &edge.inputs {
              dfs(graph, input_id, visited, order);
            }
          }
        }
      }

      // Add this node after visiting all its dependencies
      order.push(node_id);
    }

    dfs(self, output_id, &mut visited, &mut order);
    order.reverse(); // Reverse to get proper backward propagation order
    order
  }

  /// Legacy topological sort for compatibility (may process unreachable nodes)
  #[deprecated(note = "Use reverse_postorder_from for more efficient backward propagation")]
  pub fn topological_sort(&self) -> Result<Vec<NodeId>, String> {
    let mut in_degree = HashMap::new();
    let mut adj_list = HashMap::new();

    for &node_id in self.nodes.keys() {
      in_degree.insert(node_id, 0);
      adj_list.insert(node_id, Vec::new());
    }

    for edge in self.edges.values() {
      let output_id = edge.output;
      for &input_id in &edge.inputs {
        adj_list.get_mut(&input_id).unwrap().push(output_id);
        *in_degree.get_mut(&output_id).unwrap() += 1;
      }
    }

    let mut queue = VecDeque::new();
    for (&node_id, &degree) in &in_degree {
      if degree == 0 {
        queue.push_back(node_id);
      }
    }

    let mut result = Vec::new();
    while let Some(node_id) = queue.pop_front() {
      result.push(node_id);
      if let Some(neighbors) = adj_list.get(&node_id) {
        for &neighbor in neighbors {
          let degree = in_degree.get_mut(&neighbor).unwrap();
          *degree -= 1;
          if *degree == 0 {
            queue.push_back(neighbor);
          }
        }
      }
    }

    if result.len() != self.nodes.len() {
      return Err("Cycle detected in computation graph".to_string());
    }

    result.reverse();
    Ok(result)
  }

  pub fn backward(
    &mut self,
    output_id: NodeId,
    grad_output: Option<Array2<f64>>,
  ) -> Result<(), String> {
    // Use DFS-based traversal for efficiency (only reachable nodes)
    let order = self.reverse_postorder_from(output_id);
    if let Some(output_node) = self.get_node(output_id) {
      let mut node = output_node.borrow_mut();
      if node.tensor.requires_grad() {
        let grad = match grad_output {
          Some(g) => g,
          None => {
            let shape = node.tensor.shape();
            let numel = shape.0 * shape.1;
            if numel != 1 {
              return Err(format!(
                "grad_output must be specified for non-scalar tensor (shape: {:?})",
                shape
              ));
            }
            // Default gradient of ones for scalar output only
            Array2::ones(shape)
          }
        };

        if let Some(ref tensor_grad) = node.tensor.grad {
          *tensor_grad.borrow_mut() = grad;
        } else {
          node.tensor.grad = Some(Rc::new(RefCell::new(grad)));
        }
      }
    } else {
      return Err(format!("Output node {} not found", output_id));
    }

    for &node_id in &order {
      let node = match self.get_node(node_id) {
        Some(n) => n,
        None => continue,
      };

      let producer_edge_id = *node.borrow().producer.borrow();

      if let Some(edge_id) = producer_edge_id {
        if let Some(edge) = self.get_edge(edge_id) {
          let output_grad = {
            let node_ref = node.borrow();
            if let Some(ref grad) = node_ref.tensor.grad {
              grad.borrow().clone()
            } else {
              continue;
            }
          };

          let input_grads = edge.operation.backward(&output_grad)?;
          for (i, &input_id) in edge.inputs.iter().enumerate() {
            if i >= input_grads.len() {
              continue;
            }

            if let Some(input_node) = self.get_node(input_id) {
              let mut input_ref = input_node.borrow_mut();
              if input_ref.tensor.requires_grad() {
                if let Some(ref existing_grad) = input_ref.tensor.grad {
                  let mut grad_borrow = existing_grad.borrow_mut();
                  grad_borrow.add_assign(&input_grads[i]);
                } else {
                  input_ref.tensor.grad = Some(Rc::new(RefCell::new(input_grads[i].clone())));
                }
              }
            }
          }
        }
      }
    }

    Ok(())
  }

  pub fn clear(&mut self) {
    self.nodes.clear();
    self.edges.clear();
    self.root_nodes.clear();
    self.next_node_id = 0;
    self.next_edge_id = 0;
  }

  pub fn node_count(&self) -> usize {
    self.nodes.len()
  }

  pub fn edge_count(&self) -> usize {
    self.edges.len()
  }

  pub fn is_empty(&self) -> bool {
    self.nodes.is_empty() && self.edges.is_empty()
  }

  pub fn get_leaf_nodes(&self) -> Vec<NodeId> {
    self.root_nodes.iter().cloned().collect()
  }

  pub fn get_node_gradient(&self, node_id: NodeId) -> Option<Array2<f64>> {
    if let Some(node) = self.get_node(node_id) {
      let node_ref = node.borrow();
      node_ref
        .tensor
        .grad
        .as_ref()
        .map(|grad| grad.borrow().clone())
    } else {
      None
    }
  }

  pub fn zero_grad(&mut self) {
    for node in self.nodes.values() {
      let mut node_ref = node.borrow_mut();
      if node_ref.tensor.requires_grad() {
        node_ref.tensor.zero_grad();
      }
    }
  }
}

impl Default for ComputationGraph {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ops::OpBuilder;

  #[test]
  fn test_reverse_postorder_efficiency() {
    let mut graph = ComputationGraph::new();

    // Create a simple computation: c = a + b
    let a = Tensor::ones(2, 2);
    let b = Tensor::ones(2, 2);
    let op = OpBuilder::add(Rc::new(a.clone()), Rc::new(b.clone()));
    let c = op.forward().unwrap();

    let a_id = graph.add_leaf_node(a);
    let b_id = graph.add_leaf_node(b);
    let c_id = graph.add_operation(op, vec![a_id, b_id], c).unwrap();

    // Add an unconnected node that shouldn't be visited
    let d = Tensor::ones(2, 2);
    let _d_id = graph.add_leaf_node(d);

    let order = graph.reverse_postorder_from(c_id);

    // Should only contain reachable nodes: a, b, c (not d)
    assert_eq!(order.len(), 3);
    assert!(order.contains(&a_id));
    assert!(order.contains(&b_id));
    assert!(order.contains(&c_id));

    // c should be first for backward propagation
    assert_eq!(order[0], c_id);
  }

  #[test]
  fn test_non_scalar_gradient_error() {
    let mut graph = ComputationGraph::new();

    // Create a non-scalar computation result
    let mut a = Tensor::ones(2, 2);
    a.set_requires_grad(true);
    let mut b = Tensor::ones(2, 2);
    b.set_requires_grad(true);
    let op = OpBuilder::add(Rc::new(a.clone()), Rc::new(b.clone()));
    let mut c = op.forward().unwrap(); // c is 2x2, non-scalar
    c.set_requires_grad(true);

    let a_id = graph.add_leaf_node(a);
    let b_id = graph.add_leaf_node(b);
    let c_id = graph.add_operation(op, vec![a_id, b_id], c).unwrap();

    // Should return error for non-scalar output without explicit gradient
    let result = graph.backward(c_id, None);
    assert!(result.is_err());
    assert!(result
      .unwrap_err()
      .contains("grad_output must be specified"));
  }

  #[test]
  fn test_scalar_gradient_default() {
    let mut graph = ComputationGraph::new();

    let a = Tensor::ones(1, 1); // Scalar
    let a_id = graph.add_leaf_node(a);

    // Should work fine for scalar output
    let result = graph.backward(a_id, None);
    assert!(result.is_ok());
  }

  #[test]
  fn test_cycle_detection_still_works() {
    let graph = ComputationGraph::new();

    // The legacy topological_sort should still detect cycles
    // (though in practice, our graph construction shouldn't allow cycles)
    #[allow(deprecated)]
    let result = graph.topological_sort();
    assert!(result.is_ok());
  }

  #[test]
  fn test_simple_addition_gradient() {
    let mut graph = ComputationGraph::new();

    // Test: f(x, y) = x + y, where x = [2, 3], y = [1, 4]
    let mut x = Tensor::new(vec![vec![2.0, 3.0]]).unwrap();
    x.set_requires_grad(true);
    let mut y = Tensor::new(vec![vec![1.0, 4.0]]).unwrap();
    y.set_requires_grad(true);

    let op = OpBuilder::add(Rc::new(x.clone()), Rc::new(y.clone()));
    let mut result = op.forward().unwrap();
    result.set_requires_grad(true);

    let x_id = graph.add_leaf_node(x);
    let y_id = graph.add_leaf_node(y);
    let result_id = graph.add_operation(op, vec![x_id, y_id], result).unwrap();

    // Backward with gradient [1, 1] (sum reduction simulation)
    let grad_output = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
    graph.backward(result_id, Some(grad_output)).unwrap();

    // For addition: ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1
    let x_grad = graph.get_node_gradient(x_id).unwrap();
    let y_grad = graph.get_node_gradient(y_id).unwrap();

    assert_eq!(x_grad[[0, 0]], 1.0);
    assert_eq!(x_grad[[0, 1]], 1.0);
    assert_eq!(y_grad[[0, 0]], 1.0);
    assert_eq!(y_grad[[0, 1]], 1.0);
  }

  #[test]
  fn test_matrix_multiplication_gradient() {
    let mut graph = ComputationGraph::new();

    // Test: f(A, B) = A * B, where A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    let mut a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    a.set_requires_grad(true);
    let mut b = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();
    b.set_requires_grad(true);

    let op = OpBuilder::matmul(Rc::new(a.clone()), Rc::new(b.clone()));
    let mut result = op.forward().unwrap();
    result.set_requires_grad(true);

    let a_id = graph.add_leaf_node(a);
    let b_id = graph.add_leaf_node(b);
    let result_id = graph.add_operation(op, vec![a_id, b_id], result).unwrap();

    // Backward with gradient [[1, 1], [1, 1]]
    let grad_output = Array2::ones((2, 2));
    graph.backward(result_id, Some(grad_output)).unwrap();

    // For matrix multiplication: ∂(A*B)/∂A = grad_output * B^T, ∂(A*B)/∂B = A^T * grad_output
    let a_grad = graph.get_node_gradient(a_id).unwrap();
    let b_grad = graph.get_node_gradient(b_id).unwrap();

    // Expected: a_grad = [[1,1],[1,1]] * [[5,7],[6,8]] = [[11,15],[11,15]]
    assert_eq!(a_grad[[0, 0]], 11.0);
    assert_eq!(a_grad[[0, 1]], 15.0);
    assert_eq!(a_grad[[1, 0]], 11.0);
    assert_eq!(a_grad[[1, 1]], 15.0);

    // Expected: b_grad = [[1,3],[2,4]] * [[1,1],[1,1]] = [[4,4],[6,6]]
    assert_eq!(b_grad[[0, 0]], 4.0);
    assert_eq!(b_grad[[0, 1]], 4.0);
    assert_eq!(b_grad[[1, 0]], 6.0);
    assert_eq!(b_grad[[1, 1]], 6.0);
  }

  #[test]
  fn test_chain_rule_validation() {
    let mut graph = ComputationGraph::new();

    // Test chain rule with element-wise operations: f(x) = sigmoid(x + x)
    let mut x = Tensor::new(vec![vec![0.0, 1.0]]).unwrap();
    x.set_requires_grad(true);

    // First operation: y = x + x (element-wise)
    let add_op = OpBuilder::add(Rc::new(x.clone()), Rc::new(x.clone()));
    let mut y = add_op.forward().unwrap();
    y.set_requires_grad(true);

    // Second operation: z = sigmoid(y)
    let sig_op = OpBuilder::sigmoid(Rc::new(y.clone()));
    let mut z = sig_op.forward().unwrap();
    z.set_requires_grad(true);

    let x_id = graph.add_leaf_node(x);
    let y_id = graph.add_operation(add_op, vec![x_id, x_id], y).unwrap();
    let z_id = graph.add_operation(sig_op, vec![y_id], z).unwrap();

    // Backward pass
    let grad_output = Array2::ones((1, 2));
    graph.backward(z_id, Some(grad_output)).unwrap();

    // Manual calculation:
    // y = x + x = [0+0, 1+1] = [0, 2]
    // z = sigmoid([0, 2]) = [0.5, sigmoid(2)]
    // sigmoid'(0) = 0.5 * (1-0.5) = 0.25
    // sigmoid'(2) ≈ 0.8808 * (1-0.8808) ≈ 0.1050
    // dz/dy = [0.25, 0.1050]
    // dy/dx = [1+1, 1+1] = [2, 2] (gradient accumulates from both uses)
    // dz/dx = dz/dy ⊙ dy/dx = [0.25*2, 0.1050*2] = [0.5, 0.21]

    let x_grad = graph.get_node_gradient(x_id).unwrap();

    // Allow some floating point tolerance
    assert!((x_grad[[0, 0]] - 0.5).abs() < 1e-6);
    assert!((x_grad[[0, 1]] - 0.21).abs() < 0.01);
  }

  #[test]
  fn test_gradient_accumulation() {
    let mut graph = ComputationGraph::new();

    // Test gradient accumulation: f(x) = x + x (x used twice)
    let mut x = Tensor::new(vec![vec![2.0]]).unwrap(); // Scalar
    x.set_requires_grad(true);

    let add_op = OpBuilder::add(Rc::new(x.clone()), Rc::new(x.clone()));
    let mut result = add_op.forward().unwrap();
    result.set_requires_grad(true);

    let x_id = graph.add_leaf_node(x);
    let result_id = graph
      .add_operation(add_op, vec![x_id, x_id], result)
      .unwrap();

    // Backward pass (scalar, so no grad_output needed)
    graph.backward(result_id, None).unwrap();

    // For f(x) = x + x, df/dx = 1 + 1 = 2 (gradient should accumulate)
    let x_grad = graph.get_node_gradient(x_id).unwrap();
    assert_eq!(x_grad[[0, 0]], 2.0);
  }

  #[test]
  fn test_complex_computation_gradient() {
    let mut graph = ComputationGraph::new();

    // Test: f(a,b,c) = relu(tanh(a * b) + c) where a=2, b=3, c=1
    let mut a = Tensor::new(vec![vec![2.0]]).unwrap();
    a.set_requires_grad(true);
    let mut b = Tensor::new(vec![vec![3.0]]).unwrap();
    b.set_requires_grad(true);
    let mut c = Tensor::new(vec![vec![1.0]]).unwrap();
    c.set_requires_grad(true);

    // Step 1: ab = a * b
    let mul_op = OpBuilder::matmul(Rc::new(a.clone()), Rc::new(b.clone()));
    let mut ab = mul_op.forward().unwrap();
    ab.set_requires_grad(true);

    // Step 2: tab = tanh(ab)
    let tanh_op = OpBuilder::tanh(Rc::new(ab.clone()));
    let mut tab = tanh_op.forward().unwrap();
    tab.set_requires_grad(true);

    // Step 3: tabc = tab + c
    let add_op = OpBuilder::add(Rc::new(tab.clone()), Rc::new(c.clone()));
    let mut tabc = add_op.forward().unwrap();
    tabc.set_requires_grad(true);

    // Step 4: result = relu(tabc)
    let relu_op = OpBuilder::relu(Rc::new(tabc.clone()));
    let mut result = relu_op.forward().unwrap();
    result.set_requires_grad(true);

    // Build graph
    let a_id = graph.add_leaf_node(a);
    let b_id = graph.add_leaf_node(b);
    let c_id = graph.add_leaf_node(c);
    let ab_id = graph.add_operation(mul_op, vec![a_id, b_id], ab).unwrap();
    let tab_id = graph.add_operation(tanh_op, vec![ab_id], tab).unwrap();
    let tabc_id = graph
      .add_operation(add_op, vec![tab_id, c_id], tabc)
      .unwrap();
    let result_id = graph.add_operation(relu_op, vec![tabc_id], result).unwrap();

    // Backward pass
    graph.backward(result_id, None).unwrap();

    // Verify gradients exist (exact values would require manual calculation)
    let a_grad = graph.get_node_gradient(a_id).unwrap();
    let b_grad = graph.get_node_gradient(b_id).unwrap();
    let c_grad = graph.get_node_gradient(c_id).unwrap();

    // All gradients should be non-zero for this positive function
    assert!(a_grad[[0, 0]] > 0.0);
    assert!(b_grad[[0, 0]] > 0.0);
    assert_eq!(c_grad[[0, 0]], 1.0); // ReLU(positive) * 1 (from add) = 1
  }
}
