use crate::ops::OpNode;
use crate::tensor::Tensor;
use ndarray::Array2;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
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

  pub fn topological_sort(&self) -> Result<Vec<NodeId>, String> {
    let mut in_degree = HashMap::new();
    let mut adj_list = HashMap::new();

    for &node_id in self.nodes.keys() {
      in_degree.insert(node_id, 0);
      adj_list.insert(node_id, Vec::new());
    }

    for edge in self.edges.values() {
      for &input_id in &edge.inputs {
        adj_list.get_mut(&input_id).unwrap().push(edge.output);
        *in_degree.get_mut(&edge.output).unwrap() += 1;
      }
    }

    // Kahn's algorithm
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    for (&node_id, &degree) in &in_degree {
      if degree == 0 {
        queue.push_back(node_id);
      }
    }

    while let Some(node_id) = queue.pop_front() {
      result.push(node_id);

      for &neighbor in &adj_list[&node_id] {
        let degree = in_degree.get_mut(&neighbor).unwrap();
        *degree -= 1;
        if *degree == 0 {
          queue.push_back(neighbor);
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
    let topo_order = self.topological_sort()?;

    let start_idx = topo_order
      .iter()
      .position(|&id| id == output_id)
      .ok_or_else(|| format!("Output node {} not found in graph", output_id))?;
    if let Some(output_node) = self.get_node(output_id) {
      let mut node = output_node.borrow_mut();
      if node.tensor.requires_grad() {
        let grad = match grad_output {
          Some(g) => g,
          None => {
            // Default gradient of ones for scalar output
            let shape = node.tensor.shape();
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

    for &node_id in &topo_order[start_idx..] {
      let node = match self.get_node(node_id) {
        Some(n) => n,
        None => continue,
      };

      let producer_edge_id = node.borrow().producer.borrow().clone();

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
                  *grad_borrow = &*grad_borrow + &input_grads[i];
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
      if let Some(ref grad) = node_ref.tensor.grad {
        Some(grad.borrow().clone())
      } else {
        None
      }
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
