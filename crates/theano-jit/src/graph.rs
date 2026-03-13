//! Computation graph for JIT compilation.

use crate::ir::{Node, Op, Value};

/// A computation graph — sequence of operations in SSA form.
pub struct Graph {
    nodes: Vec<Node>,
    next_id: usize,
    outputs: Vec<Value>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
            outputs: Vec::new(),
        }
    }

    /// Add a node to the graph, returning its value handle.
    pub fn add_node(&mut self, op: Op, shape: Vec<usize>) -> Value {
        let id = Value(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node {
            id,
            op,
            shape,
            name: None,
        });
        id
    }

    /// Mark values as graph outputs.
    pub fn set_outputs(&mut self, outputs: Vec<Value>) {
        self.outputs = outputs;
    }

    /// Get the list of nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Get output values.
    pub fn outputs(&self) -> &[Value] {
        &self.outputs
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get a node by value.
    pub fn get_node(&self, value: Value) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == value)
    }

    /// Pretty-print the graph.
    pub fn dump(&self) -> String {
        let mut s = String::new();
        s.push_str("Graph {\n");
        for node in &self.nodes {
            s.push_str(&format!(
                "  {} = {} // shape={:?}\n",
                node.id, node.op, node.shape
            ));
        }
        s.push_str(&format!("  outputs: {:?}\n", self.outputs));
        s.push_str("}\n");
        s
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g = Graph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert!(g.outputs().is_empty());
    }

    #[test]
    fn test_add_nodes() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0, 2.0], vec![2]), vec![2]);
        let v1 = g.add_node(Op::Constant(vec![3.0, 4.0], vec![2]), vec![2]);
        let v2 = g.add_node(Op::Add(v0, v1), vec![2]);

        assert_eq!(g.len(), 3);
        assert!(!g.is_empty());
        assert_eq!(v0, Value(0));
        assert_eq!(v1, Value(1));
        assert_eq!(v2, Value(2));
    }

    #[test]
    fn test_set_outputs() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        g.set_outputs(vec![v0]);
        assert_eq!(g.outputs(), &[Value(0)]);
    }

    #[test]
    fn test_get_node() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        let node = g.get_node(v0).unwrap();
        assert_eq!(node.id, v0);
        assert_eq!(node.shape, vec![1]);

        assert!(g.get_node(Value(99)).is_none());
    }

    #[test]
    fn test_dump() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        let v1 = g.add_node(Op::Neg(v0), vec![1]);
        g.set_outputs(vec![v1]);

        let dump = g.dump();
        assert!(dump.contains("Graph {"));
        assert!(dump.contains("%0 = constant([1])"));
        assert!(dump.contains("%1 = neg(%0)"));
        assert!(dump.contains("outputs:"));
    }

    #[test]
    fn test_default() {
        let g = Graph::default();
        assert!(g.is_empty());
    }

    #[test]
    fn test_graph_linear_chain() {
        let mut g = Graph::new();
        let input = g.add_node(Op::Constant(vec![0.0; 6], vec![2, 3]), vec![2, 3]);
        let relu = g.add_node(Op::Relu(input), vec![2, 3]);
        let sum = g.add_node(Op::Sum(relu), vec![]);
        g.set_outputs(vec![sum]);

        assert_eq!(g.len(), 3);
        assert_eq!(g.outputs(), &[Value(2)]);

        let sum_node = g.get_node(sum).unwrap();
        assert_eq!(sum_node.shape, Vec::<usize>::new());
    }
}
