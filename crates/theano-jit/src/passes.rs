//! JIT optimization passes.

use crate::graph::Graph;
use crate::ir::{Op, Value};

/// Dead code elimination: remove nodes not reachable from outputs.
pub fn dead_code_elimination(graph: &Graph) -> Graph {
    let mut live = std::collections::HashSet::new();

    // Mark outputs as live
    for &out in graph.outputs() {
        mark_live(graph, out, &mut live);
    }

    // Rebuild graph with only live nodes
    let mut new_graph = Graph::new();
    let mut value_map = std::collections::HashMap::new();

    for node in graph.nodes() {
        if live.contains(&node.id) {
            let remapped_op = remap_op(&node.op, &value_map);
            let new_id = new_graph.add_node(remapped_op, node.shape.clone());
            value_map.insert(node.id, new_id);
        }
    }

    let new_outputs: Vec<Value> = graph
        .outputs()
        .iter()
        .filter_map(|v| value_map.get(v).copied())
        .collect();
    new_graph.set_outputs(new_outputs);

    new_graph
}

fn mark_live(graph: &Graph, value: Value, live: &mut std::collections::HashSet<Value>) {
    if live.contains(&value) {
        return;
    }
    live.insert(value);

    if let Some(node) = graph.get_node(value) {
        for dep in op_dependencies(&node.op) {
            mark_live(graph, dep, live);
        }
    }
}

fn op_dependencies(op: &Op) -> Vec<Value> {
    match op {
        Op::Constant(_, _) => vec![],
        Op::Neg(v)
        | Op::Exp(v)
        | Op::Log(v)
        | Op::Sqrt(v)
        | Op::Tanh(v)
        | Op::Sigmoid(v)
        | Op::Relu(v)
        | Op::Sum(v)
        | Op::Mean(v)
        | Op::Reshape(v, _) => vec![*v],
        Op::Transpose(v, _, _) => vec![*v],
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::MatMul(a, b) => vec![*a, *b],
    }
}

fn remap_op(op: &Op, map: &std::collections::HashMap<Value, Value>) -> Op {
    let r = |v: &Value| *map.get(v).unwrap_or(v);
    match op {
        Op::Constant(d, s) => Op::Constant(d.clone(), s.clone()),
        Op::Neg(v) => Op::Neg(r(v)),
        Op::Exp(v) => Op::Exp(r(v)),
        Op::Log(v) => Op::Log(r(v)),
        Op::Sqrt(v) => Op::Sqrt(r(v)),
        Op::Tanh(v) => Op::Tanh(r(v)),
        Op::Sigmoid(v) => Op::Sigmoid(r(v)),
        Op::Relu(v) => Op::Relu(r(v)),
        Op::Sum(v) => Op::Sum(r(v)),
        Op::Mean(v) => Op::Mean(r(v)),
        Op::Add(a, b) => Op::Add(r(a), r(b)),
        Op::Sub(a, b) => Op::Sub(r(a), r(b)),
        Op::Mul(a, b) => Op::Mul(r(a), r(b)),
        Op::Div(a, b) => Op::Div(r(a), r(b)),
        Op::MatMul(a, b) => Op::MatMul(r(a), r(b)),
        Op::Reshape(v, s) => Op::Reshape(r(v), s.clone()),
        Op::Transpose(v, d0, d1) => Op::Transpose(r(v), *d0, *d1),
    }
}

/// Constant folding: evaluate constant expressions at compile time.
pub fn constant_folding(graph: &Graph) -> Graph {
    // Stub — would evaluate constant subgraphs
    // For now, return unchanged
    let mut new_graph = Graph::new();
    for node in graph.nodes() {
        new_graph.add_node(node.op.clone(), node.shape.clone());
    }
    new_graph.set_outputs(graph.outputs().to_vec());
    new_graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dce_removes_dead_nodes() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        let _v1 = g.add_node(Op::Constant(vec![2.0], vec![1]), vec![1]); // dead
        let v2 = g.add_node(Op::Neg(v0), vec![1]);
        g.set_outputs(vec![v2]);

        let optimized = dead_code_elimination(&g);
        assert_eq!(optimized.len(), 2); // v0 and neg(v0), not v1
        assert_eq!(optimized.outputs().len(), 1);
    }

    #[test]
    fn test_dce_keeps_all_live_nodes() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        let v1 = g.add_node(Op::Constant(vec![2.0], vec![1]), vec![1]);
        let v2 = g.add_node(Op::Add(v0, v1), vec![1]);
        g.set_outputs(vec![v2]);

        let optimized = dead_code_elimination(&g);
        assert_eq!(optimized.len(), 3); // all nodes are live
    }

    #[test]
    fn test_dce_empty_graph() {
        let g = Graph::new();
        let optimized = dead_code_elimination(&g);
        assert!(optimized.is_empty());
        assert!(optimized.outputs().is_empty());
    }

    #[test]
    fn test_dce_multiple_dead_nodes() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        let _v1 = g.add_node(Op::Constant(vec![2.0], vec![1]), vec![1]); // dead
        let _v2 = g.add_node(Op::Constant(vec![3.0], vec![1]), vec![1]); // dead
        let _v3 = g.add_node(Op::Neg(_v1), vec![1]); // dead (depends on dead)
        let v4 = g.add_node(Op::Exp(v0), vec![1]);
        g.set_outputs(vec![v4]);

        let optimized = dead_code_elimination(&g);
        assert_eq!(optimized.len(), 2); // v0 and exp(v0)
    }

    #[test]
    fn test_dce_multiple_outputs() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        let v1 = g.add_node(Op::Constant(vec![2.0], vec![1]), vec![1]);
        let _v2 = g.add_node(Op::Constant(vec![3.0], vec![1]), vec![1]); // dead
        let v3 = g.add_node(Op::Neg(v0), vec![1]);
        let v4 = g.add_node(Op::Exp(v1), vec![1]);
        g.set_outputs(vec![v3, v4]);

        let optimized = dead_code_elimination(&g);
        assert_eq!(optimized.len(), 4); // v0, v1, neg(v0), exp(v1)
        assert_eq!(optimized.outputs().len(), 2);
    }

    #[test]
    fn test_constant_folding_preserves_graph() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        let v1 = g.add_node(Op::Neg(v0), vec![1]);
        g.set_outputs(vec![v1]);

        let folded = constant_folding(&g);
        assert_eq!(folded.len(), 2);
        assert_eq!(folded.outputs().len(), 1);
    }

    #[test]
    fn test_dce_with_binary_chain() {
        let mut g = Graph::new();
        let v0 = g.add_node(Op::Constant(vec![1.0], vec![1]), vec![1]);
        let v1 = g.add_node(Op::Constant(vec![2.0], vec![1]), vec![1]);
        let v2 = g.add_node(Op::Add(v0, v1), vec![1]);
        let v3 = g.add_node(Op::Mul(v2, v0), vec![1]);
        let _v4 = g.add_node(Op::Constant(vec![99.0], vec![1]), vec![1]); // dead
        g.set_outputs(vec![v3]);

        let optimized = dead_code_elimination(&g);
        assert_eq!(optimized.len(), 4); // v0, v1, add, mul
    }
}
