//! Graph capture via tracing.

use theano_core::Tensor;
use crate::graph::Graph;
use crate::ir::{Op, Value};

/// Trace a function to capture its computation graph.
/// The function receives traced tensors and the operations are recorded.
pub fn trace(
    input_shapes: &[Vec<usize>],
    f: impl FnOnce(&[Tensor]) -> Tensor,
) -> (Graph, Vec<Value>) {
    let mut graph = Graph::new();

    // Create input nodes
    let mut input_values = Vec::new();
    let mut input_tensors = Vec::new();
    for shape in input_shapes {
        let v = graph.add_node(
            Op::Constant(vec![0.0; shape.iter().product()], shape.clone()),
            shape.clone(),
        );
        input_values.push(v);
        input_tensors.push(Tensor::zeros(shape));
    }

    // Execute the function to determine output shape
    let output = f(&input_tensors);
    let output_shape = output.shape().to_vec();

    // In a full implementation, the trace would intercept tensor operations
    // and record them into the graph. For now, we just record the I/O shapes.

    let output_value = graph.add_node(
        Op::Constant(vec![0.0; output_shape.iter().product()], output_shape.clone()),
        output_shape,
    );

    graph.set_outputs(vec![output_value]);

    (graph, input_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_identity() {
        let (graph, inputs) = trace(&[vec![2, 3]], |tensors| tensors[0].clone());

        assert!(!graph.is_empty());
        assert_eq!(inputs.len(), 1);
        assert_eq!(graph.outputs().len(), 1);
    }

    #[test]
    fn test_trace_records_input_shapes() {
        let (graph, inputs) = trace(&[vec![4], vec![4]], |tensors| tensors[0].clone());

        assert_eq!(inputs.len(), 2);
        // Should have 2 input nodes + 1 output node
        assert_eq!(graph.len(), 3);

        let node0 = graph.get_node(inputs[0]).unwrap();
        assert_eq!(node0.shape, vec![4]);

        let node1 = graph.get_node(inputs[1]).unwrap();
        assert_eq!(node1.shape, vec![4]);
    }

    #[test]
    fn test_trace_output_shape() {
        let (graph, _) = trace(&[vec![3, 4]], |tensors| {
            // Just return the input — output shape should match
            tensors[0].clone()
        });

        let output = graph.outputs()[0];
        let output_node = graph.get_node(output).unwrap();
        assert_eq!(output_node.shape, vec![3, 4]);
    }

    #[test]
    fn test_trace_scalar_output() {
        let (graph, _) = trace(&[vec![5]], |_tensors| Tensor::scalar(0.0));

        let output = graph.outputs()[0];
        let output_node = graph.get_node(output).unwrap();
        assert_eq!(output_node.shape, Vec::<usize>::new());
    }

    #[test]
    fn test_trace_multiple_inputs() {
        let (graph, inputs) = trace(
            &[vec![2, 3], vec![3, 4], vec![1]],
            |tensors| tensors[0].clone(),
        );

        assert_eq!(inputs.len(), 3);
        assert_eq!(graph.get_node(inputs[0]).unwrap().shape, vec![2, 3]);
        assert_eq!(graph.get_node(inputs[1]).unwrap().shape, vec![3, 4]);
        assert_eq!(graph.get_node(inputs[2]).unwrap().shape, vec![1]);
    }
}
