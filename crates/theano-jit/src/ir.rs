//! SSA-based intermediate representation for JIT compilation.

use std::fmt;

/// Unique identifier for a value in the IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Value(pub usize);

/// Operations in the IR.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    // Constants
    Constant(Vec<f64>, Vec<usize>), // data, shape

    // Unary
    Neg(Value),
    Exp(Value),
    Log(Value),
    Sqrt(Value),
    Tanh(Value),
    Sigmoid(Value),
    Relu(Value),

    // Binary
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Div(Value, Value),

    // Reductions
    Sum(Value),
    Mean(Value),

    // Matrix
    MatMul(Value, Value),

    // Shape
    Reshape(Value, Vec<usize>),
    Transpose(Value, usize, usize),
}

/// A node in the computation graph.
#[derive(Clone, Debug)]
pub struct Node {
    pub id: Value,
    pub op: Op,
    pub shape: Vec<usize>,
    pub name: Option<String>,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Constant(_, shape) => write!(f, "constant({:?})", shape),
            Op::Neg(v) => write!(f, "neg({})", v),
            Op::Exp(v) => write!(f, "exp({})", v),
            Op::Log(v) => write!(f, "log({})", v),
            Op::Sqrt(v) => write!(f, "sqrt({})", v),
            Op::Tanh(v) => write!(f, "tanh({})", v),
            Op::Sigmoid(v) => write!(f, "sigmoid({})", v),
            Op::Relu(v) => write!(f, "relu({})", v),
            Op::Add(a, b) => write!(f, "add({}, {})", a, b),
            Op::Sub(a, b) => write!(f, "sub({}, {})", a, b),
            Op::Mul(a, b) => write!(f, "mul({}, {})", a, b),
            Op::Div(a, b) => write!(f, "div({}, {})", a, b),
            Op::Sum(v) => write!(f, "sum({})", v),
            Op::Mean(v) => write!(f, "mean({})", v),
            Op::MatMul(a, b) => write!(f, "matmul({}, {})", a, b),
            Op::Reshape(v, shape) => write!(f, "reshape({}, {:?})", v, shape),
            Op::Transpose(v, d0, d1) => write!(f, "transpose({}, {}, {})", v, d0, d1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", Value(0)), "%0");
        assert_eq!(format!("{}", Value(42)), "%42");
    }

    #[test]
    fn test_op_display() {
        assert_eq!(
            format!("{}", Op::Constant(vec![1.0], vec![1])),
            "constant([1])"
        );
        assert_eq!(format!("{}", Op::Neg(Value(0))), "neg(%0)");
        assert_eq!(format!("{}", Op::Add(Value(0), Value(1))), "add(%0, %1)");
        assert_eq!(
            format!("{}", Op::MatMul(Value(2), Value(3))),
            "matmul(%2, %3)"
        );
        assert_eq!(
            format!("{}", Op::Reshape(Value(0), vec![2, 3])),
            "reshape(%0, [2, 3])"
        );
        assert_eq!(
            format!("{}", Op::Transpose(Value(0), 0, 1)),
            "transpose(%0, 0, 1)"
        );
    }

    #[test]
    fn test_value_equality() {
        assert_eq!(Value(0), Value(0));
        assert_ne!(Value(0), Value(1));
    }

    #[test]
    fn test_op_clone() {
        let op = Op::Add(Value(0), Value(1));
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }

    #[test]
    fn test_node_creation() {
        let node = Node {
            id: Value(0),
            op: Op::Constant(vec![1.0, 2.0], vec![2]),
            shape: vec![2],
            name: Some("input".to_string()),
        };
        assert_eq!(node.id, Value(0));
        assert_eq!(node.shape, vec![2]);
        assert_eq!(node.name.as_deref(), Some("input"));
    }

    #[test]
    fn test_all_unary_ops_display() {
        let v = Value(0);
        assert_eq!(format!("{}", Op::Exp(v)), "exp(%0)");
        assert_eq!(format!("{}", Op::Log(v)), "log(%0)");
        assert_eq!(format!("{}", Op::Sqrt(v)), "sqrt(%0)");
        assert_eq!(format!("{}", Op::Tanh(v)), "tanh(%0)");
        assert_eq!(format!("{}", Op::Sigmoid(v)), "sigmoid(%0)");
        assert_eq!(format!("{}", Op::Relu(v)), "relu(%0)");
        assert_eq!(format!("{}", Op::Sum(v)), "sum(%0)");
        assert_eq!(format!("{}", Op::Mean(v)), "mean(%0)");
    }

    #[test]
    fn test_all_binary_ops_display() {
        let a = Value(0);
        let b = Value(1);
        assert_eq!(format!("{}", Op::Sub(a, b)), "sub(%0, %1)");
        assert_eq!(format!("{}", Op::Mul(a, b)), "mul(%0, %1)");
        assert_eq!(format!("{}", Op::Div(a, b)), "div(%0, %1)");
    }
}
