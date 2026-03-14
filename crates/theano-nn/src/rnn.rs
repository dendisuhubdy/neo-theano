//! Recurrent neural network cells.

use theano_autograd::Variable;
use theano_core::Tensor;
use theano_types::{Device, Result};
use crate::init;
use crate::module::Module;

/// RNN Cell. Like `torch.nn.RNNCell`.
/// h' = tanh(x @ W_ih^T + h @ W_hh^T + b)
pub struct RNNCell {
    input_size: usize,
    hidden_size: usize,
    w_ih: Variable,
    w_hh: Variable,
    b_ih: Variable,
    b_hh: Variable,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            w_ih: init::kaiming_uniform(&[hidden_size, input_size], input_size),
            w_hh: init::kaiming_uniform(&[hidden_size, hidden_size], hidden_size),
            b_ih: init::zeros(&[hidden_size]),
            b_hh: init::zeros(&[hidden_size]),
        }
    }

    /// Move this layer to a different device, returning a new RNNCell.
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            w_ih: self.w_ih.to(device)?,
            w_hh: self.w_hh.to(device)?,
            b_ih: self.b_ih.to(device)?,
            b_hh: self.b_hh.to(device)?,
        })
    }

    /// Move to CPU.
    pub fn cpu(&self) -> Result<Self> {
        self.to(&Device::Cpu)
    }

    /// Move to CUDA device 0.
    pub fn cuda(&self) -> Result<Self> {
        self.to(&Device::Cuda(0))
    }

    /// Forward: (input, hidden) -> new_hidden
    /// input: [batch, input_size], hidden: [batch, hidden_size]
    pub fn forward_cell(&self, input: &Variable, hidden: &Variable) -> Variable {
        let w_ih_t = self.w_ih.t().unwrap();
        let w_hh_t = self.w_hh.t().unwrap();

        let ih = input.matmul(&w_ih_t).unwrap().add(&self.b_ih).unwrap();
        let hh = hidden.matmul(&w_hh_t).unwrap().add(&self.b_hh).unwrap();
        ih.add(&hh).unwrap().tanh().unwrap()
    }
}

impl Module for RNNCell {
    fn forward(&self, input: &Variable) -> Variable {
        // Default: zero hidden state
        let batch = input.tensor().shape()[0];
        let hidden = Variable::new(Tensor::zeros(&[batch, self.hidden_size]));
        self.forward_cell(input, &hidden)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.w_ih.clone(), self.w_hh.clone(), self.b_ih.clone(), self.b_hh.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![
            ("w_ih".to_string(), self.w_ih.clone()),
            ("w_hh".to_string(), self.w_hh.clone()),
            ("b_ih".to_string(), self.b_ih.clone()),
            ("b_hh".to_string(), self.b_hh.clone()),
        ]
    }
}

/// LSTM Cell. Like `torch.nn.LSTMCell`.
pub struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    w_ih: Variable, // [4*hidden, input]
    w_hh: Variable, // [4*hidden, hidden]
    b_ih: Variable, // [4*hidden]
    b_hh: Variable, // [4*hidden]
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let gate_size = 4 * hidden_size;
        Self {
            input_size,
            hidden_size,
            w_ih: init::kaiming_uniform(&[gate_size, input_size], input_size),
            w_hh: init::kaiming_uniform(&[gate_size, hidden_size], hidden_size),
            b_ih: init::zeros(&[gate_size]),
            b_hh: init::zeros(&[gate_size]),
        }
    }

    /// Move this layer to a different device, returning a new LSTMCell.
    pub fn to(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            w_ih: self.w_ih.to(device)?,
            w_hh: self.w_hh.to(device)?,
            b_ih: self.b_ih.to(device)?,
            b_hh: self.b_hh.to(device)?,
        })
    }

    /// Move to CPU.
    pub fn cpu(&self) -> Result<Self> {
        self.to(&Device::Cpu)
    }

    /// Move to CUDA device 0.
    pub fn cuda(&self) -> Result<Self> {
        self.to(&Device::Cuda(0))
    }

    /// Reconstruct an LSTMCell from pre-trained tensors.
    pub fn from_tensors(w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Self {
        let gate_size = w_ih.shape()[0];
        let input_size = w_ih.shape()[1];
        let hidden_size = gate_size / 4;
        Self {
            input_size,
            hidden_size,
            w_ih: Variable::requires_grad(w_ih),
            w_hh: Variable::requires_grad(w_hh),
            b_ih: Variable::requires_grad(b_ih),
            b_hh: Variable::requires_grad(b_hh),
        }
    }

    /// Forward: (input, (h, c)) -> (h', c')
    pub fn forward_cell(&self, input: &Variable, h: &Variable, c: &Variable) -> (Variable, Variable) {
        let w_ih_t = self.w_ih.t().unwrap();
        let w_hh_t = self.w_hh.t().unwrap();

        let gates = input.matmul(&w_ih_t).unwrap()
            .add(&self.b_ih).unwrap()
            .add(&h.matmul(&w_hh_t).unwrap()).unwrap()
            .add(&self.b_hh).unwrap();

        // Split into 4 gates: i, f, g, o
        let gates_data = gates.tensor().to_vec_f64().unwrap();
        let batch = input.tensor().shape()[0];
        let hs = self.hidden_size;

        let mut i_data = vec![0.0f64; batch * hs];
        let mut f_data = vec![0.0f64; batch * hs];
        let mut g_data = vec![0.0f64; batch * hs];
        let mut o_data = vec![0.0f64; batch * hs];

        for b in 0..batch {
            for j in 0..hs {
                let base = b * 4 * hs;
                i_data[b * hs + j] = sigmoid(gates_data[base + j]);
                f_data[b * hs + j] = sigmoid(gates_data[base + hs + j]);
                g_data[b * hs + j] = gates_data[base + 2 * hs + j].tanh();
                o_data[b * hs + j] = sigmoid(gates_data[base + 3 * hs + j]);
            }
        }

        let c_data = c.tensor().to_vec_f64().unwrap();
        let new_c: Vec<f64> = (0..batch * hs).map(|i| {
            f_data[i] * c_data[i] + i_data[i] * g_data[i]
        }).collect();

        let new_h: Vec<f64> = (0..batch * hs).map(|i| {
            o_data[i] * new_c[i].tanh()
        }).collect();

        (
            Variable::new(Tensor::from_slice(&new_h, &[batch, hs])),
            Variable::new(Tensor::from_slice(&new_c, &[batch, hs])),
        )
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Module for LSTMCell {
    fn forward(&self, input: &Variable) -> Variable {
        let batch = input.tensor().shape()[0];
        let h = Variable::new(Tensor::zeros(&[batch, self.hidden_size]));
        let c = Variable::new(Tensor::zeros(&[batch, self.hidden_size]));
        let (new_h, _new_c) = self.forward_cell(input, &h, &c);
        new_h
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.w_ih.clone(), self.w_hh.clone(), self.b_ih.clone(), self.b_hh.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        vec![
            ("w_ih".to_string(), self.w_ih.clone()),
            ("w_hh".to_string(), self.w_hh.clone()),
            ("b_ih".to_string(), self.b_ih.clone()),
            ("b_hh".to_string(), self.b_hh.clone()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_cell() {
        let cell = RNNCell::new(10, 20);
        let input = Variable::new(Tensor::ones(&[3, 10]));
        let output = cell.forward(&input);
        assert_eq!(output.tensor().shape(), &[3, 20]);
    }

    #[test]
    fn test_lstm_cell() {
        let cell = LSTMCell::new(10, 20);
        let input = Variable::new(Tensor::ones(&[3, 10]));
        let h = Variable::new(Tensor::zeros(&[3, 20]));
        let c = Variable::new(Tensor::zeros(&[3, 20]));
        let (new_h, new_c) = cell.forward_cell(&input, &h, &c);
        assert_eq!(new_h.tensor().shape(), &[3, 20]);
        assert_eq!(new_c.tensor().shape(), &[3, 20]);
    }

    #[test]
    fn test_rnn_cell_params() {
        let cell = RNNCell::new(10, 20);
        assert_eq!(cell.parameters().len(), 4);
    }

    #[test]
    fn test_rnn_cell_to_device() {
        let cell = RNNCell::new(10, 20);
        let cell_gpu = cell.to(&Device::Cuda(0)).unwrap();
        for param in cell_gpu.parameters() {
            assert_eq!(param.device(), &Device::Cuda(0));
        }

        let cell_cpu = cell_gpu.cpu().unwrap();
        for param in cell_cpu.parameters() {
            assert_eq!(param.device(), &Device::Cpu);
        }
    }

    #[test]
    fn test_rnn_cell_named_parameters() {
        let cell = RNNCell::new(10, 20);
        let named = cell.named_parameters();
        assert_eq!(named.len(), 4);
        assert_eq!(named[0].0, "w_ih");
        assert_eq!(named[1].0, "w_hh");
        assert_eq!(named[2].0, "b_ih");
        assert_eq!(named[3].0, "b_hh");
    }

    #[test]
    fn test_lstm_cell_to_device() {
        let cell = LSTMCell::new(10, 20);
        let cell_gpu = cell.to(&Device::Cuda(0)).unwrap();
        for param in cell_gpu.parameters() {
            assert_eq!(param.device(), &Device::Cuda(0));
        }

        let cell_cpu = cell_gpu.cpu().unwrap();
        for param in cell_cpu.parameters() {
            assert_eq!(param.device(), &Device::Cpu);
        }
    }

    #[test]
    fn test_lstm_cell_named_parameters() {
        let cell = LSTMCell::new(10, 20);
        let named = cell.named_parameters();
        assert_eq!(named.len(), 4);
        assert_eq!(named[0].0, "w_ih");
        assert_eq!(named[1].0, "w_hh");
    }

    #[test]
    fn test_rnn_cell_state_dict() {
        let cell = RNNCell::new(10, 20);
        let sd = cell.state_dict();
        assert!(sd.contains_key("w_ih"));
        assert!(sd.contains_key("w_hh"));
        assert!(sd.contains_key("b_ih"));
        assert!(sd.contains_key("b_hh"));
    }
}
