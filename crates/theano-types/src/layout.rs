/// Tensor memory layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Dense contiguous storage (default).
    Dense,
    /// Sparse COO (coordinate) format.
    SparseCoo,
    /// Sparse CSR (compressed sparse row) format.
    SparseCsr,
    /// Sparse CSC (compressed sparse column) format.
    SparseCsc,
    /// Sparse BSR (block sparse row) format.
    SparseBsr,
    /// Sparse BSC (block sparse column) format.
    SparseBsc,
}

impl Default for Layout {
    fn default() -> Self {
        Layout::Dense
    }
}

impl Layout {
    /// Whether this is a dense layout.
    pub fn is_dense(self) -> bool {
        matches!(self, Layout::Dense)
    }

    /// Whether this is any sparse layout.
    pub fn is_sparse(self) -> bool {
        !self.is_dense()
    }
}
