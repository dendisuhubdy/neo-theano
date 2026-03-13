use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Check if gradient computation is currently enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Set gradient computation enabled/disabled. Returns the previous value.
pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|g| {
        let prev = g.get();
        g.set(enabled);
        prev
    })
}

/// RAII guard that disables gradient computation.
/// Like `torch.no_grad()` context manager.
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = set_grad_enabled(false);
        Self { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}

/// Execute a closure with gradient computation disabled.
/// Like `with torch.no_grad():` in Python.
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_grad() {
        assert!(is_grad_enabled());
        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled());
        }
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_closure() {
        assert!(is_grad_enabled());
        no_grad(|| {
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_nested_no_grad() {
        assert!(is_grad_enabled());
        let _g1 = NoGradGuard::new();
        assert!(!is_grad_enabled());
        {
            let _g2 = NoGradGuard::new();
            assert!(!is_grad_enabled());
        }
        // Still disabled because g1 is still alive (prev was true for g1, false for g2)
        assert!(!is_grad_enabled());
        drop(_g1);
        assert!(is_grad_enabled());
    }
}
