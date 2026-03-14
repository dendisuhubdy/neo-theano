//! Mixed precision (automatic mixed precision) support stubs.
//!
//! Like `torch.cuda.amp` in PyTorch. Provides the autocast context
//! manager and GradScaler API surface.
//!
//! This is a stub — the actual dtype conversion happens when GPU backends
//! are active. The API surface exists now so user code can be written against it.

use std::cell::RefCell;

thread_local! {
    static AUTOCAST_ENABLED: RefCell<bool> = const { RefCell::new(false) };
}

/// Check whether automatic mixed precision (autocast) is currently enabled.
///
/// Like `torch.is_autocast_enabled()`.
pub fn is_autocast_enabled() -> bool {
    AUTOCAST_ENABLED.with(|e| *e.borrow())
}

/// RAII guard that enables/disables autocast.
///
/// When the guard is created, autocast is set to the given value.
/// When the guard is dropped, autocast is restored to its previous value.
pub struct AutocastGuard {
    prev: bool,
}

impl AutocastGuard {
    /// Create a new autocast guard that sets autocast to `enabled`.
    pub fn new(enabled: bool) -> Self {
        let prev = AUTOCAST_ENABLED.with(|e| {
            let prev = *e.borrow();
            *e.borrow_mut() = enabled;
            prev
        });
        Self { prev }
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST_ENABLED.with(|e| {
            *e.borrow_mut() = self.prev;
        });
    }
}

/// Run a closure with autocast enabled.
///
/// Like `with torch.cuda.amp.autocast():` in PyTorch.
///
/// **WARNING: This is currently a no-op stub.** No tensor operations inspect
/// the autocast flag yet. The API exists so user code can be written against it,
/// but dtype conversion will not occur until GPU float16 support is implemented.
///
/// # Example
/// ```ignore
/// let output = autocast(|| {
///     model.forward(&input)
/// });
/// ```
pub fn autocast<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = AutocastGuard::new(true);
    f()
}

/// Gradient scaler for mixed precision training.
///
/// Like `torch.cuda.amp.GradScaler` in PyTorch. Scales the loss to prevent
/// underflow in float16 gradients, then unscales before optimizer step.
///
/// **WARNING: This is currently a stub.** `scale()` operates on `f64` scalars,
/// not `Tensor`/`Variable` objects. The full implementation requires GPU float16
/// support. The API exists for forward-compatibility only.
pub struct GradScaler {
    scale_factor: f64,
    _growth_factor: f64,
    _backoff_factor: f64,
    _growth_interval: usize,
    enabled: bool,
}

impl GradScaler {
    /// Create a new gradient scaler with default parameters.
    pub fn new() -> Self {
        Self {
            scale_factor: 65536.0,
            _growth_factor: 2.0,
            _backoff_factor: 0.5,
            _growth_interval: 2000,
            enabled: true,
        }
    }

    /// Create a new gradient scaler with the given initial scale.
    pub fn with_scale(init_scale: f64) -> Self {
        Self {
            scale_factor: init_scale,
            ..Self::new()
        }
    }

    /// Whether the scaler is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the current scale factor.
    pub fn get_scale(&self) -> f64 {
        self.scale_factor
    }

    /// Scale a loss value. In a real implementation, this would multiply
    /// the loss tensor by the scale factor before backward().
    pub fn scale(&self, loss: f64) -> f64 {
        if self.enabled {
            loss * self.scale_factor
        } else {
            loss
        }
    }

    /// Update the scale factor after an optimizer step.
    /// In a real implementation, this checks for inf/nan gradients
    /// and adjusts the scale factor accordingly.
    pub fn update(&mut self) {
        // Stub: no-op until GPU float16 support is implemented
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocast_default_disabled() {
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_autocast_guard() {
        assert!(!is_autocast_enabled());
        {
            let _guard = AutocastGuard::new(true);
            assert!(is_autocast_enabled());
        }
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_autocast_closure() {
        assert!(!is_autocast_enabled());
        autocast(|| {
            assert!(is_autocast_enabled());
        });
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_autocast_nested() {
        assert!(!is_autocast_enabled());
        let _g1 = AutocastGuard::new(true);
        assert!(is_autocast_enabled());
        {
            let _g2 = AutocastGuard::new(false);
            assert!(!is_autocast_enabled());
        }
        assert!(is_autocast_enabled());
    }

    #[test]
    fn test_grad_scaler_default() {
        let scaler = GradScaler::new();
        assert!(scaler.is_enabled());
        assert_eq!(scaler.get_scale(), 65536.0);
    }

    #[test]
    fn test_grad_scaler_scale() {
        let scaler = GradScaler::new();
        let scaled = scaler.scale(1.0);
        assert_eq!(scaled, 65536.0);
    }

    #[test]
    fn test_grad_scaler_custom_scale() {
        let scaler = GradScaler::with_scale(1024.0);
        assert_eq!(scaler.get_scale(), 1024.0);
        assert_eq!(scaler.scale(2.0), 2048.0);
    }

    #[test]
    fn test_grad_scaler_update() {
        let mut scaler = GradScaler::new();
        let before = scaler.get_scale();
        scaler.update();
        // Stub: update is a no-op, scale shouldn't change
        assert_eq!(scaler.get_scale(), before);
    }
}
