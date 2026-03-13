/// Trait for learning rate schedulers. Like PyTorch's _LRScheduler.
pub trait LRScheduler {
    /// Compute the learning rate for the current step/epoch.
    fn get_lr(&self) -> f64;
    /// Advance the scheduler by one step/epoch.
    fn step(&mut self);
    /// Get the base (initial) learning rate.
    fn base_lr(&self) -> f64;
}

/// Step LR: decays LR by gamma every step_size epochs.
pub struct StepLR {
    base: f64,
    step_size: usize,
    gamma: f64,
    current_epoch: usize,
}

impl StepLR {
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self { base: base_lr, step_size, gamma, current_epoch: 0 }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self) -> f64 {
        self.base * self.gamma.powi((self.current_epoch / self.step_size) as i32)
    }
    fn step(&mut self) { self.current_epoch += 1; }
    fn base_lr(&self) -> f64 { self.base }
}

/// Exponential LR: decays LR by gamma every epoch.
pub struct ExponentialLR {
    base: f64,
    gamma: f64,
    current_epoch: usize,
}

impl ExponentialLR {
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self { base: base_lr, gamma, current_epoch: 0 }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self) -> f64 {
        self.base * self.gamma.powi(self.current_epoch as i32)
    }
    fn step(&mut self) { self.current_epoch += 1; }
    fn base_lr(&self) -> f64 { self.base }
}

/// Cosine Annealing LR.
pub struct CosineAnnealingLR {
    base: f64,
    t_max: usize,
    eta_min: f64,
    current_epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f64, t_max: usize) -> Self {
        Self { base: base_lr, t_max, eta_min: 0.0, current_epoch: 0 }
    }

    pub fn eta_min(mut self, eta_min: f64) -> Self {
        self.eta_min = eta_min;
        self
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f64 {
        self.eta_min + (self.base - self.eta_min)
            * (1.0 + (std::f64::consts::PI * self.current_epoch as f64 / self.t_max as f64).cos())
            / 2.0
    }
    fn step(&mut self) { self.current_epoch += 1; }
    fn base_lr(&self) -> f64 { self.base }
}

/// Multi-step LR: decays LR by gamma at each milestone.
pub struct MultiStepLR {
    base: f64,
    milestones: Vec<usize>,
    gamma: f64,
    current_epoch: usize,
}

impl MultiStepLR {
    pub fn new(base_lr: f64, milestones: Vec<usize>, gamma: f64) -> Self {
        Self { base: base_lr, milestones, gamma, current_epoch: 0 }
    }
}

impl LRScheduler for MultiStepLR {
    fn get_lr(&self) -> f64 {
        let num_decays = self.milestones.iter().filter(|&&m| self.current_epoch >= m).count();
        self.base * self.gamma.powi(num_decays as i32)
    }
    fn step(&mut self) { self.current_epoch += 1; }
    fn base_lr(&self) -> f64 { self.base }
}

/// Linear LR warmup then constant.
pub struct LinearLR {
    base: f64,
    start_factor: f64,
    end_factor: f64,
    total_iters: usize,
    current_iter: usize,
}

impl LinearLR {
    pub fn new(base_lr: f64, start_factor: f64, end_factor: f64, total_iters: usize) -> Self {
        Self { base: base_lr, start_factor, end_factor, total_iters, current_iter: 0 }
    }
}

impl LRScheduler for LinearLR {
    fn get_lr(&self) -> f64 {
        if self.current_iter >= self.total_iters {
            self.base * self.end_factor
        } else {
            let factor = self.start_factor + (self.end_factor - self.start_factor)
                * (self.current_iter as f64 / self.total_iters as f64);
            self.base * factor
        }
    }
    fn step(&mut self) { self.current_iter += 1; }
    fn base_lr(&self) -> f64 { self.base }
}

/// Reduce LR on plateau.
pub struct ReduceLROnPlateau {
    base: f64,
    factor: f64,
    patience: usize,
    threshold: f64,
    current_lr: f64,
    best_metric: f64,
    num_bad_epochs: usize,
}

impl ReduceLROnPlateau {
    pub fn new(base_lr: f64) -> Self {
        Self {
            base: base_lr,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            current_lr: base_lr,
            best_metric: f64::INFINITY,
            num_bad_epochs: 0,
        }
    }

    pub fn factor(mut self, f: f64) -> Self { self.factor = f; self }
    pub fn patience(mut self, p: usize) -> Self { self.patience = p; self }

    /// Call with the metric value (e.g., validation loss).
    pub fn step_metric(&mut self, metric: f64) {
        if metric < self.best_metric - self.threshold {
            self.best_metric = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
            if self.num_bad_epochs > self.patience {
                self.current_lr *= self.factor;
                self.num_bad_epochs = 0;
            }
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn get_lr(&self) -> f64 { self.current_lr }
    fn step(&mut self) { /* use step_metric instead */ }
    fn base_lr(&self) -> f64 { self.base }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_lr() {
        let mut sched = StepLR::new(0.1, 10, 0.1);
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        for _ in 0..10 { sched.step(); }
        assert!((sched.get_lr() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_lr() {
        let mut sched = ExponentialLR::new(1.0, 0.9);
        assert!((sched.get_lr() - 1.0).abs() < 1e-10);
        sched.step();
        assert!((sched.get_lr() - 0.9).abs() < 1e-10);
        sched.step();
        assert!((sched.get_lr() - 0.81).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing() {
        let mut sched = CosineAnnealingLR::new(0.1, 100);
        let lr_start = sched.get_lr();
        assert!((lr_start - 0.1).abs() < 1e-10);
        for _ in 0..50 { sched.step(); }
        let lr_mid = sched.get_lr();
        assert!(lr_mid < lr_start); // should have decayed
        for _ in 0..50 { sched.step(); }
        let lr_end = sched.get_lr();
        assert!(lr_end < 1e-8); // near zero at T_max
    }

    #[test]
    fn test_multistep_lr() {
        let mut sched = MultiStepLR::new(0.1, vec![5, 10], 0.1);
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        for _ in 0..5 { sched.step(); }
        assert!((sched.get_lr() - 0.01).abs() < 1e-10);
        for _ in 0..5 { sched.step(); }
        assert!((sched.get_lr() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_linear_lr() {
        let mut sched = LinearLR::new(0.1, 0.1, 1.0, 10);
        let lr0 = sched.get_lr();
        assert!((lr0 - 0.01).abs() < 1e-10); // 0.1 * 0.1
        for _ in 0..10 { sched.step(); }
        let lr_end = sched.get_lr();
        assert!((lr_end - 0.1).abs() < 1e-10); // 0.1 * 1.0
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut sched = ReduceLROnPlateau::new(0.1).patience(2).factor(0.5);
        assert!((sched.get_lr() - 0.1).abs() < 1e-10);
        // First call sets best_metric to 1.0 (improves from INFINITY)
        sched.step_metric(1.0);
        // Next 3 calls are bad epochs (no improvement)
        sched.step_metric(1.0);
        sched.step_metric(1.0);
        sched.step_metric(1.0); // 3 bad epochs > patience=2
        assert!((sched.get_lr() - 0.05).abs() < 1e-10);
    }
}
