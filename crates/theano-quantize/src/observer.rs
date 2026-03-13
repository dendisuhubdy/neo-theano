//! Observers for collecting tensor statistics for quantization.

use crate::qconfig::QuantDType;

/// Trait for quantization observers.
pub trait Observer {
    fn observe(&mut self, values: &[f64]);
    fn compute_scale_zero_point(&self, dtype: QuantDType) -> (f64, f64);
    fn reset(&mut self);
}

/// Min-max observer: tracks running min and max values.
pub struct MinMaxObserver {
    min_val: f64,
    max_val: f64,
    initialized: bool,
}

impl MinMaxObserver {
    pub fn new() -> Self {
        Self {
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            initialized: false,
        }
    }

    pub fn min_val(&self) -> f64 {
        self.min_val
    }
    pub fn max_val(&self) -> f64 {
        self.max_val
    }
}

impl Default for MinMaxObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for MinMaxObserver {
    fn observe(&mut self, values: &[f64]) {
        for &v in values {
            self.min_val = self.min_val.min(v);
            self.max_val = self.max_val.max(v);
        }
        self.initialized = true;
    }

    fn compute_scale_zero_point(&self, dtype: QuantDType) -> (f64, f64) {
        if !self.initialized {
            return (1.0, 0.0);
        }

        let qmin = dtype.qmin();
        let qmax = dtype.qmax();

        let scale = (self.max_val - self.min_val) / (qmax - qmin);
        let scale = scale.max(1e-10); // avoid division by zero

        let zero_point = qmin - self.min_val / scale;
        let zero_point = zero_point.round().clamp(qmin, qmax);

        (scale, zero_point)
    }

    fn reset(&mut self) {
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
        self.initialized = false;
    }
}

/// Per-channel min-max observer.
pub struct PerChannelMinMaxObserver {
    num_channels: usize,
    min_vals: Vec<f64>,
    max_vals: Vec<f64>,
    initialized: bool,
}

impl PerChannelMinMaxObserver {
    pub fn new(num_channels: usize) -> Self {
        Self {
            num_channels,
            min_vals: vec![f64::INFINITY; num_channels],
            max_vals: vec![f64::NEG_INFINITY; num_channels],
            initialized: false,
        }
    }

    pub fn scales_and_zero_points(&self, dtype: QuantDType) -> Vec<(f64, f64)> {
        let qmin = dtype.qmin();
        let qmax = dtype.qmax();

        (0..self.num_channels)
            .map(|i| {
                let scale = (self.max_vals[i] - self.min_vals[i]) / (qmax - qmin);
                let scale = scale.max(1e-10);
                let zp = qmin - self.min_vals[i] / scale;
                let zp = zp.round().clamp(qmin, qmax);
                (scale, zp)
            })
            .collect()
    }
}

impl Observer for PerChannelMinMaxObserver {
    fn observe(&mut self, values: &[f64]) {
        // Assumes values are laid out as [channels, ...]
        let per_channel = values.len() / self.num_channels;
        for c in 0..self.num_channels {
            for i in 0..per_channel {
                let v = values[c * per_channel + i];
                self.min_vals[c] = self.min_vals[c].min(v);
                self.max_vals[c] = self.max_vals[c].max(v);
            }
        }
        self.initialized = true;
    }

    fn compute_scale_zero_point(&self, dtype: QuantDType) -> (f64, f64) {
        // Return global scale/zp (average across channels)
        let scales: Vec<(f64, f64)> = self.scales_and_zero_points(dtype);
        let avg_scale = scales.iter().map(|(s, _)| s).sum::<f64>() / scales.len() as f64;
        let avg_zp = scales.iter().map(|(_, z)| z).sum::<f64>() / scales.len() as f64;
        (avg_scale, avg_zp)
    }

    fn reset(&mut self) {
        self.min_vals = vec![f64::INFINITY; self.num_channels];
        self.max_vals = vec![f64::NEG_INFINITY; self.num_channels];
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_observer_basic() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&[1.0, 2.0, 3.0, -1.0]);
        assert_eq!(obs.min_val(), -1.0);
        assert_eq!(obs.max_val(), 3.0);
    }

    #[test]
    fn test_minmax_observer_multiple_observations() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&[1.0, 2.0]);
        obs.observe(&[-5.0, 10.0]);
        assert_eq!(obs.min_val(), -5.0);
        assert_eq!(obs.max_val(), 10.0);
    }

    #[test]
    fn test_minmax_observer_uninitialized() {
        let obs = MinMaxObserver::new();
        let (scale, zp) = obs.compute_scale_zero_point(QuantDType::Int8);
        assert_eq!(scale, 1.0);
        assert_eq!(zp, 0.0);
    }

    #[test]
    fn test_minmax_observer_scale_zero_point() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&[0.0, 1.0]);
        let (scale, zp) = obs.compute_scale_zero_point(QuantDType::UInt8);
        // range is 1.0, qrange is 255, so scale = 1/255
        assert!((scale - 1.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_observer_reset() {
        let mut obs = MinMaxObserver::new();
        obs.observe(&[1.0, 2.0]);
        obs.reset();
        assert_eq!(obs.min_val(), f64::INFINITY);
        assert_eq!(obs.max_val(), f64::NEG_INFINITY);
        let (scale, zp) = obs.compute_scale_zero_point(QuantDType::Int8);
        assert_eq!(scale, 1.0);
        assert_eq!(zp, 0.0);
    }

    #[test]
    fn test_per_channel_observer() {
        let mut obs = PerChannelMinMaxObserver::new(2);
        // 2 channels, 3 elements each
        obs.observe(&[1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
        let scales = obs.scales_and_zero_points(QuantDType::Int8);
        assert_eq!(scales.len(), 2);
    }

    #[test]
    fn test_per_channel_observer_reset() {
        let mut obs = PerChannelMinMaxObserver::new(3);
        obs.observe(&[1.0, 2.0, 3.0]);
        obs.reset();
        // After reset, scales_and_zero_points should use default infinity values
        let (scale, zp) = obs.compute_scale_zero_point(QuantDType::Int8);
        // With infinity values, scale will be very large
        assert!(scale > 0.0);
    }

    #[test]
    fn test_minmax_observer_default() {
        let obs = MinMaxObserver::default();
        assert_eq!(obs.min_val(), f64::INFINITY);
        assert_eq!(obs.max_val(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_per_channel_compute_global_scale() {
        let mut obs = PerChannelMinMaxObserver::new(2);
        // Channel 0: [0, 1], Channel 1: [0, 2]
        obs.observe(&[0.0, 1.0, 0.0, 2.0]);
        let (scale, _zp) = obs.compute_scale_zero_point(QuantDType::Int8);
        // Average of the two channel scales
        assert!(scale > 0.0);
    }
}
