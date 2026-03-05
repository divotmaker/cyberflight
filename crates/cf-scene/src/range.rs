use crate::camera::Camera;
use crate::grid::GridConfig;
use crate::hud::HudLayout;
use crate::target::Target;
use crate::trail::BallFlight;

/// Complete driving range scene.
#[derive(Debug)]
pub struct Range {
    pub grid: GridConfig,
    pub targets: Vec<Target>,
    pub flights: Vec<BallFlight>,
    pub camera: Camera,
    pub hud: HudLayout,
}

impl Range {
    /// Create a default driving range scene.
    #[must_use]
    pub fn new() -> Self {
        Self {
            grid: GridConfig::default(),
            targets: crate::target::default_targets(),
            flights: Vec::new(),
            camera: Camera::driving_range(),
            hud: HudLayout::driving_range(),
        }
    }

    /// Add a new ball flight to the scene.
    pub fn launch_ball(&mut self, start_time: f64) -> &mut BallFlight {
        self.flights.push(BallFlight::new(start_time));
        self.flights.last_mut().expect("just pushed")
    }

    /// Get all active (in-flight) balls.
    #[must_use]
    pub fn active_flights(&self) -> Vec<&BallFlight> {
        self.flights.iter().filter(|f| f.active).collect()
    }

    /// Remove landed flights older than `max_age` seconds.
    pub fn cleanup_old_flights(&mut self, current_time: f64, max_age: f64) {
        self.flights.retain(|f| {
            f.active || (current_time - f.start_time) < max_age
        });
    }
}

impl Default for Range {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn new_range_has_targets() {
        let range = Range::new();
        assert!(!range.targets.is_empty());
    }

    #[test]
    fn launch_and_track() {
        let mut range = Range::new();
        let flight = range.launch_ball(0.0);
        flight.update_position(Vec3::new(10.0, 5.0, 0.0), 0.1);
        assert_eq!(range.active_flights().len(), 1);

        range.flights[0].land();
        assert_eq!(range.active_flights().len(), 0);
    }

    #[test]
    fn cleanup_removes_old_landed() {
        let mut range = Range::new();
        let flight = range.launch_ball(0.0);
        flight.update_position(Vec3::new(100.0, 0.0, 0.0), 1.0);
        flight.land();

        range.cleanup_old_flights(100.0, 30.0);
        assert!(range.flights.is_empty());
    }

    #[test]
    fn cleanup_keeps_recent_landed() {
        let mut range = Range::new();
        let flight = range.launch_ball(95.0);
        flight.update_position(Vec3::new(100.0, 0.0, 0.0), 1.0);
        flight.land();

        range.cleanup_old_flights(100.0, 30.0);
        assert_eq!(range.flights.len(), 1);
    }
}
