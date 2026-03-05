use crate::trajectory::FlightResult;
use crate::units;

/// Statistics for a single shot, in display-friendly units.
#[derive(Debug, Clone, Copy)]
pub struct ShotStats {
    pub carry_yards: f64,
    pub total_yards: f64, // same as carry for driving range (no roll)
    pub apex_feet: f64,
    pub flight_time_s: f64,
    pub lateral_yards: f64, // positive = left, negative = right (for right-handed)
    pub descent_angle_deg: f64,
}

impl ShotStats {
    /// Compute shot statistics from a flight result.
    #[must_use]
    pub fn from_flight(result: &FlightResult) -> Self {
        let descent_angle = if let Some(last) = result.points.last() {
            let vy = last.vel.y;
            let vx = last.vel.x;
            let vz = last.vel.z;
            let horizontal_speed = (vx * vx + vz * vz).sqrt();
            if horizontal_speed > 1e-10 {
                units::rad_to_deg((-vy).atan2(horizontal_speed))
            } else {
                90.0
            }
        } else {
            0.0
        };

        Self {
            carry_yards: result.carry_yards,
            total_yards: result.carry_yards, // no roll on driving range
            apex_feet: units::meters_to_feet(result.apex_m),
            flight_time_s: result.flight_time,
            lateral_yards: units::meters_to_yards(result.lateral_m),
            descent_angle_deg: descent_angle,
        }
    }
}

/// Running statistics for a session of shots.
#[derive(Debug, Clone)]
pub struct SessionStats {
    shots: Vec<ShotStats>,
}

impl SessionStats {
    #[must_use]
    pub fn new() -> Self {
        Self {
            shots: Vec::new(),
        }
    }

    pub fn add_shot(&mut self, stats: ShotStats) {
        self.shots.push(stats);
    }

    #[must_use]
    pub fn shot_count(&self) -> usize {
        self.shots.len()
    }

    #[must_use]
    pub fn avg_carry_yards(&self) -> Option<f64> {
        if self.shots.is_empty() {
            return None;
        }
        Some(self.shots.iter().map(|s| s.carry_yards).sum::<f64>() / self.shots.len() as f64)
    }

    #[must_use]
    pub fn avg_apex_feet(&self) -> Option<f64> {
        if self.shots.is_empty() {
            return None;
        }
        Some(self.shots.iter().map(|s| s.apex_feet).sum::<f64>() / self.shots.len() as f64)
    }

    #[must_use]
    pub fn avg_lateral_yards(&self) -> Option<f64> {
        if self.shots.is_empty() {
            return None;
        }
        Some(self.shots.iter().map(|s| s.lateral_yards).sum::<f64>() / self.shots.len() as f64)
    }

    /// Standard deviation of carry distance.
    #[must_use]
    pub fn carry_std_dev(&self) -> Option<f64> {
        let avg = self.avg_carry_yards()?;
        if self.shots.len() < 2 {
            return None;
        }
        let variance = self
            .shots
            .iter()
            .map(|s| {
                let diff = s.carry_yards - avg;
                diff * diff
            })
            .sum::<f64>()
            / (self.shots.len() - 1) as f64;
        Some(variance.sqrt())
    }

    /// Standard deviation of lateral dispersion.
    #[must_use]
    pub fn lateral_std_dev(&self) -> Option<f64> {
        let avg = self.avg_lateral_yards()?;
        if self.shots.len() < 2 {
            return None;
        }
        let variance = self
            .shots
            .iter()
            .map(|s| {
                let diff = s.lateral_yards - avg;
                diff * diff
            })
            .sum::<f64>()
            / (self.shots.len() - 1) as f64;
        Some(variance.sqrt())
    }

    /// Get all individual shot stats.
    #[must_use]
    pub fn shots(&self) -> &[ShotStats] {
        &self.shots
    }

    /// Get the last N shots.
    #[must_use]
    pub fn last_n(&self, n: usize) -> &[ShotStats] {
        let start = self.shots.len().saturating_sub(n);
        &self.shots[start..]
    }
}

impl Default for SessionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoring for target challenge mode.
#[derive(Debug, Clone, Copy)]
pub struct TargetScore {
    /// Which ring the ball landed in (0 = bullseye, increasing = outer rings).
    /// `None` if the ball missed all rings.
    pub ring: Option<u32>,
    /// Points awarded for this shot.
    pub points: u32,
    /// Distance from target center in yards.
    pub distance_yards: f64,
}

/// Score a shot against a target.
///
/// `target_x_m`, `target_z_m`: target center in meters.
/// `ring_spacing_m`: distance between rings in meters.
/// `num_rings`: number of scoring rings (including bullseye).
/// `landing_x_m`, `landing_z_m`: where the ball landed in meters.
#[must_use]
pub fn score_target(
    target_x_m: f64,
    target_z_m: f64,
    ring_spacing_m: f64,
    num_rings: u32,
    landing_x_m: f64,
    landing_z_m: f64,
) -> TargetScore {
    let dx = landing_x_m - target_x_m;
    let dz = landing_z_m - target_z_m;
    let distance_m = (dx * dx + dz * dz).sqrt();
    let distance_yards = units::meters_to_yards(distance_m);

    let ring_index = (distance_m / ring_spacing_m) as u32;

    if ring_index >= num_rings {
        return TargetScore {
            ring: None,
            points: 0,
            distance_yards,
        };
    }

    let points = (num_rings - ring_index) * 10;

    TargetScore {
        ring: Some(ring_index),
        points,
        distance_yards,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aero::BallModel;
    use crate::environment::Environment;
    use crate::trajectory::{ShotInput, simulate_flight};

    #[test]
    fn shot_stats_from_seven_iron() {
        let result = simulate_flight(&ShotInput::seven_iron(), &BallModel::TOUR, &Environment::SEA_LEVEL);
        let stats = ShotStats::from_flight(&result);
        assert!(stats.carry_yards > 100.0);
        assert!(stats.apex_feet > 0.0);
        assert!(stats.descent_angle_deg > 0.0);
    }

    #[test]
    fn session_stats_empty() {
        let session = SessionStats::new();
        assert_eq!(session.shot_count(), 0);
        assert!(session.avg_carry_yards().is_none());
    }

    #[test]
    fn session_stats_averages() {
        let mut session = SessionStats::new();
        session.add_shot(ShotStats {
            carry_yards: 150.0,
            total_yards: 150.0,
            apex_feet: 90.0,
            flight_time_s: 5.0,
            lateral_yards: 2.0,
            descent_angle_deg: 45.0,
        });
        session.add_shot(ShotStats {
            carry_yards: 160.0,
            total_yards: 160.0,
            apex_feet: 100.0,
            flight_time_s: 5.5,
            lateral_yards: -2.0,
            descent_angle_deg: 47.0,
        });

        assert_eq!(session.shot_count(), 2);
        assert!((session.avg_carry_yards().unwrap() - 155.0).abs() < 1e-10);
        assert!((session.avg_lateral_yards().unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn score_bullseye() {
        let score = score_target(100.0, 0.0, 5.0, 5, 100.0, 0.0);
        assert_eq!(score.ring, Some(0));
        assert_eq!(score.points, 50);
        assert!(score.distance_yards < 0.1);
    }

    #[test]
    fn score_outer_ring() {
        let score = score_target(100.0, 0.0, 5.0, 5, 117.0, 0.0);
        assert_eq!(score.ring, Some(3)); // 17m / 5m = ring 3
        assert_eq!(score.points, 20); // (5 - 3) * 10
    }

    #[test]
    fn score_miss() {
        let score = score_target(100.0, 0.0, 5.0, 5, 200.0, 0.0);
        assert_eq!(score.ring, None);
        assert_eq!(score.points, 0);
    }

    #[test]
    fn last_n_shots() {
        let mut session = SessionStats::new();
        for i in 0..10 {
            session.add_shot(ShotStats {
                carry_yards: 150.0 + i as f64,
                total_yards: 150.0 + i as f64,
                apex_feet: 90.0,
                flight_time_s: 5.0,
                lateral_yards: 0.0,
                descent_angle_deg: 45.0,
            });
        }
        let last_3 = session.last_n(3);
        assert_eq!(last_3.len(), 3);
        assert!((last_3[0].carry_yards - 157.0).abs() < 1e-10);
    }
}
