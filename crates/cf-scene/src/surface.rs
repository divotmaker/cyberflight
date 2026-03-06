//! Surface types for the driving range.
//!
//! The range floor is fairway by default. Target zones (defined by each
//! target's outer radius) behave as greens — softer, more receptive, with
//! higher spin interaction and lower rolling friction.

/// Surface type at a given point on the range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Standard driving range floor. Firm, moderate rolling friction.
    Fairway,
    /// Target zone surface. Soft, receptive, low rolling friction.
    Green,
}

/// Physical parameters for ball–surface interaction.
///
/// Sources:
/// - Penner (2002), "The run of a golf ball"
/// - Haake (1989), PhD thesis, Loughborough
/// - Chamberlain & Groenland (2023), arXiv:2302.02758
/// - USGA Green Speed Physics (stimpmeter calibration)
#[derive(Debug, Clone, Copy)]
pub struct SurfaceParams {
    /// Normal coefficient of restitution. Controls how much vertical
    /// velocity is retained after bounce. Range: 0.0 (dead) to 1.0 (perfect).
    pub e_n: f64,

    /// Sliding friction coefficient during bounce contact.
    /// Controls tangential impulse and spin change at impact.
    pub mu_slide: f64,

    /// Rolling friction coefficient for ground roll.
    /// Lower = faster roll (greens), higher = shorter roll (fairway/rough).
    pub mu_roll: f64,

    /// Penner's turf compliance parameter (rad per m/s).
    /// Tilts the effective contact surface toward the ball on soft turf,
    /// steepening the rebound angle at high impact speeds ("check up" effect).
    pub kp: f64,
}

impl SurfaceParams {
    /// Standard fairway surface.
    ///
    /// Firm, fast bounce, moderate rolling friction.
    /// Equivalent to ~stimpmeter 5 (fairway roll speed).
    pub const FAIRWAY: Self = Self {
        e_n: 0.40,
        mu_slide: 0.40,
        mu_roll: 0.12,
        kp: 0.010,
    };

    /// Standard green surface.
    ///
    /// Soft, receptive, low rolling friction.
    /// Equivalent to ~stimpmeter 10 (tournament green speed).
    pub const GREEN: Self = Self {
        e_n: 0.25,
        mu_slide: 0.40,
        mu_roll: 0.055,
        kp: 0.018,
    };

    /// Get params for a surface type.
    #[must_use]
    pub fn for_surface(surface: Surface) -> Self {
        match surface {
            Surface::Fairway => Self::FAIRWAY,
            Surface::Green => Self::GREEN,
        }
    }

    /// Rolling deceleration magnitude (m/s²).
    ///
    /// `a = (5/7) × mu_roll × g`
    ///
    /// The 5/7 factor accounts for a solid sphere's moment of inertia:
    /// energy goes into both translational and rotational deceleration.
    #[must_use]
    pub fn rolling_decel(&self) -> f64 {
        (5.0 / 7.0) * self.mu_roll * cf_math::units::G
    }

    /// Sliding deceleration magnitude (m/s²).
    #[must_use]
    pub fn sliding_decel(&self) -> f64 {
        (5.0 / 7.0) * self.mu_slide * cf_math::units::G
    }
}

/// Query the surface type at a world-space point on the range.
///
/// Returns `Green` if the point is inside any target's outer radius,
/// `Fairway` otherwise. Targets are circles on the Y=0 plane.
#[must_use]
pub fn surface_at(x: f64, z: f64, targets: &[crate::target::Target]) -> Surface {
    for t in targets {
        let dx = x - f64::from(t.center[0]);
        let dz = z - f64::from(t.center[2]);
        let dist_sq = dx * dx + dz * dz;
        let r = f64::from(t.outer_radius_m());
        if dist_sq <= r * r {
            return Surface::Green;
        }
    }
    Surface::Fairway
}

/// Query surface params at a world-space point.
#[must_use]
pub fn surface_params_at(x: f64, z: f64, targets: &[crate::target::Target]) -> SurfaceParams {
    SurfaceParams::for_surface(surface_at(x, z, targets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    #[test]
    fn fairway_params() {
        let p = SurfaceParams::FAIRWAY;
        assert!(p.e_n > 0.0 && p.e_n < 1.0);
        assert!(p.mu_slide > 0.0);
        assert!(p.mu_roll > 0.0);
        assert!(p.kp > 0.0);
    }

    #[test]
    fn green_params() {
        let p = SurfaceParams::GREEN;
        assert!(p.e_n > 0.0 && p.e_n < 1.0);
        assert!(p.mu_slide > 0.0);
        assert!(p.mu_roll > 0.0);
        assert!(p.kp > 0.0);
    }

    #[test]
    fn green_softer_than_fairway() {
        assert!(SurfaceParams::GREEN.e_n < SurfaceParams::FAIRWAY.e_n);
    }

    #[test]
    fn green_rolls_farther() {
        // Lower rolling friction = longer roll
        assert!(SurfaceParams::GREEN.mu_roll < SurfaceParams::FAIRWAY.mu_roll);
    }

    #[test]
    fn green_more_compliant() {
        assert!(SurfaceParams::GREEN.kp > SurfaceParams::FAIRWAY.kp);
    }

    #[test]
    fn rolling_decel_positive() {
        assert!(SurfaceParams::FAIRWAY.rolling_decel() > 0.0);
        assert!(SurfaceParams::GREEN.rolling_decel() > 0.0);
    }

    #[test]
    fn fairway_decel_greater_than_green() {
        assert!(SurfaceParams::FAIRWAY.rolling_decel() > SurfaceParams::GREEN.rolling_decel());
    }

    #[test]
    fn surface_at_origin_is_fairway() {
        let targets = crate::target::default_targets();
        assert_eq!(surface_at(0.0, 0.0, &targets), Surface::Fairway);
    }

    #[test]
    fn surface_at_target_center_is_green() {
        let targets = crate::target::default_targets();
        // First target is at 45.72m downrange (50 yards)
        let t = &targets[0];
        assert_eq!(
            surface_at(f64::from(t.center[0]), f64::from(t.center[2]), &targets),
            Surface::Green
        );
    }

    #[test]
    fn surface_at_target_edge_is_green() {
        let targets = vec![Target::at_distance(100.0, "100")];
        // Just inside outer radius (5 rings × 5m = 25m)
        assert_eq!(surface_at(100.0, 24.9, &targets), Surface::Green);
    }

    #[test]
    fn surface_outside_target_is_fairway() {
        let targets = vec![Target::at_distance(100.0, "100")];
        // Just outside outer radius
        assert_eq!(surface_at(100.0, 25.1, &targets), Surface::Fairway);
    }

    #[test]
    fn surface_params_at_matches() {
        let targets = crate::target::default_targets();
        let p = surface_params_at(0.0, 0.0, &targets);
        assert!((p.e_n - SurfaceParams::FAIRWAY.e_n).abs() < 1e-10);
    }

    #[test]
    fn stimpmeter_sanity() {
        // Stimpmeter 10 ft: v0=1.83 m/s, distance=10*0.3048=3.048m
        // a = v0^2 / (2*d) = 1.83^2 / (2*3.048) = 0.549 m/s^2
        // Our green rolling_decel should be in the right ballpark
        let green_decel = SurfaceParams::GREEN.rolling_decel();
        assert!(
            (green_decel - 0.39).abs() < 0.2,
            "green decel {green_decel:.3} should be near stimp-10 value ~0.39"
        );
    }
}
