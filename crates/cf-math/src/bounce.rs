use glam::DVec3;

/// Golf ball radius (m). USGA min diameter = 42.67mm.
const BALL_RADIUS: f64 = 0.021_335;

/// Minimum vertical rebound speed (m/s) to continue bouncing.
/// Below this, transition to rollout.
const BOUNCE_THRESHOLD: f64 = 0.1;

/// Surface parameters needed for a bounce computation.
///
/// Kept separate from cf-scene to avoid cross-crate dependency.
/// cf-scene's `SurfaceParams` has the same fields.
#[derive(Debug, Clone, Copy)]
pub struct BounceSurface {
    /// Normal coefficient of restitution (0..1).
    pub e_n: f64,
    /// Sliding friction coefficient.
    pub mu: f64,
    /// Penner's turf compliance (rad per m/s of normal impact speed).
    pub kp: f64,
}

impl BounceSurface {
    /// Standard fairway.
    pub const FAIRWAY: Self = Self {
        e_n: 0.40,
        mu: 0.40,
        kp: 0.010,
    };

    /// Standard green.
    pub const GREEN: Self = Self {
        e_n: 0.25,
        mu: 0.40,
        kp: 0.018,
    };
}

/// Result of a single bounce computation.
#[derive(Debug, Clone, Copy)]
pub struct BounceResult {
    /// Rebound velocity (m/s) in world space.
    pub vel: DVec3,
    /// Angular velocity after bounce (rad/s).
    /// Convention: positive = topspin (contact point moves backward, promotes rolling).
    /// Negative = backspin (contact point moves forward, opposes rolling).
    pub omega: f64,
    /// Sidespin after bounce (rad/s).
    pub sidespin: f64,
    /// Whether the ball should continue bouncing (vertical speed above threshold).
    pub still_bouncing: bool,
}

/// Compute the result of a golf ball bouncing on a surface.
///
/// Uses the oblique impact model from Penner (2002) and Cross (2002):
/// - Decompose velocity into normal and tangential components
/// - Apply Penner's effective contact angle (turf compliance)
/// - Normal: COR-based rebound
/// - Tangential: slip vs grip depending on friction and spin
///
/// Coordinate system: +X downrange, +Y up, +Z lateral (right-handed).
/// Ground plane is Y=0, surface normal is +Y.
///
/// `vel`: incoming velocity (m/s), must have negative Y component (falling).
/// `backspin`: incoming backspin (rad/s, positive = backspin opposing forward motion).
/// `sidespin`: incoming sidespin (rad/s).
/// `surface`: surface parameters.
#[must_use]
pub fn compute_bounce(
    vel: DVec3,
    backspin: f64,
    sidespin: f64,
    surface: &BounceSurface,
) -> BounceResult {
    // Convert to standard angular velocity convention:
    // positive omega = topspin (contact point moves backward, promotes rolling)
    // negative omega = backspin (contact point moves forward, opposes rolling)
    let omega_in = -backspin;

    let v_n_abs = vel.y.abs(); // normal (vertical) impact speed
    let v_x = vel.x; // downrange
    let v_z = vel.z; // lateral
    let v_h = (v_x * v_x + v_z * v_z).sqrt(); // horizontal speed

    // Penner's effective contact angle: turf compliance tilts the effective
    // surface normal toward the ball, steepening the rebound angle.
    // Only applies when there's horizontal motion (direction is undefined otherwise).
    let beta = if v_h > 0.01 {
        surface.kp * v_n_abs
    } else {
        0.0
    };

    let cos_b = beta.cos();
    let sin_b = beta.sin();

    // Effective normal and tangential speeds after Penner tilt.
    // The tilt transfers some normal impulse into the tangential direction.
    let v_n_eff = v_n_abs * cos_b + v_h * sin_b;
    let v_h_eff = (v_h * cos_b - v_n_abs * sin_b).max(0.0);

    // Normal rebound
    let v_n_out = surface.e_n * v_n_eff;

    // Tangential: contact point velocity determines slip vs grip.
    //
    // Contact point velocity: v_c = v_tangential - R * omega
    // With backspin (omega < 0): v_c = v_h_eff + R*|backspin| (slides forward)
    // With topspin (omega > 0): v_c = v_h_eff - R*topspin (closer to rolling)
    let v_c = v_h_eff - BALL_RADIUS * omega_in;

    // Normal impulse (mass-normalized): J_n/m = (1 + e_n) * v_n_eff
    let j_n = (1.0 + surface.e_n) * v_n_eff;

    // Maximum friction impulse: J_t_max/m = mu * J_n/m
    let j_t_max = surface.mu * j_n;

    // Impulse needed to reach rolling condition (v_c_out = 0):
    // j_t_roll = |v_c| * (2/7) for a solid sphere (alpha = 2/5)
    let j_t_roll = (2.0 / 7.0) * v_c.abs();

    let (v_h_out, omega_out) = if j_t_roll <= j_t_max {
        // GRIP: friction is sufficient to reach rolling condition.
        // Ball leaves with v_c = 0 (rolling): v_out = R * omega_out.
        //
        // v_out = (5/7)*v_h_eff + (2/7)*R*omega_in
        // With backspin (omega_in < 0): v_out is reduced.
        // With enough backspin: v_out goes negative → spin-back.
        let v_out = (5.0 / 7.0) * v_h_eff + (2.0 / 7.0) * BALL_RADIUS * omega_in;
        let omega = v_out / BALL_RADIUS;
        (v_out, omega)
    } else {
        // SLIP: ball slides throughout bounce. Apply maximum friction impulse.
        let sign_vc = v_c.signum();
        let v_out = v_h_eff - j_t_max * sign_vc;
        let omega = omega_in + (5.0 / (2.0 * BALL_RADIUS)) * j_t_max * sign_vc;
        (v_out, omega)
    };

    // Reconstruct world-space velocity.
    // Reverse Penner tilt: effective → world coordinates.
    let v_h_abs = v_h_out.abs();
    let v_n_world = v_n_out * cos_b - v_h_abs * sin_b;
    let v_h_world = v_h_abs * cos_b + v_n_out * sin_b;

    // Preserve forward/backward direction from v_h_out sign.
    let v_h_signed = v_h_world.copysign(v_h_out);

    // Distribute horizontal velocity back into X and Z components
    // preserving the original direction ratio.
    let (out_x, out_z) = if v_h > 1e-10 {
        (v_h_signed * v_x / v_h, v_h_signed * v_z / v_h)
    } else {
        // Vertical drop: no horizontal component
        (0.0, 0.0)
    };

    // Sidespin: largely unaffected by bounce. Slight decay.
    let sidespin_out = sidespin * 0.9;

    let still_bouncing = v_n_world > BOUNCE_THRESHOLD;

    BounceResult {
        vel: DVec3::new(out_x, v_n_world, out_z),
        omega: omega_out,
        sidespin: sidespin_out,
        still_bouncing,
    }
}

/// Compute all bounces until the ball transitions to rolling.
///
/// Returns the sequence of bounce results and the final state for rollout.
/// Between bounces, the ball follows a simple parabolic arc (no aero forces
/// at bounce-speed velocities — drag is negligible below ~10 m/s).
///
/// `vel`: velocity at first ground contact (from flight simulation).
/// `backspin`: spin at landing (rad/s, positive = backspin).
/// `sidespin`: sidespin at landing (rad/s).
/// `surface`: surface parameters (assumed constant across bounces).
#[must_use]
pub fn bounce_sequence(
    vel: DVec3,
    backspin: f64,
    sidespin: f64,
    surface: &BounceSurface,
) -> Vec<BounceResult> {
    let mut results = Vec::new();
    let mut v = vel;
    let mut omega = -backspin; // convert to standard convention
    let mut ss = sidespin;

    // Safety limit: no more than 10 bounces
    for _ in 0..10 {
        // Convert omega back to backspin for compute_bounce API
        let bs = -omega;
        let result = compute_bounce(v, bs, ss, surface);
        let done = !result.still_bouncing;
        v = result.vel;
        omega = result.omega;
        ss = result.sidespin;
        results.push(result);
        if done {
            break;
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units;

    fn rpm_to_rads(rpm: f64) -> f64 {
        units::rpm_to_rads(rpm)
    }

    // ══════════════════════════════════════════════════════════════════
    //  STRUCTURAL TESTS — invariants for any surface
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn rebound_goes_up() {
        let vel = DVec3::new(0.0, -10.0, 0.0);
        let r = compute_bounce(vel, 0.0, 0.0, &BounceSurface::FAIRWAY);
        assert!(r.vel.y > 0.0, "rebound should go up: vy={:.3}", r.vel.y);
    }

    #[test]
    fn energy_decreases() {
        let vel = DVec3::new(20.0, -15.0, 0.0);
        let r = compute_bounce(vel, rpm_to_rads(5000.0), 0.0, &BounceSurface::FAIRWAY);
        assert!(
            r.vel.length() < vel.length(),
            "should lose energy: in={:.1}, out={:.1}",
            vel.length(),
            r.vel.length()
        );
    }

    #[test]
    fn no_lateral_without_sidespin() {
        let vel = DVec3::new(20.0, -15.0, 0.0);
        let r = compute_bounce(vel, rpm_to_rads(3000.0), 0.0, &BounceSurface::GREEN);
        assert!(
            r.vel.z.abs() < 1e-10,
            "no lateral without sidespin: vz={:.6}",
            r.vel.z
        );
    }

    #[test]
    fn vertical_drop_no_horizontal() {
        // Ball dropping straight down with no spin should bounce straight up.
        let vel = DVec3::new(0.0, -5.0, 0.0);
        let r = compute_bounce(vel, 0.0, 0.0, &BounceSurface::GREEN);
        assert!(r.vel.y > 0.0);
        assert!(r.vel.x.abs() < 1e-6, "no horizontal: vx={:.6}", r.vel.x);
    }

    // ══════════════════════════════════════════════════════════════════
    //  COR TESTS — normal rebound
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn green_lower_bounce_than_fairway() {
        let vel = DVec3::new(15.0, -10.0, 0.0);
        let spin = rpm_to_rads(3000.0);
        let rf = compute_bounce(vel, spin, 0.0, &BounceSurface::FAIRWAY);
        let rg = compute_bounce(vel, spin, 0.0, &BounceSurface::GREEN);
        assert!(
            rg.vel.y < rf.vel.y,
            "green bounce lower: green vy={:.3}, fairway vy={:.3}",
            rg.vel.y,
            rf.vel.y
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  SPIN TESTS — backspin effects
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn backspin_reduces_forward_speed() {
        let vel = DVec3::new(15.0, -10.0, 0.0);
        let r_low = compute_bounce(vel, rpm_to_rads(2000.0), 0.0, &BounceSurface::GREEN);
        let r_high = compute_bounce(vel, rpm_to_rads(8000.0), 0.0, &BounceSurface::GREEN);
        assert!(
            r_high.vel.x < r_low.vel.x,
            "high spin should reduce forward speed: high={:.3}, low={:.3}",
            r_high.vel.x,
            r_low.vel.x
        );
    }

    #[test]
    fn very_high_backspin_can_reverse() {
        // Wedge shot: steep angle, very high backspin on green → spin back.
        let speed = 25.0;
        let angle = units::deg_to_rad(50.0);
        let vel = DVec3::new(speed * angle.cos(), -speed * angle.sin(), 0.0);
        let backspin = rpm_to_rads(10000.0);
        let r = compute_bounce(vel, backspin, 0.0, &BounceSurface::GREEN);
        assert!(
            r.vel.x < 0.0,
            "very high backspin on green should spin back: vx={:.3}",
            r.vel.x
        );
    }

    #[test]
    fn driver_bounces_forward() {
        // Driver: shallow angle, low spin → bounces forward.
        let speed = 30.0;
        let angle = units::deg_to_rad(30.0);
        let vel = DVec3::new(speed * angle.cos(), -speed * angle.sin(), 0.0);
        let backspin = rpm_to_rads(2500.0);
        let r = compute_bounce(vel, backspin, 0.0, &BounceSurface::FAIRWAY);
        assert!(
            r.vel.x > 0.0,
            "driver should bounce forward: vx={:.3}",
            r.vel.x
        );
    }

    #[test]
    fn seven_iron_checks_on_green() {
        let speed = 25.0;
        let angle = units::deg_to_rad(45.0);
        let vel = DVec3::new(speed * angle.cos(), -speed * angle.sin(), 0.0);
        let backspin = rpm_to_rads(7000.0);
        let r = compute_bounce(vel, backspin, 0.0, &BounceSurface::GREEN);
        let r_nospin = compute_bounce(vel, 0.0, 0.0, &BounceSurface::GREEN);
        assert!(
            r.vel.x < r_nospin.vel.x,
            "7-iron with spin should check: spin vx={:.3}, nospin vx={:.3}",
            r.vel.x,
            r_nospin.vel.x
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  PENNER COMPLIANCE — steeper rebound on soft surfaces
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn compliance_steepens_rebound() {
        // Isolate kp effect: same e_n and mu, only kp differs.
        let vel = DVec3::new(20.0, -15.0, 0.0);
        let spin = rpm_to_rads(5000.0);

        let firm = BounceSurface { e_n: 0.35, mu: 0.40, kp: 0.005 };
        let soft = BounceSurface { e_n: 0.35, mu: 0.40, kp: 0.025 };

        let r_firm = compute_bounce(vel, spin, 0.0, &firm);
        let r_soft = compute_bounce(vel, spin, 0.0, &soft);

        // Higher kp → steeper rebound (higher vy/vx ratio)
        let angle_firm = r_firm.vel.y.atan2(r_firm.vel.x.abs().max(0.001));
        let angle_soft = r_soft.vel.y.atan2(r_soft.vel.x.abs().max(0.001));
        assert!(
            angle_soft > angle_firm,
            "soft surface steeper: firm={:.1}°, soft={:.1}°",
            units::rad_to_deg(angle_firm),
            units::rad_to_deg(angle_soft)
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  BOUNCE SEQUENCE
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn bounce_sequence_terminates() {
        let vel = DVec3::new(20.0, -15.0, 0.0);
        let bounces = bounce_sequence(vel, rpm_to_rads(3000.0), 0.0, &BounceSurface::FAIRWAY);
        assert!(!bounces.is_empty());
        assert!(bounces.len() <= 10);
        assert!(!bounces.last().unwrap().still_bouncing);
    }

    #[test]
    fn bounce_sequence_height_decreases() {
        let vel = DVec3::new(15.0, -10.0, 0.0);
        let bounces = bounce_sequence(vel, rpm_to_rads(2000.0), 0.0, &BounceSurface::FAIRWAY);
        for w in bounces.windows(2) {
            assert!(
                w[1].vel.y <= w[0].vel.y + 0.01,
                "bounce height should decrease: {:.3} then {:.3}",
                w[0].vel.y,
                w[1].vel.y
            );
        }
    }

    #[test]
    fn soft_drop_few_bounces() {
        // Gentle landing on green: e_n=0.25, v_n=1.0 → v_n_out=0.25 (above threshold)
        // → second bounce: 0.25*0.25=0.0625 (below threshold). Expect 2 bounces.
        let vel = DVec3::new(5.0, -1.0, 0.0);
        let bounces = bounce_sequence(vel, rpm_to_rads(500.0), 0.0, &BounceSurface::GREEN);
        assert!(
            bounces.len() <= 3,
            "soft landing: {} bounces (expected <=3)",
            bounces.len()
        );
    }
}
