use glam::DVec3;

use crate::units::G;

/// Golf ball radius (m).
const BALL_RADIUS: f64 = 0.021_335;

/// Speed threshold (m/s) below which the ball is considered stopped.
const STOP_SPEED: f64 = 0.05;

/// Threshold for transition from sliding to pure rolling.
/// When |v - R*omega| < this value, the ball is considered to be rolling.
const ROLL_TRANSITION_THRESHOLD: f64 = 0.1;

/// Maximum rollout simulation time (seconds). Safety cap.
const MAX_ROLLOUT_TIME: f64 = 30.0;

/// Reference speed (m/s) for speed-dependent rolling friction.
///
/// At speeds well below this, rolling friction equals the base `mu_roll`
/// (matching stimpmeter calibration). At higher speeds, turf deformation
/// increases the effective friction quadratically:
///
///   `mu_eff = mu_roll * (1 + (speed / V_REF)²)`
///
/// At driver post-bounce speeds (~15 m/s), this gives ~37× base friction,
/// producing realistic ~30 yard rollouts instead of 100+.
/// At stimpmeter speeds (~1.8 m/s), the multiplier is only ~1.5×.
///
/// Source: Kensrud et al. (2024), "Modeling golf ball roll with a variable
/// coefficient of friction", Sports Engineering.
const V_REF: f64 = 2.5;

/// Surface parameters needed for rollout computation.
#[derive(Debug, Clone, Copy)]
pub struct RolloutSurface {
    /// Sliding friction coefficient.
    pub mu_slide: f64,
    /// Rolling friction coefficient.
    pub mu_roll: f64,
}

impl RolloutSurface {
    /// Standard fairway.
    pub const FAIRWAY: Self = Self {
        mu_slide: 0.40,
        mu_roll: 0.12,
    };

    /// Standard green.
    pub const GREEN: Self = Self {
        mu_slide: 0.40,
        mu_roll: 0.055,
    };
}

/// A point along the rollout trajectory.
#[derive(Debug, Clone, Copy)]
pub struct RolloutPoint {
    pub time: f64,
    pub pos: DVec3,
    pub speed: f64,
}

/// Result of a rollout simulation.
#[derive(Debug, Clone)]
pub struct RolloutResult {
    /// Trajectory points (on the ground plane, Y=0).
    pub points: Vec<RolloutPoint>,
    /// Final resting position.
    pub final_pos: DVec3,
    /// Total rollout distance (m) from start to final position.
    pub rollout_m: f64,
    /// Total rollout time (s).
    pub rollout_time: f64,
}

/// Ground-plane state for integration.
///
/// Position (x, z) on the Y=0 plane. Velocity (vx, vz) horizontal.
/// Angular velocity omega in standard convention:
///   positive = topspin (promotes rolling for forward motion)
///   negative = backspin (opposes rolling)
///
/// `dir_x, dir_z` track the last meaningful velocity direction for use
/// when speed is near zero (friction from backspin can reverse the ball).
#[derive(Debug, Clone, Copy)]
struct GroundState {
    x: f64,
    z: f64,
    vx: f64,
    vz: f64,
    omega: f64,
    dir_x: f64,
    dir_z: f64,
}

/// Simulate ball rollout on the ground plane.
///
/// Starts from the post-bounce velocity and spin state. Two phases:
///
/// 1. **Sliding**: contact point slides (v != R*omega). Higher friction.
///    Spin adjusts toward rolling condition.
/// 2. **Pure rolling**: omega locked to v/R. Lower friction.
///
/// The ball stops when horizontal speed drops below `STOP_SPEED`.
///
/// `start_pos`: position at start of rollout (Y=0 plane).
/// `vel`: horizontal velocity at start (Y component ignored).
/// `omega`: angular velocity (standard convention: positive=topspin, negative=backspin).
/// `surface`: surface friction parameters.
#[must_use]
pub fn simulate_rollout(
    start_pos: DVec3,
    vel: DVec3,
    omega: f64,
    surface: &RolloutSurface,
) -> RolloutResult {
    let dt = 0.001;
    let sample_every = 10;

    let init_speed = (vel.x * vel.x + vel.z * vel.z).sqrt();
    let (init_dir_x, init_dir_z) = if init_speed > 1e-10 {
        (vel.x / init_speed, vel.z / init_speed)
    } else {
        (1.0, 0.0)
    };

    let mut state = GroundState {
        x: start_pos.x,
        z: start_pos.z,
        vx: vel.x,
        vz: vel.z,
        omega,
        dir_x: init_dir_x,
        dir_z: init_dir_z,
    };

    let mut t = 0.0;
    let mut points = Vec::with_capacity(256);

    points.push(RolloutPoint {
        time: 0.0,
        pos: DVec3::new(state.x, 0.0, state.z),
        speed: (state.vx * state.vx + state.vz * state.vz).sqrt(),
    });

    let max_steps = (MAX_ROLLOUT_TIME / dt) as usize;

    for step_num in 1..=max_steps {
        state = step_ground(state, dt, surface);
        t += dt;

        let speed = (state.vx * state.vx + state.vz * state.vz).sqrt();

        if step_num % sample_every == 0 {
            points.push(RolloutPoint {
                time: t,
                pos: DVec3::new(state.x, 0.0, state.z),
                speed,
            });
        }

        // Stop only when both translation and spin energy are exhausted.
        // Without the omega check, backspin-driven reversal would be killed
        // as the ball passes through zero forward speed.
        let spin_speed = (state.omega * BALL_RADIUS).abs();
        if speed < STOP_SPEED && spin_speed < STOP_SPEED {
            points.push(RolloutPoint {
                time: t,
                pos: DVec3::new(state.x, 0.0, state.z),
                speed: 0.0,
            });
            break;
        }
    }

    let final_pos = DVec3::new(state.x, 0.0, state.z);
    let dx = final_pos.x - start_pos.x;
    let dz = final_pos.z - start_pos.z;
    let rollout_m = (dx * dx + dz * dz).sqrt();

    RolloutResult {
        points,
        final_pos,
        rollout_m,
        rollout_time: t,
    }
}

/// Advance ground state by one timestep.
fn step_ground(state: GroundState, dt: f64, surface: &RolloutSurface) -> GroundState {
    let speed = (state.vx * state.vx + state.vz * state.vz).sqrt();
    if speed < 1e-10 && state.omega.abs() < 1e-10 {
        return state;
    }

    // Fixed "original forward" direction — set once at init, never updated.
    // The omega sign convention is defined relative to this direction,
    // so flipping it would corrupt the signed velocity decomposition
    // (especially during spin-back when the ball reverses).
    let dir_x = state.dir_x;
    let dir_z = state.dir_z;

    // Signed forward velocity: projection of velocity onto the saved direction.
    // Positive = moving in the original forward direction.
    // Negative = ball has reversed (spin-back).
    let v_fwd = state.vx * dir_x + state.vz * dir_z;

    // Contact point velocity (signed, in the forward direction).
    // v_c > 0: contact slides forward (backspin dominant or ball moving forward)
    // v_c < 0: contact slides backward (excess topspin)
    // v_c = 0: pure rolling
    let v_c = v_fwd - BALL_RADIUS * state.omega;
    let is_rolling = v_c.abs() < ROLL_TRANSITION_THRESHOLD
        && speed > STOP_SPEED;

    if is_rolling {
        // Pure rolling: omega locked to v_fwd/R, decelerate together.
        // Speed-dependent friction: turf deformation increases with speed.
        let mu_eff = surface.mu_roll * (1.0 + (speed / V_REF).powi(2));
        let a = (5.0 / 7.0) * mu_eff * G;

        if speed < 1e-10 {
            return GroundState { omega: 0.0, dir_x, dir_z, ..state };
        }

        // Decelerate speed toward zero (rolling friction opposes motion).
        let new_speed = (speed - a * dt).max(0.0);
        let ratio = new_speed / speed;

        GroundState {
            x: state.x + state.vx * (1.0 + ratio) * 0.5 * dt,
            z: state.z + state.vz * (1.0 + ratio) * 0.5 * dt,
            vx: state.vx * ratio,
            vz: state.vz * ratio,
            omega: v_fwd.signum() * new_speed / BALL_RADIUS,
            dir_x,
            dir_z,
        }
    } else {
        // Sliding: friction opposes contact point sliding direction.
        //
        // Applied along the saved direction (dir_x, dir_z) so friction
        // can push the ball through zero speed and into reverse (spin-back).
        let sign_vc = v_c.signum();
        let friction_accel = surface.mu_slide * G;

        let dv = -friction_accel * sign_vc * dt;
        let new_vx = state.vx + dir_x * dv;
        let new_vz = state.vz + dir_z * dv;

        // Spin: torque drives omega toward the rolling condition.
        let omega_accel = (5.0 * friction_accel * sign_vc) / (2.0 * BALL_RADIUS);
        let new_omega = state.omega + omega_accel * dt;

        // Clamp: don't overshoot rolling condition (v_c = 0).
        let new_v_fwd = new_vx * dir_x + new_vz * dir_z;
        let target_omega = new_v_fwd / BALL_RADIUS;
        let new_omega = if sign_vc > 0.0 {
            // omega was too low (backspin), increasing toward target
            new_omega.min(target_omega)
        } else {
            // omega was too high, decreasing toward target
            new_omega.max(target_omega)
        };

        GroundState {
            x: state.x + (state.vx + new_vx) * 0.5 * dt,
            z: state.z + (state.vz + new_vz) * 0.5 * dt,
            vx: new_vx,
            vz: new_vz,
            omega: new_omega,
            dir_x,
            dir_z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units;

    // ══════════════════════════════════════════════════════════════════
    //  STRUCTURAL TESTS
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn rollout_stops() {
        let r = simulate_rollout(
            DVec3::ZERO,
            DVec3::new(10.0, 0.0, 0.0),
            10.0 / BALL_RADIUS, // already rolling
            &RolloutSurface::FAIRWAY,
        );
        assert!(r.points.last().unwrap().speed < STOP_SPEED + 0.01);
    }

    #[test]
    fn rollout_moves_forward() {
        let r = simulate_rollout(
            DVec3::ZERO,
            DVec3::new(10.0, 0.0, 0.0),
            10.0 / BALL_RADIUS,
            &RolloutSurface::FAIRWAY,
        );
        assert!(r.final_pos.x > 0.0, "ball should roll forward");
        assert!(r.rollout_m > 0.0);
    }

    #[test]
    fn rollout_on_ground_plane() {
        let r = simulate_rollout(
            DVec3::ZERO,
            DVec3::new(10.0, 0.0, 2.0),
            0.0,
            &RolloutSurface::GREEN,
        );
        for p in &r.points {
            assert!(
                p.pos.y.abs() < 1e-10,
                "rollout should stay on Y=0: y={:.6}",
                p.pos.y
            );
        }
    }

    #[test]
    fn speed_monotonically_decreases_pure_roll() {
        // Already in pure rolling → speed should only decrease.
        let v0 = 8.0;
        let r = simulate_rollout(
            DVec3::ZERO,
            DVec3::new(v0, 0.0, 0.0),
            v0 / BALL_RADIUS,
            &RolloutSurface::GREEN,
        );
        for w in r.points.windows(2) {
            assert!(
                w[1].speed <= w[0].speed + 0.01,
                "speed should decrease: {:.3} then {:.3}",
                w[0].speed,
                w[1].speed
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════
    //  SURFACE COMPARISON
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn green_rolls_farther_than_fairway() {
        let v0 = 5.0;
        let omega = v0 / BALL_RADIUS; // pure rolling
        let r_fairway = simulate_rollout(DVec3::ZERO, DVec3::new(v0, 0.0, 0.0), omega, &RolloutSurface::FAIRWAY);
        let r_green = simulate_rollout(DVec3::ZERO, DVec3::new(v0, 0.0, 0.0), omega, &RolloutSurface::GREEN);
        assert!(
            r_green.rollout_m > r_fairway.rollout_m,
            "green should roll farther: green={:.1}m, fairway={:.1}m",
            r_green.rollout_m,
            r_fairway.rollout_m
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  BACKSPIN EFFECTS
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn backspin_reduces_rollout() {
        let v0 = 5.0;
        let no_spin = simulate_rollout(
            DVec3::ZERO,
            DVec3::new(v0, 0.0, 0.0),
            v0 / BALL_RADIUS, // pure rolling (no backspin)
            &RolloutSurface::GREEN,
        );
        let with_spin = simulate_rollout(
            DVec3::ZERO,
            DVec3::new(v0, 0.0, 0.0),
            -units::rpm_to_rads(5000.0), // backspin = negative omega
            &RolloutSurface::GREEN,
        );
        assert!(
            with_spin.rollout_m < no_spin.rollout_m,
            "backspin should reduce roll: spin={:.1}m, nospin={:.1}m",
            with_spin.rollout_m,
            no_spin.rollout_m
        );
    }

    #[test]
    fn extreme_backspin_can_reverse() {
        // Very high backspin with low forward speed → friction reverses ball.
        // omega = -838 rad/s (backspin), speed = 2 m/s
        // v_c = 2.0 - 0.021335*(-838) = 2.0 + 17.9 = 19.9 (huge forward slide)
        // Friction decelerates ball, can push it backward.
        let backspin_omega = -units::rpm_to_rads(8000.0);
        let r = simulate_rollout(
            DVec3::ZERO,
            DVec3::new(2.0, 0.0, 0.0),
            backspin_omega,
            &RolloutSurface::GREEN,
        );
        assert!(
            r.final_pos.x < 0.0,
            "extreme backspin should reverse: final_x={:.3}",
            r.final_pos.x
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  QUANTITATIVE — stimpmeter calibration
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn stimpmeter_green_distance() {
        // Stimpmeter: ball exits ramp at 1.83 m/s, already rolling.
        // On stimp-10 green: ~3.05m (10 ft).
        // Our model: mu_roll=0.055, a=(5/7)*0.055*9.81=0.386 m/s^2
        // d = v^2/(2a) = 1.83^2/(2*0.386) = 4.34m → ~14.2 ft
        // Accept 2-6m range.
        let v0 = 1.83;
        let r = simulate_rollout(
            DVec3::ZERO,
            DVec3::new(v0, 0.0, 0.0),
            v0 / BALL_RADIUS,
            &RolloutSurface::GREEN,
        );
        assert!(
            r.rollout_m > 2.0 && r.rollout_m < 6.0,
            "stimpmeter roll should be 2-6m, got {:.2}m ({:.1} ft)",
            r.rollout_m,
            units::meters_to_feet(r.rollout_m)
        );
    }

    #[test]
    fn stimpmeter_fairway_shorter() {
        let v0 = 1.83;
        let omega = v0 / BALL_RADIUS;
        let r_green = simulate_rollout(DVec3::ZERO, DVec3::new(v0, 0.0, 0.0), omega, &RolloutSurface::GREEN);
        let r_fairway = simulate_rollout(DVec3::ZERO, DVec3::new(v0, 0.0, 0.0), omega, &RolloutSurface::FAIRWAY);
        assert!(
            r_fairway.rollout_m < r_green.rollout_m,
            "fairway shorter: {:.2}m vs green {:.2}m",
            r_fairway.rollout_m,
            r_green.rollout_m
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  DRIVER ROLLOUT — sanity check
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn driver_rollout_reasonable() {
        // Post-bounce: ~10 m/s forward, low spin, nearly rolling.
        // Speed-dependent friction (Kensrud 2024) increases deceleration at
        // high speeds. At 10 m/s, mu_eff ≈ mu_roll * (1 + (10/2.5)²) ≈ 2.04,
        // giving ~14 m/s² decel. Rollout is short — realistic for firm fairway.
        let vel = DVec3::new(10.0, 0.0, 0.0);
        let omega = 8.0 / BALL_RADIUS; // slightly less than rolling (some backspin left)
        let r = simulate_rollout(DVec3::ZERO, vel, omega, &RolloutSurface::FAIRWAY);
        let rollout_yards = units::meters_to_yards(r.rollout_m);
        assert!(
            rollout_yards > 5.0 && rollout_yards < 60.0,
            "driver rollout should be 5-60 yards, got {rollout_yards:.1}"
        );
    }
}
