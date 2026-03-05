use glam::DVec3;

/// State vector for RK4 integration: position and velocity.
#[derive(Debug, Clone, Copy)]
pub struct State {
    pub pos: DVec3,
    pub vel: DVec3,
}

/// Advance state by one RK4 step.
///
/// `accel_fn(position, velocity, time) -> acceleration`
#[must_use]
pub fn step<F>(state: State, t: f64, dt: f64, accel_fn: F) -> State
where
    F: Fn(DVec3, DVec3, f64) -> DVec3,
{
    let k1_v = accel_fn(state.pos, state.vel, t);
    let k1_x = state.vel;

    let k2_v = accel_fn(
        state.pos + k1_x * (dt / 2.0),
        state.vel + k1_v * (dt / 2.0),
        t + dt / 2.0,
    );
    let k2_x = state.vel + k1_v * (dt / 2.0);

    let k3_v = accel_fn(
        state.pos + k2_x * (dt / 2.0),
        state.vel + k2_v * (dt / 2.0),
        t + dt / 2.0,
    );
    let k3_x = state.vel + k2_v * (dt / 2.0);

    let k4_v = accel_fn(
        state.pos + k3_x * dt,
        state.vel + k3_v * dt,
        t + dt,
    );
    let k4_x = state.vel + k3_v * dt;

    State {
        pos: state.pos + (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) * (dt / 6.0),
        vel: state.vel + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) * (dt / 6.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::G;

    #[test]
    fn vacuum_parabola() {
        // Launch at 45 degrees, 50 m/s, no drag
        let speed = 50.0;
        let angle = std::f64::consts::FRAC_PI_4;
        let state = State {
            pos: DVec3::ZERO,
            vel: DVec3::new(speed * angle.cos(), speed * angle.sin(), 0.0),
        };

        let dt = 0.001;
        let mut s = state;
        let mut t = 0.0;

        // Gravity only
        let gravity = |_pos: DVec3, _vel: DVec3, _t: f64| DVec3::new(0.0, -G, 0.0);

        while s.pos.y >= 0.0 || t < 0.1 {
            s = step(s, t, dt, gravity);
            t += dt;
        }

        // Analytical range: v^2 * sin(2*theta) / g
        let expected_range = speed * speed * (2.0 * angle).sin() / G;
        assert!(
            (s.pos.x - expected_range).abs() < 0.5,
            "range {:.1} vs expected {:.1}",
            s.pos.x,
            expected_range
        );
    }

    #[test]
    fn constant_acceleration() {
        let state = State {
            pos: DVec3::ZERO,
            vel: DVec3::ZERO,
        };
        let dt = 0.001;
        let a = DVec3::new(1.0, 0.0, 0.0);

        let mut s = state;
        for i in 0..1000 {
            s = step(s, i as f64 * dt, dt, |_, _, _| a);
        }

        let t = 1.0;
        let expected_pos = 0.5 * a * t * t;
        let expected_vel = a * t;

        assert!((s.pos - expected_pos).length() < 1e-6);
        assert!((s.vel - expected_vel).length() < 1e-6);
    }
}
