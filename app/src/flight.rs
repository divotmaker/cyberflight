//! Shot animation: a simulated shot's timeline, render data, telemetry and
//! chase camera, and the manager that runs every shot on screen.

use std::collections::HashMap;

use glam::{DVec3, Vec3};

use cf_render::flight::FlightRenderData;
use cf_scene::camera::Camera;
use cf_scene::frame::physics_to_scene;
use cf_scene::green::GreenConfig;
use cf_scene::ground;
use cf_scene::hud::{ShotTelemetry, UnitSystem, build_hud};
use cf_scene::shot::{ClubDelivery, ReceivedShot};
use cf_scene::trail::{TrailPoint, tracer_tail};
use dimple::aero::BallModel;
use dimple::environment::Environment;
use dimple::ground::Termination;
use dimple::trajectory::{ShotInput, ShotResult, TrajectoryPoint, interpolate_trajectory};

use crate::intake::ArrivedShot;

/// How long the chase camera lingers after the ball stops (seconds).
pub const CHASE_LINGER_TIME: f64 = 5.0;
/// Time constant for chase camera smoothing (seconds).
pub const CHASE_CAM_SMOOTHING_TAU: f64 = 0.3;
/// Default subsample stride: every Nth trajectory sample is used as a trail point.
pub const DEFAULT_TRAIL_SUBSAMPLE: usize = 1;
/// Longest a ball that dimple did not bring to rest keeps moving on the
/// ground after landing (seconds). Shots that come to rest animate to the end.
pub const MAX_UNSETTLED_GROUND_TIME_S: f64 = 8.0;

/// A dimple-frame vector as a scene-frame render position.
fn to_scene(p: DVec3) -> Vec3 {
    physics_to_scene(p).as_vec3()
}

/// Build the unified shot timeline from a ShotResult, in dimple's frame:
/// the flight points, then the ground points (hops, slide, roll). Both carry
/// absolute shot times; a ground point at or before the previous point's
/// time (the landing, shared by both) is skipped so times strictly increase.
pub fn build_timeline(result: &ShotResult) -> Vec<TrajectoryPoint> {
    let mut timeline = Vec::with_capacity(result.flight.points.len() + result.ground.points.len());
    let ground = result.ground.points.iter().map(|p| TrajectoryPoint {
        time: p.time,
        pos: p.pos,
        vel: p.vel,
    });
    for point in result.flight.points.iter().copied().chain(ground) {
        if timeline
            .last()
            .is_none_or(|last: &TrajectoryPoint| point.time > last.time)
        {
            timeline.push(point);
        }
    }
    timeline
}

/// An in-progress or recently completed shot animation.
pub struct AnimatedFlight {
    pub input: ShotInput,
    pub result: ShotResult,
    pub timeline: Vec<TrajectoryPoint>,
    pub total_time: f64,
    /// Where the animation ends: dimple's rest position, or the last shown
    /// position when the ball did not settle.
    pub rest_pos: DVec3,
    pub launch_time: f64,
    pub club: ClubDelivery,
    pub lm_carry_yards: Option<f64>,
    /// Whether the device measured the backspin (see [`ReceivedShot`]).
    pub backspin_measured: bool,
    /// Whether the device measured the sidespin (see [`ReceivedShot`]).
    pub sidespin_measured: bool,
    pub landed: bool,
    pub prev_chase: Option<(Camera, f64)>,
}

impl AnimatedFlight {
    pub fn from_received(
        shot: &ReceivedShot,
        launch_time: f64,
        ball: &BallModel,
        env: &Environment,
        green: Option<GreenConfig>,
    ) -> Self {
        let result = ground::simulate_shot(&shot.input, ball, env, green);
        Self::from_result(shot, result, launch_time)
    }

    /// Animate a simulated shot. A ball that did not come to rest stops
    /// [`MAX_UNSETTLED_GROUND_TIME_S`] after landing at the latest.
    fn from_result(shot: &ReceivedShot, result: ShotResult, launch_time: f64) -> Self {
        let mut timeline = build_timeline(&result);
        if result.ground.termination != Termination::Rest {
            let end = result.flight.flight_time + MAX_UNSETTLED_GROUND_TIME_S;
            if let Some(cut) = timeline
                .last()
                .filter(|last| last.time > end)
                .and_then(|_| interpolate_trajectory(&timeline, end))
            {
                timeline.retain(|p| p.time < end);
                // The animation stops the ball here.
                timeline.push(TrajectoryPoint {
                    time: end,
                    pos: cut,
                    vel: DVec3::ZERO,
                });
            }
            eprintln!(
                "WARN: shot #{} from {}: ball did not come to rest ({:?}), animation ends at {:.1}s",
                shot.shot_number,
                shot.device,
                result.ground.termination,
                timeline.last().map_or(0.0, |p| p.time),
            );
        }
        let total_time = timeline.last().map_or(0.0, |p| p.time);
        let rest_pos = timeline.last().map_or(result.final_pos, |p| p.pos);

        let rollout_yards = result.total_yards - result.flight.carry_yards;
        eprintln!(
            "Shot #{} from {}: {:.1} carry + {:.1} rollout = {:.1} total yds, {:.2}s",
            shot.shot_number,
            shot.device,
            result.flight.carry_yards,
            rollout_yards,
            result.total_yards,
            total_time,
        );

        Self {
            input: shot.input,
            result,
            timeline,
            total_time,
            rest_pos,
            launch_time,
            club: shot.club.clone(),
            lm_carry_yards: shot.lm_carry_yards,
            backspin_measured: shot.backspin_measured,
            sidespin_measured: shot.sidespin_measured,
            landed: false,
            prev_chase: None,
        }
    }

    pub fn telemetry(&self, now: f64) -> ShotTelemetry {
        let shot_t = (now - self.launch_time).max(0.0);
        let t_clamped = shot_t.min(self.total_time);

        // Live position while the ball moves; dimple's rest position once it
        // has stopped.
        let pos = if shot_t >= self.total_time {
            self.rest_pos
        } else {
            interpolate_trajectory(&self.timeline, t_clamped).unwrap_or(DVec3::ZERO)
        };

        let total_m = pos.x.hypot(pos.z);
        let downrange_m = pos.x;

        let carry_yards = if shot_t < self.result.flight.flight_time {
            dimple::units::meters_to_yards(total_m)
        } else {
            self.result.flight.carry_yards
        };

        ShotTelemetry {
            club_speed_mph: self.club.club_speed_mph,
            ball_speed_mph: self.input.ball_speed_mph,
            smash_factor: self.club.smash_factor,
            launch_angle_deg: self.input.launch_angle_deg,
            launch_azimuth_deg: self.input.launch_azimuth_deg,
            backspin_rpm: self.backspin_measured.then_some(self.input.backspin_rpm),
            sidespin_rpm: self.sidespin_measured.then_some(self.input.sidespin_rpm),
            club_path_deg: self.club.path_deg,
            attack_angle_deg: self.club.attack_angle_deg,
            face_angle_deg: self.club.face_angle_deg,
            dynamic_loft_deg: self.club.dynamic_loft_deg,
            apex_m: if shot_t < self.result.flight.flight_time {
                pos.y.max(0.0)
            } else {
                self.result.flight.apex_m
            },
            carry_yards,
            lm_carry_yards: self.lm_carry_yards,
            total_yards: dimple::units::meters_to_yards(total_m),
            downrange_yards: dimple::units::meters_to_yards(downrange_m),
            lateral_yards: dimple::units::meters_to_yards(pos.z),
            flight_time_s: self.result.flight.flight_time,
            elapsed_s: t_clamped,
        }
    }

    pub fn render_data(&self, now: f64, trail_subsample: usize) -> Option<FlightRenderData> {
        let shot_t = now - self.launch_time;
        if shot_t < 0.0 {
            return None;
        }

        let t_clamped = shot_t.min(self.total_time);

        let ball_pos = interpolate_trajectory(&self.timeline, t_clamped).map(to_scene)?;

        let samples = self.timeline.iter().map(|p| TrailPoint {
            position: to_scene(p.pos),
            time: p.time,
        });
        let trail_points = tracer_tail(samples, t_clamped, ball_pos, trail_subsample);

        Some(FlightRenderData {
            ball_pos,
            trail_points,
            current_time: shot_t,
        })
    }

    pub fn chase_camera(&mut self, now: f64) -> Option<Camera> {
        let shot_t = now - self.launch_time;
        if shot_t < 0.0 {
            return None;
        }

        let final_pos = to_scene(self.rest_pos);

        let shot_dir = if final_pos.length_squared() > 0.01 {
            final_pos.normalize()
        } else {
            Vec3::Z
        };

        let ideal = if shot_t < self.total_time {
            let ball_pos = interpolate_trajectory(&self.timeline, shot_t).map(to_scene)?;

            let idx = self.timeline.partition_point(|p| p.time <= shot_t);
            let vel_idx = if idx > 0 { idx - 1 } else { 0 };
            let vel = if vel_idx + 1 < self.timeline.len() {
                let a = &self.timeline[vel_idx];
                let b = &self.timeline[vel_idx + 1];
                let dt = (b.time - a.time).max(1e-6);
                (b.pos - a.pos) / dt
            } else {
                DVec3::new(1.0, 0.0, 0.0)
            };
            let velocity = to_scene(vel);

            let horiz_speed = Vec3::new(velocity.x, 0.0, velocity.z).length();
            let direction = if horiz_speed > 2.0 {
                velocity
            } else {
                shot_dir
            };

            Camera::chase(ball_pos, direction, final_pos)
        } else if shot_t < self.total_time + CHASE_LINGER_TIME {
            Camera::chase(final_pos, shot_dir, final_pos)
        } else {
            self.prev_chase = None;
            return None;
        };

        // During carry, snap to the ball's direction of travel so hooks/slices
        // stay visible. Smoothing only kicks in after landing to avoid jarring
        // camera motion from hops and slow roll direction changes.
        let in_carry = shot_t < self.result.flight.flight_time;
        let smoothed = if in_carry {
            ideal
        } else if let Some((prev, prev_time)) = &self.prev_chase {
            let dt = (now - prev_time).clamp(0.001, 0.1);
            let alpha = (dt / (CHASE_CAM_SMOOTHING_TAU + dt)) as f32;
            Camera {
                position: prev.position.lerp(ideal.position, alpha),
                target: prev.target.lerp(ideal.target, alpha),
                ..ideal
            }
        } else {
            ideal
        };

        self.prev_chase = Some((smoothed, now));
        Some(smoothed)
    }

    pub fn is_animating(&self, now: f64) -> bool {
        let shot_t = now - self.launch_time;
        shot_t >= 0.0 && shot_t < self.total_time
    }

    /// Total distance where the animation ends (yards): dimple's total for
    /// a ball that came to rest, the cut position's for one that did not.
    /// The HUD shows the same value once the ball has stopped.
    pub fn shown_total_yards(&self) -> f64 {
        dimple::units::meters_to_yards(self.rest_pos.x.hypot(self.rest_pos.z))
    }

    pub fn is_expired(&self, now: f64) -> bool {
        let shot_t = now - self.launch_time;
        shot_t > self.total_time + CHASE_LINGER_TIME
    }
}

/// A shot taken in this much later than it arrived is reported (seconds).
const LATE_INTAKE_S: f64 = 0.25;

/// Shared flight manager: handles animation, telemetry, and camera for a list of flights.
pub struct FlightManager {
    pub flights: Vec<AnimatedFlight>,
    pub last_telemetry: Option<ShotTelemetry>,
    pub trail_subsample: usize,
}

impl Default for FlightManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FlightManager {
    pub fn new() -> Self {
        Self {
            flights: Vec::new(),
            last_telemetry: None,
            trail_subsample: DEFAULT_TRAIL_SUBSAMPLE,
        }
    }

    /// Launch a shot that arrived, as of `now`: it starts at its arrival
    /// time (never after `now`), so a shot taken in late is where it would
    /// have been had it been taken in at once, and one that came to rest
    /// meanwhile is not shown flying.
    pub fn take_in(
        &mut self,
        arrived: &ArrivedShot,
        now: f64,
        env: &Environment,
        green: Option<GreenConfig>,
    ) {
        let launch_time = arrived.arrival.min(now);
        let late = now - launch_time;
        if late > LATE_INTAKE_S {
            eprintln!(
                "Shot #{} from {} taken in {late:.2}s after it arrived",
                arrived.shot.shot_number, arrived.shot.device
            );
        }
        self.launch(&arrived.shot, launch_time, env, green);
    }

    /// Simulate `shot` and animate it from `launch_time`.
    ///
    /// Earlier balls keep animating to rest; the newest ball drives the chase
    /// camera and the HUD.
    pub fn launch(
        &mut self,
        shot: &ReceivedShot,
        launch_time: f64,
        env: &Environment,
        green: Option<GreenConfig>,
    ) {
        if self.is_animating(launch_time) {
            eprintln!(
                "Shot #{} from {} launched while an earlier ball is still moving",
                shot.shot_number, shot.device
            );
        }
        self.flights.push(AnimatedFlight::from_received(
            shot,
            launch_time,
            &BallModel::TOUR,
            env,
            green,
        ));
    }

    /// Update flight state: mark landed, remove expired, build render data.
    pub fn update(&mut self, now: f64) -> Vec<FlightRenderData> {
        // Mark completed shots
        for flight in &mut self.flights {
            if !flight.landed && !flight.is_animating(now) && now > flight.launch_time {
                flight.landed = true;
                let total_yards = flight.shown_total_yards();
                eprintln!(
                    "  Stopped: {:.1} total yds ({:.1} carry + {:.1} rollout)",
                    total_yards,
                    flight.result.flight.carry_yards,
                    total_yards - flight.result.flight.carry_yards,
                );
            }
        }

        // The HUD keeps the newest shot's numbers after its flight expires.
        if let Some(newest) = self.flights.last() {
            self.last_telemetry = Some(newest.telemetry(now));
        }

        // Remove expired flights
        self.flights.retain(|f| !f.is_expired(now));

        // Build render data
        self.flights
            .iter()
            .filter_map(|f| f.render_data(now, self.trail_subsample))
            .collect()
    }

    pub fn is_animating(&self, now: f64) -> bool {
        self.flights.iter().any(|f| f.is_animating(now))
    }

    /// Compute the chase camera for the most recent active flight (if any).
    pub fn chase_camera(&mut self, now: f64) -> Option<Camera> {
        self.flights
            .iter_mut()
            .rev()
            .find_map(|f| f.chase_camera(now))
    }

    /// Build HUD geometry for the current frame from the newest shot.
    pub fn build_hud(
        &mut self,
        now: f64,
        chase_active: bool,
        lm_states: &HashMap<String, bool>,
        frp_connected: bool,
        screen_w: f32,
        screen_h: f32,
    ) -> cf_scene::hud::HudGeometry {
        if let Some(f) = self.flights.last() {
            self.last_telemetry = Some(f.telemetry(now));
        }
        build_hud(
            self.last_telemetry.as_ref(),
            UnitSystem::Imperial,
            chase_active,
            lm_states,
            frp_connected,
            screen_w,
            screen_h,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cf_scene::range::Range;
    use cf_scene::shot::ClubDelivery;

    fn received(input: ShotInput) -> ReceivedShot {
        ReceivedShot {
            device: "test.0".to_owned(),
            shot_number: 1,
            input,
            club: ClubDelivery::default(),
            lm_carry_yards: None,
            backspin_measured: true,
            sidespin_measured: true,
            clamped: Vec::new(),
        }
    }

    fn animate(input: ShotInput, green: Option<GreenConfig>) -> AnimatedFlight {
        AnimatedFlight::from_received(
            &received(input),
            0.0,
            &BallModel::TOUR,
            &Environment::SEA_LEVEL,
            green,
        )
    }

    #[test]
    fn a_shot_taken_in_late_is_where_it_would_have_been() {
        let mut fm = FlightManager::new();
        let shot = ArrivedShot {
            shot: received(ShotInput::seven_iron()),
            arrival: 1.0,
        };
        fm.take_in(&shot, 2.5, &Environment::SEA_LEVEL, None);
        assert_eq!(fm.flights[0].launch_time, 1.0);

        // 1.5 s into its flight, as if it had launched on arrival.
        let on_time = animate(ShotInput::seven_iron(), None);
        let late = fm.update(2.5);
        let expected = on_time.render_data(1.5, 1).expect("in flight");
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].ball_pos, expected.ball_pos);
    }

    #[test]
    fn a_shot_from_a_long_gap_is_not_animated_as_new() {
        // Two shots arrived while nothing was drawn; both came to rest long ago.
        let mut fm = FlightManager::new();
        let env = Environment::SEA_LEVEL;
        for (n, arrival, input) in [
            (1, 1.0, ShotInput::driver()),
            (2, 12.0, ShotInput::seven_iron()),
        ] {
            let mut shot = received(input);
            shot.shot_number = n;
            fm.take_in(&ArrivedShot { shot, arrival }, 100.0, &env, None);
        }
        assert!(fm.update(100.0).is_empty(), "no ball is drawn");
        assert!(fm.flights.is_empty(), "both have expired");
        assert!(fm.chase_camera(100.0).is_none());
        assert!(!fm.is_animating(100.0));
        // The HUD shows the newest shot, at rest.
        let shown = fm.last_telemetry.as_ref().expect("telemetry");
        let seven_iron = animate(ShotInput::seven_iron(), None);
        assert!((shown.ball_speed_mph - seven_iron.telemetry(0.0).ball_speed_mph).abs() < 1e-9);
    }

    #[test]
    fn a_shot_is_not_launched_after_now() {
        let mut fm = FlightManager::new();
        let shot = ArrivedShot {
            shot: received(ShotInput::seven_iron()),
            arrival: 5.0,
        };
        fm.take_in(&shot, 4.99, &Environment::SEA_LEVEL, None);
        assert_eq!(fm.flights[0].launch_time, 4.99);
    }

    #[test]
    fn shot_during_rollout_is_launched() {
        let mut fm = FlightManager::new();
        let env = Environment::SEA_LEVEL;
        let green = Range::new().green;
        fm.launch(&received(ShotInput::driver()), 0.0, &env, green);

        // The drive is on the ground but still rolling.
        let now = fm.flights[0].result.flight.flight_time + 1.0;
        assert!(fm.is_animating(now));
        let second = ShotInput::seven_iron();
        fm.launch(&received(second), now, &env, green);
        assert_eq!(fm.flights.len(), 2, "second shot is launched, not dropped");

        // Both balls are drawn; the HUD and the chase camera follow the newest.
        let t = now + 0.5;
        assert_eq!(fm.update(t).len(), 2);
        let _ = fm.build_hud(t, true, &HashMap::new(), true, 1920.0, 1080.0);
        let shown = fm.last_telemetry.as_ref().expect("telemetry");
        assert!((shown.ball_speed_mph - second.ball_speed_mph).abs() < 1e-9);
        let chase = fm.chase_camera(t).expect("newest ball is followed");
        let newest = &fm.flights[1];
        assert_eq!(chase.target, to_scene(newest.rest_pos));
        let ball = newest.render_data(t, 1).expect("in flight").ball_pos;
        assert!(((chase.position - ball).length() - 125f32.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn timeline_is_flight_then_ground() {
        let f = animate(ShotInput::driver(), Range::new().green);
        let r = &f.result;
        assert!(
            f.timeline.windows(2).all(|w| w[1].time > w[0].time),
            "times strictly increase"
        );
        assert_eq!(f.timeline[0].pos, DVec3::ZERO);
        let landing = r.flight.points.last().expect("flight has points");
        let i = f
            .timeline
            .iter()
            .position(|p| p.time == landing.time)
            .expect("landing in timeline");
        assert_eq!(f.timeline[i].pos, landing.pos);
        assert!(f.timeline[i + 1..].iter().all(|p| p.pos.y >= -1e-9));
        assert!(f.timeline.len() > r.flight.points.len());
        let last = f.timeline.last().expect("timeline has points");
        assert_eq!(last.pos, r.final_pos);
        assert_eq!(f.total_time, last.time);
        assert!(f.total_time > r.flight.flight_time + 1.0, "ball runs out");
    }

    #[test]
    fn unsettled_ball_stops_animating() {
        let input = ShotInput::driver();
        let mut result =
            ground::simulate_shot(&input, &BallModel::TOUR, &Environment::SEA_LEVEL, None);
        assert_eq!(result.ground.termination, Termination::Rest);
        // A ground phase that runs into dimple's time limit.
        let mut late = *result.ground.points.last().expect("ground points");
        late.time += 100.0;
        late.pos.x += 50.0;
        result.ground.points.push(late);
        result.ground.termination = Termination::MaxTime;
        result.final_pos = late.pos;
        result.total_yards = dimple::units::meters_to_yards(late.pos.x.hypot(late.pos.z));

        let f = AnimatedFlight::from_result(&received(input), result, 0.0);
        let end = f.result.flight.flight_time + MAX_UNSETTLED_GROUND_TIME_S;
        // The log reports the distance the HUD shows, not dimple's.
        let shown = f.telemetry(end + 1.0).total_yards;
        assert!((f.shown_total_yards() - shown).abs() < 1e-9);
        assert!(f.shown_total_yards() < f.result.total_yards - 1.0);
        assert!(
            (f.total_time - end).abs() < 1e-9,
            "total {:.1}",
            f.total_time
        );
        let last = f.timeline.last().expect("timeline");
        assert_eq!(last.time, f.total_time);
        assert_eq!(last.pos, f.rest_pos);
        assert!(f.timeline.windows(2).all(|w| w[1].time > w[0].time));
        assert!(!f.is_animating(end + 0.01));
        assert!(f.is_expired(end + CHASE_LINGER_TIME + 0.01));
    }

    #[test]
    fn settled_ball_animates_to_rest() {
        let f = animate(ShotInput::driver(), None);
        assert_eq!(f.result.ground.termination, Termination::Rest);
        assert_eq!(f.rest_pos, f.result.final_pos);
        assert!(f.total_time > f.result.flight.flight_time + 1.0);
    }

    #[test]
    fn chase_camera_follows_a_hook_then_lets_go() {
        let input = ShotInput {
            launch_azimuth_deg: -2.0,
            sidespin_rpm: -900.0,
            ..ShotInput::driver()
        };
        let mut f = animate(input, Range::new().green);
        assert!(f.rest_pos.z < -10.0, "hook finishes left");
        let end = f.total_time + CHASE_LINGER_TIME;
        let mut t = 0.0;
        while t < end {
            let cam = f.chase_camera(t).expect("followed until the linger ends");
            assert!(
                cam.position.is_finite() && cam.target.is_finite(),
                "t = {t:.2}: {cam:?}"
            );
            assert!((cam.position - cam.target).length() > 1.0);
            t += 0.05;
        }
        assert!(f.chase_camera(end + 0.01).is_none());
        assert!(f.prev_chase.is_none());
    }

    /// `ReceivedShot::from_frp` rejects a non-positive speed; this covers a
    /// zero-speed input reaching the animation directly.
    #[test]
    fn motionless_shot_is_a_single_point() {
        let input = ShotInput {
            ball_speed_mph: 0.0,
            backspin_rpm: 0.0,
            sidespin_rpm: 0.0,
            ..ShotInput::driver()
        };
        let f = animate(input, None);
        assert_eq!(f.timeline.len(), 1);
        assert_eq!(f.total_time, 0.0);
        assert!(!f.is_animating(1.0));
    }

    #[test]
    fn interpolates_through_the_ground_phase() {
        let f = animate(ShotInput::seven_iron(), None);
        let t = f.result.flight.flight_time + 0.5;
        let pos = interpolate_trajectory(&f.timeline, t).expect("inside timeline");
        assert!(pos.x > f.result.flight.carry_m * 0.99);
        assert!(pos.x < f.result.final_pos.x + 1e-9);
    }

    #[test]
    fn telemetry_at_rest_reports_dimple_results() {
        let input = ShotInput {
            launch_azimuth_deg: 1.0,
            sidespin_rpm: 600.0,
            ..ShotInput::driver()
        };
        let f = animate(input, Range::new().green);
        let t = f.telemetry(f.total_time + 1.0);
        let r = &f.result;
        assert!((t.carry_yards - r.flight.carry_yards).abs() < 1e-9);
        assert!((t.total_yards - r.total_yards).abs() < 1e-9);
        let lateral = dimple::units::meters_to_yards(r.final_pos.z);
        assert!((t.lateral_yards - lateral).abs() < 1e-9);
        assert!(t.lateral_yards > 0.0, "slice finishes right");
        assert!(
            (t.elapsed_s - f.total_time).abs() < 1e-9,
            "clock runs to rest"
        );
    }

    #[test]
    fn slice_is_drawn_right_of_the_line() {
        let input = ShotInput {
            launch_azimuth_deg: 1.0,
            sidespin_rpm: 600.0,
            ..ShotInput::driver()
        };
        let f = animate(input, Range::new().green);
        let data = f.render_data(f.total_time, 1).expect("launched");
        // Scene +X is the golfer's left (see `cf_scene::frame`).
        assert!(data.ball_pos.x < -10.0, "ball at {:?}", data.ball_pos);
        let camera = Camera::driving_range();
        let mut proj = camera.projection_matrix(16.0 / 9.0);
        proj.y_axis.y *= -1.0;
        let clip = proj * camera.view_matrix() * data.ball_pos.extend(1.0);
        assert!(clip.x / clip.w > 0.0, "slice is on the right of the screen");
    }
}
