//! Shot animation: a simulated shot's timeline, render data, telemetry and
//! chase camera, and the manager that runs every shot on screen.

use std::collections::HashMap;

use glam::Vec3;

use cf_render::flight::FlightRenderData;
use cf_scene::camera::Camera;
use cf_scene::frame::physics_to_scene;
use cf_scene::ground;
use cf_scene::hole::{Cut, Hole, Stop};
use cf_scene::hud::{HoleTelemetry, ShotTelemetry, build_hud};
use cf_scene::range::Ground;
use cf_scene::shot::{ClubDelivery, ReceivedShot};
use cf_scene::trail::{TrailPoint, tracer_tail};
use dimple::AeroModel;
use dimple::ground::Termination;
use dimple::trajectory::{ShotInput, ShotResult, TrajectoryPoint, interpolate_trajectory};
use dimple::units::meters_to_yards;

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
fn to_scene(p: dimple::Vec3) -> Vec3 {
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

/// End `timeline` at `end` with the ball stopped at `pos`: points from
/// `end` on are dropped.
fn end_timeline_at(timeline: &mut Vec<TrajectoryPoint>, end: f64, pos: dimple::Vec3) {
    timeline.retain(|p| p.time < end);
    timeline.push(TrajectoryPoint {
        time: end,
        pos,
        vel: dimple::Vec3::ZERO,
    });
}

/// An in-progress or recently completed shot animation.
pub struct AnimatedFlight {
    pub input: ShotInput,
    pub result: ShotResult,
    pub timeline: Vec<TrajectoryPoint>,
    pub total_time: f64,
    /// Where the animation ends: dimple's rest position, or the last shown
    /// position when the ball did not settle.
    pub rest_pos: dimple::Vec3,
    pub launch_time: f64,
    pub club: ClubDelivery,
    pub lm_carry_yards: Option<f64>,
    /// Whether the device measured the backspin (see [`ReceivedShot`]).
    pub backspin_measured: bool,
    /// Whether the device measured the sidespin (see [`ReceivedShot`]).
    pub sidespin_measured: bool,
    /// Whether the shot is a ground ball, rolled instead of flown (see
    /// [`ReceivedShot`]). Its launch angle is not shown.
    pub rolled: bool,
    pub landed: bool,
    pub prev_chase: Option<(Camera, f64)>,
    /// The hole the shot is played on; `None` on the driving range.
    pub hole: Option<Hole>,
    /// How the shot stopped early on the hole (cup, water or tree), if it
    /// did.
    pub stop: Option<Stop>,
}

impl AnimatedFlight {
    pub fn from_received(
        shot: &ReceivedShot,
        launch_time: f64,
        model: &dyn AeroModel,
        ground: &Ground,
    ) -> Self {
        let result = if shot.rolled {
            ground::simulate_roll(&shot.input, model, ground)
        } else {
            ground::simulate_shot(&shot.input, model, ground)
        };
        let hole = ground.hole();
        let cut = hole.and_then(|h| {
            h.stop(
                &build_timeline(&result),
                &result.ground.points,
                model.radius_m(),
                model.gravity(),
            )
        });
        Self::from_result(shot, result, launch_time, hole.cloned(), cut)
    }

    /// Animate a simulated shot on `hole` (`None` on the driving range).
    /// A shot `cut` short (cup, water, tree) ends as the cut says; a ball that
    /// did not come to rest stops [`MAX_UNSETTLED_GROUND_TIME_S`] after
    /// landing at the latest.
    fn from_result(
        shot: &ReceivedShot,
        result: ShotResult,
        launch_time: f64,
        hole: Option<Hole>,
        cut: Option<Cut>,
    ) -> Self {
        let mut timeline = build_timeline(&result);
        let stop = cut.as_ref().map(|c| c.stop);
        if let Some(cut) = cut {
            let time = cut.time();
            timeline.retain(|p| p.time < time);
            timeline.extend(cut.path);
        } else if result.ground.termination != Termination::Rest {
            let end = result.flight.flight_time + MAX_UNSETTLED_GROUND_TIME_S;
            if let Some(cut) = timeline
                .last()
                .filter(|last| last.time > end)
                .and_then(|_| interpolate_trajectory(&timeline, end))
            {
                // The animation stops the ball here.
                end_timeline_at(&mut timeline, end, cut);
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
        let rest_pos = timeline.last().map_or(result.ground.rest.pos, |p| p.pos);

        let carry_yards = meters_to_yards(result.flight.carry_m);
        let total_yards = meters_to_yards(result.total_m);
        let rollout_yards = total_yards - carry_yards;
        let on_hole = hole.as_ref().map_or_else(String::new, |h| {
            let rest = physics_to_scene(rest_pos);
            let (to_pin, lie) = if stop == Some(Stop::Cup) {
                (0.0, "HOLED")
            } else {
                (h.to_cup(rest.x, rest.z), h.lie_at(rest.x, rest.z).label())
            };
            let tree = if stop == Some(Stop::Tree) {
                ", hit a tree"
            } else {
                ""
            };
            format!(", {:.1} yds to pin, {lie}{tree}", meters_to_yards(to_pin))
        });
        eprintln!(
            "Shot #{} from {}: {:.1} carry + {:.1} rollout = {:.1} total yds, {:.2}s{on_hole}",
            shot.shot_number, shot.device, carry_yards, rollout_yards, total_yards, total_time,
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
            rolled: shot.rolled,
            landed: false,
            prev_chase: None,
            stop,
            hole,
        }
    }

    /// Whether the ball dropped into the hole's cup.
    #[must_use]
    pub fn holed(&self) -> bool {
        self.stop == Some(Stop::Cup)
    }

    pub fn telemetry(&self, now: f64) -> ShotTelemetry {
        let shot_t = (now - self.launch_time).max(0.0);
        let t_clamped = shot_t.min(self.total_time);

        // Live position while the ball moves; dimple's rest position once it
        // has stopped.
        let pos = if shot_t >= self.total_time {
            self.rest_pos
        } else {
            interpolate_trajectory(&self.timeline, t_clamped).unwrap_or(dimple::Vec3::ZERO)
        };

        let total_m = pos.x.hypot(pos.z);
        let downrange_m = pos.x;

        let carry_yards = if shot_t < self.result.flight.flight_time {
            dimple::units::meters_to_yards(total_m)
        } else {
            meters_to_yards(self.result.flight.carry_m)
        };

        ShotTelemetry {
            club_speed_mph: self.club.club_speed_mph,
            ball_speed_mph: self.input.ball_speed_mph,
            smash_factor: self.club.smash_factor,
            launch_angle_deg: (!self.rolled).then_some(self.input.launch_angle_deg),
            launch_azimuth_deg: self.input.launch_azimuth_deg,
            backspin_rpm: self.backspin_measured.then_some(self.input.backspin_rpm),
            sidespin_rpm: self.sidespin_measured.then_some(self.input.sidespin_rpm),
            club_path_deg: self.club.path_deg,
            attack_angle_deg: self.club.attack_angle_deg,
            face_angle_deg: self.club.face_angle_deg,
            dynamic_loft_deg: self.club.dynamic_loft_deg,
            apex_m: if shot_t < self.result.flight.flight_time {
                // The highest point so far, so the row never falls after the apex.
                let passed = self.timeline.partition_point(|p| p.time <= t_clamped);
                self.timeline[..passed]
                    .iter()
                    .map(|p| p.pos.y)
                    .fold(pos.y.max(0.0), f64::max)
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
            hole: self.hole.as_ref().map(|h| {
                let stopped = shot_t >= self.total_time;
                let at = physics_to_scene(pos);
                let to_pin = if stopped && self.holed() {
                    0.0
                } else {
                    h.to_cup(at.x, at.z)
                };
                HoleTelemetry {
                    to_pin_yards: meters_to_yards(to_pin),
                    lie: stopped.then(|| {
                        if self.holed() {
                            "HOLED"
                        } else {
                            h.lie_at(at.x, at.z).label()
                        }
                    }),
                }
            }),
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
                dimple::Vec3::X
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
    /// The newest launched shot's flight has expired: the HUD keeps its last
    /// numbers and the chase camera lets go, whatever older shots remain.
    newest_expired: bool,
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
            newest_expired: false,
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
        model: &dyn AeroModel,
        ground: &Ground,
    ) {
        let launch_time = arrived.arrival.min(now);
        let late = now - launch_time;
        if late > LATE_INTAKE_S {
            eprintln!(
                "Shot #{} from {} taken in {late:.2}s after it arrived",
                arrived.shot.shot_number, arrived.shot.device
            );
        }
        self.launch(&arrived.shot, launch_time, model, ground);
    }

    /// Simulate `shot` and animate it from `launch_time`.
    ///
    /// Earlier balls keep animating to rest; the newest ball drives the chase
    /// camera and the HUD.
    pub fn launch(
        &mut self,
        shot: &ReceivedShot,
        launch_time: f64,
        model: &dyn AeroModel,
        ground: &Ground,
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
            model,
            ground,
        ));
        self.newest_expired = false;
    }

    /// Update flight state: mark landed, remove expired, build render data.
    pub fn update(&mut self, now: f64) -> Vec<FlightRenderData> {
        // Mark completed shots
        for flight in &mut self.flights {
            if !flight.landed && !flight.is_animating(now) && now > flight.launch_time {
                flight.landed = true;
                let total_yards = flight.shown_total_yards();
                let carry_yards = meters_to_yards(flight.result.flight.carry_m);
                eprintln!(
                    "  Stopped: {:.1} total yds ({:.1} carry + {:.1} rollout)",
                    total_yards,
                    carry_yards,
                    total_yards - carry_yards,
                );
            }
        }

        // The HUD keeps the newest shot's numbers after its flight expires.
        if !self.newest_expired
            && let Some(newest) = self.flights.last()
        {
            self.last_telemetry = Some(newest.telemetry(now));
            self.newest_expired = newest.is_expired(now);
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

    /// Compute the chase camera for the newest shot, while its flight lasts.
    pub fn chase_camera(&mut self, now: f64) -> Option<Camera> {
        if self.newest_expired {
            return None;
        }
        self.flights.last_mut()?.chase_camera(now)
    }

    /// Build HUD geometry for the current frame from the newest shot's
    /// telemetry as of the last [`update`](Self::update).
    pub fn build_hud(
        &self,
        chase_active: bool,
        lm_states: &HashMap<String, bool>,
        frp_connected: bool,
        screen_w: f32,
        screen_h: f32,
    ) -> cf_scene::hud::HudGeometry {
        build_hud(
            self.last_telemetry.as_ref(),
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
    use dimple::Tour;

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
            rolled: false,
        }
    }

    /// Flat ground with no green: the default range.
    const FLAT: Ground = Ground::Flat { green: None };

    fn animate(input: ShotInput, ground: &Ground) -> AnimatedFlight {
        AnimatedFlight::from_received(&received(input), 0.0, &Tour::SEA_LEVEL, ground)
    }

    #[test]
    fn a_shot_taken_in_late_is_where_it_would_have_been() {
        let mut fm = FlightManager::new();
        let shot = ArrivedShot {
            shot: received(ShotInput::seven_iron()),
            arrival: 1.0,
        };
        fm.take_in(&shot, 2.5, &Tour::SEA_LEVEL, &FLAT);
        assert_eq!(fm.flights[0].launch_time, 1.0);

        // 1.5 s into its flight, as if it had launched on arrival.
        let on_time = animate(ShotInput::seven_iron(), &FLAT);
        let late = fm.update(2.5);
        let expected = on_time.render_data(1.5, 1).expect("in flight");
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].ball_pos, expected.ball_pos);
    }

    #[test]
    fn a_shot_from_a_long_gap_is_not_animated_as_new() {
        // Two shots arrived while nothing was drawn; both came to rest long ago.
        let mut fm = FlightManager::new();
        let model = Tour::SEA_LEVEL;
        for (n, arrival, input) in [
            (1, 1.0, ShotInput::driver()),
            (2, 12.0, ShotInput::seven_iron()),
        ] {
            let mut shot = received(input);
            shot.shot_number = n;
            fm.take_in(&ArrivedShot { shot, arrival }, 100.0, &model, &FLAT);
        }
        assert!(fm.update(100.0).is_empty(), "no ball is drawn");
        assert!(fm.flights.is_empty(), "both have expired");
        assert!(fm.chase_camera(100.0).is_none());
        assert!(!fm.is_animating(100.0));
        // The HUD shows the newest shot, at rest.
        let shown = fm.last_telemetry.as_ref().expect("telemetry");
        let seven_iron = animate(ShotInput::seven_iron(), &FLAT);
        assert!((shown.ball_speed_mph - seven_iron.telemetry(0.0).ball_speed_mph).abs() < 1e-9);
    }

    #[test]
    fn the_hud_and_chase_camera_follow_the_newest_shot_only() {
        // A driver at 0 s, a short chip at 1 s: the chip's flight expires
        // while the driver's is still on screen.
        let mut fm = FlightManager::new();
        let model = Tour::SEA_LEVEL;
        let driver = received(ShotInput::driver());
        let chip = received(ShotInput {
            ball_speed_mph: 40.0,
            launch_angle_deg: 25.0,
            ..ShotInput::seven_iron()
        });
        fm.launch(&driver, 0.0, &model, &FLAT);
        fm.launch(&chip, 1.0, &model, &FLAT);
        let chip_flight = animate(chip.input, &FLAT);
        let chip_expires = 1.0 + chip_flight.total_time + CHASE_LINGER_TIME;
        let driver_expires = fm.flights[0].total_time + CHASE_LINGER_TIME;
        assert!(
            chip_expires + 1.0 < driver_expires,
            "the chip expires first"
        );

        fm.update(chip_expires - 1.0);
        fm.update(chip_expires + 1.0);
        assert_eq!(fm.flights.len(), 1, "only the driver remains");
        let shown = fm
            .last_telemetry
            .as_ref()
            .expect("telemetry")
            .ball_speed_mph;
        assert!((shown - chip.input.ball_speed_mph).abs() < 1e-9);
        assert!(fm.chase_camera(chip_expires + 1.0).is_none());

        fm.update(chip_expires + 2.0);
        let shown = fm
            .last_telemetry
            .as_ref()
            .expect("telemetry")
            .ball_speed_mph;
        assert!((shown - chip.input.ball_speed_mph).abs() < 1e-9);

        // A newer shot takes both back.
        fm.launch(&driver, chip_expires + 3.0, &model, &FLAT);
        fm.update(chip_expires + 3.5);
        let shown = fm
            .last_telemetry
            .as_ref()
            .expect("telemetry")
            .ball_speed_mph;
        assert!((shown - driver.input.ball_speed_mph).abs() < 1e-9);
        assert!(fm.chase_camera(chip_expires + 3.5).is_some());
    }

    #[test]
    fn a_shot_is_not_launched_after_now() {
        let mut fm = FlightManager::new();
        let shot = ArrivedShot {
            shot: received(ShotInput::seven_iron()),
            arrival: 5.0,
        };
        fm.take_in(&shot, 4.99, &Tour::SEA_LEVEL, &FLAT);
        assert_eq!(fm.flights[0].launch_time, 4.99);
    }

    #[test]
    fn shot_during_rollout_is_launched() {
        let mut fm = FlightManager::new();
        let model = Tour::SEA_LEVEL;
        let ground = Range::new().ground;
        fm.launch(&received(ShotInput::driver()), 0.0, &model, &ground);

        // The drive is on the ground but still rolling.
        let now = fm.flights[0].result.flight.flight_time + 1.0;
        assert!(fm.is_animating(now));
        let second = ShotInput::seven_iron();
        fm.launch(&received(second), now, &model, &ground);
        assert_eq!(fm.flights.len(), 2, "second shot is launched, not dropped");

        // Both balls are drawn; the HUD and the chase camera follow the newest.
        let t = now + 0.5;
        assert_eq!(fm.update(t).len(), 2);
        let _ = fm.build_hud(true, &HashMap::new(), true, 1920.0, 1080.0);
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
        let f = animate(ShotInput::driver(), &Range::new().ground);
        let r = &f.result;
        assert!(
            f.timeline.windows(2).all(|w| w[1].time > w[0].time),
            "times strictly increase"
        );
        assert_eq!(f.timeline[0].pos, dimple::Vec3::ZERO);
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
        assert_eq!(last.pos, r.ground.rest.pos);
        assert_eq!(f.total_time, last.time);
        assert!(f.total_time > r.flight.flight_time + 1.0, "ball runs out");
    }

    /// The first straight drive (11°, 2500 rpm) from 160 to 163 mph in
    /// 0.05 mph steps that drops into the hole's cup.
    fn holed_drive(ground: &Ground) -> AnimatedFlight {
        (0..=60)
            .map(|i| ShotInput {
                ball_speed_mph: 160.0 + 0.05 * f64::from(i),
                launch_angle_deg: 11.0,
                launch_azimuth_deg: 0.0,
                backspin_rpm: 2500.0,
                sidespin_rpm: 0.0,
            })
            .map(|input| animate(input, ground))
            .find(AnimatedFlight::holed)
            .expect("a holed drive")
    }

    #[test]
    fn a_holed_shot_ends_in_the_cup() {
        let range = Range::hole();
        let hole = range.ground.hole().expect("a hole").clone();
        let f = holed_drive(&range.ground);
        let last = f.timeline.last().expect("timeline");
        assert_eq!(last.time, f.total_time);
        assert!(f.total_time < f.result.ground.rest.time);
        let at = physics_to_scene(last.pos);
        assert!(hole.to_cup(at.x, at.z) < 1e-9);
        assert!(at.y < hole.height(at.x, at.z));

        let moving = f
            .telemetry(f.result.flight.flight_time + 0.5)
            .hole
            .expect("hole rows");
        assert_eq!(moving.lie, None);
        assert!(moving.to_pin_yards > 0.0);
        let stopped = f.telemetry(f.total_time + 0.1).hole.expect("hole rows");
        assert_eq!(stopped.lie, Some("HOLED"));
        assert_eq!(stopped.to_pin_yards, 0.0);
    }

    #[test]
    fn a_shot_that_misses_reads_its_lie() {
        let range = Range::hole();
        let f = animate(ShotInput::wedge(), &range.ground);
        assert!(!f.holed());
        let stopped = f.telemetry(f.total_time + 0.1).hole.expect("hole rows");
        assert_eq!(stopped.lie, Some("FAIRWAY"));
        assert!(
            animate(ShotInput::wedge(), &FLAT)
                .telemetry(0.0)
                .hole
                .is_none()
        );
    }

    #[test]
    fn a_shot_into_the_water_stops_there() {
        let range = Range::hole();
        let f = animate(
            ShotInput {
                ball_speed_mph: 120.0,
                launch_angle_deg: 13.0,
                launch_azimuth_deg: -9.0,
                backspin_rpm: 3000.0,
                sidespin_rpm: 0.0,
            },
            &range.ground,
        );
        assert_eq!(f.stop, Some(Stop::Water));
        assert!(f.total_time < f.result.ground.rest.time);
        assert_eq!(f.timeline.last().map(|p| p.pos), Some(f.rest_pos));
        let stopped = f.telemetry(f.total_time + 0.1).hole.expect("hole rows");
        assert_eq!(stopped.lie, Some("WATER"));
        assert!(stopped.to_pin_yards > 50.0);
    }

    #[test]
    fn a_shot_into_the_trees_drops_and_reads_its_lie() {
        let range = Range::hole();
        let f = animate(
            ShotInput {
                ball_speed_mph: 140.0,
                launch_angle_deg: 13.0,
                launch_azimuth_deg: 9.0,
                backspin_rpm: 3000.0,
                sidespin_rpm: 0.0,
            },
            &range.ground,
        );
        assert_eq!(f.stop, Some(Stop::Tree));
        assert!(f.timeline.windows(2).all(|p| p[1].time > p[0].time));
        assert_eq!(f.timeline.last().map(|p| p.pos), Some(f.rest_pos));
        let stopped = f.telemetry(f.total_time + 0.1).hole.expect("hole rows");
        assert_eq!(stopped.lie, Some("ROUGH"));
    }

    #[test]
    fn unsettled_ball_stops_animating() {
        let input = ShotInput::driver();
        let mut result =
            ground::simulate_shot(&input, &Tour::SEA_LEVEL, &Ground::Flat { green: None });
        assert_eq!(result.ground.termination, Termination::Rest);
        // A ground phase that runs into dimple's time limit.
        let mut late = *result.ground.points.last().expect("ground points");
        late.time += 100.0;
        late.pos.x += 50.0;
        result.ground.points.push(late);
        result.ground.termination = Termination::MaxTime;
        result.ground.rest.pos = late.pos;
        result.total_m = late.pos.x.hypot(late.pos.z);

        let f = AnimatedFlight::from_result(&received(input), result, 0.0, None, None);
        let end = f.result.flight.flight_time + MAX_UNSETTLED_GROUND_TIME_S;
        // The log reports the distance the HUD shows, not dimple's.
        let shown = f.telemetry(end + 1.0).total_yards;
        assert!((f.shown_total_yards() - shown).abs() < 1e-9);
        assert!(f.shown_total_yards() < meters_to_yards(f.result.total_m) - 1.0);
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
        let f = animate(ShotInput::driver(), &FLAT);
        assert_eq!(f.result.ground.termination, Termination::Rest);
        assert_eq!(f.rest_pos, f.result.ground.rest.pos);
        assert!(f.total_time > f.result.flight.flight_time + 1.0);
    }

    #[test]
    fn chase_camera_follows_a_hook_then_lets_go() {
        let input = ShotInput {
            launch_azimuth_deg: -2.0,
            sidespin_rpm: -900.0,
            ..ShotInput::driver()
        };
        let mut f = animate(input, &Range::new().ground);
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

    /// A 4.5 mph putt started 1.2° right, as `ReceivedShot::from_frp` makes
    /// it from a putting mat's launch speed and azimuth.
    fn putt(ground: &Ground) -> AnimatedFlight {
        let shot = ReceivedShot {
            backspin_measured: false,
            sidespin_measured: false,
            rolled: true,
            ..received(ShotInput {
                ball_speed_mph: 4.5,
                launch_angle_deg: 0.0,
                launch_azimuth_deg: 1.2,
                backspin_rpm: 0.0,
                sidespin_rpm: 0.0,
            })
        };
        AnimatedFlight::from_received(&shot, 0.0, &Tour::SEA_LEVEL, ground)
    }

    #[test]
    fn a_putt_rolls_along_the_ground_from_the_tee() {
        let f = putt(&Range::new().ground);
        assert_eq!(f.timeline[0].time, 0.0);
        assert_eq!(f.timeline[0].pos, dimple::Vec3::ZERO);
        assert!(f.timeline.iter().all(|p| p.pos.y == 0.0));
        assert_eq!(f.result.ground.termination, Termination::Rest);
        assert!(f.rest_pos.z > 0.0, "started right, finishes right");

        let rolling = f.telemetry(0.5);
        assert_eq!(rolling.apex_m, 0.0);
        assert_eq!(rolling.carry_yards, 0.0);
        assert_eq!(rolling.launch_angle_deg, None);
        assert_eq!(rolling.backspin_rpm, None);
        assert_eq!(rolling.ball_speed_mph, 4.5);
        assert_eq!(rolling.launch_azimuth_deg, 1.2);
        assert!(rolling.total_yards > 0.0);
        assert!(rolling.total_yards < f.shown_total_yards());
        assert_eq!(f.render_data(0.5, 1).expect("launched").ball_pos.y, 0.0);

        let rest = f.telemetry(f.total_time + 1.0);
        assert_eq!(rest.total_yards, f.shown_total_yards());
        assert_eq!(rest.elapsed_s, f.total_time);
        assert_eq!(rest.launch_angle_deg, None);
    }

    #[test]
    fn a_putt_rolls_on_green_whatever_the_range_green() {
        let green = Some(cf_scene::green::GreenConfig {
            distance_m: 100.0,
            diameter_m: 20.0,
        });
        assert_eq!(
            putt(&Ground::Flat { green: None }).rest_pos,
            putt(&Ground::Flat { green }).rest_pos
        );
    }

    #[test]
    fn the_chase_camera_keeps_a_rolling_putt_in_view() {
        let mut f = putt(&Range::new().ground);
        // The inset is the right fifth of a 16:9 frame.
        let aspect = 0.2 * 16.0 / 9.0;
        let end = f.total_time + CHASE_LINGER_TIME;
        let mut t = 0.0;
        while t < end {
            let cam = f.chase_camera(t).expect("followed until the linger ends");
            let ball = f.render_data(t, 1).expect("launched").ball_pos;
            let clip = cam.projection_matrix(aspect) * cam.view_matrix() * ball.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!(
                clip.w > 0.0 && ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0,
                "t = {t:.2}: ball at {ndc:?}"
            );
            t += 0.05;
        }
        assert!(f.chase_camera(end + 0.01).is_none());
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
        let f = animate(input, &FLAT);
        assert_eq!(f.timeline.len(), 1);
        assert_eq!(f.total_time, 0.0);
        assert!(!f.is_animating(1.0));
    }

    #[test]
    fn interpolates_through_the_ground_phase() {
        let f = animate(ShotInput::seven_iron(), &FLAT);
        let t = f.result.flight.flight_time + 0.5;
        let pos = interpolate_trajectory(&f.timeline, t).expect("inside timeline");
        assert!(pos.x > f.result.flight.carry_m * 0.99);
        assert!(pos.x < f.result.ground.rest.pos.x + 1e-9);
    }

    #[test]
    fn telemetry_at_rest_reports_dimple_results() {
        let input = ShotInput {
            launch_azimuth_deg: 1.0,
            sidespin_rpm: 600.0,
            ..ShotInput::driver()
        };
        let f = animate(input, &Range::new().ground);
        let t = f.telemetry(f.total_time + 1.0);
        let r = &f.result;
        assert!((t.carry_yards - meters_to_yards(r.flight.carry_m)).abs() < 1e-9);
        assert!((t.total_yards - meters_to_yards(r.total_m)).abs() < 1e-9);
        let lateral = meters_to_yards(r.ground.rest.pos.z);
        assert!((t.lateral_yards - lateral).abs() < 1e-9);
        assert!(t.lateral_yards > 0.0, "slice finishes right");
        assert!(
            (t.elapsed_s - f.total_time).abs() < 1e-9,
            "clock runs to rest"
        );
    }

    #[test]
    fn the_apex_row_holds_the_highest_point_reached() {
        let f = animate(ShotInput::driver(), &FLAT);
        let flight_time = f.result.flight.flight_time;
        let apex = f.result.flight.apex_m;
        let mut shown = 0.0_f64;
        let mut t = 0.0;
        while t < flight_time {
            let now = f.telemetry(t).apex_m;
            assert!(now >= shown, "apex fell from {shown} to {now} at {t} s");
            shown = now;
            t += 0.05;
        }
        // Past the apex the row stays within a sample of it, not at the
        // ball's lower current height.
        assert!(shown > apex - 0.5, "{shown} vs apex {apex}");
        assert!(f.telemetry(flight_time * 0.95).apex_m > apex - 0.5);
        assert!((f.telemetry(flight_time + 0.1).apex_m - apex).abs() < 1e-9);
    }

    #[test]
    fn slice_is_drawn_right_of_the_line() {
        let input = ShotInput {
            launch_azimuth_deg: 1.0,
            sidespin_rpm: 600.0,
            ..ShotInput::driver()
        };
        let f = animate(input, &Range::new().ground);
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
