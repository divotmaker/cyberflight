//! Shared flight animation logic used by both windowed and streaming modes.

use std::collections::HashMap;

use glam::{DVec3, Vec3};

use cf_math::aero::BallModel;
use cf_math::bounce::BounceSurface;
use cf_math::environment::Environment;
use cf_math::rollout::RolloutSurface;
use cf_math::trajectory::{simulate_shot, ShotInput, ShotResult};
use cf_render::render::FlightRenderData;
use cf_scene::camera::Camera;
use cf_scene::hud::{build_hud, ShotTelemetry, UnitSystem};
use cf_scene::shot::{ClubDelivery, ReceivedShot};
use cf_scene::surface;
use cf_scene::trail::TrailPoint;

use flightrelay::{FrpClient, FrpEvent, FrpMessage, ShotAggregator};

/// FRP spec version we support.
const FRP_VERSION: &str = flightrelay::SPEC_VERSION;
/// How long the chase camera lingers after the ball stops (seconds).
pub const CHASE_LINGER_TIME: f64 = 5.0;
/// Time constant for chase camera smoothing (seconds).
pub const CHASE_CAM_SMOOTHING_TAU: f64 = 0.3;
/// Default subsample stride: every Nth trajectory sample is used as a trail point.
pub const DEFAULT_TRAIL_SUBSAMPLE: usize = 1;
/// Trail tail as a fraction of elapsed flight time.
pub const TRAIL_TAIL_FRACTION: f64 = 1.2;
/// Gravity for parabolic bounce arcs.
pub const G: f64 = cf_math::units::G;
/// Default smash factor for estimating club speed.
pub const DEFAULT_SMASH_FACTOR: f64 = 1.40;

/// A point in the unified shot timeline (flight + bounce arcs + rollout).
#[derive(Debug, Clone, Copy)]
pub struct TimelinePoint {
    pub time: f64,
    pub pos: DVec3,
}

/// Map trajectory coordinates (x=forward, y=up, z=left) to
/// scene coordinates (+Z=downrange, +Y=up, +X=right).
pub fn traj_to_scene(p: DVec3) -> Vec3 {
    Vec3::new(-p.z as f32, p.y as f32, p.x as f32)
}

/// Interpolate position in the unified timeline.
pub fn interpolate_timeline(points: &[TimelinePoint], t: f64) -> Option<DVec3> {
    if points.is_empty() || t < points[0].time {
        return None;
    }
    let last = points.last()?;
    if t >= last.time {
        return Some(last.pos);
    }
    let idx = points.partition_point(|p| p.time <= t);
    if idx == 0 {
        return Some(points[0].pos);
    }
    let a = &points[idx - 1];
    let b = &points[idx];
    let frac = (t - a.time) / (b.time - a.time);
    Some(a.pos.lerp(b.pos, frac))
}

/// Build a unified timeline from a ShotResult.
pub fn build_timeline(result: &ShotResult) -> Vec<TimelinePoint> {
    let mut timeline =
        Vec::with_capacity(result.flight.points.len() + result.rollout.points.len() + 200);

    for p in &result.flight.points {
        timeline.push(TimelinePoint {
            time: p.time,
            pos: p.pos,
        });
    }

    let mut t_offset = result.flight.flight_time;

    for (i, bounce) in result.bounces.iter().enumerate() {
        let is_last = i + 1 == result.bounces.len();
        if is_last {
            break;
        }
        let vel = bounce.vel_out;
        if vel.y <= 0.0 {
            continue;
        }
        let t_arc = 2.0 * vel.y / G;
        let num_samples = ((t_arc / 0.01) as usize).clamp(2, 200);

        for s in 1..=num_samples {
            let dt = t_arc * s as f64 / num_samples as f64;
            let pos = DVec3::new(
                bounce.pos.x + vel.x * dt,
                (bounce.pos.y + vel.y * dt - 0.5 * G * dt * dt).max(0.0),
                bounce.pos.z + vel.z * dt,
            );
            timeline.push(TimelinePoint {
                time: t_offset + dt,
                pos,
            });
        }
        t_offset += t_arc;
    }

    for rp in &result.rollout.points {
        timeline.push(TimelinePoint {
            time: t_offset + rp.time,
            pos: rp.pos,
        });
    }

    timeline
}

/// An in-progress or recently completed shot animation.
pub struct AnimatedFlight {
    pub input: ShotInput,
    pub result: ShotResult,
    pub timeline: Vec<TimelinePoint>,
    pub total_time: f64,
    pub launch_time: f64,
    pub club: ClubDelivery,
    pub lm_carry_yards: Option<f64>,
    pub landed: bool,
    pub prev_chase: Option<(Camera, f64)>,
}

impl AnimatedFlight {
    pub fn from_received(
        shot: &ReceivedShot,
        launch_time: f64,
        ball: &BallModel,
        env: &Environment,
        targets: &[cf_scene::target::Target],
    ) -> Self {
        let input = &shot.input;

        let flight = cf_math::trajectory::simulate_flight(input, ball, env);
        let landing_x = flight.points.last().map_or(0.0, |p| p.pos.x);
        let landing_z = flight.points.last().map_or(0.0, |p| p.pos.z);

        let (bounce_surface, rollout_surface) =
            match surface::surface_at(landing_x, landing_z, targets) {
                surface::Surface::Green => (BounceSurface::GREEN, RolloutSurface::GREEN),
                surface::Surface::Fairway => (BounceSurface::FAIRWAY, RolloutSurface::FAIRWAY),
            };

        let result = simulate_shot(input, ball, env, &bounce_surface, &rollout_surface);
        let timeline = build_timeline(&result);
        let total_time = timeline.last().map_or(0.0, |p| p.time);

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
            input: *input,
            result,
            timeline,
            total_time,
            launch_time,
            club: shot.club.clone(),
            lm_carry_yards: shot.lm_carry_yards,
            landed: false,
            prev_chase: None,
        }
    }

    pub fn telemetry(&self, now: f64) -> ShotTelemetry {
        let shot_t = (now - self.launch_time).max(0.0);
        let t_clamped = shot_t.min(self.total_time);

        let pos = interpolate_timeline(&self.timeline, t_clamped).unwrap_or(DVec3::ZERO);

        let total_m = (pos.x * pos.x + pos.z * pos.z).sqrt();
        let downrange_m = pos.x;

        let carry_yards = if shot_t < self.result.flight.flight_time {
            cf_math::units::meters_to_yards(total_m)
        } else {
            self.result.flight.carry_yards
        };

        let club_speed_mph = self
            .club
            .club_speed_mph
            .unwrap_or(self.input.ball_speed_mph / DEFAULT_SMASH_FACTOR);

        ShotTelemetry {
            club_speed_mph,
            ball_speed_mph: self.input.ball_speed_mph,
            smash_factor: self.club.smash_factor,
            launch_angle_deg: self.input.launch_angle_deg,
            launch_azimuth_deg: self.input.launch_azimuth_deg,
            backspin_rpm: self.input.backspin_rpm,
            sidespin_rpm: self.input.sidespin_rpm,
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
            total_yards: cf_math::units::meters_to_yards(total_m),
            downrange_yards: cf_math::units::meters_to_yards(downrange_m),
            lateral_m: -pos.z,
            flight_time_s: self.result.flight.flight_time,
            elapsed_s: shot_t.min(self.result.flight.flight_time),
            in_flight: self.is_animating(now),
        }
    }

    pub fn render_data(&self, now: f64, trail_subsample: usize) -> Option<FlightRenderData> {
        let shot_t = now - self.launch_time;
        if shot_t < 0.0 {
            return None;
        }

        let t_clamped = shot_t.min(self.total_time);

        let ball_pos = interpolate_timeline(&self.timeline, t_clamped).map(traj_to_scene)?;

        let tail_duration = t_clamped * TRAIL_TAIL_FRACTION;
        let tail_start = (t_clamped - tail_duration).max(0.0);
        let mut trail_points: Vec<TrailPoint> = self
            .timeline
            .iter()
            .filter(|p| p.time >= tail_start && p.time <= t_clamped)
            .enumerate()
            .filter(|(i, _)| i % trail_subsample == 0)
            .map(|(_, p)| TrailPoint {
                position: traj_to_scene(p.pos),
                time: p.time,
            })
            .collect();

        if trail_points
            .last()
            .is_none_or(|last| (last.position - ball_pos).length() > 0.01)
        {
            trail_points.push(TrailPoint {
                position: ball_pos,
                time: t_clamped,
            });
        }

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

        let final_pos = traj_to_scene(self.result.final_pos);

        let shot_dir = if final_pos.length_squared() > 0.01 {
            final_pos.normalize()
        } else {
            Vec3::Z
        };

        let ideal = if shot_t < self.total_time {
            let ball_pos = interpolate_timeline(&self.timeline, shot_t).map(traj_to_scene)?;

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
            let velocity = Vec3::new(vel.z as f32, vel.y as f32, vel.x as f32);

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
        // camera motion from bounces and slow rollout direction changes.
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

    pub fn is_expired(&self, now: f64) -> bool {
        let shot_t = now - self.launch_time;
        shot_t > self.total_time + CHASE_LINGER_TIME
    }
}

/// Shared FRP connection state used by both windowed and streaming modes.
pub struct FrpState {
    pub client: Option<FrpClient>,
    pub shots: ShotAggregator,
    pub lm_states: HashMap<String, bool>,
    pub connected: bool,
    pub last_reconnect_attempt: f64,
}

impl FrpState {
    pub fn new() -> Self {
        let (client, connected) = Self::try_connect();
        Self {
            client,
            shots: ShotAggregator::new(),
            lm_states: HashMap::new(),
            connected,
            last_reconnect_attempt: 0.0,
        }
    }

    pub fn try_connect() -> (Option<FrpClient>, bool) {
        eprintln!("Connecting to FRP at {}...", flightrelay::DEFAULT_URL);
        match FrpClient::connect(flightrelay::DEFAULT_URL, "cyberflight", &[FRP_VERSION]) {
            Ok(client) => {
                eprintln!("Connected (FRP {})", client.version());
                if let Err(e) = client.set_nonblocking(true) {
                    eprintln!("set_nonblocking failed: {e}");
                    return (None, false);
                }
                (Some(client), true)
            }
            Err(e) => {
                eprintln!("FRP not available (expected {FRP_VERSION}): {e}");
                (None, false)
            }
        }
    }

    /// Poll FRP for new shots. Returns any completed ReceivedShot.
    pub fn poll(&mut self, now: f64, animating: bool) -> Option<ReceivedShot> {
        // Reconnect if disconnected (try every 5 seconds).
        if self.client.is_none() && now - self.last_reconnect_attempt >= 5.0 {
            self.last_reconnect_attempt = now;
            let (client, connected) = Self::try_connect();
            self.client = client;
            self.connected = connected;
        }

        let mut result = None;
        let mut disconnected = false;

        if let Some(client) = &mut self.client {
            loop {
                match client.try_recv() {
                    Ok(Some(msg)) => {
                        // Update launch monitor ready state from device telemetry.
                        if let FrpMessage::Envelope(env) = &msg {
                            if let FrpEvent::DeviceTelemetry {
                                telemetry: Some(t), ..
                            } = &env.event
                            {
                                if let Some(ready) = t.get(flightrelay::types::telemetry::READY) {
                                    self.lm_states.insert(env.device.clone(), ready == "true");
                                }
                            }
                        }
                        if !animating {
                            if let Some(completed) = self.shots.feed(&msg) {
                                result = ReceivedShot::from_frp(&completed);
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(flightrelay::FrpError::Json(_)) => {
                        // some messages may not be parsable due to extension
                    }

                    Err(e) => {
                        eprintln!("FRP disconnected: {e}");
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        if disconnected {
            self.connected = false;
            self.client = None;
            self.lm_states.clear();
        }

        result
    }
}

/// Shared flight manager: handles animation, telemetry, and camera for a list of flights.
pub struct FlightManager {
    pub flights: Vec<AnimatedFlight>,
    pub last_telemetry: Option<ShotTelemetry>,
    pub trail_subsample: usize,
}

impl FlightManager {
    pub fn new() -> Self {
        Self {
            flights: Vec::new(),
            last_telemetry: None,
            trail_subsample: DEFAULT_TRAIL_SUBSAMPLE,
        }
    }

    /// Update flight state: mark landed, remove expired, build render data.
    pub fn update(&mut self, now: f64) -> Vec<FlightRenderData> {
        // Mark completed shots
        for flight in &mut self.flights {
            if !flight.landed && !flight.is_animating(now) && now > flight.launch_time {
                flight.landed = true;
                eprintln!(
                    "  Stopped: {:.1} total yds ({:.1} carry + {:.1} rollout)",
                    flight.result.total_yards,
                    flight.result.flight.carry_yards,
                    flight.result.total_yards - flight.result.flight.carry_yards,
                );
            }
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

    /// Build HUD geometry for the current frame.
    /// Compute the chase camera for the most recent active flight (if any).
    pub fn chase_camera(&mut self, now: f64) -> Option<Camera> {
        self.flights
            .iter_mut()
            .rev()
            .find_map(|f| f.chase_camera(now))
    }

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
