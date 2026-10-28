//! Shared flight animation logic used by both windowed and streaming modes.

use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;

use glam::{DVec3, Vec3};

use cf_render::render::FlightRenderData;
use cf_scene::camera::Camera;
use cf_scene::frame::physics_to_scene;
use cf_scene::ground;
use cf_scene::hud::{ShotTelemetry, UnitSystem, build_hud};
use cf_scene::shot::{ClubDelivery, ReceivedShot};
use cf_scene::trail::TrailPoint;
use dimple::aero::BallModel;
use dimple::environment::Environment;
use dimple::ground::Termination;
use dimple::trajectory::{ShotInput, ShotResult};

use flightrelay::{
    EndpointConfig, FrpConnection, FrpEndpoint, FrpEvent, FrpMessage, Role, ShotAggregator,
    Transport,
};

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
/// Longest a ball that dimple did not bring to rest keeps moving on the
/// ground after landing (seconds). Shots that come to rest animate to the end.
pub const MAX_UNSETTLED_GROUND_TIME_S: f64 = 8.0;
/// Default smash factor for estimating club speed.
pub const DEFAULT_SMASH_FACTOR: f64 = 1.40;

/// A point in the unified shot timeline (flight, then hops, slide and roll),
/// in dimple's frame.
#[derive(Debug, Clone, Copy)]
pub struct TimelinePoint {
    pub time: f64,
    pub pos: DVec3,
}

/// A dimple-frame vector as a scene-frame render position.
fn to_scene(p: DVec3) -> Vec3 {
    physics_to_scene(p).as_vec3()
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

/// Build the unified timeline from a ShotResult: the flight points, then the
/// ground points (hops, slide, roll). Both carry absolute shot times; a
/// ground point at or before the previous point's time (the landing, shared
/// by both) is skipped so times strictly increase.
pub fn build_timeline(result: &ShotResult) -> Vec<TimelinePoint> {
    let mut timeline = Vec::with_capacity(result.flight.points.len() + result.ground.points.len());
    let flight = result.flight.points.iter().map(|p| (p.time, p.pos));
    let ground = result.ground.points.iter().map(|p| (p.time, p.pos));
    for (time, pos) in flight.chain(ground) {
        if timeline
            .last()
            .is_none_or(|last: &TimelinePoint| time > last.time)
        {
            timeline.push(TimelinePoint { time, pos });
        }
    }
    timeline
}

/// An in-progress or recently completed shot animation.
pub struct AnimatedFlight {
    pub input: ShotInput,
    pub result: ShotResult,
    pub timeline: Vec<TimelinePoint>,
    pub total_time: f64,
    /// Where the animation ends: dimple's rest position, or the last shown
    /// position when the ball did not settle.
    pub rest_pos: DVec3,
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
        let result = ground::simulate_shot(&shot.input, ball, env, targets);
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
                .and_then(|_| interpolate_timeline(&timeline, end))
            {
                timeline.retain(|p| p.time < end);
                timeline.push(TimelinePoint {
                    time: end,
                    pos: cut,
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
            interpolate_timeline(&self.timeline, t_clamped).unwrap_or(DVec3::ZERO)
        };

        let total_m = pos.x.hypot(pos.z);
        let downrange_m = pos.x;

        let carry_yards = if shot_t < self.result.flight.flight_time {
            dimple::units::meters_to_yards(total_m)
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
            total_yards: dimple::units::meters_to_yards(total_m),
            downrange_yards: dimple::units::meters_to_yards(downrange_m),
            lateral_m: pos.z,
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

        let ball_pos = interpolate_timeline(&self.timeline, t_clamped).map(to_scene)?;

        let tail_duration = t_clamped * TRAIL_TAIL_FRACTION;
        let tail_start = (t_clamped - tail_duration).max(0.0);
        let mut trail_points: Vec<TrailPoint> = self
            .timeline
            .iter()
            .filter(|p| p.time >= tail_start && p.time <= t_clamped)
            .enumerate()
            .filter(|(i, _)| i % trail_subsample == 0)
            .map(|(_, p)| TrailPoint {
                position: to_scene(p.pos),
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

        let final_pos = to_scene(self.rest_pos);

        let shot_dir = if final_pos.length_squared() > 0.01 {
            final_pos.normalize()
        } else {
            Vec3::Z
        };

        let ideal = if shot_t < self.total_time {
            let ball_pos = interpolate_timeline(&self.timeline, shot_t).map(to_scene)?;

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

    pub fn is_expired(&self, now: f64) -> bool {
        let shot_t = now - self.launch_time;
        shot_t > self.total_time + CHASE_LINGER_TIME
    }
}

/// How often a lost or refused FRP connection is retried (seconds).
pub const RECONNECT_INTERVAL_S: f64 = 5.0;

/// The FRP device URL: the value after `--frp` in `args`, or
/// [`flightrelay::DEFAULT_URL`].
pub fn frp_url(args: &[String]) -> String {
    args.iter()
        .position(|a| a == "--frp")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| flightrelay::DEFAULT_URL.to_owned())
}

/// Open the FRP endpoint cyberflight dials at `url`.
///
/// Cyberflight is an FRP *controller*: it receives shot data and drives the
/// device. Opening a dialing endpoint performs no I/O, so this only fails on a
/// malformed configuration.
///
/// # Errors
///
/// Returns an error if the endpoint cannot be opened.
pub fn frp_endpoint(url: &str) -> Result<FrpEndpoint, flightrelay::FrpError> {
    FrpEndpoint::open(
        EndpointConfig::new(Role::Controller, Transport::connect(url))
            .with_name("cyberflight")
            .with_versions(&[FRP_VERSION]),
    )
}

/// The endpoint back from a dialing thread, with the connection it made.
type DialResult = (FrpEndpoint, Result<FrpConnection, flightrelay::FrpError>);

/// Dial the device and complete the FRP handshake on a background thread,
/// so a slow or silent device cannot stall the render loop. The endpoint
/// and the outcome come back on the returned channel.
fn spawn_dial(mut endpoint: FrpEndpoint, url: &str) -> mpsc::Receiver<DialResult> {
    eprintln!("Connecting to FRP at {url}...");
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = endpoint.establish().and_then(|conn| {
            conn.set_nonblocking(true)?;
            Ok(conn)
        });
        // The receiver is gone only when the app is shutting down.
        let _ = tx.send((endpoint, result));
    });
    rx
}

/// Turns FRP messages into shots: tracks launch monitor ready states and
/// aggregates each shot's lifecycle events.
///
/// Every message is fed, whatever is on screen, so a shot that arrives while
/// an earlier ball is still flying or rolling is never lost.
#[derive(Debug, Default)]
pub struct ShotIntake {
    shots: ShotAggregator,
    /// Last known ready state per launch monitor device ID.
    pub lm_states: HashMap<String, bool>,
}

impl ShotIntake {
    /// Process one FRP message. Returns a shot once its `shot_finished`
    /// arrives and it converts to a simulation input.
    pub fn ingest(&mut self, msg: &FrpMessage) -> Option<ReceivedShot> {
        if let FrpMessage::Envelope(env) = msg {
            if let FrpEvent::DeviceTelemetry {
                telemetry: Some(t), ..
            } = &env.event
            {
                if let Some(ready) = t.get(flightrelay::types::telemetry::READY) {
                    self.lm_states.insert(env.device.clone(), ready == "true");
                }
            }
        }
        let completed = self.shots.feed(msg)?;
        match ReceivedShot::from_frp(&completed) {
            Ok(shot) => {
                for clamp in &shot.clamped {
                    eprintln!(
                        "WARN: shot #{} from {}: {clamp}",
                        shot.shot_number, shot.device
                    );
                }
                Some(shot)
            }
            Err(reason) => {
                eprintln!(
                    "WARN: shot #{} from {} not shown: {reason}",
                    completed.key.shot_number, completed.device
                );
                None
            }
        }
    }
}

/// Shared FRP connection state used by both windowed and streaming modes.
///
/// Connects in the background and reconnects every
/// [`RECONNECT_INTERVAL_S`] while disconnected.
pub struct FrpState {
    url: String,
    /// Outlives the connections it produces. `None` while a dialing thread
    /// holds it.
    endpoint: Option<FrpEndpoint>,
    /// Outcome of the dial in progress, if any.
    dial: Option<mpsc::Receiver<DialResult>>,
    conn: Option<FrpConnection>,
    pub intake: ShotIntake,
    pub connected: bool,
    last_dial: f64,
}

impl FrpState {
    /// FRP state for the device at `url`. The first [`poll`](Self::poll)
    /// starts connecting.
    ///
    /// # Errors
    ///
    /// Returns an error if the FRP endpoint cannot be opened.
    pub fn new(url: &str) -> Result<Self, flightrelay::FrpError> {
        Ok(Self {
            url: url.to_owned(),
            endpoint: Some(frp_endpoint(url)?),
            dial: None,
            conn: None,
            intake: ShotIntake::default(),
            connected: false,
            last_dial: f64::NEG_INFINITY,
        })
    }

    /// Pick up a finished dial, or start one when disconnected and the
    /// retry interval has passed. Never blocks.
    fn maintain_connection(&mut self, now: f64) {
        if let Some(dial) = &self.dial {
            match dial.try_recv() {
                Ok((endpoint, result)) => {
                    self.endpoint = Some(endpoint);
                    self.dial = None;
                    match result {
                        Ok(conn) => {
                            eprintln!("Connected (FRP {})", conn.version());
                            self.conn = Some(conn);
                            self.connected = true;
                        }
                        Err(e) => {
                            eprintln!(
                                "FRP not available at {} (expected {FRP_VERSION}): {e}",
                                self.url
                            );
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    // The dialing thread died with the endpoint; open a new one.
                    self.dial = None;
                    match frp_endpoint(&self.url) {
                        Ok(endpoint) => self.endpoint = Some(endpoint),
                        Err(e) => eprintln!("WARN: cannot reopen FRP endpoint: {e}"),
                    }
                }
            }
        }

        if self.conn.is_none() && now - self.last_dial >= RECONNECT_INTERVAL_S {
            if let Some(endpoint) = self.endpoint.take() {
                self.last_dial = now;
                self.dial = Some(spawn_dial(endpoint, &self.url));
            }
        }
    }

    /// Drain all pending FRP messages. Returns the shots completed by them,
    /// oldest first.
    pub fn poll(&mut self, now: f64) -> Vec<ReceivedShot> {
        self.maintain_connection(now);

        let mut shots = Vec::new();
        let mut disconnected = false;

        if let Some(conn) = &mut self.conn {
            loop {
                match conn.try_recv() {
                    Ok(Some(msg)) => shots.extend(self.intake.ingest(&msg)),
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
            self.conn = None;
            self.intake.lm_states.clear();
        }

        shots
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

    /// Simulate `shot` and start animating it at `now`.
    ///
    /// Earlier balls keep animating to rest; the newest ball drives the chase
    /// camera and the HUD.
    pub fn launch(
        &mut self,
        shot: &ReceivedShot,
        now: f64,
        env: &Environment,
        targets: &[cf_scene::target::Target],
    ) {
        if self.is_animating(now) {
            eprintln!(
                "Shot #{} from {} launched while an earlier ball is still moving",
                shot.shot_number, shot.device
            );
        }
        self.flights.push(AnimatedFlight::from_received(
            shot,
            now,
            &BallModel::TOUR,
            env,
            targets,
        ));
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
    use cf_scene::shot::ClubDelivery;
    use cf_scene::target::{Target, default_targets};

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

    fn animate(input: ShotInput, targets: &[Target]) -> AnimatedFlight {
        AnimatedFlight::from_received(
            &received(input),
            0.0,
            &BallModel::TOUR,
            &Environment::SEA_LEVEL,
            targets,
        )
    }

    fn envelope(event: FrpEvent) -> FrpMessage {
        FrpMessage::Envelope(flightrelay::FrpEnvelope::new("test.0", event))
    }

    /// The lifecycle messages of shot `n`: trigger, ball flight, finished.
    fn shot_messages(n: u32, speed_mph: f64, elevation_deg: f64) -> Vec<FrpMessage> {
        let key = flightrelay::ShotKey {
            shot_id: format!("shot-{n}"),
            shot_number: n,
        };
        let ball = flightrelay::BallFlight {
            launch_speed: Some(flightrelay::Velocity::MilesPerHour(speed_mph)),
            launch_elevation: Some(elevation_deg),
            launch_azimuth: Some(0.0),
            backspin_rpm: Some(3000),
            sidespin_rpm: Some(0),
            ..flightrelay::BallFlight::default()
        };
        vec![
            envelope(FrpEvent::shot_trigger(key.clone())),
            envelope(FrpEvent::ball_flight(key.clone(), ball)),
            envelope(FrpEvent::ShotFinished { key }),
        ]
    }

    /// Feed `msgs` to `intake`, launching every completed shot at `now`.
    fn deliver(intake: &mut ShotIntake, fm: &mut FlightManager, msgs: &[FrpMessage], now: f64) {
        for msg in msgs {
            if let Some(shot) = intake.ingest(msg) {
                fm.launch(&shot, now, &Environment::SEA_LEVEL, &default_targets());
            }
        }
    }

    #[test]
    fn frp_url_comes_from_the_flag() {
        let args = |a: &[&str]| a.iter().map(|&s| s.to_owned()).collect::<Vec<_>>();
        assert_eq!(frp_url(&args(&["cyberflight"])), flightrelay::DEFAULT_URL);
        assert_eq!(
            frp_url(&args(&[
                "cyberflight",
                "--stream",
                "--frp",
                "ws://pc:5880/frp"
            ])),
            "ws://pc:5880/frp"
        );
        assert_eq!(
            frp_url(&args(&["cyberflight", "--frp"])),
            flightrelay::DEFAULT_URL
        );
    }

    /// Poll `state` until `done` or a few seconds pass; returns the shots.
    fn poll_until(
        state: &mut FrpState,
        done: impl Fn(&FrpState, &[ReceivedShot]) -> bool,
    ) -> Vec<ReceivedShot> {
        let start = std::time::Instant::now();
        let mut shots = Vec::new();
        while !done(state, &shots) && start.elapsed().as_secs() < 5 {
            shots.extend(state.poll(0.0));
            thread::sleep(std::time::Duration::from_millis(5));
        }
        shots
    }

    #[test]
    fn a_silent_device_does_not_block_poll() {
        // Accepts TCP but never answers the WebSocket upgrade.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!("ws://{}/frp", listener.local_addr().expect("addr"));
        let mut state = FrpState::new(&url).expect("endpoint");
        let start = std::time::Instant::now();
        assert!(state.poll(0.0).is_empty());
        assert!(state.poll(1.0).is_empty());
        assert!(start.elapsed().as_millis() < 500, "poll returned at once");
        assert!(!state.connected);
    }

    #[test]
    fn connects_in_the_background_and_receives_shots() {
        let mut device = FrpEndpoint::open(EndpointConfig::new(
            Role::Device,
            Transport::listen("127.0.0.1:0"),
        ))
        .expect("device endpoint");
        let addr = device.local_addr().expect("addr").expect("bound");
        let server = thread::spawn(move || {
            let mut conn = device.establish().expect("handshake");
            for msg in shot_messages(7, 140.0, 14.0) {
                conn.send(&msg).expect("send");
            }
            conn
        });

        let mut state = FrpState::new(&format!("ws://{addr}/frp")).expect("endpoint");
        let shots = poll_until(&mut state, |s, shots| s.connected && !shots.is_empty());
        assert!(state.connected);
        let _conn = server.join().expect("device thread");
        assert_eq!(shots.len(), 1);
        assert_eq!(shots[0].shot_number, 7);
    }

    #[test]
    fn shot_during_rollout_is_launched() {
        let mut intake = ShotIntake::default();
        let mut fm = FlightManager::new();
        deliver(&mut intake, &mut fm, &shot_messages(1, 160.0, 11.0), 0.0);
        assert_eq!(fm.flights.len(), 1);

        // The drive is on the ground but still rolling.
        let now = fm.flights[0].result.flight.flight_time + 1.0;
        assert!(fm.is_animating(now));
        deliver(&mut intake, &mut fm, &shot_messages(2, 100.0, 24.0), now);
        assert_eq!(fm.flights.len(), 2, "second shot is launched, not dropped");

        // Both balls are drawn; the HUD and the chase camera follow the newest.
        let t = now + 0.5;
        assert_eq!(fm.update(t).len(), 2);
        let _ = fm.build_hud(t, true, &intake.lm_states, true, 1920.0, 1080.0);
        let shown = fm.last_telemetry.as_ref().expect("telemetry");
        assert!((shown.ball_speed_mph - 100.0).abs() < 1e-9);
        let chase = fm.chase_camera(t).expect("newest ball is followed");
        let newest = &fm.flights[1];
        assert_eq!(chase.target, to_scene(newest.rest_pos));
        let ball = newest.render_data(t, 1).expect("in flight").ball_pos;
        assert!(((chase.position - ball).length() - 125f32.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn shot_without_speed_is_not_launched() {
        let mut intake = ShotIntake::default();
        let mut fm = FlightManager::new();
        let mut msgs = shot_messages(1, 160.0, 11.0);
        if let FrpMessage::Envelope(env) = &mut msgs[1] {
            if let FrpEvent::BallFlight { ball, .. } = &mut env.event {
                ball.launch_speed = None;
            }
        }
        deliver(&mut intake, &mut fm, &msgs, 0.0);
        assert!(fm.flights.is_empty());
    }

    #[test]
    fn timeline_is_flight_then_ground() {
        let f = animate(ShotInput::driver(), &default_targets());
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
            ground::simulate_shot(&input, &BallModel::TOUR, &Environment::SEA_LEVEL, &[]);
        assert_eq!(result.ground.termination, Termination::Rest);
        // A ground phase that runs into dimple's time limit.
        let mut late = *result.ground.points.last().expect("ground points");
        late.time += 100.0;
        late.pos.x += 50.0;
        result.ground.points.push(late);
        result.ground.termination = Termination::MaxTime;

        let f = AnimatedFlight::from_result(&received(input), result, 0.0);
        let end = f.result.flight.flight_time + MAX_UNSETTLED_GROUND_TIME_S;
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
        let f = animate(ShotInput::driver(), &[]);
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
        let mut f = animate(input, &default_targets());
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

    #[test]
    fn motionless_shot_is_a_single_point() {
        let input = ShotInput {
            ball_speed_mph: 0.0,
            backspin_rpm: 0.0,
            sidespin_rpm: 0.0,
            ..ShotInput::driver()
        };
        let f = animate(input, &[]);
        assert_eq!(f.timeline.len(), 1);
        assert_eq!(f.total_time, 0.0);
        assert!(!f.is_animating(1.0));
    }

    #[test]
    fn interpolates_through_the_ground_phase() {
        let f = animate(ShotInput::seven_iron(), &[]);
        let t = f.result.flight.flight_time + 0.5;
        let pos = interpolate_timeline(&f.timeline, t).expect("inside timeline");
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
        let f = animate(input, &default_targets());
        let t = f.telemetry(f.total_time + 1.0);
        let r = &f.result;
        assert!((t.carry_yards - r.flight.carry_yards).abs() < 1e-9);
        assert!((t.total_yards - r.total_yards).abs() < 1e-9);
        assert!((t.lateral_m - r.final_pos.z).abs() < 1e-9);
        assert!(t.lateral_m > 0.0, "slice finishes right");
        assert!(!t.in_flight);
    }

    #[test]
    fn slice_is_drawn_right_of_the_line() {
        let input = ShotInput {
            launch_azimuth_deg: 1.0,
            sidespin_rpm: 600.0,
            ..ShotInput::driver()
        };
        let f = animate(input, &default_targets());
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
