//! Shared flight animation logic used by both windowed and streaming modes.

use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;

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
/// Longest a ball that dimple did not bring to rest keeps moving on the
/// ground after landing (seconds). Shots that come to rest animate to the end.
pub const MAX_UNSETTLED_GROUND_TIME_S: f64 = 8.0;

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
            interpolate_timeline(&self.timeline, t_clamped).unwrap_or(DVec3::ZERO)
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

        let ball_pos = interpolate_timeline(&self.timeline, t_clamped).map(to_scene)?;

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

/// How often a lost or refused FRP connection is retried.
pub const RECONNECT_INTERVAL: Duration = Duration::from_secs(5);

/// How often a connect that has not finished is reported.
pub const DIAL_WARN_INTERVAL: Duration = Duration::from_secs(30);

/// A shot taken in this much later than it arrived is reported (seconds).
const LATE_INTAKE_S: f64 = 0.25;

/// The FRP device dialled without `--frp`: flighthook on this machine.
///
/// An IPv4 address rather than `localhost`: flighthook listens on IPv4
/// only, and where `localhost` resolves to `::1` first (Windows), every dial
/// waits for the refused IPv6 attempt (about 2 s) before trying IPv4.
pub const DEFAULT_FRP_URL: &str = "ws://127.0.0.1:5880/frp";

/// The FRP device URL: the value after `--frp` in `args`, or
/// [`DEFAULT_FRP_URL`] without the flag.
///
/// # Errors
///
/// Returns an error if `--frp` has no value or the value is not a
/// `ws://` or `wss://` URL.
pub fn frp_url(args: &[String]) -> anyhow::Result<String> {
    let Some(i) = args.iter().position(|a| a == "--frp") else {
        return Ok(DEFAULT_FRP_URL.to_owned());
    };
    let url = args
        .get(i + 1)
        .ok_or_else(|| anyhow::anyhow!("--frp needs a URL, e.g. --frp ws://<host>:5880/frp"))?;
    if url.starts_with("ws://") || url.starts_with("wss://") {
        Ok(url.clone())
    } else {
        anyhow::bail!("--frp {url}: expected a ws:// or wss:// URL, e.g. ws://<host>:5880/frp")
    }
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
/// so the caller can report a handshake that does not finish. The endpoint
/// and the outcome come back on the returned channel.
///
/// # Errors
///
/// Returns the endpoint and the error when the thread cannot be started.
fn spawn_dial(
    endpoint: FrpEndpoint,
) -> Result<mpsc::Receiver<DialResult>, (FrpEndpoint, std::io::Error)> {
    let (tx, rx) = mpsc::channel();
    // The endpoint is handed over only once the thread runs, so a failed
    // spawn leaves it with the caller.
    let (start_tx, start_rx) = mpsc::channel::<FrpEndpoint>();
    let spawned = thread::Builder::new()
        .name("frp-dial".to_owned())
        .spawn(move || {
            let Ok(mut endpoint) = start_rx.recv() else {
                return;
            };
            let result = endpoint.establish();
            // The receiver is gone only when the intake has stopped.
            let _ = tx.send((endpoint, result));
        });
    if let Err(e) = spawned {
        return Err((endpoint, e));
    }
    start_tx
        .send(endpoint)
        .map_err(|mpsc::SendError(endpoint)| {
            (
                endpoint,
                std::io::Error::other("FRP connect thread exited before starting"),
            )
        })?;
    Ok(rx)
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

/// A shot and when it arrived.
#[derive(Debug)]
pub struct ArrivedShot {
    pub shot: ReceivedShot,
    /// When its last message arrived, in seconds on the app's clock (since
    /// the `start` given to [`FrpState::new`]).
    pub arrival: f64,
}

/// What the intake thread tells the app.
#[derive(Debug)]
enum IntakeEvent {
    /// A dial has started: not connected.
    Connecting,
    /// The FRP handshake completed.
    Connected,
    /// The connection was lost.
    Disconnected,
    /// The launch monitor ready states changed.
    LmStates(HashMap<String, bool>),
    Shot(ArrivedShot),
}

/// How often the intake thread dials and reports.
#[derive(Debug, Clone, Copy)]
struct IntakeTiming {
    /// Least time from the start of one dial to the start of the next.
    reconnect: Duration,
    /// How often a dial that has not finished is reported.
    dial_warn: Duration,
}

impl IntakeTiming {
    const DEFAULT: Self = Self {
        reconnect: RECONNECT_INTERVAL,
        dial_warn: DIAL_WARN_INTERVAL,
    };
}

/// Shot intake from an FRP device, used by both windowed and streaming modes.
///
/// A dedicated thread holds the connection and reads it with blocking reads,
/// whatever the app is doing: it connects, reconnects every
/// [`RECONNECT_INTERVAL`] while disconnected, aggregates shots and stamps
/// each with its arrival time. [`poll`](Self::poll) never blocks.
///
/// The thread is never joined. It ends at its next report after this is
/// dropped (a shot, a ready-state change, a disconnect or a dial), or with
/// the process.
pub struct FrpState {
    events: mpsc::Receiver<IntakeEvent>,
    /// Whether the device is connected (the FRP handshake completed).
    pub connected: bool,
    /// Last known ready state per launch monitor device ID; empty while
    /// disconnected.
    pub lm_states: HashMap<String, bool>,
    /// The intake thread has ended (reported once).
    stopped: bool,
}

impl FrpState {
    /// Start taking in shots from the device at `url`. Arrival times are
    /// seconds since `start`; `wake` is called after each shot, connection
    /// change or ready-state change, from the intake thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the FRP endpoint cannot be opened or the thread
    /// cannot be started.
    pub fn new(
        url: &str,
        start: Instant,
        wake: impl Fn() + Send + 'static,
    ) -> anyhow::Result<Self> {
        let (events, _thread) = spawn_intake(url, start, wake, IntakeTiming::DEFAULT)?;
        Ok(Self::from_events(events))
    }

    fn from_events(events: mpsc::Receiver<IntakeEvent>) -> Self {
        Self {
            events,
            connected: false,
            lm_states: HashMap::new(),
            stopped: false,
        }
    }

    /// Apply what the intake thread reported since the last call. Returns
    /// the shots that arrived, oldest first.
    pub fn poll(&mut self) -> Vec<ArrivedShot> {
        let mut shots = Vec::new();
        loop {
            match self.events.try_recv() {
                Ok(IntakeEvent::Connecting | IntakeEvent::Disconnected) => {
                    self.connected = false;
                    self.lm_states.clear();
                }
                Ok(IntakeEvent::Connected) => self.connected = true,
                Ok(IntakeEvent::LmStates(states)) => self.lm_states = states,
                Ok(IntakeEvent::Shot(shot)) => shots.push(shot),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if !self.stopped {
                        eprintln!("WARN: the FRP intake has stopped; no more shots are taken in");
                        self.stopped = true;
                    }
                    self.connected = false;
                    self.lm_states.clear();
                    break;
                }
            }
        }
        shots
    }
}

/// Start the intake thread for the device at `url`. Returns its event
/// channel and handle.
fn spawn_intake(
    url: &str,
    start: Instant,
    wake: impl Fn() + Send + 'static,
    timing: IntakeTiming,
) -> anyhow::Result<(mpsc::Receiver<IntakeEvent>, thread::JoinHandle<()>)> {
    let endpoint = frp_endpoint(url).context("open the FRP endpoint")?;
    let (tx, rx) = mpsc::channel();
    let url = url.to_owned();
    let handle = thread::Builder::new()
        .name("frp-intake".to_owned())
        .spawn(move || run_intake(&url, endpoint, start, &tx, &wake, timing))
        .context("start the FRP intake thread")?;
    Ok((rx, handle))
}

/// The intake thread: dial, read until the connection is lost, repeat.
/// Returns once the app has dropped its end of `tx`.
fn run_intake(
    url: &str,
    endpoint: FrpEndpoint,
    start: Instant,
    tx: &mpsc::Sender<IntakeEvent>,
    wake: &dyn Fn(),
    timing: IntakeTiming,
) {
    // Report `event` and wake the app; false once the app is gone.
    let report = |event: IntakeEvent| {
        let sent = tx.send(event).is_ok();
        if sent {
            wake();
        }
        sent
    };
    let mut endpoint = Some(endpoint);
    let mut last_dial: Option<Instant> = None;
    loop {
        if let Some(dialed) = last_dial {
            thread::sleep(timing.reconnect.saturating_sub(dialed.elapsed()));
        }
        // Nothing to wake the app for: it is not connected either way.
        if tx.send(IntakeEvent::Connecting).is_err() {
            return;
        }
        last_dial = Some(Instant::now());
        let ep = match endpoint.take() {
            Some(ep) => ep,
            None => match frp_endpoint(url) {
                Ok(ep) => ep,
                Err(e) => {
                    eprintln!("WARN: cannot reopen FRP endpoint: {e}");
                    continue;
                }
            },
        };
        eprintln!("Connecting to FRP at {url}...");
        let (ep, conn) = dial(url, ep, timing.dial_warn);
        endpoint = ep;
        let Some(mut conn) = conn else {
            continue;
        };
        eprintln!("Connected (FRP {})", conn.version());
        if !report(IntakeEvent::Connected) {
            return;
        }

        let mut intake = ShotIntake::default();
        // The ready states the app has; it clears them when a dial starts.
        let mut reported = HashMap::new();
        let reason = loop {
            match conn.recv() {
                Ok(msg) => {
                    let arrival = start.elapsed().as_secs_f64();
                    let shot = intake.ingest(&msg);
                    if intake.lm_states != reported {
                        reported.clone_from(&intake.lm_states);
                        if !report(IntakeEvent::LmStates(reported.clone())) {
                            return;
                        }
                    }
                    if let Some(shot) = shot {
                        eprintln!(
                            "Shot #{} from {} arrived at {arrival:.3}s",
                            shot.shot_number, shot.device
                        );
                        if !report(IntakeEvent::Shot(ArrivedShot { shot, arrival })) {
                            return;
                        }
                    }
                }
                Err(flightrelay::FrpError::Json(_)) => {
                    // some messages may not be parsable due to extension
                }
                Err(e) => break e,
            }
        };
        // Device states and partly received shots go with the connection.
        eprintln!("FRP disconnected: {reason}");
        if !report(IntakeEvent::Disconnected) {
            return;
        }
    }
}

/// Dial `url` and wait for the handshake, reporting every `warn` that it
/// has not finished. A dial that does not finish is left running: the
/// handshake has no timeout, so a redial would only add a second waiting
/// thread.
///
/// Returns the endpoint for the next dial (`None` if it was lost and could
/// not be reopened) and the connection, if one was made.
fn dial(
    url: &str,
    endpoint: FrpEndpoint,
    warn: Duration,
) -> (Option<FrpEndpoint>, Option<FrpConnection>) {
    let pending = match spawn_dial(endpoint) {
        Ok(pending) => pending,
        Err((endpoint, e)) => {
            eprintln!("WARN: cannot start the FRP connect thread: {e}");
            return (Some(endpoint), None);
        }
    };
    let started = Instant::now();
    loop {
        match pending.recv_timeout(warn) {
            Ok((endpoint, Ok(conn))) => return (Some(endpoint), Some(conn)),
            Ok((endpoint, Err(e))) => {
                eprintln!("FRP not available at {url} (expected {FRP_VERSION}): {e}");
                return (Some(endpoint), None);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                eprintln!(
                    "WARN: still connecting to FRP at {url} after {:.0}s: \
                     the device has not answered",
                    started.elapsed().as_secs_f64()
                );
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The dialing thread died with the endpoint; open a new one.
                let endpoint = frp_endpoint(url)
                    .inspect_err(|e| eprintln!("WARN: cannot reopen FRP endpoint: {e}"))
                    .ok();
                return (endpoint, None);
            }
        }
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
                fm.launch(&shot, now, &Environment::SEA_LEVEL, Range::new().green);
            }
        }
    }

    #[test]
    fn frp_url_comes_from_the_flag() {
        let args = |a: &[&str]| a.iter().map(|&s| s.to_owned()).collect::<Vec<_>>();
        assert_eq!(
            frp_url(&args(&["cyberflight"])).expect("default"),
            DEFAULT_FRP_URL
        );
        assert_eq!(
            frp_url(&args(&[
                "cyberflight",
                "--stream",
                "--frp",
                "ws://pc:5880/frp"
            ]))
            .expect("ws"),
            "ws://pc:5880/frp"
        );
        assert_eq!(
            frp_url(&args(&["cyberflight", "--frp", "wss://pc/frp"])).expect("wss"),
            "wss://pc/frp"
        );
    }

    #[test]
    fn frp_url_must_be_a_websocket_url() {
        let args = |a: &[&str]| a.iter().map(|&s| s.to_owned()).collect::<Vec<_>>();
        let missing = frp_url(&args(&["cyberflight", "--frp"])).unwrap_err();
        assert!(missing.to_string().contains("needs a URL"), "{missing}");
        for bad in ["--stream", "pc:5880", "http://pc:5880/frp"] {
            let err = frp_url(&args(&["cyberflight", "--frp", bad])).unwrap_err();
            assert!(err.to_string().contains("ws://"), "{bad}: {err}");
        }
    }

    /// A device listening on a free local port, and the URL to dial it.
    fn device() -> (FrpEndpoint, String) {
        let device = FrpEndpoint::open(EndpointConfig::new(
            Role::Device,
            Transport::listen("127.0.0.1:0"),
        ))
        .expect("device endpoint");
        let addr = device.local_addr().expect("addr").expect("bound");
        (device, format!("ws://{addr}/frp"))
    }

    /// Quick redials and no slow-dial reports, so tests run in milliseconds.
    const QUICK: IntakeTiming = IntakeTiming {
        reconnect: Duration::from_millis(50),
        dial_warn: Duration::from_secs(60),
    };

    /// An intake of `url` on a clock started now, with [`QUICK`] timing and
    /// a count of its wakes.
    fn intake(url: &str) -> (FrpState, Instant, Arc<AtomicUsize>) {
        let start = Instant::now();
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&wakes);
        let wake = move || {
            counter.fetch_add(1, Ordering::SeqCst);
        };
        let (events, _thread) = spawn_intake(url, start, wake, QUICK).expect("intake");
        (FrpState::from_events(events), start, wakes)
    }

    /// Wait until `done` or a few seconds pass.
    fn wait_until(done: impl Fn() -> bool) {
        let start = Instant::now();
        while !done() && start.elapsed().as_secs() < 5 {
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// Poll `state` until `done` or a few seconds pass; returns the shots.
    fn poll_until(
        state: &mut FrpState,
        done: impl Fn(&FrpState, &[ArrivedShot]) -> bool,
    ) -> Vec<ArrivedShot> {
        let start = Instant::now();
        let mut shots = Vec::new();
        while !done(state, &shots) && start.elapsed().as_secs() < 5 {
            shots.extend(state.poll());
            thread::sleep(Duration::from_millis(5));
        }
        shots
    }

    /// A free local port: nothing listens on a port that was just released.
    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .expect("free port")
            .port()
    }

    #[test]
    fn the_default_device_is_dialled_over_ipv4() {
        assert!(DEFAULT_FRP_URL.starts_with("ws://127.0.0.1:"));
        assert!(DEFAULT_FRP_URL.ends_with(&format!(
            ":{}{}",
            flightrelay::DEFAULT_PORT,
            flightrelay::DEFAULT_PATH
        )));
    }

    // The dial this test starts stays blocked in the handshake for the rest
    // of the test process: the FRP handshake has no timeout. That is
    // intended; nothing joins it.
    #[test]
    fn a_silent_device_is_dialled_once() {
        // Accepts TCP but never answers the WebSocket upgrade.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("ws://{}/frp", listener.local_addr().expect("addr"));
        let (mut state, _, _) = intake(&url);
        let polled = Instant::now();
        assert!(state.poll().is_empty());
        assert!(
            polled.elapsed() < Duration::from_millis(100),
            "poll returns at once"
        );

        // Many reconnect intervals pass; the outstanding dial is not repeated.
        let mut accepted = Vec::new();
        let start = Instant::now();
        while start.elapsed() < QUICK.reconnect * 8 {
            if let Ok((socket, _)) = listener.accept() {
                accepted.push(socket);
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(accepted.len(), 1);
        assert!(state.poll().is_empty());
        assert!(!state.connected);
    }

    #[test]
    fn a_refused_device_is_redialled_until_it_listens() {
        let port = free_port();
        let (mut state, _, _) = intake(&format!("ws://127.0.0.1:{port}/frp"));
        // Several dials are refused meanwhile.
        thread::sleep(QUICK.reconnect * 4);
        assert!(state.poll().is_empty());
        assert!(!state.connected);

        let mut device = FrpEndpoint::open(EndpointConfig::new(
            Role::Device,
            Transport::listen(format!("127.0.0.1:{port}")),
        ))
        .expect("device endpoint");
        let server = thread::spawn(move || device.establish().expect("handshake"));
        let _ = poll_until(&mut state, |s, _| s.connected);
        assert!(state.connected);
        let _conn = server.join().expect("device thread");
    }

    #[test]
    fn connects_in_the_background_and_receives_shots() {
        let (mut device, url) = device();
        let server = thread::spawn(move || {
            let mut conn = device.establish().expect("handshake");
            for msg in shot_messages(7, 140.0, 14.0) {
                conn.send(&msg).expect("send");
            }
            conn
        });

        let (mut state, _, wakes) = intake(&url);
        let shots = poll_until(&mut state, |s, shots| s.connected && !shots.is_empty());
        assert!(state.connected);
        let _conn = server.join().expect("device thread");
        assert_eq!(shots.len(), 1);
        assert_eq!(shots[0].shot.shot_number, 7);
        assert!(shots[0].arrival > 0.0);
        assert_eq!(wakes.load(Ordering::SeqCst), 2, "connected, then the shot");
    }

    #[test]
    fn shots_keep_their_arrival_times() {
        let (mut device, url) = device();
        let gap = Duration::from_millis(300);
        let server = thread::spawn(move || {
            let mut conn = device.establish().expect("handshake");
            for (i, n) in [1, 2].into_iter().enumerate() {
                if i > 0 {
                    thread::sleep(gap);
                }
                for msg in shot_messages(n, 140.0, 14.0) {
                    conn.send(&msg).expect("send");
                }
            }
            conn
        });

        // Nothing is polled while the shots arrive: the app is not drawing.
        let (mut state, start, wakes) = intake(&url);
        wait_until(|| wakes.load(Ordering::SeqCst) >= 3);
        let _conn = server.join().expect("device thread");
        thread::sleep(Duration::from_millis(200));
        let polled = start.elapsed().as_secs_f64();
        let shots = state.poll();

        assert!(state.connected);
        assert_eq!(shots.len(), 2);
        assert_eq!(shots[0].shot.shot_number, 1);
        assert_eq!(shots[1].shot.shot_number, 2);
        // Loopback Nagle + delayed ACK can hold the first shot back ~40 ms.
        let apart = shots[1].arrival - shots[0].arrival;
        assert!(
            apart >= gap.as_secs_f64() - 0.1,
            "arrived {apart:.3}s apart"
        );
        assert!(
            polled - shots[1].arrival >= 0.19,
            "arrival {:.3}s is not the poll time {polled:.3}s",
            shots[1].arrival
        );
    }

    #[test]
    fn the_intake_reconnects_after_the_peer_closes() {
        let (mut device, url) = device();
        let server = thread::spawn(move || {
            let mut conn = device.establish().expect("handshake");
            let ready = HashMap::from([(
                flightrelay::types::telemetry::READY.to_owned(),
                "true".to_owned(),
            )]);
            conn.send(&envelope(FrpEvent::DeviceTelemetry {
                manufacturer: None,
                model: None,
                firmware: None,
                telemetry: Some(ready),
            }))
            .expect("send");
            // Shot 3 is cut off by the close; its end comes on the next connection.
            let shot3 = shot_messages(3, 140.0, 14.0);
            for msg in &shot3[..2] {
                conn.send(msg).expect("send");
            }
            drop(conn);

            let mut conn = device.establish().expect("second handshake");
            conn.send(&shot3[2]).expect("send");
            for msg in shot_messages(4, 140.0, 14.0) {
                conn.send(&msg).expect("send");
            }
            conn
        });

        let (events, _thread) = spawn_intake(&url, Instant::now(), || {}, QUICK).expect("intake");
        let mut seen = Vec::new();
        loop {
            let event = events
                .recv_timeout(Duration::from_secs(5))
                .expect("an event");
            let last = matches!(&event, IntakeEvent::Shot(_));
            seen.push(event);
            if last {
                break;
            }
        }
        let _conn = server.join().expect("device thread accepted twice");
        // Repeated dial starts are left out: refused redials are timing-dependent.
        let seen: Vec<String> = seen
            .iter()
            .filter(|e| !matches!(e, IntakeEvent::Connecting))
            .map(|e| match e {
                IntakeEvent::Shot(s) => format!("shot {}", s.shot.shot_number),
                IntakeEvent::LmStates(states) => format!("ready {states:?}"),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            seen,
            [
                "Connected",
                r#"ready {"test.0": true}"#,
                "Disconnected",
                "Connected",
                // Shot 3 was cut off by the close and is forgotten.
                "shot 4",
            ]
        );

        // The app forgets the ready states with the connection.
        let (tx, rx) = mpsc::channel();
        let mut state = FrpState::from_events(rx);
        let ready = HashMap::from([("test.0".to_owned(), true)]);
        for event in [IntakeEvent::Connected, IntakeEvent::LmStates(ready.clone())] {
            tx.send(event).expect("send");
        }
        assert!(state.poll().is_empty());
        assert!(state.connected);
        assert_eq!(state.lm_states, ready);
        tx.send(IntakeEvent::Disconnected).expect("send");
        assert!(state.poll().is_empty());
        assert!(!state.connected);
        assert!(state.lm_states.is_empty());
    }

    #[test]
    fn a_stopped_intake_is_not_connected() {
        let (tx, rx) = mpsc::channel();
        let mut state = FrpState::from_events(rx);
        tx.send(IntakeEvent::Connected).expect("send");
        drop(tx);
        assert!(state.poll().is_empty());
        assert!(!state.connected);
        assert!(state.stopped);
    }

    #[test]
    fn the_intake_ends_when_the_app_is_gone() {
        // Disconnected: it ends at its next dial.
        let port = free_port();
        let (events, thread) = spawn_intake(
            &format!("ws://127.0.0.1:{port}/frp"),
            Instant::now(),
            || {},
            QUICK,
        )
        .expect("intake");
        thread::sleep(QUICK.reconnect * 2);
        drop(events);
        wait_until(|| thread.is_finished());
        assert!(thread.is_finished(), "ended while disconnected");

        // Connected: it ends at the next shot.
        let (mut device, url) = device();
        let (events, thread) = spawn_intake(&url, Instant::now(), || {}, QUICK).expect("intake");
        let mut conn = device.establish().expect("handshake");
        drop(events);
        for msg in shot_messages(1, 140.0, 14.0) {
            conn.send(&msg).expect("send");
        }
        wait_until(|| thread.is_finished());
        assert!(thread.is_finished(), "ended while connected");
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
