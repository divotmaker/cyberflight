use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::{DVec3, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use cf_math::aero::BallModel;
use cf_math::bounce::BounceSurface;
use cf_math::environment::Environment;
use cf_math::rollout::RolloutSurface;
use cf_math::trajectory::{ShotInput, ShotResult, simulate_shot};
use cf_render::context::{GpuConfig, GpuContext};
use cf_render::render::{FlightRenderData, Renderer};
use cf_render::window::Swapchain;
use cf_scene::camera::Camera;
use cf_scene::hud::{ShotTelemetry, UnitSystem, build_hud};
use cf_scene::range::Range;
use cf_scene::shot::{ClubDelivery, ReceivedShot};
use cf_scene::surface;
use cf_scene::trail::TrailPoint;

use flighthook::{FlighthookClient, FlighthookEvent, ShotAggregator};

/// Default flighthook WebSocket URL.
const FLIGHTHOOK_URL: &str = "ws://localhost:3030/api/ws";
/// How long the chase camera lingers after the ball stops (seconds).
const CHASE_LINGER_TIME: f64 = 5.0;
/// Time constant for chase camera smoothing (seconds). Higher = smoother but laggier.
const CHASE_CAM_SMOOTHING_TAU: f64 = 0.3;
/// Default subsample stride: every Nth trajectory sample is used as a trail point.
const DEFAULT_TRAIL_SUBSAMPLE: usize = 1;
/// Trail tail as a fraction of elapsed flight time.
const TRAIL_TAIL_FRACTION: f64 = 1.2;

/// Gravity for parabolic bounce arcs.
const G: f64 = cf_math::units::G;

/// A point in the unified shot timeline (flight + bounce arcs + rollout).
#[derive(Debug, Clone, Copy)]
struct TimelinePoint {
    time: f64,
    pos: DVec3,
}

/// Map trajectory coordinates (x=forward, y=up, z=left) to
/// scene coordinates (+Z=downrange, +Y=up, +X=right).
fn traj_to_scene(p: DVec3) -> Vec3 {
    Vec3::new(-p.z as f32, p.y as f32, p.x as f32)
}

/// Interpolate position in the unified timeline.
fn interpolate_timeline(points: &[TimelinePoint], t: f64) -> Option<DVec3> {
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

/// Default smash factor for estimating club speed when no real data is available.
const DEFAULT_SMASH_FACTOR: f64 = 1.40;

/// Build a unified timeline from a ShotResult.
///
/// Stitches together:
/// 1. Flight trajectory points (0..flight_time)
/// 2. Parabolic bounce arcs (flight_time..bounce_end)
/// 3. Rollout points (bounce_end..total_time)
fn build_timeline(result: &ShotResult) -> Vec<TimelinePoint> {
    let mut timeline = Vec::with_capacity(
        result.flight.points.len() + result.rollout.points.len() + 200,
    );

    // Phase 1: Flight
    for p in &result.flight.points {
        timeline.push(TimelinePoint { time: p.time, pos: p.pos });
    }

    let mut t_offset = result.flight.flight_time;

    // Phase 2: Bounce arcs
    for (i, bounce) in result.bounces.iter().enumerate() {
        // Each bounce event has a position (where it hit) and vel_out (rebound velocity).
        // If still bouncing (not the last bounce), trace a parabolic arc.
        let is_last = i + 1 == result.bounces.len();
        if is_last {
            // Last bounce transitions to rollout — no arc needed.
            break;
        }

        let vel = bounce.vel_out;
        if vel.y <= 0.0 {
            // No upward velocity — skip arc
            continue;
        }

        // Time for parabolic arc: ball goes up and comes back to Y=0.
        let t_arc = 2.0 * vel.y / G;
        let num_samples = ((t_arc / 0.01) as usize).clamp(2, 200);

        for s in 1..=num_samples {
            let dt = t_arc * s as f64 / num_samples as f64;
            let pos = DVec3::new(
                bounce.pos.x + vel.x * dt,
                (bounce.pos.y + vel.y * dt - 0.5 * G * dt * dt).max(0.0),
                bounce.pos.z + vel.z * dt,
            );
            timeline.push(TimelinePoint { time: t_offset + dt, pos });
        }
        t_offset += t_arc;
    }

    // Phase 3: Rollout
    for rp in &result.rollout.points {
        timeline.push(TimelinePoint {
            time: t_offset + rp.time,
            pos: rp.pos,
        });
    }

    timeline
}

/// An in-progress or recently completed shot animation.
struct AnimatedFlight {
    input: ShotInput,
    result: ShotResult,
    /// Unified position timeline covering all phases.
    timeline: Vec<TimelinePoint>,
    /// Total animation duration (flight + bounce + rollout).
    total_time: f64,
    launch_time: f64,
    /// Club delivery data from launch monitor.
    club: ClubDelivery,
    /// Carry distance reported by the launch monitor (yards), if available.
    lm_carry_yards: Option<f64>,
    landed: bool,
    /// Previous chase camera state for smoothing (camera, timestamp).
    prev_chase: Option<(Camera, f64)>,
}

impl AnimatedFlight {
    fn from_received(
        shot: &ReceivedShot,
        launch_time: f64,
        ball: &BallModel,
        env: &Environment,
        targets: &[cf_scene::target::Target],
    ) -> Self {
        let input = &shot.input;

        // Pre-simulate flight to determine landing surface.
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
            shot.shot_number, shot.source,
            result.flight.carry_yards, rollout_yards, result.total_yards, total_time,
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

    fn telemetry(&self, now: f64) -> ShotTelemetry {
        let shot_t = (now - self.launch_time).max(0.0);
        let t_clamped = shot_t.min(self.total_time);

        // Live distances from current ball position in trajectory space.
        // Trajectory: X = downrange, Y = up, Z = lateral.
        let pos = interpolate_timeline(&self.timeline, t_clamped)
            .unwrap_or(DVec3::ZERO);

        let total_m = (pos.x * pos.x + pos.z * pos.z).sqrt();
        let downrange_m = pos.x; // downrange axis only

        // Carry = live distance during flight, locked to final carry after landing
        let carry_yards = if shot_t < self.result.flight.flight_time {
            cf_math::units::meters_to_yards(total_m)
        } else {
            self.result.flight.carry_yards
        };

        let club_speed_mph = self.club.club_speed_mph
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
            // Live apex: current height during flight, final apex after landing
            apex_m: if shot_t < self.result.flight.flight_time {
                pos.y.max(0.0)
            } else {
                self.result.flight.apex_m
            },
            carry_yards,
            lm_carry_yards: self.lm_carry_yards,
            total_yards: cf_math::units::meters_to_yards(total_m),
            downrange_yards: cf_math::units::meters_to_yards(downrange_m),
            lateral_m: -pos.z, // physics +Z=left, golf positive=right
            flight_time_s: self.result.flight.flight_time,
            // Time measures carry flight only, not bounce/rollout
            elapsed_s: shot_t.min(self.result.flight.flight_time),
            in_flight: self.is_animating(now),
        }
    }

    /// Compute render data for the current time.
    fn render_data(&self, now: f64, trail_subsample: usize) -> Option<FlightRenderData> {
        let shot_t = now - self.launch_time;
        if shot_t < 0.0 {
            return None;
        }

        let t_clamped = shot_t.min(self.total_time);

        let ball_pos = interpolate_timeline(&self.timeline, t_clamped)
            .map(traj_to_scene)?;

        // Trail: last TRAIL_TAIL_FRACTION of elapsed time, subsampled
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

        // Snap trail end to ball position
        if trail_points
            .last()
            .is_none_or(|last| (last.position - ball_pos).length() > 0.01)
        {
            trail_points.push(TrailPoint { position: ball_pos, time: t_clamped });
        }

        Some(FlightRenderData {
            ball_pos,
            trail_points,
            current_time: shot_t,
        })
    }

    /// Chase camera that follows the ball through all phases, with smoothing.
    fn chase_camera(&mut self, now: f64) -> Option<Camera> {
        let shot_t = now - self.launch_time;
        if shot_t < 0.0 {
            return None;
        }

        let final_pos = traj_to_scene(self.result.final_pos);

        // Overall shot direction (tee at origin) — stable fallback for low speeds
        let shot_dir = if final_pos.length_squared() > 0.01 {
            final_pos.normalize()
        } else {
            Vec3::Z
        };

        let ideal = if shot_t < self.total_time {
            // Shot in progress: chase the ball
            let ball_pos = interpolate_timeline(&self.timeline, shot_t)
                .map(traj_to_scene)?;

            // Estimate velocity from adjacent timeline points
            let idx = self.timeline.partition_point(|p| p.time <= shot_t);
            let vel_idx = if idx > 0 { idx - 1 } else { 0 };
            let vel = if vel_idx + 1 < self.timeline.len() {
                let a = &self.timeline[vel_idx];
                let b = &self.timeline[vel_idx + 1];
                let dt = (b.time - a.time).max(1e-6);
                (b.pos - a.pos) / dt
            } else {
                DVec3::new(1.0, 0.0, 0.0) // fallback
            };
            let velocity = Vec3::new(vel.z as f32, vel.y as f32, vel.x as f32);

            // At low horizontal speed (rollout deceleration), lock to shot direction
            let horiz_speed = Vec3::new(velocity.x, 0.0, velocity.z).length();
            let direction = if horiz_speed > 2.0 { velocity } else { shot_dir };

            Camera::chase(ball_pos, direction, final_pos)
        } else if shot_t < self.total_time + CHASE_LINGER_TIME {
            // Shot complete: hold camera using overall shot direction
            Camera::chase(final_pos, shot_dir, final_pos)
        } else {
            self.prev_chase = None;
            return None;
        };

        // Exponential smoothing: lerp toward ideal position/target each frame
        let smoothed = if let Some((prev, prev_time)) = &self.prev_chase {
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

    /// Whether the shot is still animating (any phase).
    fn is_animating(&self, now: f64) -> bool {
        let shot_t = now - self.launch_time;
        shot_t >= 0.0 && shot_t < self.total_time
    }

    fn is_expired(&self, now: f64) -> bool {
        let shot_t = now - self.launch_time;
        shot_t > self.total_time + CHASE_LINGER_TIME
    }
}

struct App {
    windowed: bool,
    window: Option<Window>,
    renderer: Option<Renderer>,
    range: Range,
    environment: Environment,
    flights: Vec<AnimatedFlight>,
    /// Last shot telemetry — persists until the next shot arrives.
    last_telemetry: Option<ShotTelemetry>,
    /// Live WebSocket connection to flighthook server (None when disconnected).
    client: Option<FlighthookClient>,
    /// Accumulates shot lifecycle events into complete ShotData.
    shots: ShotAggregator,
    /// Last known armed state per launch monitor source ID.
    lm_states: HashMap<String, bool>,
    /// Whether the flighthook WebSocket connection is alive.
    flighthook_connected: bool,
    start_time: Instant,
    last_launch: f64,
    /// Monotonic time of last reconnect attempt.
    last_reconnect_attempt: f64,
    frame_count: u64,
    fps_timer: Instant,
    trail_subsample: usize,
}

impl App {
    fn new(windowed: bool) -> Result<Self> {
        let (client, connected) = Self::try_connect();

        Ok(Self {
            windowed,
            window: None,
            renderer: None,
            range: Range::new(),
            environment: Environment::SEA_LEVEL,
            flights: Vec::new(),
            last_telemetry: None,
            client,
            shots: ShotAggregator::new(),
            lm_states: HashMap::new(),
            flighthook_connected: connected,
            start_time: Instant::now(),
            last_launch: 0.0,
            last_reconnect_attempt: 0.0,
            frame_count: 0,
            fps_timer: Instant::now(),
            trail_subsample: DEFAULT_TRAIL_SUBSAMPLE,
        })
    }

    /// Attempt to connect to flighthook. Returns (client, connected).
    fn try_connect() -> (Option<FlighthookClient>, bool) {
        eprintln!("Connecting to flighthook at {FLIGHTHOOK_URL}...");
        match FlighthookClient::connect(FLIGHTHOOK_URL, "cyberflight") {
            Ok(client) => {
                eprintln!("Connected as {}", client.source_id());
                if let Err(e) = client.set_nonblocking(true) {
                    eprintln!("set_nonblocking failed: {e}");
                    return (None, false);
                }
                (Some(client), true)
            }
            Err(e) => {
                eprintln!("flighthook not available: {e}");
                (None, false)
            }
        }
    }

    fn init_renderer(&mut self, window: &Window) {
        let size = window.inner_size();
        let config = GpuConfig {
            width: size.width,
            height: size.height,
            enable_raytracing: true,
            ..GpuConfig::default()
        };

        let display_handle = window
            .display_handle()
            .expect("display handle")
            .as_raw();
        let window_handle = window
            .window_handle()
            .expect("window handle")
            .as_raw();

        match GpuContext::new(config, display_handle, window_handle) {
            Err(cf_render::error::RenderError::Loading(_)) => {
                eprintln!("ERROR: Vulkan loader not found.");
                eprintln!("  Linux:   sudo apt install libvulkan1 mesa-vulkan-drivers");
                eprintln!("  Windows: Install or update your GPU drivers");
            }
            Ok((gpu, surface, surface_loader)) => {
                match Swapchain::new(
                    &gpu.instance,
                    &gpu.device,
                    gpu.physical_device,
                    surface,
                    surface_loader,
                    gpu.queue_family_index,
                    size.width,
                    size.height,
                ) {
                    Ok(swapchain) => match Renderer::new(gpu, swapchain, &self.range.grid) {
                        Ok(renderer) => {
                            eprintln!(
                                "Renderer initialized: {}x{}",
                                renderer.swapchain.extent.width,
                                renderer.swapchain.extent.height
                            );
                            self.renderer = Some(renderer);
                        }
                        Err(e) => eprintln!("Failed to create renderer: {e}"),
                    },
                    Err(e) => eprintln!("Failed to create swapchain: {e}"),
                }
            }
            Err(e) => eprintln!("Failed to create GPU context: {e}"),
        }
    }

    fn update(&mut self) {
        let now = self.start_time.elapsed().as_secs_f64();

        // Reconnect to flighthook if disconnected (try every 5 seconds).
        if self.client.is_none() && now - self.last_reconnect_attempt >= 5.0 {
            self.last_reconnect_attempt = now;
            let (client, connected) = Self::try_connect();
            self.client = client;
            self.flighthook_connected = connected;
        }

        // Always drain flighthook messages to prevent OS buffer overflow.
        // If a shot arrives while one is already in flight, ignore it.
        let animating = self.flights.iter().any(|f| f.is_animating(now));
        let mut disconnected = false;
        if let Some(client) = &mut self.client {
            loop {
                match client.try_recv() {
                    Ok(Some(msg)) => {
                        // Always update launch monitor status.
                        match &msg.event {
                            FlighthookEvent::LaunchMonitorState { armed, .. } => {
                                self.lm_states.insert(msg.source.clone(), *armed);
                            }
                            FlighthookEvent::ActorStatus { status, .. }
                                if *status == flighthook::ActorStatus::Disconnected =>
                            {
                                self.lm_states.remove(&msg.source);
                            }
                            _ => {}
                        }
                        // Only feed the shot aggregator when idle.
                        if !animating {
                            if let Some(shot_data) = self.shots.feed(&msg) {
                                let received = ReceivedShot::from_flighthook(&shot_data);
                                self.flights.push(AnimatedFlight::from_received(
                                    &received,
                                    now,
                                    &BallModel::TOUR,
                                    &self.environment,
                                    &self.range.targets,
                                ));
                                self.last_launch = now;
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("flighthook disconnected: {e}");
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        if disconnected {
            self.flighthook_connected = false;
            self.client = None;
            self.lm_states.clear();
        }

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

        // Build render data for all visible flights
        let render_data: Vec<FlightRenderData> = self
            .flights
            .iter()
            .filter_map(|f| f.render_data(now, self.trail_subsample))
            .collect();

        // Build telemetry from the most recent flight, or keep the last snapshot
        if let Some(f) = self.flights.last() {
            self.last_telemetry = Some(f.telemetry(now));
        }
        let telemetry = self.last_telemetry.as_ref();

        // Chase camera: follow the most recent active ball
        let chase_camera = self
            .flights
            .iter_mut()
            .rev()
            .find_map(|f| f.chase_camera(now));

        if let Some(renderer) = &mut self.renderer {
            renderer.update_flight_geometry(&render_data, &self.range.camera);
            renderer.set_chase_camera(chase_camera);

            let screen_w = renderer.swapchain.extent.width as f32;
            let screen_h = renderer.swapchain.extent.height as f32;
            let hud = build_hud(telemetry, UnitSystem::Imperial, chase_camera.is_some(), &self.lm_states, self.flighthook_connected, screen_w, screen_h);
            renderer.update_hud(&hud.lines, &hud.fills);
        }

        // FPS counter
        self.frame_count += 1;
        let fps_elapsed = self.fps_timer.elapsed().as_secs_f64();
        if fps_elapsed >= 5.0 {
            let fps = self.frame_count as f64 / fps_elapsed;
            eprintln!("FPS: {fps:.0}");
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }
    }
}

impl App {
    fn load_window_icon() -> Option<winit::window::Icon> {
        let png_bytes = include_bytes!("../assets/icon_256.png");
        let img = image::load_from_memory(png_bytes).ok()?.into_rgba8();
        let (w, h) = img.dimensions();
        winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let icon = Self::load_window_icon();
            let fullscreen = if self.windowed {
                None
            } else {
                Some(winit::window::Fullscreen::Borderless(None))
            };
            let attrs = Window::default_attributes()
                .with_title("cyberflight")
                .with_window_icon(icon)
                .with_fullscreen(fullscreen)
                .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));
            match event_loop.create_window(attrs) {
                Ok(window) => {
                    eprintln!("Window created: {:?}", window.inner_size());
                    if !self.windowed {
                        window.set_cursor_visible(false);
                    }
                    self.init_renderer(&window);
                    self.window = Some(window);
                }
                Err(e) => {
                    eprintln!("Failed to create window: {e}");
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
            {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.update();

                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.draw_frame(&self.range.camera) {
                        eprintln!("Render error: {e}");
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::KeyR) =>
            {
                if let Some(renderer) = &mut self.renderer {
                    use cf_render::mode::RenderMode;
                    let new_mode = if renderer.render_mode == RenderMode::Rasterized {
                        RenderMode::RayTraced
                    } else {
                        RenderMode::Rasterized
                    };
                    renderer.set_render_mode(new_mode);
                    eprintln!("Render mode: {:?}", renderer.render_mode);
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::BracketRight) =>
            {
                self.trail_subsample = (self.trail_subsample + 1).min(20);
                eprintln!("Trail subsample: {} (every {}th point)", self.trail_subsample,
                    self.trail_subsample);
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::BracketLeft) =>
            {
                self.trail_subsample = (self.trail_subsample - 1).max(1);
                eprintln!("Trail subsample: {} (every {}th point)", self.trail_subsample,
                    self.trail_subsample);
            }
            WindowEvent::Resized(size) => {
                eprintln!("Resized: {}x{}", size.width, size.height);
                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.resize(size.width, size.height) {
                        eprintln!("Resize error: {e}");
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let windowed = std::env::args().any(|a| a == "--windowed");

    let event_loop = {
        #[cfg(target_os = "linux")]
        {
            // Default to winit's auto-detection (Wayland if available, else X11).
            // Set CYBERFLIGHT_X11=1 to force X11 (useful when wayland libs are
            // missing or incompatible).
            if std::env::var_os("CYBERFLIGHT_X11").is_some_and(|v| v == "1") {
                use winit::platform::x11::EventLoopBuilderExtX11;
                EventLoop::builder().with_x11().build()
            } else {
                EventLoop::new()
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            EventLoop::new()
        }
    }
    .context("failed to create event loop")?;
    let mut app = App::new(windowed).context("failed to initialize app")?;
    event_loop.run_app(&mut app).context("event loop error")?;
    Ok(())
}
