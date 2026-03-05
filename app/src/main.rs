use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec3;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use cf_math::aero::BallModel;
use cf_math::environment::Environment;
use cf_math::trajectory::{FlightResult, ShotInput, interpolate_trajectory, simulate_flight};
use cf_render::context::{GpuConfig, GpuContext};
use cf_render::render::{FlightRenderData, Renderer};
use cf_render::window::Swapchain;
use cf_scene::camera::Camera;
use cf_scene::hud::{ShotTelemetry, UnitSystem, build_hud};
use cf_scene::range::Range;
use cf_scene::trail::TrailPoint;

/// Seconds between automatic shot launches.
const SHOT_INTERVAL: f64 = 30.0;
/// Delay before the first shot (seconds after app start).
const FIRST_SHOT_DELAY: f64 = 10.0;
/// How long a landed flight stays visible before cleanup (seconds).
const LANDED_DISPLAY_TIME: f64 = 10.0;
/// How long the chase camera lingers after the ball lands (seconds).
const CHASE_LINGER_TIME: f64 = 5.0;
/// Default subsample stride: every Nth trajectory sample is used as a trail point.
/// At the default 10ms physics timestep, stride=1 uses every sample (smoothest),
/// stride=3 uses every 30ms (lighter), etc. Increase this on lower-end hardware
/// to reduce trail vertex count and improve frame rate — the trail glow generates
/// 6 vertices × 10 shells per segment, so halving segments roughly halves the
/// per-frame geometry upload. Adjustable at runtime with [ and ] keys.
const DEFAULT_TRAIL_SUBSAMPLE: usize = 1;
/// Trail tail as a fraction of elapsed flight time.
const TRAIL_TAIL_FRACTION: f64 = 1.2;

/// Map trajectory coordinates (x=forward, y=up, z=lateral) to
/// scene coordinates (+Z=downrange, +Y=up, +X=lateral).
fn traj_to_scene(p: glam::DVec3) -> Vec3 {
    Vec3::new(p.z as f32, p.y as f32, p.x as f32)
}

/// Smash factor estimates for deriving club speed from ball speed.
fn smash_factor(club_name: &str) -> f64 {
    match club_name {
        "DRIVER" => 1.48,
        "7I" => 1.33,
        "PW" => 1.25,
        _ => 1.35,
    }
}

/// An in-progress or recently landed flight animation.
struct AnimatedFlight {
    input: ShotInput,
    result: FlightResult,
    launch_time: f64,
    club_name: &'static str,
    landed: bool,
}

impl AnimatedFlight {
    fn new(input: &ShotInput, launch_time: f64, club_name: &'static str, ball: &BallModel, env: &Environment) -> Self {
        let result = simulate_flight(input, ball, env);
        eprintln!(
            "Launched {club_name}: {:.1} yds carry, {:.1}m apex, {:.2}s flight",
            result.carry_yards, result.apex_m, result.flight_time,
        );
        Self {
            input: *input,
            result,
            launch_time,
            club_name,
            landed: false,
        }
    }

    fn telemetry(&self, now: f64) -> ShotTelemetry {
        let elapsed = (now - self.launch_time).max(0.0);
        ShotTelemetry {
            club_name: self.club_name.to_uppercase(),
            club_speed_mph: self.input.ball_speed_mph / smash_factor(self.club_name),
            ball_speed_mph: self.input.ball_speed_mph,
            launch_angle_deg: self.input.launch_angle_deg,
            launch_azimuth_deg: self.input.launch_azimuth_deg,
            backspin_rpm: self.input.backspin_rpm,
            sidespin_rpm: self.input.sidespin_rpm,
            apex_m: self.result.apex_m,
            carry_yards: self.result.carry_yards,
            lateral_m: self.result.lateral_m,
            flight_time_s: self.result.flight_time,
            elapsed_s: elapsed.min(self.result.flight_time),
            in_flight: self.is_in_flight(now),
        }
    }

    /// Compute render data for the current time. Returns None if not yet launched or expired.
    fn render_data(&self, now: f64, trail_subsample: usize) -> Option<FlightRenderData> {
        let flight_t = now - self.launch_time;
        if flight_t < 0.0 {
            return None;
        }

        // Clamp to flight end
        let t_clamped = flight_t.min(self.result.flight_time);

        let ball_pos = interpolate_trajectory(&self.result.points, t_clamped)
            .map(traj_to_scene)?;

        // Trail: last TRAIL_TAIL_FRACTION of elapsed time, subsampled
        let tail_duration = t_clamped * TRAIL_TAIL_FRACTION;
        let tail_start = (t_clamped - tail_duration).max(0.0);
        let mut trail_points: Vec<TrailPoint> = self
            .result
            .points
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
            // Use unclamped elapsed time so the trail continues aging after landing.
            // Trail point timestamps are flight-local (0..flight_time), so as
            // current_time grows past flight_time the age formula fades them out.
            current_time: flight_t,
        })
    }

    /// Compute a chase camera for this flight, or None if past the linger window.
    ///
    /// During flight: follows the ball from behind. After landing: holds position
    /// looking at where the ball stopped for `CHASE_LINGER_TIME` seconds.
    fn chase_camera(&self, now: f64) -> Option<Camera> {
        let flight_t = now - self.launch_time;
        if flight_t < 0.0 {
            return None;
        }

        let landing_pos = traj_to_scene(self.result.points.last()?.pos);

        if flight_t < self.result.flight_time {
            // In flight: chase the ball
            let ball_pos = interpolate_trajectory(&self.result.points, flight_t)
                .map(traj_to_scene)?;

            let idx = self.result.points.partition_point(|p| p.time <= flight_t);
            let vel_idx = if idx > 0 { idx - 1 } else { 0 };
            let vel = self.result.points[vel_idx].vel;
            let velocity = Vec3::new(vel.z as f32, vel.y as f32, vel.x as f32);

            Some(Camera::chase(ball_pos, velocity, landing_pos))
        } else if flight_t < self.result.flight_time + CHASE_LINGER_TIME {
            // Landed: hold the final chase position, looking at where the ball stopped
            let last_vel = self.result.points.last()?.vel;
            let velocity = Vec3::new(last_vel.z as f32, last_vel.y as f32, last_vel.x as f32);

            Some(Camera::chase(landing_pos, velocity, landing_pos))
        } else {
            None
        }
    }

    fn is_in_flight(&self, now: f64) -> bool {
        let flight_t = now - self.launch_time;
        flight_t >= 0.0 && flight_t < self.result.flight_time
    }

    fn is_expired(&self, now: f64) -> bool {
        let flight_t = now - self.launch_time;
        flight_t > self.result.flight_time + LANDED_DISPLAY_TIME
    }
}

/// Pick a random club and generate a slightly varied shot input.
fn random_shot(rng: &mut SimpleRng) -> (ShotInput, &'static str) {
    let club = rng.next_u32() % 3;
    let (base, name) = match club {
        0 => (ShotInput::wedge(), "PW"),
        1 => (ShotInput::seven_iron(), "7i"),
        _ => (ShotInput::driver(), "Driver"),
    };

    // Add realistic variation: ±5% speed, ±10% spin, ±2° azimuth, ±1° launch
    let speed_var = 1.0 + rng.next_f64_range(-0.05, 0.05);
    let spin_var = 1.0 + rng.next_f64_range(-0.10, 0.10);
    let azimuth_offset = rng.next_f64_range(-2.0, 2.0);
    let launch_offset = rng.next_f64_range(-1.0, 1.0);

    let input = ShotInput {
        ball_speed_mph: base.ball_speed_mph * speed_var,
        launch_angle_deg: base.launch_angle_deg + launch_offset,
        launch_azimuth_deg: base.launch_azimuth_deg + azimuth_offset,
        backspin_rpm: base.backspin_rpm * spin_var,
        sidespin_rpm: rng.next_f64_range(-1500.0, 1500.0),
    };

    (input, name)
}

/// Minimal xorshift64 RNG (no external dependency).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn from_time() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            state: seed | 1, // must be non-zero
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 16) as u32
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn next_f64_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    range: Range,
    environment: Environment,
    flights: Vec<AnimatedFlight>,
    rng: SimpleRng,
    start_time: Instant,
    last_launch: f64,
    frame_count: u64,
    fps_timer: Instant,
    /// Trail subsample stride: every Nth trajectory sample becomes a trail point.
    /// Lower = more segments (smoother, heavier). Higher = fewer segments (faster).
    trail_subsample: usize,
}

impl App {
    fn new() -> Self {
        let range = Range::new();
        let environment = Environment::SEA_LEVEL;
        let mut rng = SimpleRng::from_time();

        // First shot launches after FIRST_SHOT_DELAY seconds
        let (input, name) = random_shot(&mut rng);
        let first_flight = AnimatedFlight::new(&input, FIRST_SHOT_DELAY, name, &BallModel::TOUR, &environment);

        Self {
            window: None,
            renderer: None,
            range,
            environment,
            flights: vec![first_flight],
            rng,
            start_time: Instant::now(),
            last_launch: FIRST_SHOT_DELAY,
            frame_count: 0,
            fps_timer: Instant::now(),
            trail_subsample: DEFAULT_TRAIL_SUBSAMPLE,
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

        // Launch a new shot every SHOT_INTERVAL seconds
        if now - self.last_launch >= SHOT_INTERVAL {
            let (input, name) = random_shot(&mut self.rng);
            self.flights.push(AnimatedFlight::new(&input, now, name, &BallModel::TOUR, &self.environment));
            self.last_launch = now;
        }

        // Mark landed flights
        for flight in &mut self.flights {
            if !flight.landed && !flight.is_in_flight(now) && now > flight.launch_time {
                flight.landed = true;
                eprintln!(
                    "  {} landed: {:.1} yds",
                    flight.club_name, flight.result.carry_yards
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

        // Build telemetry from the most recent flight
        let telemetry = self.flights.last().map(|f| f.telemetry(now));

        // Chase camera: follow the most recent in-flight ball
        let chase_camera = self
            .flights
            .iter()
            .rev()
            .find_map(|f| f.chase_camera(now));

        if let Some(renderer) = &mut self.renderer {
            renderer.update_flight_geometry(&render_data, &self.range.camera);
            renderer.set_chase_camera(chase_camera);

            let screen_w = renderer.swapchain.extent.width as f32;
            let screen_h = renderer.swapchain.extent.height as f32;
            let hud = build_hud(telemetry.as_ref(), UnitSystem::Imperial, screen_w, screen_h);
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("cyberflight")
                .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));
            match event_loop.create_window(attrs) {
                Ok(window) => {
                    eprintln!("Window created: {:?}", window.inner_size());
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
                // ] = fewer trail segments (faster)
                self.trail_subsample = (self.trail_subsample + 1).min(20);
                eprintln!("Trail subsample: {} (every {}th point)", self.trail_subsample,
                    self.trail_subsample);
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::BracketLeft) =>
            {
                // [ = more trail segments (smoother)
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
    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut app = App::new();
    event_loop.run_app(&mut app).context("event loop error")?;
    Ok(())
}
