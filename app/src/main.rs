mod flight;
mod stream;

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use cf_math::aero::BallModel;
use cf_math::environment::Environment;
use cf_render::context::{GpuConfig, GpuContext};
use cf_render::render::{FlightRenderData, Renderer};
use cf_render::window::Swapchain;
use cf_scene::hud::{build_hud, ShotTelemetry, UnitSystem};
use cf_scene::range::Range;
use cf_scene::shot::ReceivedShot;

use flightrelay::{FrpClient, FrpEvent, FrpMessage, ShotAggregator};

use flight::{AnimatedFlight, DEFAULT_TRAIL_SUBSAMPLE};

struct App {
    windowed: bool,
    window: Option<Window>,
    renderer: Option<Renderer>,
    range: Range,
    environment: Environment,
    flights: Vec<AnimatedFlight>,
    /// Last shot telemetry — persists until the next shot arrives.
    last_telemetry: Option<ShotTelemetry>,
    /// Live WebSocket connection to FRP device (None when disconnected).
    client: Option<FrpClient>,
    /// Accumulates shot lifecycle events into complete shots.
    shots: ShotAggregator,
    /// Last known ready state per launch monitor device ID.
    lm_states: HashMap<String, bool>,
    /// Whether the FRP WebSocket connection is alive.
    frp_connected: bool,
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
            frp_connected: connected,
            start_time: Instant::now(),
            last_launch: 0.0,
            last_reconnect_attempt: 0.0,
            frame_count: 0,
            fps_timer: Instant::now(),
            trail_subsample: DEFAULT_TRAIL_SUBSAMPLE,
        })
    }

    /// Attempt to connect to FRP device. Returns (client, connected).
    fn try_connect() -> (Option<FrpClient>, bool) {
        eprintln!("Connecting to FRP at {}...", flightrelay::DEFAULT_URL);
        match FrpClient::connect(
            flightrelay::DEFAULT_URL,
            "cyberflight",
            &[flightrelay::SPEC_VERSION],
        ) {
            Ok(client) => {
                eprintln!("Connected (FRP {})", client.version());
                if let Err(e) = client.set_nonblocking(true) {
                    eprintln!("set_nonblocking failed: {e}");
                    return (None, false);
                }
                (Some(client), true)
            }
            Err(e) => {
                eprintln!(
                    "FRP not available (expected {}): {e}",
                    flightrelay::SPEC_VERSION
                );
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

        let display_handle = window.display_handle().expect("display handle").as_raw();
        let window_handle = window.window_handle().expect("window handle").as_raw();

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
                                renderer.swapchain.extent.width, renderer.swapchain.extent.height
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

        // Reconnect to FRP if disconnected (try every 5 seconds).
        if self.client.is_none() && now - self.last_reconnect_attempt >= 5.0 {
            self.last_reconnect_attempt = now;
            let (client, connected) = Self::try_connect();
            self.client = client;
            self.frp_connected = connected;
        }

        // Always drain FRP messages to prevent OS buffer overflow.
        // If a shot arrives while one is already in flight, ignore it.
        let animating = self.flights.iter().any(|f| f.is_animating(now));
        let mut disconnected = false;
        if let Some(client) = &mut self.client {
            loop {
                match client.try_recv() {
                    Ok(Some(msg)) => {
                        // Always update launch monitor status from device telemetry.
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
                        // Only feed the shot aggregator when idle.
                        if !animating {
                            if let Some(completed) = self.shots.feed(&msg) {
                                if let Some(received) = ReceivedShot::from_frp(&completed) {
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
            self.frp_connected = false;
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
            let hud = build_hud(
                telemetry,
                UnitSystem::Imperial,
                chase_camera.is_some(),
                &self.lm_states,
                self.frp_connected,
                screen_w,
                screen_h,
            );
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
                eprintln!(
                    "Trail subsample: {} (every {}th point)",
                    self.trail_subsample, self.trail_subsample
                );
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::BracketLeft) =>
            {
                self.trail_subsample = (self.trail_subsample - 1).max(1);
                eprintln!(
                    "Trail subsample: {} (every {}th point)",
                    self.trail_subsample, self.trail_subsample
                );
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
    if std::env::args().any(|a| a == "--stream") {
        return stream::run_stream();
    }

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
