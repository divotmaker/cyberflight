mod flight;
mod stream;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use cf_render::context::{GpuConfig, GpuContext};
use cf_render::render::Renderer;
use cf_render::window::Swapchain;
use cf_scene::range::Range;
use dimple::environment::Environment;

use flight::{FlightManager, FrpState};

/// How often a minimised window wakes to take in shots and advance the
/// flights without drawing.
const MINIMISED_TICK: Duration = Duration::from_millis(200);

/// Whether a window of `size` is minimised: Windows reports a minimised
/// window as a resize to zero width and height.
fn is_minimised(size: PhysicalSize<u32>) -> bool {
    size.width == 0 || size.height == 0
}

/// How the event loop waits: on window events while visible (redraws keep
/// the loop running), on a [`MINIMISED_TICK`] timer while minimised.
fn control_flow(minimised: bool, now: Instant) -> ControlFlow {
    if minimised {
        ControlFlow::WaitUntil(now + MINIMISED_TICK)
    } else {
        ControlFlow::Wait
    }
}

struct App {
    windowed: bool,
    window: Option<Window>,
    renderer: Option<Renderer>,
    range: Range,
    environment: Environment,
    frp: FrpState,
    flights: FlightManager,
    start_time: Instant,
    frame_count: u64,
    fps_timer: Instant,
    /// The window is minimised: nothing is drawn, and [`App::advance`] runs
    /// on a timer instead of per frame.
    minimised: bool,
    /// Why the event loop was stopped, when the app could not run.
    fatal: Option<anyhow::Error>,
}

impl App {
    fn new(windowed: bool, frp_url: &str) -> Result<Self> {
        Ok(Self {
            windowed,
            window: None,
            renderer: None,
            range: Range::new(),
            environment: Environment::SEA_LEVEL,
            frp: FrpState::new(frp_url).context("open FRP endpoint")?,
            flights: FlightManager::new(),
            start_time: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            minimised: false,
            fatal: None,
        })
    }

    fn init_renderer(&mut self, window: &Window) -> Result<()> {
        let size = window.inner_size();
        let config = GpuConfig {
            enable_raytracing: true,
            ..GpuConfig::default()
        };

        let display_handle = window
            .display_handle()
            .context("window has no display handle")?
            .as_raw();
        let window_handle = window
            .window_handle()
            .context("window has no window handle")?
            .as_raw();

        let (gpu, surface, surface_loader) =
            match GpuContext::new(config, display_handle, window_handle) {
                Err(cf_render::error::RenderError::Loading(e)) => {
                    eprintln!("ERROR: Vulkan loader not found.");
                    eprintln!("  Linux:   sudo apt install libvulkan1 mesa-vulkan-drivers");
                    eprintln!("  Windows: Install or update your GPU drivers");
                    return Err(e).context("failed to load Vulkan");
                }
                other => other.context("failed to create GPU context")?,
            };
        let swapchain = Swapchain::new(
            &gpu.instance,
            &gpu.device,
            gpu.physical_device,
            surface,
            surface_loader,
            gpu.queue_family_index,
            size.width,
            size.height,
        )
        .context("failed to create swapchain")?;
        let mut renderer =
            Renderer::new(gpu, swapchain, &self.range.grid).context("failed to create renderer")?;
        renderer
            .set_green(self.range.green.as_ref())
            .context("failed to upload the green")?;
        eprintln!(
            "Renderer initialized: {}x{}",
            renderer.extent().width,
            renderer.extent().height
        );
        self.renderer = Some(renderer);
        Ok(())
    }

    /// Stop the event loop because the app cannot run; `main` returns `err`,
    /// which prints it.
    fn fail(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.fatal = Some(err);
        event_loop.exit();
    }

    /// Track the minimised state from a window resize. Returns whether the
    /// window was restored, so drawing has to be restarted.
    fn track_minimised(&mut self, size: PhysicalSize<u32>) -> bool {
        let was_minimised = self.minimised;
        self.minimised = is_minimised(size);
        was_minimised && !self.minimised
    }

    /// Take in shots and advance the flights to `now` (seconds since
    /// start), without touching the renderer.
    fn advance(&mut self, now: f64) -> Vec<cf_render::flight::FlightRenderData> {
        for shot in self.frp.poll(now) {
            self.flights
                .launch(&shot, now, &self.environment, self.range.green);
        }
        self.flights.update(now)
    }

    fn update(&mut self) {
        let now = self.start_time.elapsed().as_secs_f64();

        let render_data = self.advance(now);
        let chase_camera = self.flights.chase_camera(now);

        if let Some(renderer) = &mut self.renderer {
            renderer.update_flight(&render_data, &self.range.camera);
            renderer.set_chase_camera(chase_camera);

            let screen_w = renderer.extent().width as f32;
            let screen_h = renderer.extent().height as f32;
            let hud = self.flights.build_hud(
                now,
                chase_camera.is_some(),
                &self.frp.intake.lm_states,
                self.frp.connected,
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
                    let init = self.init_renderer(&window);
                    self.window = Some(window);
                    if let Err(e) = init {
                        self.fail(event_loop, e);
                    }
                }
                Err(e) => {
                    self.fail(
                        event_loop,
                        anyhow::Error::new(e).context("failed to create window"),
                    );
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
            // Without a renderer nothing paces the loop, so it is not redrawn
            // (the app is exiting; see `fail`). A minimised window is not
            // drawn and stops requesting redraws until it is restored;
            // `about_to_wait` keeps the flights going meanwhile.
            WindowEvent::RedrawRequested if self.renderer.is_some() && !self.minimised => {
                self.update();

                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.draw_frame(&self.range.camera) {
                        self.fail(event_loop, anyhow::Error::new(e).context("draw frame"));
                        return;
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
                    let new_mode = if renderer.render_mode() == RenderMode::Rasterized {
                        RenderMode::RayTraced
                    } else {
                        RenderMode::Rasterized
                    };
                    renderer.set_render_mode(new_mode);
                    eprintln!("Render mode: {:?}", renderer.render_mode());
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::BracketRight) =>
            {
                self.flights.trail_subsample = (self.flights.trail_subsample + 1).min(20);
                eprintln!(
                    "Trail subsample: {} (every {}th point)",
                    self.flights.trail_subsample, self.flights.trail_subsample
                );
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::BracketLeft) =>
            {
                self.flights.trail_subsample = (self.flights.trail_subsample - 1).max(1);
                eprintln!(
                    "Trail subsample: {} (every {}th point)",
                    self.flights.trail_subsample, self.flights.trail_subsample
                );
            }
            WindowEvent::Resized(size) => {
                eprintln!("Resized: {}x{}", size.width, size.height);
                let restored = self.track_minimised(size);
                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.resize(size.width, size.height) {
                        eprintln!("Resize error: {e}");
                    }
                }
                if restored {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.minimised && self.renderer.is_some() {
            let now = self.start_time.elapsed().as_secs_f64();
            let _ = self.advance(now);
        }
        event_loop.set_control_flow(control_flow(self.minimised, Instant::now()));
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let frp_url = flight::frp_url(&args)?;
    if args.iter().any(|a| a == "--stream") {
        return stream::run_stream(&frp_url);
    }

    let windowed = args.iter().any(|a| a == "--windowed");

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
    let mut app = App::new(windowed, &frp_url).context("failed to initialize app")?;
    event_loop.run_app(&mut app).context("event loop error")?;
    match app.fatal.take() {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flightrelay::{
        BallFlight, EndpointConfig, FrpEndpoint, FrpEnvelope, FrpEvent, FrpMessage, Role, ShotKey,
        Transport, Velocity,
    };

    /// An app whose FRP device is never dialled (no I/O until `advance`).
    fn app() -> App {
        App::new(true, "ws://127.0.0.1:9/frp").expect("app")
    }

    #[test]
    fn a_zero_sized_window_is_minimised() {
        assert!(is_minimised(PhysicalSize::new(0, 0)));
        assert!(is_minimised(PhysicalSize::new(0, 1080)));
        assert!(is_minimised(PhysicalSize::new(1920, 0)));
        assert!(!is_minimised(PhysicalSize::new(1920, 1080)));
    }

    #[test]
    fn drawing_restarts_only_on_restore() {
        let mut app = app();
        let visible = PhysicalSize::new(1920, 1080);
        let hidden = PhysicalSize::new(0, 0);
        assert!(!app.track_minimised(visible), "resize while visible");
        assert!(!app.minimised);
        assert!(!app.track_minimised(hidden), "minimise");
        assert!(app.minimised);
        assert!(!app.track_minimised(hidden), "still minimised");
        assert!(app.track_minimised(visible), "restore");
        assert!(!app.minimised);
    }

    #[test]
    fn a_minimised_window_waits_on_a_timer() {
        let now = Instant::now();
        assert_eq!(
            control_flow(true, now),
            ControlFlow::WaitUntil(now + MINIMISED_TICK)
        );
        assert_eq!(control_flow(false, now), ControlFlow::Wait);
    }

    /// Shots that arrive while nothing is drawn launch when they arrive.
    #[test]
    fn advance_launches_shots_without_a_renderer() {
        let mut device = FrpEndpoint::open(EndpointConfig::new(
            Role::Device,
            Transport::listen("127.0.0.1:0"),
        ))
        .expect("device endpoint");
        let addr = device.local_addr().expect("addr").expect("bound");
        let server = std::thread::spawn(move || {
            let mut conn = device.establish().expect("handshake");
            let key = ShotKey {
                shot_id: "shot-1".to_owned(),
                shot_number: 1,
            };
            let ball = BallFlight {
                launch_speed: Some(Velocity::MilesPerHour(140.0)),
                launch_elevation: Some(14.0),
                backspin_rpm: Some(5000),
                sidespin_rpm: Some(0),
                ..BallFlight::default()
            };
            for event in [
                FrpEvent::shot_trigger(key.clone()),
                FrpEvent::ball_flight(key.clone(), ball),
                FrpEvent::ShotFinished { key },
            ] {
                conn.send(&FrpMessage::Envelope(FrpEnvelope::new("test.0", event)))
                    .expect("send");
            }
            conn
        });

        let mut app = App::new(true, &format!("ws://{addr}/frp")).expect("app");
        app.minimised = true;
        let start = Instant::now();
        let mut now = 0.0;
        while app.flights.flights.is_empty() && start.elapsed().as_secs() < 5 {
            now += 0.2;
            let _ = app.advance(now);
            std::thread::sleep(Duration::from_millis(5));
        }
        let _conn = server.join().expect("device thread");
        assert!(app.renderer.is_none());
        assert_eq!(app.flights.flights.len(), 1);
        let flight = &app.flights.flights[0];
        assert_eq!(flight.launch_time, now, "launched at the tick it arrived");

        // Later ticks animate it in game time; it lands while minimised.
        let landed = now + flight.total_time + 0.1;
        let _ = app.advance(landed);
        assert!(app.flights.flights[0].landed);
        assert_eq!(app.frame_count, 0, "nothing was drawn");
    }
}
