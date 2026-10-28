mod flight;
mod stream;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use cf_render::context::{GpuConfig, GpuContext};
use cf_render::mode::RenderMode;
use cf_render::render::Renderer;
use cf_render::window::Swapchain;
use cf_scene::hud::build_notice;
use cf_scene::range::Range;
use dimple::environment::Environment;

use flight::{FlightManager, FrpState};

/// How often a minimised window wakes to take in shots and advance the
/// flights without drawing.
const MINIMISED_TICK: Duration = Duration::from_millis(200);

/// How long the notice of a key press stays on screen.
const NOTICE_DURATION: Duration = Duration::from_millis(2500);

/// The reflections mode a press of the reflections key (`R`) switches to.
fn next_reflections(current: RenderMode) -> RenderMode {
    match current {
        RenderMode::Rasterized => RenderMode::RayTraced,
        RenderMode::RayTraced => RenderMode::Rasterized,
    }
}

/// The notice that says which reflections are drawn, after a press of the
/// reflections key left the renderer in `mode`.
fn reflections_notice(mode: RenderMode, ray_tracing_available: bool) -> &'static str {
    match (mode, ray_tracing_available) {
        (RenderMode::RayTraced, _) => "REFLECTIONS: RAY TRACED",
        (RenderMode::Rasterized, true) => "REFLECTIONS: MIRROR",
        (RenderMode::Rasterized, false) => "REFLECTIONS: MIRROR - RAY TRACING UNAVAILABLE",
    }
}

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
    /// The notice shown over the scene and when it was raised.
    notice: Option<(&'static str, Instant)>,
    /// Why the event loop was stopped, when the app could not run.
    fatal: Option<anyhow::Error>,
}

impl App {
    /// The app, taking in shots from `frp`, whose arrival times are seconds
    /// since `start_time`.
    fn new(windowed: bool, frp: FrpState, start_time: Instant) -> Self {
        Self {
            windowed,
            window: None,
            renderer: None,
            range: Range::new(),
            environment: Environment::SEA_LEVEL,
            frp,
            flights: FlightManager::new(),
            start_time,
            frame_count: 0,
            fps_timer: Instant::now(),
            minimised: false,
            notice: None,
            fatal: None,
        }
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

    /// Launch the shots that have arrived, each from its arrival time, as of
    /// `now` (seconds since start).
    fn take_in(&mut self, now: f64) {
        for shot in self.frp.poll() {
            self.flights
                .take_in(&shot, now, &self.environment, self.range.green);
        }
    }

    /// Take in shots and advance the flights to `now` (seconds since
    /// start), without touching the renderer.
    fn advance(&mut self, now: f64) -> Vec<cf_render::flight::FlightRenderData> {
        self.take_in(now);
        self.flights.update(now)
    }

    fn update(&mut self) {
        let now = self.start_time.elapsed().as_secs_f64();

        let render_data = self.advance(now);
        let chase_camera = self.flights.chase_camera(now);

        if let Some(renderer) = &mut self.renderer {
            renderer.update_flight(&render_data);
            renderer.set_chase_camera(chase_camera);

            let screen_w = renderer.extent().width as f32;
            let screen_h = renderer.extent().height as f32;
            let hud = self.flights.build_hud(
                now,
                chase_camera.is_some(),
                &self.frp.lm_states,
                self.frp.connected,
                screen_w,
                screen_h,
            );
            let mut lines = hud.lines;
            let mut fills = hud.fills;
            self.notice
                .take_if(|(_, raised)| raised.elapsed() >= NOTICE_DURATION);
            if let Some((text, _)) = self.notice {
                let notice = build_notice(text, screen_w, screen_h);
                lines.extend(notice.lines);
                fills.extend(notice.fills);
            }
            renderer.update_hud(&lines, &fills);
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
                    && !event.repeat
                    && event.physical_key == PhysicalKey::Code(KeyCode::KeyR) =>
            {
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_render_mode(next_reflections(renderer.render_mode()));
                    let text = reflections_notice(
                        renderer.render_mode(),
                        renderer.ray_tracing_available(),
                    );
                    eprintln!("{text}");
                    self.notice = Some((text, Instant::now()));
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

    /// The FRP intake woke the loop: a shot arrived or the connection
    /// changed.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, (): ()) {
        let now = self.start_time.elapsed().as_secs_f64();
        self.take_in(now);
        if self.renderer.is_some() && !self.minimised {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = self.start_time.elapsed().as_secs_f64();
        if self.minimised && self.renderer.is_some() {
            let _ = self.advance(now);
        } else {
            self.take_in(now);
        }
        event_loop.set_control_flow(control_flow(self.minimised, Instant::now()));
    }
}

/// Wake the event loop from the FRP intake thread.
fn wake_on_intake(proxy: EventLoopProxy<()>) -> impl Fn() + Send + 'static {
    move || {
        // Fails only once the event loop has exited.
        let _ = proxy.send_event(());
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
    let start_time = Instant::now();
    let frp = FrpState::new(
        &frp_url,
        start_time,
        wake_on_intake(event_loop.create_proxy()),
    )
    .context("failed to start the FRP intake")?;
    let mut app = App::new(windowed, frp, start_time);
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

    /// An app whose FRP device at `url` is dialled from now.
    fn app_of(url: &str) -> App {
        let start = Instant::now();
        let frp = FrpState::new(url, start, || {}).expect("intake");
        App::new(true, frp, start)
    }

    /// An app whose FRP device refuses the connection.
    fn app() -> App {
        app_of("ws://127.0.0.1:9/frp")
    }

    #[test]
    fn the_reflections_key_alternates_the_modes() {
        assert_eq!(
            next_reflections(RenderMode::RayTraced),
            RenderMode::Rasterized
        );
        assert_eq!(
            next_reflections(RenderMode::Rasterized),
            RenderMode::RayTraced
        );
    }

    #[test]
    fn the_notice_names_the_reflections_drawn() {
        assert_eq!(
            reflections_notice(RenderMode::RayTraced, true),
            "REFLECTIONS: RAY TRACED"
        );
        assert_eq!(
            reflections_notice(RenderMode::Rasterized, true),
            "REFLECTIONS: MIRROR"
        );
        // Pressing the key without a ray tracer stays on the mirror and says why.
        let stays = next_reflections(RenderMode::Rasterized);
        assert_eq!(stays, RenderMode::RayTraced, "the key asks for ray tracing");
        assert_eq!(
            reflections_notice(RenderMode::Rasterized, false),
            "REFLECTIONS: MIRROR - RAY TRACING UNAVAILABLE"
        );
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

    /// Shots that arrive while nothing is drawn launch when they arrived.
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

        let mut app = app_of(&format!("ws://{addr}/frp"));
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
        // The ticks run ahead of the app's clock, so the tick that took the
        // shot in is later than its arrival.
        assert!(flight.launch_time > 0.0);
        assert!(
            flight.launch_time < now,
            "launched at its arrival, not the tick"
        );

        // Later ticks animate it in game time; it lands while minimised.
        let landed = flight.launch_time + flight.total_time + 0.1;
        let _ = app.advance(landed);
        assert!(app.flights.flights[0].landed);
        assert_eq!(app.frame_count, 0, "nothing was drawn");
    }
}
