mod flight;
mod stream;

use std::time::Instant;

use anyhow::{Context, Result};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use cf_render::context::{GpuConfig, GpuContext};
use cf_render::render::Renderer;
use cf_render::window::Swapchain;
use cf_scene::range::Range;
use dimple::environment::Environment;

use flight::{FlightManager, FrpState};

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
            fatal: None,
        })
    }

    fn init_renderer(&mut self, window: &Window) -> Result<()> {
        let size = window.inner_size();
        let config = GpuConfig {
            width: size.width,
            height: size.height,
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
            .set_targets(&self.range.targets, &self.range.grid)
            .context("failed to upload target rings")?;
        eprintln!(
            "Renderer initialized: {}x{}",
            renderer.swapchain.extent.width, renderer.swapchain.extent.height
        );
        self.renderer = Some(renderer);
        Ok(())
    }

    /// Stop the event loop because the app cannot run; `main` returns `err`.
    fn fail(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        eprintln!("ERROR: {err:#}");
        self.fatal = Some(err);
        event_loop.exit();
    }

    fn update(&mut self) {
        let now = self.start_time.elapsed().as_secs_f64();

        for shot in self.frp.poll(now) {
            self.flights
                .launch(&shot, now, &self.environment, &self.range.targets);
        }

        let render_data = self.flights.update(now);
        let chase_camera = self.flights.chase_camera(now);

        if let Some(renderer) = &mut self.renderer {
            renderer.update_flight_geometry(&render_data, &self.range.camera);
            renderer.set_chase_camera(chase_camera);

            let screen_w = renderer.swapchain.extent.width as f32;
            let screen_h = renderer.swapchain.extent.height as f32;
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
            // (the app is exiting; see `fail`).
            WindowEvent::RedrawRequested if self.renderer.is_some() => {
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
    let args: Vec<String> = std::env::args().collect();
    let frp_url = flight::frp_url(&args);
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
