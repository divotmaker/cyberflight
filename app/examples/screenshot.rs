//! Headless screenshot generator for the driving range.
//!
//! Renders driving range views to PNG files using offscreen Vulkan rendering.
//! Run: `cargo run -p cyberflight --example screenshot [-- --out <dir>]`
//!
//! Output, in `screenshots/` or the `--out` directory:
//!   - `cyberflight.png` — **HERO**: the hero shot mid-flight from the tee
//!     box, chase camera + HUD, ray-traced when supported, 2× supersampled
//!   - `driving_range.png` — empty range with ball on tee box
//!     (the plain `.png` views have the floor's reflections mirrored)
//!   - `flight.png` — 7-iron mid-flight with tracer trail (close side view)
//!   - `flight_tee.png` — 7-iron mid-flight from tee box (golfer's view)
//!   - `flight_wide.png` — 7-iron full tracer in frame (wide side view)
//!   - `flight_wide_45.png` — 7-iron full tracer from 45° overhead angle
//!   - `flight_hud.png` — 7-iron mid-flight from tee box with telemetry HUD
//!   - `driving_range_rt.png`, `flight_rt.png`, `flight_tee_rt.png`,
//!     `flight_wide_rt.png`, `flight_wide_45_rt.png` — ray-traced versions,
//!     when the GPU supports ray tracing
//!   - `stream_flight.png` — the hero shot rendered through the streaming
//!     calls (`update_flight`, `update_hud`) on one renderer configured like
//!     `--stream` (ray-traced when supported)
//!   - `stream_idle.png` — the same renderer after the flight clears: no
//!     flight, no chase camera, idle HUD
//!
//! Compare two runs with `cargo run -p cf-test -- compare <dir_a> <dir_b>`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use glam::Vec3;
use image::{ImageBuffer, Rgba};

use cf_render::flight::FlightRenderData;
use cf_render::mode::{RenderMode, RtMode};
use cf_render::offscreen::OffscreenRenderer;
use cf_render::readback::Frame;
use cf_scene::camera::Camera;
use cf_scene::hud::build_hud;
use cf_scene::range::Range;
use cf_scene::shot::{ClubDelivery, ReceivedShot};
use cyberflight::flight::{AnimatedFlight, DEFAULT_TRAIL_SUBSAMPLE};
use dimple::Tour;
use dimple::trajectory::ShotInput;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
/// Hero screenshot renders at 2x then downscales for anti-aliasing.
const HERO_SCALE: u32 = 2;
/// Output directory used without `--out`, relative to the working directory.
const DEFAULT_OUTPUT_DIR: &str = "screenshots";

/// Output directory, set once from the command line in `main`.
static OUTPUT_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Path of the screenshot `name` in the output directory.
fn out_path(name: &str) -> String {
    let dir = OUTPUT_DIR
        .get()
        .map_or(Path::new(DEFAULT_OUTPUT_DIR), PathBuf::as_path);
    dir.join(name).display().to_string()
}

/// Parse `--out <dir>` (or `--out=<dir>`). Returns `None` for `--help`.
fn parse_out_dir(mut args: impl Iterator<Item = String>) -> Result<Option<PathBuf>> {
    let mut out = PathBuf::from(DEFAULT_OUTPUT_DIR);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--out" => match args.next() {
                Some(dir) => out = dir.into(),
                None => bail!("--out needs a directory"),
            },
            _ => match arg.strip_prefix("--out=") {
                Some(dir) => out = dir.into(),
                None => bail!("unknown argument '{arg}' (usage: screenshot [--out <dir>])"),
            },
        }
    }
    Ok(Some(out))
}

fn save_png(frame: &Frame<'_>, path: &str) -> Result<()> {
    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(frame.width, frame.height, frame.pixels.to_vec())
            .context("failed to create image buffer")?;
    img.save(path).context("failed to save PNG")?;

    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  Saved: {} ({WIDTH}x{HEIGHT}, {:.1} KB)",
        Path::new(path).display(),
        size as f64 / 1024.0
    );
    Ok(())
}

/// A shot as the app animates it: `input` launched at 0 s on `range`, and
/// the moment `frac` of its flight time in.
fn animate(range: &Range, input: ShotInput, frac: f64) -> (AnimatedFlight, f64) {
    let shot = ReceivedShot {
        device: "mevo.0".to_owned(),
        shot_number: 1,
        input,
        club: ClubDelivery {
            club_speed_mph: Some(input.ball_speed_mph / 1.33), // smash ~1.33
            path_deg: Some(-1.8),
            attack_angle_deg: Some(-4.2),
            face_angle_deg: Some(0.5),
            dynamic_loft_deg: Some(21.3),
            smash_factor: Some(1.33),
        },
        lm_carry_yards: None,
        backspin_measured: true,
        sidespin_measured: true,
        clamped: Vec::new(),
    };
    let flight = AnimatedFlight::from_received(&shot, 0.0, &Tour::SEA_LEVEL, range.green);
    let now = flight.result.flight.flight_time * frac;
    (flight, now)
}

/// The hero shot: a slight draw, sampled at 55 % of its flight (past the
/// apex, descending toward the target).
fn hero_shot(range: &Range) -> (AnimatedFlight, f64) {
    let input = ShotInput {
        ball_speed_mph: 110.0,
        launch_angle_deg: 18.0,
        launch_azimuth_deg: -1.0, // slight draw
        backspin_rpm: 6500.0,
        sidespin_rpm: -250.0, // draw spin
    };
    animate(range, input, 0.55)
}

/// A 7-iron sampled at 40 % of its flight (ascending, approaching the apex).
fn seven_iron(range: &Range) -> (AnimatedFlight, f64) {
    animate(range, ShotInput::seven_iron(), 0.40)
}

/// The renderers' flight input for `flight` at `now`, as the app draws it.
fn render_data(flight: &AnimatedFlight, now: f64) -> Result<FlightRenderData> {
    flight
        .render_data(now, DEFAULT_TRAIL_SUBSAMPLE)
        .context("the shot has not launched")
}

/// A camera at `position` looking at `target`, 45° field of view.
fn look_at(position: Vec3, target: Vec3) -> Camera {
    Camera {
        position,
        target,
        up: Vec3::Y,
        fov_y: std::f32::consts::FRAC_PI_4,
        near: 0.01,
        far: 600.0,
    }
}

/// One 1920×1080 screenshot: file name, camera, whether it may ray-trace,
/// and whether it shows the 7-iron and the HUD.
struct Shot {
    name: &'static str,
    camera: Camera,
    rt: RtMode,
    flight: bool,
    hud: bool,
}

/// The screenshots of the empty range and the 7-iron drawn as `data`: the
/// raster set (mirrored reflections), and the ray-traced set (`_rt`) for
/// GPUs that ray-trace.
fn range_shots(data: &FlightRenderData) -> (Vec<Shot>, Vec<Shot>) {
    use RtMode::{Auto, Off};

    let ball = data.ball_pos;
    // Side-on camera 8 m away to show the tracer arc clearly.
    let side = look_at(Vec3::new(8.0, ball.y, ball.z), ball);
    // Cameras that frame the whole tracer, centered on its midpoint: to the
    // side, and above and to the side (a 45° down angle).
    let tail_start = data.trail_points.first().map_or(Vec3::ZERO, |p| p.position);
    let mid = (tail_start + ball) * 0.5;
    let dist = (ball - tail_start).length() * 0.9;
    let wide = look_at(Vec3::new(mid.x + dist, mid.y, mid.z), mid);
    let wide_45 = look_at(
        Vec3::new(mid.x + dist * 0.7, mid.y + dist * 0.7, mid.z),
        mid,
    );
    let tee = Camera::driving_range();

    let table = |rows: &[(&'static str, Camera, RtMode, bool, bool)]| {
        rows.iter()
            .map(|&(name, camera, rt, flight, hud)| Shot {
                name,
                camera,
                rt,
                flight,
                hud,
            })
            .collect()
    };
    let raster = table(&[
        ("driving_range", tee, Off, false, false),
        ("flight", side, Off, true, false),
        ("flight_tee", tee, Off, true, false),
        ("flight_wide", wide, Off, true, false),
        ("flight_wide_45", wide_45, Off, true, false),
        ("flight_hud", tee, Off, true, true),
    ]);
    let ray_traced = table(&[
        ("driving_range_rt", tee, Auto, false, false),
        ("flight_rt", side, Auto, true, false),
        ("flight_tee_rt", tee, Auto, true, false),
        ("flight_wide_rt", wide, Auto, true, false),
        ("flight_wide_45_rt", wide_45, Auto, true, false),
    ]);
    (raster, ray_traced)
}

/// Render `shot` of `flight` at `now` (or of the empty range) on its own
/// renderer. Returns `false`, without rendering, when the shot asks for ray
/// tracing and the renderer does not ray-trace.
fn render_shot(range: &Range, flight: &AnimatedFlight, now: f64, shot: &Shot) -> Result<bool> {
    eprintln!("Rendering {}...", shot.name);
    let mut renderer = OffscreenRenderer::new(WIDTH, HEIGHT, &range.grid, shot.rt)
        .context("failed to create renderer")?;
    if shot.rt == RtMode::Auto && renderer.render_mode() != RenderMode::RayTraced {
        return Ok(false);
    }
    if shot.flight {
        renderer.update_flight(&[render_data(flight, now)?]);
    }
    renderer
        .set_green(range.green.as_ref())
        .context("failed to upload the green")?;
    if shot.hud {
        let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);
        let hud = build_hud(
            Some(&flight.telemetry(now)),
            false,
            &lm,
            true,
            WIDTH as f32,
            HEIGHT as f32,
        );
        renderer.update_hud(&hud.lines, &hud.fills);
    }
    let frame = renderer.render(&shot.camera).context("render failed")?;
    save_png(&frame, &out_path(&format!("{}.png", shot.name)))?;
    Ok(true)
}

/// The app's chase camera of `flight` at `now`.
fn chase_camera(flight: &mut AnimatedFlight, now: f64) -> Result<Camera> {
    flight
        .chase_camera(now)
        .context("the shot is not followed at this time")
}

fn render_hero(range: &Range, flight: &mut AnimatedFlight, now: f64) -> Result<()> {
    eprintln!("Rendering hero screenshot (tee box + chase camera + HUD, RT)...");

    let chase = chase_camera(flight, now)?;
    let main_camera = Camera::driving_range();

    // Render at 2x resolution for supersampled anti-aliasing, then downscale
    let render_w = WIDTH * HERO_SCALE;
    let render_h = HEIGHT * HERO_SCALE;

    let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);
    let hud = build_hud(
        Some(&flight.telemetry(now)),
        true,
        &lm,
        true,
        render_w as f32,
        render_h as f32,
    );

    // Reflections, main camera at the tee box, chase camera on the right 20%
    let mut renderer = OffscreenRenderer::new(render_w, render_h, &range.grid, RtMode::Auto)
        .context("failed to create hero renderer")?;
    renderer.update_flight(&[render_data(flight, now)?]);
    renderer
        .set_green(range.green.as_ref())
        .context("failed to upload the green")?;

    renderer.set_chase_camera(Some(chase));
    renderer.update_hud(&hud.lines, &hud.fills);

    let frame = renderer.render(&main_camera).context("render failed")?;

    // Downscale from render resolution to output resolution
    let hi_res: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(render_w, render_h, frame.pixels.to_vec())
            .context("failed to create hi-res buffer")?;
    let downscaled = image::imageops::resize(
        &hi_res,
        WIDTH,
        HEIGHT,
        image::imageops::FilterType::Lanczos3,
    );
    let path = out_path("cyberflight.png");
    downscaled.save(&path).context("failed to save PNG")?;
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "  Saved: {path} ({WIDTH}x{HEIGHT} from {render_w}x{render_h}, {:.1} KB)",
        size as f64 / 1024.0
    );
    Ok(())
}

/// Render the hero shot, then the idle range, through the streaming calls on
/// one renderer set up exactly like `--stream`.
fn render_stream(range: &Range, flight: &mut AnimatedFlight, now: f64) -> Result<()> {
    eprintln!("Rendering streaming frames (flight, then idle)...");
    let camera = Camera::driving_range();
    let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);

    let mut renderer = OffscreenRenderer::new(WIDTH, HEIGHT, &range.grid, RtMode::Auto)
        .context("failed to create stream renderer")?;
    renderer
        .set_green(range.green.as_ref())
        .context("failed to upload the green")?;

    renderer.update_flight(&[render_data(flight, now)?]);
    renderer.set_chase_camera(Some(chase_camera(flight, now)?));
    let hud = build_hud(
        Some(&flight.telemetry(now)),
        true,
        &lm,
        true,
        WIDTH as f32,
        HEIGHT as f32,
    );
    renderer.update_hud(&hud.lines, &hud.fills);
    let frame = renderer.render(&camera).context("render failed")?;
    save_png(&frame, &out_path("stream_flight.png"))?;

    renderer.update_flight(&[]);
    renderer.set_chase_camera(None);
    let hud = build_hud(None, false, &lm, true, WIDTH as f32, HEIGHT as f32);
    renderer.update_hud(&hud.lines, &hud.fills);
    let frame = renderer.render(&camera).context("render failed")?;
    save_png(&frame, &out_path("stream_idle.png"))
}

fn main() -> Result<()> {
    let Some(out_dir) = parse_out_dir(std::env::args().skip(1))? else {
        println!("usage: screenshot [--out <dir>]   (default: {DEFAULT_OUTPUT_DIR}/)");
        return Ok(());
    };
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;
    if OUTPUT_DIR.set(out_dir).is_err() {
        bail!("output directory set twice");
    }

    let range = Range::new();

    eprintln!("Initializing offscreen renderers ({WIDTH}x{HEIGHT})...");

    // ── Hero screenshot (tee box + chase camera + HUD) ──
    eprintln!("Simulating hero shot...");
    let (mut hero, hero_now) = hero_shot(&range);
    render_hero(&range, &mut hero, hero_now)?;

    // ── Rasterized screenshots ──
    eprintln!("Simulating 7-iron...");
    let (seven, now) = seven_iron(&range);
    let (raster, ray_traced) = range_shots(&render_data(&seven, now)?);
    for shot in &raster {
        render_shot(&range, &seven, now, shot)?;
    }

    // ── Ray-traced screenshots, when the first one's renderer ray-traces ──
    let mut ray_traced = ray_traced.iter();
    let first = ray_traced.next().context("no ray-traced screenshots")?;
    if render_shot(&range, &seven, now, first)? {
        eprintln!("Ray tracing supported — generating RT screenshots");
        for shot in ray_traced {
            if !render_shot(&range, &seven, now, shot)? {
                bail!("{}: ray tracing failed to set up", shot.name);
            }
        }
    } else {
        eprintln!("Ray tracing not supported — skipping RT screenshots");
    }

    // ── Streaming path (dynamic geometry, as `--stream` renders it) ──
    render_stream(&range, &mut hero, hero_now)?;

    Ok(())
}
