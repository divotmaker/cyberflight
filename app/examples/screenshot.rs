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
//!   - `hole_tee.png` — the hole range, idle, from its tee camera; with
//!     `hole_tee_fov30.png` (eye height, 30° field of view) and
//!     `hole_tee_raised.png` (8 m up, 14 m behind the tee, 40°) to compare
//!   - `hole_flight.png` — a 140 mph drive on the hole at 55 % of its
//!     flight, through the streaming calls, with HUD and chase camera
//!   - `hole_green.png` — the green, its bank and bunkers and the pin, from
//!     beside the fairway
//!   - `hole_hud.png` — the same drive at rest, with the hole's HUD rows
//!     (`TO PIN`, `LIE`) and the chase camera
//!   - `hole_holed.png` — a straight drive that drops into the cup, at rest
//!     (the first holed one scanning 150-185 mph in 0.05 mph steps)
//!   - `hole_water.png` — a pulled 120 mph drive at rest in the pond, from
//!     20 m short of the ball, over the fairway side
//!   - `hole_trees.png` — a 140 mph drive pushed 9° right into the right
//!     tree row, at rest under the tree, seen along the row
//!
//!   - `menu.png` — the startup menu over the idle driving range (raster)
//!
//! The hole images are raster (`RtMode::Off`): the hole draws no
//! reflections.
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
use cf_scene::frame::physics_to_scene;
use cf_scene::hud::{build_hud, build_menu};
use cf_scene::range::Range;
use cf_scene::shot::{ClubDelivery, ReceivedShot};
use cyberflight::flight::{AnimatedFlight, DEFAULT_TRAIL_SUBSAMPLE};
use cyberflight::menu::MENU_ITEMS;
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

/// The frame's RGBA pixels with every alpha byte set to 255. The framebuffer's alpha is what
/// blending left behind, not coverage; the window and the stream show every pixel opaque.
fn opaque(pixels: &[u8]) -> Vec<u8> {
    let mut pixels = pixels.to_vec();
    for px in pixels.as_chunks_mut::<4>().0 {
        px[3] = u8::MAX;
    }
    pixels
}

fn save_png(frame: &Frame<'_>, path: &str) -> Result<()> {
    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(frame.width, frame.height, opaque(frame.pixels))
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
        rolled: false,
    };
    let flight = AnimatedFlight::from_received(&shot, 0.0, &Tour::SEA_LEVEL, &range.ground);
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
    let mut renderer = OffscreenRenderer::new(WIDTH, HEIGHT, &range.scenery(), shot.rt)
        .context("failed to create renderer")?;
    if shot.rt == RtMode::Auto && renderer.render_mode() != RenderMode::RayTraced {
        return Ok(false);
    }
    if shot.flight {
        renderer.update_flight(&[render_data(flight, now)?]);
    }
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
    let mut renderer = OffscreenRenderer::new(render_w, render_h, &range.scenery(), RtMode::Auto)
        .context("failed to create hero renderer")?;
    renderer.update_flight(&[render_data(flight, now)?]);

    renderer.set_chase_camera(Some(chase));
    renderer.update_hud(&hud.lines, &hud.fills);

    let frame = renderer.render(&main_camera).context("render failed")?;

    // Downscale from render resolution to output resolution
    let hi_res: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(render_w, render_h, opaque(frame.pixels))
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

    let mut renderer = OffscreenRenderer::new(WIDTH, HEIGHT, &range.scenery(), RtMode::Auto)
        .context("failed to create stream renderer")?;

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

/// Render the hole range: the idle tee cameras and the green, then a drive
/// through the streaming calls.
fn render_hole() -> Result<()> {
    let range = Range::hole();
    let pin = range.camera.target;
    let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);
    let mut renderer = OffscreenRenderer::new(WIDTH, HEIGHT, &range.scenery(), RtMode::Off)
        .context("failed to create hole renderer")?;
    let idle = build_hud(None, false, &lm, true, WIDTH as f32, HEIGHT as f32);
    renderer.update_hud(&idle.lines, &idle.fills);

    let fov30 = Camera {
        fov_y: 30f32.to_radians(),
        ..range.camera
    };
    let raised = Camera {
        position: Vec3::new(0.0, 8.0, -14.0),
        fov_y: 40f32.to_radians(),
        ..range.camera
    };
    let green = look_at(Vec3::new(-30.0, 4.0, 235.0), pin);
    for (name, camera) in [
        ("hole_tee", range.camera),
        ("hole_tee_fov30", fov30),
        ("hole_tee_raised", raised),
        ("hole_green", green),
    ] {
        eprintln!("Rendering {name}...");
        let frame = renderer.render(&camera).context("render failed")?;
        save_png(&frame, &out_path(&format!("{name}.png")))?;
    }

    eprintln!("Rendering hole_flight...");
    let drive = ShotInput {
        ball_speed_mph: 140.0,
        launch_angle_deg: 13.0,
        launch_azimuth_deg: 0.0,
        backspin_rpm: 3000.0,
        sidespin_rpm: 0.0,
    };
    let (mut flight, now) = animate(&range, drive, 0.55);
    renderer.update_flight(&[render_data(&flight, now)?]);
    renderer.set_chase_camera(Some(chase_camera(&mut flight, now)?));
    let hud = build_hud(
        Some(&flight.telemetry(now)),
        true,
        &lm,
        true,
        WIDTH as f32,
        HEIGHT as f32,
    );
    renderer.update_hud(&hud.lines, &hud.fills);
    let frame = renderer.render(&range.camera).context("render failed")?;
    save_png(&frame, &out_path("hole_flight.png"))?;

    eprintln!("Rendering hole_hud...");
    render_at_rest(&mut renderer, range.camera, &mut flight, "hole_hud")?;

    eprintln!("Finding a holed drive...");
    let mut holed = (0..=700)
        .map(|i| ShotInput {
            ball_speed_mph: 150.0 + 0.05 * f64::from(i),
            launch_angle_deg: 11.0,
            launch_azimuth_deg: 0.0,
            backspin_rpm: 2500.0,
            sidespin_rpm: 0.0,
        })
        .map(|input| animate(&range, input, 0.0).0)
        .find(AnimatedFlight::holed)
        .context("no holed drive from 150 to 185 mph")?;
    eprintln!("Rendering hole_holed...");
    render_at_rest(&mut renderer, range.camera, &mut holed, "hole_holed")?;

    eprintln!("Rendering hole_water...");
    let pulled = ShotInput {
        ball_speed_mph: 120.0,
        launch_azimuth_deg: -9.0,
        ..drive
    };
    let mut wet = animate(&range, pulled, 0.0).0;
    let ball = physics_to_scene(wet.rest_pos).as_vec3();
    let pond = look_at(ball + Vec3::new(-12.0, 4.0, -16.0), ball);
    render_at_rest(&mut renderer, pond, &mut wet, "hole_water")?;

    eprintln!("Rendering hole_trees...");
    let pushed = ShotInput {
        launch_azimuth_deg: 9.0,
        ..drive
    };
    let mut treed = animate(&range, pushed, 0.0).0;
    let row = look_at(Vec3::new(-20.0, 0.0, 110.0), Vec3::new(-33.0, -8.0, 220.0));
    render_at_rest(&mut renderer, row, &mut treed, "hole_trees")
}

/// Render the startup menu over the idle driving range, as the windowed app
/// draws it before a range is picked.
fn render_menu(range: &Range) -> Result<()> {
    eprintln!("Rendering menu...");
    let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);
    let mut renderer = OffscreenRenderer::new(WIDTH, HEIGHT, &range.scenery(), RtMode::Off)
        .context("failed to create menu renderer")?;
    let mut hud = build_hud(None, false, &lm, true, WIDTH as f32, HEIGHT as f32);
    let panel = build_menu(&MENU_ITEMS, 0, WIDTH as f32, HEIGHT as f32);
    hud.lines.extend(panel.lines);
    hud.fills.extend(panel.fills);
    renderer.update_hud(&hud.lines, &hud.fills);
    let frame = renderer.render(&range.camera).context("render failed")?;
    save_png(&frame, &out_path("menu.png"))
}

/// Render `flight` just after it stopped, from `camera` with its HUD and
/// chase camera, as the screenshot `name`.
fn render_at_rest(
    renderer: &mut OffscreenRenderer,
    camera: Camera,
    flight: &mut AnimatedFlight,
    name: &str,
) -> Result<()> {
    let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);
    let now = flight.total_time + 0.5;
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
    save_png(&frame, &out_path(&format!("{name}.png")))
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

    // ── The startup menu, then the hole range ──
    render_menu(&range)?;
    render_hole()
}
