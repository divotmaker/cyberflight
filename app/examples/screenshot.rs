//! Headless screenshot generator for the driving range.
//!
//! Renders driving range views to PNG files using offscreen Vulkan rendering.
//! Run: `cargo run -p cyberflight --example screenshot [-- --out <dir>]`
//!
//! Output, in `screenshots/` or the `--out` directory:
//!   - `cyberflight.png` — **HERO**: the hero shot mid-flight from the tee
//!     box, chase camera + HUD, ray-traced when supported, 2× supersampled
//!   - `driving_range.png` — empty range with ball on tee box
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
use cf_scene::frame::physics_to_scene;
use cf_scene::hud::{ShotTelemetry, UnitSystem, build_hud};
use cf_scene::range::Range;
use cf_scene::trail::{TrailPoint, tracer_tail};
use dimple::aero::BallModel;
use dimple::environment::Environment;
use dimple::trajectory::{ShotInput, simulate_flight};

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

/// A dimple-frame trajectory point as a scene-frame render position.
fn traj_to_scene(p: glam::DVec3) -> Vec3 {
    physics_to_scene(p).as_vec3()
}

/// Every how many trajectory samples (10 ms apart) a tracer point is kept:
/// 30 ms apart, so the glow ribbons don't overlap additively.
const TRAIL_SUBSAMPLE: usize = 3;

/// Simulated flight snapshot: ball position, tracer tail, and shot data for HUD.
struct FlightSnapshot {
    ball_pos: Vec3,
    trail_points: Vec<TrailPoint>,
    input: ShotInput,
    carry_yards: f64,
    apex_m: f64,
    lateral_m: f64,
    flight_time: f64,
    elapsed: f64,
}

impl FlightSnapshot {
    /// The snapshot as the renderers' per-frame flight input.
    fn render_data(&self) -> FlightRenderData {
        FlightRenderData {
            ball_pos: self.ball_pos,
            trail_points: self.trail_points.clone(),
            current_time: self.elapsed,
        }
    }

    /// Where the tracer starts (its oldest point).
    fn tail_start(&self) -> Vec3 {
        self.trail_points
            .first()
            .map(|p| p.position)
            .unwrap_or(Vec3::ZERO)
    }
}

/// Simulate `input` (flight phase only) and freeze it at `frac` of its
/// flight time, with the tracer tail the live renderers draw.
fn snapshot(input: ShotInput, frac: f64) -> FlightSnapshot {
    let result = simulate_flight(&input, &BallModel::TOUR, &Environment::SEA_LEVEL);
    eprintln!(
        "  Flight: {:.1} yards carry, {:.1}m apex, {:.2}s total",
        result.carry_yards, result.apex_m, result.flight_time
    );

    let t_current = result.flight_time * frac;
    let ball_pos = dimple::trajectory::interpolate_trajectory(&result.points, t_current)
        .map(traj_to_scene)
        .unwrap_or(Vec3::ZERO);
    let samples = result.points.iter().map(|p| TrailPoint {
        position: traj_to_scene(p.pos),
        time: p.time,
    });
    let trail_points = tracer_tail(samples, t_current, ball_pos, TRAIL_SUBSAMPLE);

    eprintln!(
        "  Ball at ({:.1}, {:.1}, {:.1}) with {} trail points",
        ball_pos.x,
        ball_pos.y,
        ball_pos.z,
        trail_points.len()
    );

    FlightSnapshot {
        ball_pos,
        trail_points,
        input,
        carry_yards: result.carry_yards,
        apex_m: result.apex_m,
        lateral_m: result.lateral_m,
        flight_time: result.flight_time,
        elapsed: t_current,
    }
}

/// The hero shot: a slight draw, sampled at 55 % of its flight (past the
/// apex, descending toward the target).
fn hero_shot() -> FlightSnapshot {
    let input = ShotInput {
        ball_speed_mph: 110.0,
        launch_angle_deg: 18.0,
        launch_azimuth_deg: -1.0, // slight draw
        backspin_rpm: 6500.0,
        sidespin_rpm: -250.0, // draw spin
    };
    snapshot(input, 0.55)
}

/// A 7-iron sampled at 40 % of its flight (ascending, approaching the apex).
fn seven_iron() -> FlightSnapshot {
    snapshot(ShotInput::seven_iron(), 0.40)
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
/// and whether it shows the 7-iron snapshot and the HUD.
struct Shot {
    name: &'static str,
    camera: Camera,
    rt: RtMode,
    flight: bool,
    hud: bool,
}

/// The screenshots of the empty range and the 7-iron `snap`: the raster
/// set, and the ray-traced set (`_rt`) for GPUs that ray-trace.
fn range_shots(snap: &FlightSnapshot) -> (Vec<Shot>, Vec<Shot>) {
    use RtMode::{Auto, Off};

    // Side-on camera 8 m away to show the tracer arc clearly.
    let side = look_at(
        Vec3::new(8.0, snap.ball_pos.y, snap.ball_pos.z),
        snap.ball_pos,
    );
    // Cameras that frame the whole tracer, centred on its midpoint: to the
    // side, and above and to the side (a 45° down angle).
    let tail_start = snap.tail_start();
    let mid = (tail_start + snap.ball_pos) * 0.5;
    let dist = (snap.ball_pos - tail_start).length() * 0.9;
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

/// Render `shot` of `snap` (or of the empty range) on its own renderer.
/// Returns `false`, without rendering, when the shot asks for ray tracing
/// and the renderer does not ray-trace.
fn render_shot(range: &Range, snap: &FlightSnapshot, shot: &Shot) -> Result<bool> {
    eprintln!("Rendering {}...", shot.name);
    let mut renderer = OffscreenRenderer::new(WIDTH, HEIGHT, &range.grid, shot.rt)
        .context("failed to create renderer")?;
    if shot.rt == RtMode::Auto && renderer.render_mode() != RenderMode::RayTraced {
        return Ok(false);
    }
    if shot.flight {
        renderer.update_flight(&[snap.render_data()], &shot.camera);
    }
    renderer
        .set_green(range.green.as_ref())
        .context("failed to upload the green")?;
    if shot.hud {
        let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);
        let hud = build_hud(
            Some(&telemetry(snap, false)),
            UnitSystem::Imperial,
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

/// Chase camera of the hero shot: behind and above the ball, looking toward the landing zone.
fn hero_chase_camera(snap: &FlightSnapshot) -> Camera {
    let velocity = Vec3::new(
        -snap.input.launch_azimuth_deg.to_radians().sin() as f32,
        0.0,
        snap.input.launch_azimuth_deg.to_radians().cos() as f32,
    ) * snap.input.ball_speed_mph as f32;
    let landing_pos = snap
        .trail_points
        .last()
        .map(|p| p.position)
        .unwrap_or(Vec3::new(0.0, 0.0, 200.0));
    Camera::chase(snap.ball_pos, velocity, landing_pos)
}

/// HUD telemetry of `snap`, with the full carry, or with `live` the carry
/// so far.
fn telemetry(snap: &FlightSnapshot, live: bool) -> ShotTelemetry {
    let carry_yards = if live {
        snap.carry_yards * (snap.elapsed / snap.flight_time)
    } else {
        snap.carry_yards
    };
    ShotTelemetry {
        club_speed_mph: Some(snap.input.ball_speed_mph / 1.33), // smash ~1.33
        ball_speed_mph: snap.input.ball_speed_mph,
        smash_factor: Some(1.33),
        launch_angle_deg: snap.input.launch_angle_deg,
        launch_azimuth_deg: snap.input.launch_azimuth_deg,
        backspin_rpm: Some(snap.input.backspin_rpm),
        sidespin_rpm: Some(snap.input.sidespin_rpm),
        club_path_deg: Some(-1.8),
        attack_angle_deg: Some(-4.2),
        face_angle_deg: Some(0.5),
        dynamic_loft_deg: Some(21.3),
        apex_m: snap.apex_m,
        carry_yards,
        lm_carry_yards: None,
        total_yards: carry_yards,
        downrange_yards: carry_yards,
        lateral_yards: dimple::units::meters_to_yards(snap.lateral_m),
        flight_time_s: snap.flight_time,
        elapsed_s: snap.elapsed,
    }
}

fn render_hero(range: &Range, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering hero screenshot (tee box + chase camera + HUD, RT)...");

    let chase = hero_chase_camera(snap);
    let main_camera = Camera::driving_range();

    // Render at 2x resolution for supersampled anti-aliasing, then downscale
    let render_w = WIDTH * HERO_SCALE;
    let render_h = HEIGHT * HERO_SCALE;

    let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);
    let hud = build_hud(
        Some(&telemetry(snap, true)),
        UnitSystem::Imperial,
        true,
        &lm,
        true,
        render_w as f32,
        render_h as f32,
    );

    // Reflections, main camera at the tee box, chase camera on the right 20%
    let mut renderer = OffscreenRenderer::new(render_w, render_h, &range.grid, RtMode::Auto)
        .context("failed to create hero renderer")?;
    renderer.update_flight(&[snap.render_data()], &main_camera);
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
fn render_stream(range: &Range, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering streaming frames (flight, then idle)...");
    let camera = Camera::driving_range();
    let lm = std::collections::HashMap::from([("mevo.0".to_owned(), true)]);

    let mut renderer = OffscreenRenderer::new(WIDTH, HEIGHT, &range.grid, RtMode::Auto)
        .context("failed to create stream renderer")?;
    renderer
        .set_green(range.green.as_ref())
        .context("failed to upload the green")?;

    renderer.update_flight(&[snap.render_data()], &camera);
    renderer.set_chase_camera(Some(hero_chase_camera(snap)));
    let hud = build_hud(
        Some(&telemetry(snap, true)),
        UnitSystem::Imperial,
        true,
        &lm,
        true,
        WIDTH as f32,
        HEIGHT as f32,
    );
    renderer.update_hud(&hud.lines, &hud.fills);
    let frame = renderer.render(&camera).context("render failed")?;
    save_png(&frame, &out_path("stream_flight.png"))?;

    renderer.update_flight(&[], &camera);
    renderer.set_chase_camera(None);
    let hud = build_hud(
        None,
        UnitSystem::Imperial,
        false,
        &lm,
        true,
        WIDTH as f32,
        HEIGHT as f32,
    );
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
    let hero_snap = hero_shot();
    render_hero(&range, &hero_snap)?;

    // ── Rasterized screenshots ──
    eprintln!("Simulating 7-iron...");
    let snap = seven_iron();
    let (raster, ray_traced) = range_shots(&snap);
    for shot in &raster {
        render_shot(&range, &snap, shot)?;
    }

    // ── Ray-traced screenshots, when the first one's renderer ray-traces ──
    let mut ray_traced = ray_traced.iter();
    let first = ray_traced.next().context("no ray-traced screenshots")?;
    if render_shot(&range, &snap, first)? {
        eprintln!("Ray tracing supported — generating RT screenshots");
        for shot in ray_traced {
            if !render_shot(&range, &snap, shot)? {
                bail!("{}: ray tracing failed to set up", shot.name);
            }
        }
    } else {
        eprintln!("Ray tracing not supported — skipping RT screenshots");
    }

    // ── Streaming path (dynamic geometry, as `--stream` renders it) ──
    render_stream(&range, &hero_snap)?;

    Ok(())
}
