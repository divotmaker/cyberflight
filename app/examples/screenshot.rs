//! Headless screenshot generator for the driving range.
//!
//! Renders driving range views to PNG files using offscreen Vulkan rendering.
//! Run: `cargo run -p cyberflight --example screenshot`
//!
//! Output:
//!   - `screenshots/driving_range.png` — empty range with ball on tee box
//!   - `screenshots/flight.png` — 7-iron mid-flight with tracer trail (close side view)
//!   - `screenshots/flight_tee.png` — 7-iron mid-flight from tee box (golfer's view)
//!   - `screenshots/flight_wide.png` — 7-iron full tracer in frame (wide side view)
//!   - `screenshots/flight_wide_45.png` — 7-iron full tracer from 45° overhead angle
//!   - `screenshots/flight_hud.png` — 7-iron mid-flight from tee box with telemetry HUD

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use glam::Vec3;
use image::{ImageBuffer, Rgba};

use cf_math::aero::BallModel;
use cf_math::environment::Environment;
use cf_math::trajectory::{ShotInput, simulate_flight};
use cf_render::offscreen::OffscreenRenderer;
use cf_render::readback::FrameBuffers;
use cf_scene::camera::Camera;
use cf_scene::grid::GridConfig;
use cf_scene::hud::{ShotTelemetry, UnitSystem, build_hud};
use cf_scene::trail::{DEFAULT_TRAIL_LIFETIME, TrailPoint};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const OUTPUT_DIR: &str = "screenshots";

fn save_png(frame: &FrameBuffers, path: &str) -> Result<()> {
    let img: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::from_raw(frame.width, frame.height, frame.color.clone())
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

/// Map trajectory coordinates (x=forward, y=up, z=lateral) to
/// scene coordinates (+Z=downrange, +Y=up, +X=lateral).
fn traj_to_scene(p: glam::DVec3) -> Vec3 {
    Vec3::new(p.z as f32, p.y as f32, p.x as f32)
}

fn render_driving_range(grid: &GridConfig, camera: &Camera) -> Result<()> {
    eprintln!("Rendering driving range...");
    let renderer =
        OffscreenRenderer::new(WIDTH, HEIGHT, grid).context("failed to create renderer")?;
    let frame = renderer.render(camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/driving_range.png"))
}

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
    /// Extract just the positions (for RT renderer which doesn't need timestamps).
    fn trail_positions(&self) -> Vec<Vec3> {
        self.trail_points.iter().map(|p| p.position).collect()
    }
}

fn simulate_seven_iron() -> FlightSnapshot {
    let input = ShotInput::seven_iron();
    let result = simulate_flight(&input, &BallModel::TOUR, &Environment::SEA_LEVEL);
    eprintln!(
        "  Flight: {:.1} yards carry, {:.1}m apex, {:.2}s total",
        result.carry_yards, result.apex_m, result.flight_time
    );

    // Sample at 40% of flight time (ascending, approaching apex)
    let t_current = result.flight_time * 0.40;

    let ball_pos = cf_math::trajectory::interpolate_trajectory(&result.points, t_current)
        .map(|p| traj_to_scene(p))
        .unwrap_or(Vec3::ZERO);

    // Tracer tail: last 120% of elapsed flight time (long tail visible at distance).
    // Subsample every 3rd point (30ms intervals) so ribbons don't overlap additively.
    let tail_duration = t_current * 1.2;
    let tail_start = (t_current - tail_duration).max(0.0);
    let mut trail_points: Vec<TrailPoint> = result
        .points
        .iter()
        .filter(|p| p.time >= tail_start && p.time <= t_current)
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, p)| TrailPoint {
            position: traj_to_scene(p.pos),
            time: p.time,
        })
        .collect();
    // Ensure the trail ends exactly at the ball (connects to ball glow).
    if trail_points.last().map_or(true, |last| (last.position - ball_pos).length() > 0.01) {
        trail_points.push(TrailPoint { position: ball_pos, time: t_current });
    }

    eprintln!(
        "  Ball at ({:.1}, {:.1}, {:.1}) with {} trail points",
        ball_pos.x, ball_pos.y, ball_pos.z, trail_points.len()
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

fn render_flight(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (side view)...");

    // Side-on camera 8m away to show the tracer arc clearly.
    let flight_camera = Camera {
        position: Vec3::new(8.0, snap.ball_pos.y, snap.ball_pos.z),
        target: snap.ball_pos,
        up: Vec3::Y,
        fov_y: std::f32::consts::FRAC_PI_4,
        near: 0.01,
        far: 600.0,
    };

    let renderer = OffscreenRenderer::new_with_flight(
        WIDTH, HEIGHT, grid, snap.ball_pos, &snap.trail_points, snap.elapsed, DEFAULT_TRAIL_LIFETIME, &flight_camera,
    )
    .context("failed to create flight renderer")?;

    let frame = renderer.render(&flight_camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight.png"))
}

fn render_flight_wide(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (wide side view)...");

    // Camera to the side, far enough to frame the entire tracer + ball.
    // Trail goes from oldest point to ball_pos — center the view on the midpoint.
    let tail_start = snap.trail_points.first().map(|p| p.position).unwrap_or(Vec3::ZERO);
    let mid = (tail_start + snap.ball_pos) * 0.5;
    let trail_len = (snap.ball_pos - tail_start).length();
    // Distance = enough to fit the trail in frame with 45° FOV
    let dist = trail_len * 0.9;

    let wide_camera = Camera {
        position: Vec3::new(mid.x + dist, mid.y, mid.z),
        target: mid,
        up: Vec3::Y,
        fov_y: std::f32::consts::FRAC_PI_4,
        near: 0.01,
        far: 600.0,
    };

    let renderer = OffscreenRenderer::new_with_flight(
        WIDTH, HEIGHT, grid, snap.ball_pos, &snap.trail_points, snap.elapsed, DEFAULT_TRAIL_LIFETIME, &wide_camera,
    )
    .context("failed to create flight renderer")?;

    let frame = renderer.render(&wide_camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight_wide.png"))
}

fn render_flight_wide_45(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (wide 45° overhead view)...");

    // Camera above and to the side, looking down at ~45° onto the flight arc.
    let tail_start = snap.trail_points.first().map(|p| p.position).unwrap_or(Vec3::ZERO);
    let mid = (tail_start + snap.ball_pos) * 0.5;
    let trail_len = (snap.ball_pos - tail_start).length();
    let dist = trail_len * 0.9;

    // Position: offset laterally (+X) and above (+Y) by equal amounts → 45° down angle
    let wide45_camera = Camera {
        position: Vec3::new(mid.x + dist * 0.7, mid.y + dist * 0.7, mid.z),
        target: mid,
        up: Vec3::Y,
        fov_y: std::f32::consts::FRAC_PI_4,
        near: 0.01,
        far: 600.0,
    };

    let renderer = OffscreenRenderer::new_with_flight(
        WIDTH, HEIGHT, grid, snap.ball_pos, &snap.trail_points, snap.elapsed, DEFAULT_TRAIL_LIFETIME, &wide45_camera,
    )
    .context("failed to create flight renderer")?;

    let frame = renderer.render(&wide45_camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight_wide_45.png"))
}

fn render_flight_tee(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (tee box view)...");

    let renderer = OffscreenRenderer::new_with_flight(
        WIDTH,
        HEIGHT,
        grid,
        snap.ball_pos,
        &snap.trail_points,
        snap.elapsed,
        DEFAULT_TRAIL_LIFETIME,
        &Camera::driving_range(),
    )
    .context("failed to create flight renderer")?;

    let frame = renderer
        .render(&Camera::driving_range())
        .context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight_tee.png"))
}

fn render_flight_hud(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (tee box view + HUD)...");

    let telemetry = ShotTelemetry {
        club_name: "7I".to_owned(),
        club_speed_mph: snap.input.ball_speed_mph / 1.33,
        ball_speed_mph: snap.input.ball_speed_mph,
        launch_angle_deg: snap.input.launch_angle_deg,
        launch_azimuth_deg: snap.input.launch_azimuth_deg,
        backspin_rpm: snap.input.backspin_rpm,
        sidespin_rpm: snap.input.sidespin_rpm,
        apex_m: snap.apex_m,
        carry_yards: snap.carry_yards,
        lateral_m: snap.lateral_m,
        flight_time_s: snap.flight_time,
        elapsed_s: snap.elapsed,
        in_flight: true,
    };

    let hud = build_hud(Some(&telemetry), UnitSystem::Imperial, WIDTH as f32, HEIGHT as f32);

    let mut renderer = OffscreenRenderer::new_with_flight(
        WIDTH,
        HEIGHT,
        grid,
        snap.ball_pos,
        &snap.trail_points,
        snap.elapsed,
        DEFAULT_TRAIL_LIFETIME,
        &Camera::driving_range(),
    )
    .context("failed to create flight renderer")?;

    renderer
        .set_hud(&hud.lines, &hud.fills)
        .context("failed to set HUD")?;

    let frame = renderer
        .render(&Camera::driving_range())
        .context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight_hud.png"))
}

/// Check if RT rendering is available by attempting to create an RT context.
fn rt_available() -> bool {
    OffscreenRenderer::new_rt(64, 64, &GridConfig::default())
        .map(|r| r.has_rt())
        .unwrap_or(false)
}

fn render_rt_driving_range(grid: &GridConfig, camera: &Camera) -> Result<()> {
    eprintln!("Rendering driving range (RT)...");
    let renderer =
        OffscreenRenderer::new_rt(WIDTH, HEIGHT, grid).context("failed to create RT renderer")?;
    eprintln!("  RT compositing: {}", if renderer.has_rt() { "enabled" } else { "fallback to raster" });
    let frame = renderer.render(camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/driving_range_rt.png"))
}

fn render_rt_flight(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (RT, side view)...");
    let flight_camera = Camera {
        position: Vec3::new(8.0, snap.ball_pos.y, snap.ball_pos.z),
        target: snap.ball_pos,
        up: Vec3::Y,
        fov_y: std::f32::consts::FRAC_PI_4,
        near: 0.01,
        far: 600.0,
    };
    let renderer = OffscreenRenderer::new_with_flight_rt(
        WIDTH, HEIGHT, grid, snap.ball_pos, &snap.trail_points, snap.elapsed, DEFAULT_TRAIL_LIFETIME, &flight_camera,
    )
    .context("failed to create RT flight renderer")?;
    let frame = renderer.render(&flight_camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight_rt.png"))
}

fn render_rt_flight_tee(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (RT, tee box view)...");
    let camera = Camera::driving_range();
    let renderer = OffscreenRenderer::new_with_flight_rt(
        WIDTH, HEIGHT, grid, snap.ball_pos, &snap.trail_points, snap.elapsed, DEFAULT_TRAIL_LIFETIME, &camera,
    )
    .context("failed to create RT flight renderer")?;
    let frame = renderer.render(&camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight_tee_rt.png"))
}

fn render_rt_flight_wide(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (RT, wide side view)...");
    let tail_start = snap.trail_points.first().map(|p| p.position).unwrap_or(Vec3::ZERO);
    let mid = (tail_start + snap.ball_pos) * 0.5;
    let trail_len = (snap.ball_pos - tail_start).length();
    let dist = trail_len * 0.9;
    let wide_camera = Camera {
        position: Vec3::new(mid.x + dist, mid.y, mid.z),
        target: mid,
        up: Vec3::Y,
        fov_y: std::f32::consts::FRAC_PI_4,
        near: 0.01,
        far: 600.0,
    };
    let renderer = OffscreenRenderer::new_with_flight_rt(
        WIDTH, HEIGHT, grid, snap.ball_pos, &snap.trail_points, snap.elapsed, DEFAULT_TRAIL_LIFETIME, &wide_camera,
    )
    .context("failed to create RT flight renderer")?;
    let frame = renderer.render(&wide_camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight_wide_rt.png"))
}

fn render_rt_flight_wide_45(grid: &GridConfig, snap: &FlightSnapshot) -> Result<()> {
    eprintln!("Rendering flight (RT, wide 45 overhead view)...");
    let tail_start = snap.trail_points.first().map(|p| p.position).unwrap_or(Vec3::ZERO);
    let mid = (tail_start + snap.ball_pos) * 0.5;
    let trail_len = (snap.ball_pos - tail_start).length();
    let dist = trail_len * 0.9;
    let wide45_camera = Camera {
        position: Vec3::new(mid.x + dist * 0.7, mid.y + dist * 0.7, mid.z),
        target: mid,
        up: Vec3::Y,
        fov_y: std::f32::consts::FRAC_PI_4,
        near: 0.01,
        far: 600.0,
    };
    let renderer = OffscreenRenderer::new_with_flight_rt(
        WIDTH, HEIGHT, grid, snap.ball_pos, &snap.trail_points, snap.elapsed, DEFAULT_TRAIL_LIFETIME, &wide45_camera,
    )
    .context("failed to create RT flight renderer")?;
    let frame = renderer.render(&wide45_camera).context("render failed")?;
    save_png(&frame, &format!("{OUTPUT_DIR}/flight_wide_45_rt.png"))
}

fn main() -> Result<()> {
    let grid = GridConfig::default();
    let camera = Camera::driving_range();

    fs::create_dir_all(OUTPUT_DIR).context("failed to create screenshots dir")?;
    eprintln!("Initializing offscreen renderers ({WIDTH}x{HEIGHT})...");

    // ── Rasterized screenshots ──
    render_driving_range(&grid, &camera)?;

    eprintln!("Simulating 7-iron...");
    let snap = simulate_seven_iron();
    render_flight(&grid, &snap)?;
    render_flight_tee(&grid, &snap)?;
    render_flight_wide(&grid, &snap)?;
    render_flight_wide_45(&grid, &snap)?;
    render_flight_hud(&grid, &snap)?;

    // ── Ray-traced screenshots (if hardware supports it) ──
    eprintln!("Checking ray tracing support...");
    if rt_available() {
        eprintln!("Ray tracing supported — generating RT screenshots");
        render_rt_driving_range(&grid, &camera)?;
        render_rt_flight(&grid, &snap)?;
        render_rt_flight_tee(&grid, &snap)?;
        render_rt_flight_wide(&grid, &snap)?;
        render_rt_flight_wide_45(&grid, &snap)?;
    } else {
        eprintln!("Ray tracing not supported — skipping RT screenshots");
    }

    Ok(())
}
