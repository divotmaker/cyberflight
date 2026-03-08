//! Headless streaming mode: renders to HLS video via ffmpeg.

use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use cf_math::aero::BallModel;
use cf_math::environment::Environment;
use cf_render::offscreen::OffscreenRenderer;
use cf_render::stream::{OutputMode, StreamConfig, StreamEncoder, StreamPreset};
use cf_scene::range::Range;

use crate::flight::{AnimatedFlight, FlightManager, FrpState};

struct StreamArgs {
    config: StreamConfig,
    serve_port: Option<u16>,
}

/// Parse streaming CLI args after `--stream`.
/// Supports: `--preset <name>`, `--mode <hls|udp>`, `--serve`, `--port <n>`
fn parse_stream_args() -> StreamArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut preset = StreamPreset::Quality;
    let mut mode: Option<OutputMode> = None;
    let mut serve = false;
    let mut port: u16 = 8080;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--preset" => {
                if let Some(name) = args.get(i + 1) {
                    if let Some(p) = StreamPreset::parse(name) {
                        preset = p;
                    } else {
                        eprintln!("Unknown preset '{name}', using quality. Options: quality, balanced, lite");
                    }
                }
                i += 2;
            }
            "--mode" => {
                if let Some(name) = args.get(i + 1) {
                    if let Some(m) = OutputMode::parse(name) {
                        mode = Some(m);
                    } else {
                        eprintln!("Unknown mode '{name}', using udp. Options: hls, udp");
                    }
                }
                i += 2;
            }
            "--serve" => {
                serve = true;
                i += 1;
            }
            "--port" => {
                if let Some(p) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    port = p;
                }
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut config = StreamConfig::from_preset(preset);
    if let Some(m) = mode {
        config.mode = m;
    }

    StreamArgs {
        config,
        serve_port: if serve { Some(port) } else { None },
    }
}

pub fn run_stream() -> Result<()> {
    let stream_args = parse_stream_args();
    let config = stream_args.config;

    eprintln!("cyberflight streaming mode");
    eprintln!("  Resolution: {}x{} @ {}fps, {}bps", config.width, config.height, config.fps, config.bitrate);

    let range = Range::new();
    let environment = Environment::SEA_LEVEL;

    // Create headless renderer with streaming buffers.
    eprintln!("Initializing headless Vulkan renderer...");
    let mut renderer = OffscreenRenderer::new(config.width, config.height, &range.grid)
        .context("failed to create offscreen renderer")?;
    renderer.init_streaming()
        .context("failed to initialize streaming buffers")?;

    // Start ffmpeg encoder.
    let mut encoder = StreamEncoder::new(config.clone())
        .context("failed to start stream encoder")?;

    // Start built-in HTTP server for HLS if requested.
    if let Some(port) = stream_args.serve_port {
        let dir = config.output_dir.clone();
        start_http_server(port, dir)?;
    }

    eprintln!("Streaming to: {}", encoder.destination());
    match &config.mode {
        OutputMode::Hls => {
            if let Some(port) = stream_args.serve_port {
                eprintln!("  Play: http://<host>:{port}/stream.m3u8");
            } else {
                eprintln!("  Serve with: python3 -m http.server 8080 --directory stream/");
                eprintln!("  Play: http://<host>:8080/stream.m3u8");
            }
        }
        OutputMode::Udp { dest } => {
            eprintln!("  Play: vlc --network-caching=100 udp://@{dest}");
        }
    }

    // Connect to FRP device.
    let mut fh = FrpState::new();
    let mut fm = FlightManager::new();

    let start = Instant::now();
    let frame_duration = Duration::from_secs_f64(1.0 / f64::from(config.fps));
    let mut frame_count: u64 = 0;
    let mut fps_timer = Instant::now();
    let mut render_us_sum: u64 = 0;
    let mut encode_us_sum: u64 = 0;

    eprintln!("Streaming started. Press Ctrl+C to stop.");

    loop {
        let frame_start = Instant::now();
        let now = start.elapsed().as_secs_f64();

        // Check encoder health, restart if crashed.
        if !encoder.is_healthy() {
            eprintln!("Encoder crashed, restarting...");
            if let Err(e) = encoder.restart() {
                eprintln!("Encoder restart failed: {e}");
                // Skip this frame, retry next loop.
                std::thread::sleep(Duration::from_secs(1));
                continue;
            }
        }

        // Poll FRP for new shots.
        let animating = fm.is_animating(now);
        if let Some(received) = fh.poll(now, animating) {
            fm.flights.push(AnimatedFlight::from_received(
                &received, now, &BallModel::TOUR, &environment, &range.targets,
            ));
        }

        // Update flight animations.
        let render_data = fm.update(now);

        // Update renderer geometry.
        renderer.update_flight(&render_data, &range.camera);

        // Chase camera: follow the most recent active ball.
        let chase_camera = fm.chase_camera(now);
        if let Some(cam) = chase_camera {
            renderer.set_chase_camera(cam);
        }

        // Build and update HUD.
        let hud = fm.build_hud(now, chase_camera.is_some(), &fh.lm_states, fh.connected, config.width as f32, config.height as f32);
        renderer.update_hud_streaming(&hud.lines, &hud.fills);

        // Render frame.
        let render_t = Instant::now();
        let frame = renderer.render(&range.camera)
            .context("render failed")?;
        let render_us = render_t.elapsed().as_micros() as u64;

        // Pipe to ffmpeg.
        match encoder.write_frame(&frame.color) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("Encode error (dropping frame): {e}");
                encoder.frames_dropped += 1;
                continue;
            }
        }
        let encode_us = encoder.last_write_us;

        // Accumulate timing stats.
        render_us_sum += render_us;
        encode_us_sum += encode_us;
        frame_count += 1;

        // FPS + timing telemetry.
        let fps_elapsed = fps_timer.elapsed().as_secs_f64();
        if fps_elapsed >= 5.0 {
            let fps = frame_count as f64 / fps_elapsed;
            let avg_render_ms = render_us_sum as f64 / frame_count as f64 / 1000.0;
            let avg_encode_ms = encode_us_sum as f64 / frame_count as f64 / 1000.0;
            eprintln!(
                "FPS: {fps:.0} | render {avg_render_ms:.1}ms, encode {avg_encode_ms:.1}ms | {} frames, {} dropped",
                encoder.frame_count(),
                encoder.frames_dropped,
            );
            frame_count = 0;
            render_us_sum = 0;
            encode_us_sum = 0;
            fps_timer = Instant::now();
        }

        // Frame pacing.
        let elapsed = frame_start.elapsed();
        if elapsed < frame_duration {
            std::thread::sleep(frame_duration - elapsed);
        }
    }
}

/// Start a static file HTTP server on a background thread.
fn start_http_server(port: u16, dir: PathBuf) -> Result<()> {
    let addr = format!("0.0.0.0:{port}");
    let server = tiny_http::Server::http(&addr)
        .map_err(|e| anyhow::anyhow!("failed to start HTTP server on {addr}: {e}"))?;

    eprintln!("HTTP server listening on port {port} (serving {}/)", dir.display());

    thread::spawn(move || {
        for request in server.incoming_requests() {
            let url_path = request.url().trim_start_matches('/');
            // Prevent path traversal.
            if url_path.contains("..") {
                let _ = request.respond(tiny_http::Response::from_string("forbidden").with_status_code(403));
                continue;
            }

            let file_path = dir.join(if url_path.is_empty() { "stream.m3u8" } else { url_path });

            let content_type = match file_path.extension().and_then(|e| e.to_str()) {
                Some("m3u8") => "application/vnd.apple.mpegurl",
                Some("ts") => "video/mp2t",
                _ => "application/octet-stream",
            };

            match std::fs::File::open(&file_path) {
                Ok(file) => {
                    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
                    let response = tiny_http::Response::from_file(file)
                        .with_header(tiny_http::Header::from_bytes("Content-Type", content_type).unwrap())
                        .with_header(tiny_http::Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap())
                        .with_status_code(200);
                    let _ = request.respond(response.with_header(
                        tiny_http::Header::from_bytes("Content-Length", len.to_string()).unwrap(),
                    ));
                }
                Err(_) => {
                    let _ = request.respond(tiny_http::Response::from_string("not found").with_status_code(404));
                }
            }
        }
    });

    Ok(())
}
