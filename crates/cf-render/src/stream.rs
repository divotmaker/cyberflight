use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, Command, ExitStatus, Stdio};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::error::RenderError;

/// Output transport mode.
#[derive(Debug, Clone)]
pub enum OutputMode {
    /// HLS segments — wide device compatibility (Apple TV native), 2-4s latency.
    Hls,
    /// MPEG-TS over UDP — low latency (~100-200ms), requires VLC/ffplay on receiver.
    /// The address is a multicast or unicast destination (e.g. "239.0.0.1:5004").
    Udp { dest: String },
}

impl Default for OutputMode {
    fn default() -> Self {
        Self::Udp {
            dest: "127.0.0.1:5004".to_owned(),
        }
    }
}

impl OutputMode {
    /// Parse from CLI string. "hls" or "udp" (with optional `:address:port`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("hls") {
            Some(Self::Hls)
        } else if s.eq_ignore_ascii_case("udp") {
            Some(Self::Udp {
                dest: "127.0.0.1:5004".to_owned(),
            })
        } else {
            None
        }
    }
}

/// Streaming quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPreset {
    /// 1080p60, 8 Mbps
    Quality,
    /// 720p60, 4 Mbps
    Balanced,
    /// 720p30, 2 Mbps
    Lite,
}

impl StreamPreset {
    /// Parse from a string (case-insensitive). Accepts short forms: q, b, l.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "quality" | "q" => Some(Self::Quality),
            "balanced" | "b" => Some(Self::Balanced),
            "lite" | "l" => Some(Self::Lite),
            _ => None,
        }
    }
}

/// Streaming video encoder configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target frames per second.
    pub fps: u32,
    /// Directory for HLS output (.m3u8 + .ts segments).
    pub output_dir: PathBuf,
    /// HLS segment duration in seconds.
    pub hls_segment_secs: u32,
    /// Number of HLS segments to keep in the playlist.
    pub hls_list_size: u32,
    /// Video bitrate (e.g. "8M", "4M").
    pub bitrate: String,
    /// Strip alpha channel (RGBA → RGB24) before piping. Saves 25% pipe bandwidth.
    pub strip_alpha: bool,
    /// Output transport: HLS (segments) or UDP (low-latency MPEG-TS).
    pub mode: OutputMode,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 60,
            output_dir: PathBuf::from("stream"),
            hls_segment_secs: 1,
            hls_list_size: 5,
            bitrate: "8M".to_owned(),
            strip_alpha: true,
            mode: OutputMode::default(),
        }
    }
}

impl StreamConfig {
    /// Create config from a named preset.
    #[must_use]
    pub fn from_preset(preset: StreamPreset) -> Self {
        match preset {
            StreamPreset::Quality => Self::default(),
            StreamPreset::Balanced => Self {
                width: 1280,
                height: 720,
                fps: 60,
                bitrate: "4M".to_owned(),
                ..Self::default()
            },
            StreamPreset::Lite => Self {
                width: 1280,
                height: 720,
                fps: 30,
                bitrate: "2M".to_owned(),
                ..Self::default()
            },
        }
    }
}

/// How long `Drop` waits for ffmpeg to finish after closing its stdin
/// before killing it.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Longest ffmpeg log line forwarded; the rest of the line is dropped.
const MAX_LOG_LINE: usize = 512;

/// At most this many ffmpeg log lines are forwarded per `LOG_WINDOW`;
/// the rest are counted and reported as suppressed.
const MAX_LOG_LINES_PER_WINDOW: u32 = 20;
const LOG_WINDOW: Duration = Duration::from_secs(10);

/// A running ffmpeg process and the thread draining its stderr.
///
/// stderr is always drained so ffmpeg can never block on a full pipe (which
/// would stop it reading frames from stdin).
struct FfmpegProcess {
    child: Child,
    stderr_thread: Option<JoinHandle<()>>,
}

impl FfmpegProcess {
    fn new(mut child: Child) -> Result<Self, RenderError> {
        let stderr_thread = match child.stderr.take() {
            Some(stderr) => match spawn_stderr_forwarder(stderr) {
                Ok(handle) => Some(handle),
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(RenderError::Stream(format!(
                        "failed to start ffmpeg stderr reader: {e}"
                    )));
                }
            },
            None => None,
        };
        Ok(Self {
            child,
            stderr_thread,
        })
    }

    /// Close stdin, give ffmpeg up to `grace` to exit on its own, kill it if
    /// it is still running, then join the stderr thread.
    ///
    /// Returns the exit status, or `None` if it could not be collected.
    fn shutdown(&mut self, grace: Duration) -> Option<ExitStatus> {
        drop(self.child.stdin.take());

        let deadline = Instant::now() + grace;
        let status = loop {
            match self.child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(10));
                }
                Ok(None) => {
                    if !grace.is_zero() {
                        eprintln!(
                            "StreamEncoder: ffmpeg still running after {}s, killing it",
                            grace.as_secs()
                        );
                    }
                    let _ = self.child.kill();
                    break self.child.wait().ok();
                }
                Err(e) => {
                    eprintln!("StreamEncoder: failed to poll ffmpeg: {e}");
                    let _ = self.child.kill();
                    break self.child.wait().ok();
                }
            }
        };

        // The child has exited, so its end of the stderr pipe is closed and
        // the reader thread reaches EOF.
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
        status
    }
}

/// Forward ffmpeg's stderr to our stderr on a background thread.
fn spawn_stderr_forwarder(stderr: ChildStderr) -> std::io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("ffmpeg-stderr".to_owned())
        .spawn(move || {
            let mut limiter = LogLimiter::new(Instant::now());
            forward_lines(stderr, |line| {
                limiter.emit(Instant::now(), line, |msg| eprintln!("ffmpeg: {msg}"));
            });
            limiter.finish(|msg| eprintln!("ffmpeg: {msg}"));
        })
}

/// Read `reader` to EOF, calling `on_line` for each non-empty line.
///
/// Lines end at `\n` or `\r` (ffmpeg uses `\r` for progress updates) and are
/// truncated to `MAX_LOG_LINE` bytes. Read errors end the stream.
fn forward_lines<R: Read>(mut reader: R, mut on_line: impl FnMut(&str)) {
    let mut buf = [0u8; 4096];
    let mut line: Vec<u8> = Vec::with_capacity(MAX_LOG_LINE);
    let mut flush = |line: &mut Vec<u8>| {
        let text = String::from_utf8_lossy(line);
        let text = text.trim();
        if !text.is_empty() {
            on_line(text);
        }
        line.clear();
    };
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        for &b in &buf[..n] {
            if b == b'\n' || b == b'\r' {
                flush(&mut line);
            } else if line.len() < MAX_LOG_LINE {
                line.push(b);
            }
        }
    }
    flush(&mut line);
}

/// Rate limiter for forwarded log lines: at most `MAX_LOG_LINES_PER_WINDOW`
/// lines per `LOG_WINDOW`, with a count of anything dropped.
struct LogLimiter {
    window_start: Instant,
    emitted: u32,
    suppressed: u64,
}

impl LogLimiter {
    fn new(now: Instant) -> Self {
        Self {
            window_start: now,
            emitted: 0,
            suppressed: 0,
        }
    }

    fn emit(&mut self, now: Instant, line: &str, mut out: impl FnMut(&str)) {
        if now.duration_since(self.window_start) >= LOG_WINDOW {
            self.finish(&mut out);
            self.window_start = now;
            self.emitted = 0;
        }
        if self.emitted < MAX_LOG_LINES_PER_WINDOW {
            self.emitted += 1;
            out(line);
        } else {
            self.suppressed += 1;
        }
    }

    /// Report (and reset) the count of suppressed lines, if any.
    fn finish(&mut self, mut out: impl FnMut(&str)) {
        if self.suppressed > 0 {
            out(&format!("({} more lines suppressed)", self.suppressed));
            self.suppressed = 0;
        }
    }
}

/// Pipes raw RGBA frames to an ffmpeg subprocess for HLS streaming.
pub struct StreamEncoder {
    process: FfmpegProcess,
    config: StreamConfig,
    frame_count: u64,
    /// Scratch buffer for RGBA → RGB24 conversion.
    rgb_scratch: Vec<u8>,
    /// Duration of the last `write_frame` call in microseconds.
    pub last_write_us: u64,
    /// Number of frames dropped due to backpressure or encoder crash.
    pub frames_dropped: u64,
}

impl StreamEncoder {
    /// Start the encoder. Probes for NVENC hardware acceleration, falls back to libx264.
    ///
    /// # Errors
    /// Returns `RenderError` if ffmpeg cannot be started.
    pub fn new(config: StreamConfig) -> Result<Self, RenderError> {
        std::fs::create_dir_all(&config.output_dir)
            .map_err(|e| RenderError::Stream(format!("create output dir: {e}")))?;

        let process = FfmpegProcess::new(spawn_encoder(&config)?)?;
        let rgb_len = if config.strip_alpha {
            (config.width * config.height * 3) as usize
        } else {
            0
        };

        let dest_desc = match &config.mode {
            OutputMode::Hls => format!("{}", config.output_dir.join("stream.m3u8").display()),
            OutputMode::Udp { dest } => format!("udp://{dest}"),
        };
        eprintln!(
            "StreamEncoder: {}x{} @ {}fps, {}bps, alpha_strip={} → {}",
            config.width, config.height, config.fps, config.bitrate, config.strip_alpha, dest_desc,
        );

        Ok(Self {
            process,
            config,
            frame_count: 0,
            rgb_scratch: vec![0u8; rgb_len],
            last_write_us: 0,
            frames_dropped: 0,
        })
    }

    /// Write a single RGBA frame to the encoder.
    ///
    /// # Errors
    /// Returns `RenderError` if the frame size is wrong or the pipe is broken.
    pub fn write_frame(&mut self, rgba: &[u8]) -> Result<(), RenderError> {
        let expected = (self.config.width * self.config.height * 4) as usize;
        if rgba.len() != expected {
            return Err(RenderError::Stream(format!(
                "frame size mismatch: got {} bytes, expected {expected}",
                rgba.len()
            )));
        }

        let t = Instant::now();

        let stdin = self
            .process
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| RenderError::Stream("ffmpeg stdin closed".to_owned()))?;

        if self.config.strip_alpha {
            strip_alpha(rgba, &mut self.rgb_scratch);
            stdin
                .write_all(&self.rgb_scratch)
                .map_err(|e| RenderError::Stream(format!("write frame: {e}")))?;
        } else {
            stdin
                .write_all(rgba)
                .map_err(|e| RenderError::Stream(format!("write frame: {e}")))?;
        }

        self.last_write_us = t.elapsed().as_micros() as u64;
        self.frame_count += 1;
        Ok(())
    }

    /// Check if the ffmpeg subprocess is still running.
    #[must_use]
    pub fn is_healthy(&mut self) -> bool {
        match self.process.child.try_wait() {
            Ok(None) => true,
            Ok(Some(status)) => {
                eprintln!("StreamEncoder: ffmpeg exited with {status}");
                false
            }
            Err(e) => {
                eprintln!("StreamEncoder: health check failed: {e}");
                false
            }
        }
    }

    /// Restart the ffmpeg subprocess after a crash.
    ///
    /// # Errors
    /// Returns `RenderError` if ffmpeg cannot be restarted.
    pub fn restart(&mut self) -> Result<(), RenderError> {
        eprintln!("StreamEncoder: restarting ffmpeg...");
        self.process.shutdown(Duration::ZERO);

        self.process = FfmpegProcess::new(spawn_encoder(&self.config)?)?;
        eprintln!(
            "StreamEncoder: restarted (was at frame {})",
            self.frame_count
        );
        Ok(())
    }

    /// Number of frames written so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Human-readable description of where the stream is going.
    #[must_use]
    pub fn destination(&self) -> String {
        match &self.config.mode {
            OutputMode::Hls => format!("{}", self.config.output_dir.join("stream.m3u8").display()),
            OutputMode::Udp { dest } => format!("udp://{dest}"),
        }
    }
}

impl Drop for StreamEncoder {
    fn drop(&mut self) {
        // Closing stdin signals EOF; ffmpeg flushes its output and exits.
        match self.process.shutdown(SHUTDOWN_GRACE) {
            Some(status) if !status.success() => {
                eprintln!("ffmpeg exited with status: {status}");
            }
            Some(_) => {}
            None => eprintln!("failed to wait for ffmpeg"),
        }
    }
}

/// Strip alpha channel: RGBA → RGB24 in-place into pre-allocated buffer.
fn strip_alpha(rgba: &[u8], rgb: &mut [u8]) {
    let mut j = 0;
    for pixel in rgba.chunks_exact(4) {
        rgb[j] = pixel[0];
        rgb[j + 1] = pixel[1];
        rgb[j + 2] = pixel[2];
        j += 3;
    }
}

/// Spawn an ffmpeg encoder process with auto-detected codec.
fn spawn_encoder(config: &StreamConfig) -> Result<Child, RenderError> {
    let has_nvenc = probe_encoder("h264_nvenc");
    let input_fmt = if config.strip_alpha { "rgb24" } else { "rgba" };

    let is_udp = matches!(config.mode, OutputMode::Udp { .. });

    let (codec, codec_opts): (&str, Vec<&str>) = if has_nvenc {
        eprintln!("StreamEncoder: using h264_nvenc (hardware)");
        let mut opts = vec!["-preset", "ll"];
        if is_udp {
            // Low-latency: zero delay, keyframe every 0.5s (30 frames at 60fps).
            opts.extend_from_slice(&[
                "-delay",
                "0",
                "-zerolatency",
                "1",
                "-g",
                "30",
                "-keyint_min",
                "30",
            ]);
        }
        ("h264_nvenc", opts)
    } else {
        eprintln!("StreamEncoder: using libx264 (software)");
        let mut opts = vec!["-preset", "ultrafast", "-tune", "zerolatency"];
        if is_udp {
            opts.extend_from_slice(&["-g", "30", "-keyint_min", "30"]);
        }
        ("libx264", opts)
    };

    spawn_ffmpeg(config, codec, &codec_opts, input_fmt)
        .map_err(|e| RenderError::Stream(format!("failed to start ffmpeg: {e}")))
}

/// Check if ffmpeg supports a given encoder.
fn probe_encoder(encoder: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map(|out| {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.contains(encoder)
        })
        .unwrap_or(false)
}

fn spawn_ffmpeg(
    config: &StreamConfig,
    codec: &str,
    codec_opts: &[&str],
    input_pix_fmt: &str,
) -> Result<Child, std::io::Error> {
    let size = format!("{}x{}", config.width, config.height);
    let fps = config.fps.to_string();

    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        input_pix_fmt,
        "-s",
        &size,
        "-r",
        &fps,
        "-i",
        "pipe:0",
        "-c:v",
        codec,
    ]);
    cmd.args(codec_opts);
    cmd.args(["-b:v", &config.bitrate, "-pix_fmt", "yuv420p"]);

    // Output format depends on mode.
    match &config.mode {
        OutputMode::Hls => {
            let hls_time = config.hls_segment_secs.to_string();
            let hls_list = config.hls_list_size.to_string();
            let output = config.output_dir.join("stream.m3u8");
            cmd.args([
                "-f",
                "hls",
                "-hls_time",
                &hls_time,
                "-hls_list_size",
                &hls_list,
                "-hls_flags",
                "delete_segments",
            ]);
            cmd.arg(output);
        }
        OutputMode::Udp { dest } => {
            let udp_url = format!("udp://{dest}?pkt_size=1316");
            cmd.args([
                "-f",
                "mpegts",
                "-flush_packets",
                "1",
                "-mpegts_flags",
                "resend_headers",
            ]);
            cmd.arg(&udp_url);
        }
    }

    cmd.stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.spawn()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(input: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        forward_lines(input, |l| lines.push(l.to_owned()));
        lines
    }

    #[test]
    fn forward_lines_splits_on_cr_and_lf() {
        let lines = collect(b"first\nsecond\r\nthird\rlast");
        assert_eq!(lines, ["first", "second", "third", "last"]);
    }

    #[test]
    fn forward_lines_skips_blank_and_truncates_long_lines() {
        let mut input = b"\n  \n".to_vec();
        input.extend(std::iter::repeat_n(b'x', MAX_LOG_LINE * 3));
        input.extend_from_slice(b"\nnext\n");
        let lines = collect(&input);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].len(), MAX_LOG_LINE);
        assert_eq!(lines[1], "next");
    }

    #[test]
    fn log_limiter_caps_lines_per_window_and_reports_suppressed() {
        let t0 = Instant::now();
        let mut limiter = LogLimiter::new(t0);
        let mut out = Vec::new();
        for i in 0..(MAX_LOG_LINES_PER_WINDOW + 5) {
            limiter.emit(t0, &format!("line {i}"), |m| out.push(m.to_owned()));
        }
        assert_eq!(out.len(), MAX_LOG_LINES_PER_WINDOW as usize);

        // Next window: suppressed count is reported, then lines flow again.
        limiter.emit(t0 + LOG_WINDOW, "fresh", |m| out.push(m.to_owned()));
        assert_eq!(out[out.len() - 2], "(5 more lines suppressed)");
        assert_eq!(out[out.len() - 1], "fresh");

        limiter.finish(|m| out.push(m.to_owned()));
        assert_eq!(out.last().unwrap(), "fresh");
    }
}
