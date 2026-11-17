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

/// An h264_nvenc ffmpeg process that exits within this long of starting
/// counts as an early exit, which triggers a fresh NVENC test encode.
const NVENC_EARLY_EXIT: Duration = Duration::from_secs(5);

/// After this many consecutive early h264_nvenc exits the encoder uses
/// libx264 for the rest of the session, whatever the test encode says.
const NVENC_MAX_EARLY_EXITS: u32 = 2;

/// Upper bound on the NVENC test encode.
const NVENC_PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// Frame size of the NVENC test encode.
const NVENC_PROBE_SIZE: (u32, u32) = (256, 256);

/// H.264 encoder driven by the ffmpeg subprocess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    /// NVIDIA NVENC hardware encoder.
    Nvenc,
    /// libx264 software encoder.
    X264,
}

impl Codec {
    fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::Nvenc => "h264_nvenc",
            Self::X264 => "libx264",
        }
    }

    /// Encoder options for this codec.
    fn options(self, low_latency: bool) -> Vec<&'static str> {
        match self {
            Self::Nvenc => {
                let mut opts = vec!["-preset", "ll"];
                if low_latency {
                    // Zero delay, keyframe every 0.5s (30 frames at 60fps).
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
                opts
            }
            Self::X264 => {
                let mut opts = vec!["-preset", "ultrafast", "-tune", "zerolatency"];
                if low_latency {
                    opts.extend_from_slice(&["-g", "30", "-keyint_min", "30"]);
                }
                opts
            }
        }
    }

    /// Pick the encoder: NVENC if a one-frame test encode succeeds, else libx264.
    ///
    /// # Errors
    /// Returns `RenderError` if ffmpeg is not installed.
    fn detect(config: &StreamConfig) -> Result<Self, RenderError> {
        match probe_nvenc(config) {
            Ok(()) => {
                eprintln!("StreamEncoder: using h264_nvenc (hardware): test encode succeeded");
                Ok(Self::Nvenc)
            }
            Err(ProbeError::FfmpegMissing) => Err(ffmpeg_missing()),
            Err(ProbeError::Failed(reason)) => {
                eprintln!(
                    "StreamEncoder: using libx264 (software): h264_nvenc test encode failed: {reason}"
                );
                Ok(Self::X264)
            }
        }
    }
}

/// Codec to use for the next ffmpeg process, plus a log line explaining the
/// choice when it is not simply "same as before".
#[derive(Debug, PartialEq, Eq)]
struct CodecDecision {
    codec: Codec,
    note: Option<String>,
}

/// Tracks the codec across encoder restarts.
///
/// An h264_nvenc process that exits early triggers a fresh test encode: if it
/// fails, libx264 is used from then on; if it passes, NVENC is retried, up to
/// `NVENC_MAX_EARLY_EXITS` consecutive early exits, after which libx264 is
/// used for the rest of the session. libx264 is never left once chosen.
#[derive(Debug)]
struct CodecSelector {
    codec: Codec,
    early_nvenc_exits: u32,
}

impl CodecSelector {
    fn new(codec: Codec) -> Self {
        Self {
            codec,
            early_nvenc_exits: 0,
        }
    }

    /// Decide the codec after the encoder process exited `lifetime` after
    /// starting. `reprobe` runs a fresh NVENC test encode and is called only
    /// when that result decides the outcome.
    fn on_exit(
        &mut self,
        lifetime: Duration,
        reprobe: impl FnOnce() -> Result<(), String>,
    ) -> CodecDecision {
        if self.codec == Codec::X264 {
            return self.decision(None);
        }
        if lifetime >= NVENC_EARLY_EXIT {
            self.early_nvenc_exits = 0;
            return self.decision(None);
        }

        self.early_nvenc_exits += 1;
        let secs = lifetime.as_secs_f64();
        if self.early_nvenc_exits >= NVENC_MAX_EARLY_EXITS {
            self.codec = Codec::X264;
            return self.decision(Some(format!(
                "switching to libx264 (software) for this session: h264_nvenc exited early \
                 {} times in a row (last after {secs:.1}s)",
                self.early_nvenc_exits
            )));
        }

        match reprobe() {
            Ok(()) => self.decision(Some(format!(
                "retrying h264_nvenc: it exited {secs:.1}s after starting but the test encode \
                 still succeeds (early exit {} of {NVENC_MAX_EARLY_EXITS})",
                self.early_nvenc_exits
            ))),
            Err(reason) => {
                self.codec = Codec::X264;
                self.decision(Some(format!(
                    "switching to libx264 (software) for this session: h264_nvenc exited \
                     {secs:.1}s after starting and the test encode now fails: {reason}"
                )))
            }
        }
    }

    fn decision(&self, note: Option<String>) -> CodecDecision {
        CodecDecision {
            codec: self.codec,
            note,
        }
    }
}

/// A running ffmpeg process and the thread draining its stderr.
///
/// stderr is always drained so ffmpeg can never block on a full pipe (which
/// would stop it reading frames from stdin).
struct FfmpegProcess {
    child: Child,
    stderr_thread: Option<JoinHandle<()>>,
    started: Instant,
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
            started: Instant::now(),
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
    codec: CodecSelector,
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
    /// Start the encoder. Uses NVENC if a test encode with it succeeds,
    /// otherwise libx264.
    ///
    /// # Errors
    /// Returns `RenderError` if ffmpeg cannot be started.
    pub fn new(config: StreamConfig) -> Result<Self, RenderError> {
        std::fs::create_dir_all(&config.output_dir)
            .map_err(|e| RenderError::Stream(format!("create output dir: {e}")))?;

        let codec = Codec::detect(&config)?;
        let process = FfmpegProcess::new(spawn_encoder(&config, codec)?)?;
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
            "StreamEncoder: {}x{} @ {}fps, {}bps, alpha_strip={} -> {}",
            config.width, config.height, config.fps, config.bitrate, config.strip_alpha, dest_desc,
        );

        Ok(Self {
            process,
            codec: CodecSelector::new(codec),
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
    /// An h264_nvenc process that exited within `NVENC_EARLY_EXIT` of
    /// starting is retried only while a fresh test encode succeeds, and at
    /// most `NVENC_MAX_EARLY_EXITS` times in a row; otherwise the restart
    /// (and every later one) uses libx264.
    ///
    /// # Errors
    /// Returns `RenderError` if ffmpeg cannot be restarted.
    pub fn restart(&mut self) -> Result<(), RenderError> {
        eprintln!("StreamEncoder: restarting ffmpeg...");
        let lifetime = self.process.started.elapsed();
        self.process.shutdown(Duration::ZERO);

        let config = &self.config;
        let decision = self.codec.on_exit(lifetime, || {
            eprintln!(
                "StreamEncoder: h264_nvenc exited {:.1}s after starting, re-running the test encode",
                lifetime.as_secs_f64()
            );
            probe_nvenc(config).map_err(|e| e.to_string())
        });
        if let Some(note) = &decision.note {
            eprintln!("StreamEncoder: {note}");
        }

        self.process = FfmpegProcess::new(spawn_encoder(&self.config, decision.codec)?)?;
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

/// Spawn an ffmpeg encoder process using `codec`.
fn spawn_encoder(config: &StreamConfig, codec: Codec) -> Result<Child, RenderError> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-hide_banner", "-loglevel", "warning"]);
    cmd.args(encode_args(config, codec, (config.width, config.height)));
    cmd.args(output_args(config));
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            ffmpeg_missing()
        } else {
            RenderError::Stream(format!("failed to start ffmpeg: {e}"))
        }
    })
}

fn ffmpeg_missing() -> RenderError {
    RenderError::Stream("ffmpeg not found on PATH".to_owned())
}

/// Raw input pixel format piped to ffmpeg, and its bytes per pixel.
fn input_pix_fmt(config: &StreamConfig) -> (&'static str, usize) {
    if config.strip_alpha {
        ("rgb24", 3)
    } else {
        ("rgba", 4)
    }
}

/// ffmpeg arguments from the raw stdin input through the encoder settings,
/// shared by the real encode and the NVENC test encode. `size` is the frame
/// size in pixels.
fn encode_args(config: &StreamConfig, codec: Codec, size: (u32, u32)) -> Vec<String> {
    let (pix_fmt, _) = input_pix_fmt(config);
    let low_latency = matches!(config.mode, OutputMode::Udp { .. });
    let size = format!("{}x{}", size.0, size.1);
    let fps = config.fps.to_string();
    let mut args = owned(&[
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        &size,
        "-r",
        &fps,
        "-i",
        "pipe:0",
        "-c:v",
        codec.ffmpeg_name(),
    ]);
    args.extend(owned(&codec.options(low_latency)));
    args.extend(owned(&["-b:v", &config.bitrate, "-pix_fmt", "yuv420p"]));
    args
}

/// ffmpeg output arguments for the configured transport.
fn output_args(config: &StreamConfig) -> Vec<String> {
    match &config.mode {
        OutputMode::Hls => {
            let hls_time = config.hls_segment_secs.to_string();
            let hls_list = config.hls_list_size.to_string();
            let mut args = owned(&[
                "-f",
                "hls",
                "-hls_time",
                &hls_time,
                "-hls_list_size",
                &hls_list,
                "-hls_flags",
                "delete_segments",
            ]);
            let output = config.output_dir.join("stream.m3u8");
            args.push(output.to_string_lossy().into_owned());
            args
        }
        OutputMode::Udp { dest } => {
            let mut args = owned(&[
                "-f",
                "mpegts",
                "-flush_packets",
                "1",
                "-mpegts_flags",
                "resend_headers",
            ]);
            args.push(format!("udp://{dest}?pkt_size=1316"));
            args
        }
    }
}

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|&a| a.to_owned()).collect()
}

/// Why the NVENC test encode did not succeed.
#[derive(Debug)]
enum ProbeError {
    /// The `ffmpeg` executable does not exist.
    FfmpegMissing,
    /// The test encode failed; a short reason.
    Failed(String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FfmpegMissing => f.write_str("ffmpeg not found on PATH"),
            Self::Failed(reason) => f.write_str(reason),
        }
    }
}

/// Check that h264_nvenc actually works by encoding one small frame with the
/// same input format and encoder options as the real encode.
///
/// Listing h264_nvenc in `ffmpeg -encoders` only means ffmpeg was built with
/// it; the encode fails without a usable NVIDIA GPU and driver.
///
/// # Errors
/// `ProbeError::FfmpegMissing` if ffmpeg is not installed, otherwise
/// `ProbeError::Failed` with a short reason (ffmpeg's first error line,
/// timeout, or a spawn/thread failure).
fn probe_nvenc(config: &StreamConfig) -> Result<(), ProbeError> {
    eprintln!(
        "StreamEncoder: probing h264_nvenc with a test encode (up to {}s)...",
        NVENC_PROBE_TIMEOUT.as_secs()
    );

    let (w, h) = NVENC_PROBE_SIZE;
    let (_, bytes_per_pixel) = input_pix_fmt(config);
    let frame = vec![0u8; w as usize * h as usize * bytes_per_pixel];

    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error"])
        .args(encode_args(config, Codec::Nvenc, NVENC_PROBE_SIZE))
        .args(["-frames:v", "1", "-f", "null", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ProbeError::FfmpegMissing
            } else {
                ProbeError::Failed(format!("cannot run ffmpeg: {e}"))
            }
        })?;

    let abort = |child: &mut Child, reason: String| {
        let _ = child.kill();
        let _ = child.wait();
        ProbeError::Failed(reason)
    };

    // Feed the frame and collect the first error line on threads, so neither
    // a stalled nor a chatty ffmpeg can block us past the timeout. A write
    // error just means ffmpeg exited early; its status says why.
    let writer = match child.stdin.take().map(|mut stdin| {
        thread::Builder::new()
            .name("ffmpeg-probe-in".to_owned())
            .spawn(move || {
                let _ = stdin.write_all(&frame);
            })
    }) {
        Some(Err(e)) => return Err(abort(&mut child, format!("cannot start probe thread: {e}"))),
        Some(Ok(handle)) => Some(handle),
        None => None,
    };
    let reader = match child.stderr.take().map(|stderr| {
        thread::Builder::new()
            .name("ffmpeg-probe-err".to_owned())
            .spawn(move || {
                let mut first: Option<String> = None;
                forward_lines(stderr, |line| {
                    if first.is_none() {
                        first = Some(line.to_owned());
                    }
                });
                first
            })
    }) {
        Some(Err(e)) => {
            let err = abort(&mut child, format!("cannot start probe thread: {e}"));
            if let Some(w) = writer {
                let _ = w.join();
            }
            return Err(err);
        }
        Some(Ok(handle)) => Some(handle),
        None => None,
    };

    let deadline = Instant::now() + NVENC_PROBE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                break Err(abort(
                    &mut child,
                    format!("timed out after {}s", NVENC_PROBE_TIMEOUT.as_secs()),
                ));
            }
            Err(e) => break Err(abort(&mut child, format!("cannot wait for ffmpeg: {e}"))),
        }
    };
    // The child has exited, so both pipes are closed and the threads finish.
    if let Some(w) = writer {
        let _ = w.join();
    }
    let first_line = reader.and_then(|r| r.join().ok()).flatten();

    match status? {
        status if status.success() => Ok(()),
        status => Err(ProbeError::Failed(
            first_line.unwrap_or_else(|| format!("ffmpeg exited with {status}")),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(input: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        forward_lines(input, |l| lines.push(l.to_owned()));
        lines
    }

    const EARLY: Duration = Duration::from_millis(200);
    const LATE: Duration = Duration::from_secs(60);

    fn probe_must_not_run() -> Result<(), String> {
        panic!("test encode should not run")
    }

    #[test]
    fn selector_keeps_x264_without_probing() {
        let mut sel = CodecSelector::new(Codec::X264);
        for lifetime in [EARLY, LATE, EARLY] {
            let d = sel.on_exit(lifetime, probe_must_not_run);
            assert_eq!(
                d,
                CodecDecision {
                    codec: Codec::X264,
                    note: None
                }
            );
        }
    }

    #[test]
    fn selector_restarts_nvenc_after_late_exit_without_probing() {
        let mut sel = CodecSelector::new(Codec::Nvenc);
        let d = sel.on_exit(LATE, probe_must_not_run);
        assert_eq!(d.codec, Codec::Nvenc);
        assert!(d.note.is_none());
    }

    #[test]
    fn selector_downgrades_when_reprobe_fails() {
        let mut sel = CodecSelector::new(Codec::Nvenc);
        let d = sel.on_exit(EARLY, || Err("no device".to_owned()));
        assert_eq!(d.codec, Codec::X264);
        let note = d.note.unwrap();
        assert!(note.contains("libx264"), "{note}");
        assert!(note.contains("no device"), "{note}");

        // Permanent: later exits stay on libx264 and never probe again.
        assert_eq!(sel.on_exit(EARLY, probe_must_not_run).codec, Codec::X264);
        assert_eq!(sel.on_exit(LATE, probe_must_not_run).codec, Codec::X264);
    }

    #[test]
    fn selector_retries_nvenc_then_downgrades_after_consecutive_early_exits() {
        let mut sel = CodecSelector::new(Codec::Nvenc);
        let mut probes = 0;
        let d = sel.on_exit(EARLY, || {
            probes += 1;
            Ok(())
        });
        assert_eq!(d.codec, Codec::Nvenc);
        assert!(d.note.unwrap().contains("retrying h264_nvenc"));
        assert_eq!(probes, 1);

        // Second early exit in a row: libx264 without another probe.
        let d = sel.on_exit(EARLY, probe_must_not_run);
        assert_eq!(d.codec, Codec::X264);
        assert!(d.note.unwrap().contains("2 times in a row"));
        assert_eq!(sel.on_exit(EARLY, probe_must_not_run).codec, Codec::X264);
    }

    #[test]
    fn selector_resets_early_exit_count_after_a_long_run() {
        let mut sel = CodecSelector::new(Codec::Nvenc);
        assert_eq!(sel.on_exit(EARLY, || Ok(())).codec, Codec::Nvenc);
        assert_eq!(sel.on_exit(LATE, probe_must_not_run).codec, Codec::Nvenc);
        assert_eq!(sel.on_exit(EARLY, || Ok(())).codec, Codec::Nvenc);
        assert_eq!(sel.on_exit(EARLY, probe_must_not_run).codec, Codec::X264);
    }

    #[test]
    fn selector_always_settles_on_x264_under_repeated_early_exits() {
        let mut sel = CodecSelector::new(Codec::Nvenc);
        let mut probes = 0;
        for _ in 0..100 {
            let _ = sel.on_exit(EARLY, || {
                probes += 1;
                Ok(())
            });
        }
        assert_eq!(sel.codec, Codec::X264);
        assert_eq!(probes, NVENC_MAX_EARLY_EXITS - 1);
    }

    #[test]
    fn probe_args_match_real_encode_except_frame_size() {
        for mode in [
            OutputMode::Hls,
            OutputMode::Udp {
                dest: "127.0.0.1:5004".to_owned(),
            },
        ] {
            for strip_alpha in [true, false] {
                let config = StreamConfig {
                    mode: mode.clone(),
                    strip_alpha,
                    ..StreamConfig::default()
                };
                let real = encode_args(&config, Codec::Nvenc, (config.width, config.height));
                let probe = encode_args(&config, Codec::Nvenc, NVENC_PROBE_SIZE);
                let size_at = real.iter().position(|a| a == "-s").unwrap() + 1;
                assert_eq!(real[size_at], "1920x1080");
                assert_eq!(probe[size_at], "256x256");
                let differing: Vec<usize> =
                    (0..real.len()).filter(|&i| real[i] != probe[i]).collect();
                assert_eq!(differing, [size_at]);

                let joined = probe.join(" ");
                assert!(joined.contains("-i pipe:0"), "{joined}");
                assert!(joined.contains("-c:v h264_nvenc -preset ll"), "{joined}");
                assert!(joined.ends_with("-pix_fmt yuv420p"), "{joined}");
                let pix_fmt = if strip_alpha { "rgb24" } else { "rgba" };
                assert!(joined.contains(&format!("-pix_fmt {pix_fmt} ")), "{joined}");
                let udp = matches!(mode, OutputMode::Udp { .. });
                assert_eq!(joined.contains("-zerolatency 1"), udp, "{joined}");
            }
        }
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
