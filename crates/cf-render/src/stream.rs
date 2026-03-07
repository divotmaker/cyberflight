use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Instant;

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
        Self::Udp { dest: "127.0.0.1:5004".to_owned() }
    }
}

impl OutputMode {
    /// Parse from CLI string. "hls" or "udp" (with optional `:address:port`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("hls") {
            Some(Self::Hls)
        } else if s.eq_ignore_ascii_case("udp") {
            Some(Self::Udp { dest: "127.0.0.1:5004".to_owned() })
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

/// Pipes raw RGBA frames to an ffmpeg subprocess for HLS streaming.
pub struct StreamEncoder {
    child: Child,
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

        let child = spawn_encoder(&config)?;
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
            child,
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
        match self.child.try_wait() {
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
        drop(self.child.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();

        self.child = spawn_encoder(&self.config)?;
        eprintln!("StreamEncoder: restarted (was at frame {})", self.frame_count);
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
        // Close stdin to signal EOF to ffmpeg.
        drop(self.child.stdin.take());
        match self.child.wait() {
            Ok(status) => {
                if !status.success() {
                    eprintln!("ffmpeg exited with status: {status}");
                }
            }
            Err(e) => eprintln!("failed to wait for ffmpeg: {e}"),
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
            opts.extend_from_slice(&["-delay", "0", "-zerolatency", "1",
                "-g", "30", "-keyint_min", "30"]);
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
        "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s", &size,
        "-r", &fps,
        "-i", "pipe:0",
        "-c:v", codec,
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
                "-f", "hls",
                "-hls_time", &hls_time,
                "-hls_list_size", &hls_list,
                "-hls_flags", "delete_segments",
            ]);
            cmd.arg(output);
        }
        OutputMode::Udp { dest } => {
            let udp_url = format!("udp://{dest}?pkt_size=1316");
            cmd.args([
                "-f", "mpegts",
                "-flush_packets", "1",
                "-mpegts_flags", "resend_headers",
            ]);
            cmd.arg(&udp_url);
        }
    }

    cmd.stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.spawn()
}
