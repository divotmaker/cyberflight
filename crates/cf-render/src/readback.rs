/// CPU-accessible frame data read back from GPU.
#[derive(Debug)]
pub struct FrameBuffers {
    /// Color buffer (RGBA8, row-major).
    pub color: Vec<u8>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
}

impl FrameBuffers {
    /// Get pixel color at (x, y). Returns [R, G, B, A].
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * self.width + x) * 4) as usize;
        [
            self.color[idx],
            self.color[idx + 1],
            self.color[idx + 2],
            self.color[idx + 3],
        ]
    }
}
