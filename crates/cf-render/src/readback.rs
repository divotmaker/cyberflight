/// A rendered frame: RGBA8 sRGB, rows top to bottom, `width * 4` bytes per row.
/// Borrows the renderer's readback memory until the next `render`.
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    /// `width * height * 4` bytes.
    pub pixels: &'a [u8],
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
}
