use glam::Vec4;

// Neon color palette for the cyberflight aesthetic.
// All colors are linear RGB with alpha.

/// Neon cyan — primary grid and UI color.
pub const CYAN: Vec4 = Vec4::new(0.0, 1.0, 1.0, 1.0);

/// Neon magenta — ball and flight accents.
pub const MAGENTA: Vec4 = Vec4::new(1.0, 0.0, 1.0, 1.0);

/// Green edge — neon green, #39FF14 as linear RGB (the colours here are
/// linear; the sRGB target encodes them on output) at 90% alpha. The hue
/// sits well clear of the cyan grid, so the thin ring reads as its own line
/// without competing with the ball tracer.
pub const GREEN_EDGE: Vec4 = Vec4::new(0.041, 1.0, 0.007, 0.9);

/// Black background.
pub const BACKGROUND: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0);
