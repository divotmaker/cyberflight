use glam::Vec4;

// Neon color palette for the cyberflight aesthetic.
// All colors are linear RGB with alpha.

/// Neon cyan — primary grid and UI color.
pub const CYAN: Vec4 = Vec4::new(0.0, 1.0, 1.0, 1.0);

/// Neon magenta — ball and flight accents.
pub const MAGENTA: Vec4 = Vec4::new(1.0, 0.0, 1.0, 1.0);

/// Green edge — neon green, #39FF14 as linear RGB (the colors here are
/// linear; the sRGB target encodes them on output) at 90% alpha. The hue
/// sits well clear of the cyan grid, so the thin ring reads as its own line
/// without competing with the ball tracer.
pub const GREEN_EDGE: Vec4 = Vec4::new(0.041, 1.0, 0.007, 0.9);

/// Sand — orange, #FF8800 as linear RGB: a hole's bunkers.
pub const SAND: Vec4 = Vec4::new(1.0, 0.246, 0.0, 1.0);

/// Water — neon blue, #3070FF as linear RGB: a hole's ponds. Bluer and
/// darker than the cyan grid, so the water reads as its own surface.
pub const WATER: Vec4 = Vec4::new(0.030, 0.162, 1.0, 1.0);

/// Tree — dim violet, #9D4DFF as linear RGB at 75% alpha: a hole's
/// trees. Clear of the green ring's green and the tracer's magenta, and
/// dimmer than both.
pub const TREE: Vec4 = Vec4::new(0.337, 0.074, 1.0, 0.75);

/// Pin — white: a hole's cup rim, pin and flag.
pub const PIN: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0);

/// Black background.
pub const BACKGROUND: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0);
