use glam::Vec4;

// Neon color palette for the cyberflight aesthetic.
// All colors are linear RGB with alpha.

/// Neon cyan — primary grid and UI color.
pub const CYAN: Vec4 = Vec4::new(0.0, 1.0, 1.0, 1.0);

/// Neon magenta — ball and flight accents.
pub const MAGENTA: Vec4 = Vec4::new(1.0, 0.0, 1.0, 1.0);

/// Green edge — the orange accent held below full strength, so the ring
/// reads against the cyan grid without competing with the ball tracer.
pub const GREEN_EDGE: Vec4 = Vec4::new(1.0, 0.533, 0.0, 0.6);

/// Black background.
pub const BACKGROUND: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0);
