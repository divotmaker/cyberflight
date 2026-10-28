use glam::Vec4;

// Neon color palette for the cyberflight aesthetic.
// All colors are linear RGB with alpha.

/// Neon cyan — primary grid and UI color.
pub const CYAN: Vec4 = Vec4::new(0.0, 1.0, 1.0, 1.0);

/// Neon magenta — accents, bullseye targets.
pub const MAGENTA: Vec4 = Vec4::new(1.0, 0.0, 1.0, 1.0);

/// Target rings — the orange accent held below full strength, so rings read
/// against the cyan grid without competing with the ball tracer.
pub const TARGET_RING: Vec4 = Vec4::new(1.0, 0.533, 0.0, 0.6);

/// Black background.
pub const BACKGROUND: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0);
