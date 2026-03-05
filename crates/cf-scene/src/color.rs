use glam::Vec4;

// Neon color palette for the cyberflight aesthetic.
// All colors are linear RGB with alpha. HDR values (>1.0) for bloom sources.

/// Neon cyan — primary grid and UI color.
pub const CYAN: Vec4 = Vec4::new(0.0, 1.0, 1.0, 1.0);

/// Neon magenta — accents, bullseye targets.
pub const MAGENTA: Vec4 = Vec4::new(1.0, 0.0, 1.0, 1.0);

/// Dimmed magenta — floor reflections.
pub const MAGENTA_DIM: Vec4 = Vec4::new(0.66, 0.0, 0.66, 0.66);

/// Bright white — ball, trail core.
pub const WHITE: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0);

/// Hot orange — warning, fade shot trails.
pub const ORANGE: Vec4 = Vec4::new(1.0, 0.533, 0.0, 1.0);

/// Cool blue — draw shot trails.
pub const BLUE: Vec4 = Vec4::new(0.0, 0.5, 1.0, 1.0);

/// Black background.
pub const BACKGROUND: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0);

/// HUD panel background — semi-transparent dark.
pub const PANEL_BG: Vec4 = Vec4::new(0.0, 0.0, 0.1, 0.7);

/// HDR multiplier for bloom-emitting elements.
pub const BLOOM_INTENSITY: f32 = 2.5;

/// Apply bloom intensity to a color (multiply RGB, keep alpha).
#[must_use]
pub fn bloom(color: Vec4) -> Vec4 {
    Vec4::new(
        color.x * BLOOM_INTENSITY,
        color.y * BLOOM_INTENSITY,
        color.z * BLOOM_INTENSITY,
        color.w,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_multiplies_rgb() {
        let c = bloom(CYAN);
        assert!(c.x < 0.01); // cyan has no red
        assert!(c.y > 2.0); // green is amplified
        assert!(c.z > 2.0); // blue is amplified
        assert!((c.w - 1.0).abs() < 1e-6); // alpha unchanged
    }
}
