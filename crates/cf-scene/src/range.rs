use crate::camera::Camera;
use crate::green::{DEFAULT_GREEN, GreenConfig};
use crate::grid::GridConfig;

/// Complete driving range scene.
#[derive(Debug)]
pub struct Range {
    pub grid: GridConfig,
    /// The green, if any: the only part of the floor that plays as green
    /// (see [`ground`](crate::ground)) and the only circle drawn on it.
    /// `None` plays the whole range as fairway and draws no circle.
    pub green: Option<GreenConfig>,
    pub camera: Camera,
}

impl Range {
    /// Create a default driving range scene with [`DEFAULT_GREEN`] (no
    /// green, so the floor is all fairway).
    #[must_use]
    pub fn new() -> Self {
        Self {
            grid: GridConfig::default(),
            green: DEFAULT_GREEN,
            camera: Camera::driving_range(),
        }
    }
}

impl Default for Range {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_range_has_no_green() {
        assert_eq!(Range::new().green, None);
    }
}
