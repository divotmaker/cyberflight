use crate::camera::Camera;
use crate::green::{DEFAULT_GREEN, GreenConfig};
use crate::grid::GridConfig;
use crate::target::Target;

/// Complete driving range scene.
#[derive(Debug)]
pub struct Range {
    pub grid: GridConfig,
    /// Aiming targets: drawn as rings, no effect on the range floor.
    pub targets: Vec<Target>,
    /// The green, if any: the only part of the floor that plays as green
    /// (see [`ground`](crate::ground)). `None` plays the whole range as
    /// fairway.
    pub green: Option<GreenConfig>,
    pub camera: Camera,
}

impl Range {
    /// Create a default driving range scene: the standard targets and
    /// [`DEFAULT_GREEN`] (no green, so the floor is all fairway).
    #[must_use]
    pub fn new() -> Self {
        Self {
            grid: GridConfig::default(),
            targets: crate::target::default_targets(),
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
    fn new_range_has_targets() {
        let range = Range::new();
        assert!(!range.targets.is_empty());
    }

    #[test]
    fn new_range_has_no_green() {
        assert_eq!(Range::new().green, None);
    }
}
