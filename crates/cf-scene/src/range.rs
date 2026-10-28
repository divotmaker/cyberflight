use crate::camera::Camera;
use crate::grid::GridConfig;
use crate::target::Target;

/// Complete driving range scene.
#[derive(Debug)]
pub struct Range {
    pub grid: GridConfig,
    pub targets: Vec<Target>,
    pub camera: Camera,
}

impl Range {
    /// Create a default driving range scene.
    #[must_use]
    pub fn new() -> Self {
        Self {
            grid: GridConfig::default(),
            targets: crate::target::default_targets(),
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
}
