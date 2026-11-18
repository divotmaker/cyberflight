use crate::camera::Camera;
use crate::green::{DEFAULT_GREEN, GreenConfig};
use crate::grid::GridConfig;
use crate::hole::Hole;

/// The ground a range is played on.
#[derive(Debug, Clone, PartialEq)]
pub enum Ground {
    /// Level ground at Y = 0. The green, if any, is the only part of the
    /// floor that plays as green (see [`ground`](crate::ground)) and the
    /// only circle drawn on it. `None` plays the whole range as fairway and
    /// draws no circle.
    Flat { green: Option<GreenConfig> },
    /// A hole's shaped ground (see [`hole`](crate::hole)).
    Hole(Hole),
}

impl Ground {
    /// The green of flat ground, `None` on a hole.
    #[must_use]
    pub fn flat_green(&self) -> Option<GreenConfig> {
        match self {
            Ground::Flat { green } => *green,
            Ground::Hole(_) => None,
        }
    }

    /// The hole, if this ground is one.
    #[must_use]
    pub fn hole(&self) -> Option<&Hole> {
        match self {
            Ground::Flat { .. } => None,
            Ground::Hole(hole) => Some(hole),
        }
    }
}

/// Complete range scene.
#[derive(Debug)]
pub struct Range {
    pub grid: GridConfig,
    pub ground: Ground,
    pub camera: Camera,
}

impl Range {
    /// Create a default driving range scene with [`DEFAULT_GREEN`] (no
    /// green, so the floor is all fairway).
    #[must_use]
    pub fn new() -> Self {
        Self {
            grid: GridConfig::default(),
            ground: Ground::Flat {
                green: DEFAULT_GREEN,
            },
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
        assert_eq!(Range::new().ground, Ground::Flat { green: None });
    }
}
