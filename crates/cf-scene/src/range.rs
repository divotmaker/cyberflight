use crate::camera::Camera;
use crate::color;
use crate::green::{DEFAULT_GREEN, GreenConfig, generate_green_edge};
use crate::grid::{GridConfig, generate_grid_vertices};
use crate::hole::Hole;
use crate::hole_draw;
use crate::scenery::{Layer, Scenery};

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

impl Range {
    /// The hole range: [`Hole::no_1`] on a grid 100 yd wide, seen from the
    /// tee camera aimed at the foot of the pin.
    #[must_use]
    pub fn hole() -> Self {
        let hole = Hole::no_1();
        let (cx, cz) = hole.cup;
        let camera = Camera::tee(glam::Vec3::new(
            cx as f32,
            hole.height(cx, cz) as f32,
            cz as f32,
        ));
        Self {
            grid: GridConfig {
                spacing: 10.0,
                downrange: 450.0,
                lateral: 50.0,
            },
            ground: Ground::Hole(hole),
            camera,
        }
    }

    /// The static scene to draw. Flat ground: the grid in cyan, then the
    /// green's edge when there is a green, over the reflective floor of the
    /// grid. A hole: the black terrain mesh and the cup's disc as fills;
    /// the draped grid (cyan), the fairway outline (cyan), the bunkers
    /// (sand), the green's ring and the cup's rim, pin and flag (white) as
    /// lines; no floor, so no reflections.
    #[must_use]
    pub fn scenery(&self) -> Scenery {
        match &self.ground {
            Ground::Flat { green } => {
                let mut lines = vec![Layer {
                    color: color::CYAN,
                    vertices: generate_grid_vertices(&self.grid),
                }];
                if let Some(green) = green {
                    lines.push(Layer {
                        color: color::GREEN_EDGE,
                        vertices: generate_green_edge(green, &self.grid),
                    });
                }
                Scenery {
                    fills: Vec::new(),
                    lines,
                    floor: Some(self.grid),
                }
            }
            Ground::Hole(hole) => {
                let layer = |color, vertices| Layer { color, vertices };
                let bunkers = hole
                    .bunkers
                    .iter()
                    .flat_map(|b| hole_draw::bunker_lines(hole, b))
                    .collect();
                Scenery {
                    fills: vec![
                        layer(color::BACKGROUND, hole_draw::terrain_mesh(hole, &self.grid)),
                        layer(color::BACKGROUND, hole_draw::cup_disc(hole)),
                    ],
                    lines: vec![
                        layer(color::CYAN, hole_draw::draped_grid(hole, &self.grid)),
                        layer(color::CYAN, hole_draw::fairway_outline(hole)),
                        layer(color::SAND, bunkers),
                        layer(color::GREEN_EDGE, hole_draw::green_ring(hole)),
                        layer(color::PIN, hole_draw::cup_and_pin(hole)),
                    ],
                    floor: None,
                }
            }
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

    use crate::green::RING_SEGMENTS;
    use glam::Vec3;

    #[test]
    fn default_range_draws_the_grid_over_a_floor() {
        let range = Range::new();
        let scenery = range.scenery();
        assert!(scenery.fills.is_empty());
        assert_eq!(scenery.lines.len(), 1);
        assert_eq!(scenery.lines[0].color, color::CYAN);
        assert!(scenery.floor.is_some());
    }

    #[test]
    fn a_green_draws_exactly_one_circle() {
        let mut range = Range::new();
        let green = GreenConfig {
            distance_m: 137.16,
            diameter_m: 30.0,
        };
        range.ground = Ground::Flat { green: Some(green) };
        let scenery = range.scenery();
        assert_eq!(scenery.lines.len(), 2);
        let edge = &scenery.lines[1];
        assert_eq!(edge.color, color::GREEN_EDGE);
        assert_eq!(edge.vertices.len(), 2 * RING_SEGMENTS as usize);
        let center = green.center_vec3();
        for v in &edge.vertices {
            let p = Vec3::from(v.position);
            let r = Vec3::new(p.x - center.x, 0.0, p.z - center.z).length();
            assert!((r - green.radius_m()).abs() < 1e-3, "radius {r}");
        }
    }

    #[test]
    fn the_hole_has_no_floor_and_its_mesh_lies_under_the_ground() {
        let range = Range::hole();
        let scenery = range.scenery();
        assert!(scenery.floor.is_none());
        let hole = range.ground.hole().expect("a hole");
        for v in scenery.fills[0].vertices.iter().step_by(13) {
            let [x, y, z] = v.position.map(f64::from);
            let want = hole.height(x, z) - hole_draw::MESH_DROP_M;
            assert!((y - want).abs() < 1e-3, "({x}, {z})");
        }
    }

    #[test]
    fn new_range_has_no_green() {
        assert_eq!(Range::new().ground, Ground::Flat { green: None });
    }
}
