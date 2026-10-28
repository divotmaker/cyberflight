//! The range's optional green: one circle on the target line.
//!
//! A [`Range`](crate::range::Range) has at most one green. Inside it the
//! range floor is [`Surface::GREEN`](dimple::ground::Surface::GREEN), and
//! the renderers draw its edge as a single ring ([`generate_green_edge`]),
//! the only circle on the range floor. A new range has none
//! ([`DEFAULT_GREEN`] is `None`), so the whole floor is fairway and no
//! circle is drawn.
//!
//! [`Range::green`](crate::range::Range::green) is the knob: set it to a
//! [`GreenConfig`] before the range is drawn and played, e.g.
//! `range.green = Some(GreenConfig { distance_m: 137.16, diameter_m: 30.0 })`
//! for a 30 m green 150 yd out. [`DEFAULT_GREEN`] is the value
//! [`Range::new`](crate::range::Range::new) starts with; changing it
//! changes the default for every range.

use glam::Vec3;

use crate::grid::{GridConfig, GridVertex};

/// A circular green centred on the target line.
///
/// `diameter_m` must be positive and finite (checked by a debug assertion
/// when the radius is taken); `distance_m` must be finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreenConfig {
    /// Distance downrange from the tee to the green's centre, in metres.
    pub distance_m: f32,
    /// Diameter of the green, in metres.
    pub diameter_m: f32,
}

/// The green of [`Range::new`](crate::range::Range::new): none, so the
/// range plays as fairway.
pub const DEFAULT_GREEN: Option<GreenConfig> = None;

impl GreenConfig {
    /// Radius of the green in metres: half the diameter.
    #[must_use]
    pub fn radius_m(&self) -> f32 {
        debug_assert!(
            self.diameter_m.is_finite() && self.diameter_m > 0.0,
            "green diameter must be positive and finite: {}",
            self.diameter_m
        );
        self.diameter_m * 0.5
    }

    /// Centre in the scene frame (see [`frame`](crate::frame)), on the
    /// Y = 0 ground plane.
    #[must_use]
    pub fn center_vec3(&self) -> Vec3 {
        Vec3::new(0.0, 0.0, self.distance_m)
    }

    /// Whether the scene-frame ground point (`x`, `z`) is on the green
    /// (edge included).
    #[must_use]
    pub fn contains(&self, x: f64, z: f64) -> bool {
        let dz = z - f64::from(self.distance_m);
        let r = f64::from(self.radius_m());
        x * x + dz * dz <= r * r
    }
}

/// Line segments in the green's edge.
pub const RING_SEGMENTS: u32 = 96;

/// The edge of `green` as `LINE_LIST` vertex pairs on the range floor:
/// one ring of [`GreenConfig::radius_m`] with [`RING_SEGMENTS`] segments,
/// 1 cm above Y = 0 so it draws over the grid, with the grid's distance
/// fade over `grid`'s [`fade_distance_m`](GridConfig::fade_distance_m)
/// (see [`GridVertex`]).
#[must_use]
pub fn generate_green_edge(green: &GreenConfig, grid: &GridConfig) -> Vec<GridVertex> {
    let center = green.center_vec3();
    let radius = green.radius_m();
    let fade_dist = grid.fade_distance_m();
    let point = |seg: u32| {
        let angle = (seg as f32 / RING_SEGMENTS as f32) * std::f32::consts::TAU;
        let p = Vec3::new(
            center.x + radius * angle.cos(),
            0.01,
            center.z + radius * angle.sin(),
        );
        GridVertex {
            position: p.into(),
            fade: (p.length() / fade_dist).min(1.0),
        }
    };
    let mut verts = Vec::with_capacity(2 * RING_SEGMENTS as usize);
    for seg in 0..RING_SEGMENTS {
        verts.push(point(seg));
        verts.push(point(seg + 1));
    }
    verts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "positive and finite")]
    fn non_positive_diameter_is_rejected() {
        let green = GreenConfig {
            distance_m: 100.0,
            diameter_m: -10.0,
        };
        let _ = green.radius_m();
    }

    #[test]
    fn green_is_a_circle_on_the_target_line() {
        let green = GreenConfig {
            distance_m: 120.0,
            diameter_m: 20.0,
        };
        assert!((green.radius_m() - 10.0).abs() < 1e-6);
        assert_eq!(green.center_vec3(), Vec3::new(0.0, 0.0, 120.0));
        assert!(green.contains(0.0, 120.0));
        assert!(green.contains(9.99, 120.0));
        assert!(green.contains(0.0, 110.01));
        assert!(!green.contains(10.01, 120.0));
        assert!(!green.contains(0.0, 130.01));
        assert!(!green.contains(7.2, 127.2));
    }

    #[test]
    fn edge_is_one_ring_of_half_the_diameter() {
        let green = GreenConfig {
            distance_m: 140.0,
            diameter_m: 28.0,
        };
        let grid = GridConfig::default();
        let verts = generate_green_edge(&green, &grid);
        assert_eq!(verts.len(), 2 * RING_SEGMENTS as usize);
        let center = green.center_vec3();
        let fade_dist = grid.fade_distance_m();
        for v in &verts {
            let p = Vec3::from(v.position);
            assert!(p.y > 0.0, "edge should be slightly above Y=0");
            let r = Vec3::new(p.x - center.x, 0.0, p.z - center.z).length();
            assert!((r - 14.0).abs() < 1e-3, "radius {r}");
            // The grid's distance fade.
            let expected = (p.length() / fade_dist).min(1.0);
            assert!((v.fade - expected).abs() < 1e-6);
        }
        // Just inside the drawn edge is green; just outside is fairway.
        let edge = |r: f64| crate::ground::surface_at(r, 140.0, Some(green));
        assert_eq!(edge(13.99), dimple::ground::Surface::GREEN);
        assert_eq!(edge(14.01), dimple::ground::Surface::FAIRWAY);
    }
}
