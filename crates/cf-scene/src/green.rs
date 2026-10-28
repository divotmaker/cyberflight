//! The range's optional green: one circle on the target line.
//!
//! A [`Range`](crate::range::Range) has at most one green. Inside it the
//! range floor is [`Surface::GREEN`](dimple::ground::Surface::GREEN), and
//! the renderers draw its edge as a single ring ([`generate_green_edge`]).
//! The default range has none ([`DEFAULT_GREEN`] is `None`), so the whole
//! floor is fairway.
//!
//! To play a green, set [`DEFAULT_GREEN`] to a [`GreenConfig`], e.g.
//! `Some(GreenConfig { distance_m: 137.16, diameter_m: 30.0 })` for a 30 m
//! green 150 yd out, or set [`Range::green`](crate::range::Range::green)
//! on a range before it is drawn and played.

use glam::Vec3;

use crate::grid::{GridConfig, GridVertex};
use crate::target::push_ring;

/// A circular green centred on the target line.
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

/// The edge of `green` as `LINE_LIST` vertex pairs on the range floor:
/// one full-strength ring of [`GreenConfig::radius_m`] with
/// [`RING_SEGMENTS`](crate::target::RING_SEGMENTS) segments, 1 cm above
/// Y = 0 and faded like `grid`, in the style of the target rings
/// ([`generate_target_rings`](crate::target::generate_target_rings)).
#[must_use]
pub fn generate_green_edge(green: &GreenConfig, grid: &GridConfig) -> Vec<GridVertex> {
    let mut verts = Vec::new();
    push_ring(
        &mut verts,
        green.center_vec3(),
        green.radius_m(),
        1.0,
        grid.fade_distance_m(),
    );
    verts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::RING_SEGMENTS;

    #[test]
    fn default_range_has_no_green() {
        assert_eq!(DEFAULT_GREEN, None);
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
    fn edge_is_one_full_strength_ring_of_half_the_diameter() {
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
            // Full strength: the plain grid distance fade.
            let expected = (p.length() / fade_dist).min(1.0);
            assert!((v.fade - expected).abs() < 1e-6);
        }
        // Just inside the drawn edge is green; just outside is fairway.
        let edge = |r: f64| crate::ground::surface_at(r, 140.0, Some(green));
        assert_eq!(edge(13.99), dimple::ground::Surface::GREEN);
        assert_eq!(edge(14.01), dimple::ground::Surface::FAIRWAY);
    }
}
