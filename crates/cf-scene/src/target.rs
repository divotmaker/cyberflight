use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::grid::{GridConfig, GridVertex};

/// A target zone on the driving range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    /// Target center in the scene frame (+X left of the target line, +Y up,
    /// +Z downrange; see [`frame`](crate::frame)), on the Y=0 ground plane.
    pub center: [f32; 3],
    /// Number of scoring rings (including bullseye).
    pub num_rings: u32,
    /// Ring spacing in meters (distance between each ring).
    pub ring_spacing_m: f32,
    /// Label text (e.g., "150").
    pub label: String,
}

impl Target {
    /// Create a target at a given distance downrange (on centerline).
    #[must_use]
    pub fn at_distance(distance_m: f32, label: &str) -> Self {
        Self {
            center: [0.0, 0.0, distance_m],
            num_rings: 5,
            ring_spacing_m: 5.0,
            label: label.to_owned(),
        }
    }

    /// Outer radius of the target in meters.
    #[must_use]
    pub fn outer_radius_m(&self) -> f32 {
        self.num_rings as f32 * self.ring_spacing_m
    }

    /// Center as a Vec3.
    #[must_use]
    pub fn center_vec3(&self) -> Vec3 {
        Vec3::from(self.center)
    }
}

/// Standard driving range target layout.
#[must_use]
pub fn default_targets() -> Vec<Target> {
    vec![
        Target::at_distance(45.72, "50"),   // 50 yards
        Target::at_distance(91.44, "100"),  // 100 yards
        Target::at_distance(137.16, "150"), // 150 yards
        Target::at_distance(182.88, "200"), // 200 yards
        Target::at_distance(228.60, "250"), // 250 yards
    ]
}

/// Line segments per ring circle.
pub const RING_SEGMENTS: u32 = 96;

/// Brightness of the inner rings relative to the outer ring, which marks the
/// green's edge.
pub const INNER_RING_BRIGHTNESS: f32 = 0.45;

/// Ring lines for `targets` as `LINE_LIST` vertex pairs on the range floor.
///
/// Each target gets `num_rings` concentric circles `ring_spacing_m` apart;
/// the outermost has [`Target::outer_radius_m`], the edge of the green that
/// [`ground::surface_at`](crate::ground::surface_at) uses. Rings sit 1 cm
/// above Y = 0 so they draw over the grid.
///
/// `fade` is the grid's distance fade (see [`GridVertex`]) over
/// `grid`'s [`fade_distance_m`](GridConfig::fade_distance_m), lowered on
/// the inner rings so they read at [`INNER_RING_BRIGHTNESS`] of the outer
/// ring (the shader's alpha is `(1 − fade)²`).
#[must_use]
pub fn generate_target_rings(targets: &[Target], grid: &GridConfig) -> Vec<GridVertex> {
    let fade_dist = grid.fade_distance_m();
    let inner_scale = INNER_RING_BRIGHTNESS.sqrt();
    let mut verts = Vec::new();
    for target in targets {
        let center = target.center_vec3();
        for ring in 0..target.num_rings {
            let radius = (ring + 1) as f32 * target.ring_spacing_m;
            let scale = if ring + 1 == target.num_rings {
                1.0
            } else {
                inner_scale
            };
            let point = |seg: u32| {
                let angle = (seg as f32 / RING_SEGMENTS as f32) * std::f32::consts::TAU;
                let p = Vec3::new(
                    center.x + radius * angle.cos(),
                    0.01,
                    center.z + radius * angle.sin(),
                );
                let near = 1.0 - (p.length() / fade_dist).min(1.0);
                GridVertex {
                    position: p.into(),
                    fade: 1.0 - near * scale,
                }
            };
            for seg in 0..RING_SEGMENTS {
                verts.push(point(seg));
                verts.push(point(seg + 1));
            }
        }
    }
    verts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_targets_count() {
        let targets = default_targets();
        assert_eq!(targets.len(), 5);
    }

    #[test]
    fn target_outer_radius() {
        let t = Target::at_distance(100.0, "test");
        assert!((t.outer_radius_m() - 25.0).abs() < 1e-6); // 5 rings * 5m
    }

    fn ring_radii(target: &Target) -> Vec<f32> {
        let center = target.center_vec3();
        let verts = generate_target_rings(std::slice::from_ref(target), &GridConfig::default());
        verts
            .chunks(2 * RING_SEGMENTS as usize)
            .map(|ring| {
                let p = Vec3::from(ring[0].position);
                Vec3::new(p.x - center.x, 0.0, p.z - center.z).length()
            })
            .collect()
    }

    #[test]
    fn rings_are_line_pairs_above_ground() {
        let targets = default_targets();
        let verts = generate_target_rings(&targets, &GridConfig::default());
        let rings: u32 = targets.iter().map(|t| t.num_rings).sum();
        assert_eq!(verts.len(), (rings * RING_SEGMENTS * 2) as usize);
        for v in &verts {
            assert!(v.position[1] > 0.0, "ring should be slightly above Y=0");
            assert!((0.0..=1.0).contains(&v.fade));
        }
    }

    #[test]
    fn outer_ring_is_the_green_edge() {
        let target = Target {
            center: [-8.0, 0.0, 120.0],
            num_rings: 3,
            ring_spacing_m: 4.0,
            label: "T".to_owned(),
        };
        let radii = ring_radii(&target);
        assert_eq!(radii.len(), 3);
        let outer = *radii.last().expect("rings");
        assert!((outer - target.outer_radius_m()).abs() < 1e-3);

        // Just inside the drawn outer ring is green; just outside is fairway.
        let c = target.center_vec3();
        let targets = [target];
        let edge = |r: f32| crate::ground::surface_at(f64::from(c.x + r), f64::from(c.z), &targets);
        assert_eq!(edge(outer - 0.01), dimple::ground::Surface::GREEN);
        assert_eq!(edge(outer + 0.01), dimple::ground::Surface::FAIRWAY);
    }

    #[test]
    fn rings_are_concentric_and_evenly_spaced() {
        let t = Target::at_distance(100.0, "test");
        let radii = ring_radii(&t);
        for (i, r) in radii.iter().enumerate() {
            assert!((r - (i + 1) as f32 * t.ring_spacing_m).abs() < 1e-3);
        }
    }

    #[test]
    fn inner_rings_are_dimmer_than_the_outer_ring() {
        let t = Target::at_distance(100.0, "test");
        let verts = generate_target_rings(std::slice::from_ref(&t), &GridConfig::default());
        let per_ring = 2 * RING_SEGMENTS as usize;
        // Same angle on the innermost and outermost ring.
        let alpha = |v: &GridVertex| (1.0 - v.fade).powi(2);
        let inner = &verts[0];
        let outer = &verts[(t.num_rings as usize - 1) * per_ring];
        let dist = |v: &GridVertex| Vec3::from(v.position).length();
        let fade_dist = GridConfig::default().fade_distance_m();
        let expected_inner = (1.0 - dist(inner) / fade_dist).powi(2) * INNER_RING_BRIGHTNESS;
        let expected_outer = (1.0 - dist(outer) / fade_dist).powi(2);
        assert!((alpha(inner) - expected_inner).abs() < 1e-5);
        assert!((alpha(outer) - expected_outer).abs() < 1e-5);
        // The inner ring is nearer the tee, so it fades less with distance;
        // only its lower brightness can make it the dimmer ring.
        assert!(dist(inner) < dist(outer));
        assert!(alpha(inner) < alpha(outer));
    }

    #[test]
    fn target_serialization_roundtrip() {
        let t = Target::at_distance(150.0, "150");
        let json = serde_json::to_string(&t).unwrap();
        let t2: Target = serde_json::from_str(&json).unwrap();
        assert_eq!(t.label, t2.label);
        assert!((t.center[2] - t2.center[2]).abs() < 1e-6);
    }
}
