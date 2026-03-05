use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A target zone on the driving range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    /// Target center position in world space (Y=0 ground plane).
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
            center: [distance_m, 0.0, 0.0],
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

/// Vertex for rendering target rings.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TargetVertex {
    pub position: [f32; 3],
    /// Which ring this vertex belongs to (0 = bullseye, outer = num_rings-1).
    pub ring_index: f32,
}

/// Generate ring vertices for a target.
///
/// Returns vertices for line strips forming concentric circles.
/// Each ring is `segments` line segments.
#[must_use]
pub fn generate_target_ring_vertices(target: &Target, segments: u32) -> Vec<TargetVertex> {
    let mut verts = Vec::new();
    let center = target.center_vec3();

    for ring in 0..target.num_rings {
        let radius = (ring + 1) as f32 * target.ring_spacing_m;
        for seg in 0..=segments {
            let angle = (seg as f32 / segments as f32) * std::f32::consts::TAU;
            let x = center.x + radius * angle.cos();
            let z = center.z + radius * angle.sin();
            verts.push(TargetVertex {
                position: [x, 0.01, z], // slightly above ground to avoid z-fighting
                ring_index: ring as f32,
            });
        }
    }

    verts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_vertex_size() {
        assert_eq!(std::mem::size_of::<TargetVertex>(), 16);
    }

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

    #[test]
    fn ring_vertices_nonempty() {
        let t = Target::at_distance(100.0, "test");
        let verts = generate_target_ring_vertices(&t, 32);
        assert!(!verts.is_empty());
    }

    #[test]
    fn ring_vertices_above_ground() {
        let t = Target::at_distance(100.0, "test");
        let verts = generate_target_ring_vertices(&t, 32);
        for v in &verts {
            assert!(v.position[1] > 0.0, "ring should be slightly above Y=0");
        }
    }

    #[test]
    fn target_serialization_roundtrip() {
        let t = Target::at_distance(150.0, "150");
        let json = serde_json::to_string(&t).unwrap();
        let t2: Target = serde_json::from_str(&json).unwrap();
        assert_eq!(t.label, t2.label);
        assert!((t.center[0] - t2.center[0]).abs() < 1e-6);
    }
}
