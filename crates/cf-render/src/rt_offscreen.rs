use cf_scene::grid::GridConfig;
use cf_scene::tee::{TeeBox, generate_tee_border, generate_tee_fill};
use cf_scene::trail::TrailPoint;

use crate::rt_pipeline::{
    GEOM_FLOOR, GEOM_TEE_BOX, GEOM_TEE_BOX_BORDER, GEOM_TRAIL, RtGeometry, RtPrimitive,
};

// ── Scene geometry builders ──

/// Extract packed f32x3 positions from `GridVertex` data.
pub fn grid_verts_to_positions(verts: &[cf_scene::grid::GridVertex]) -> Vec<f32> {
    let mut positions = Vec::with_capacity(verts.len() * 3);
    for v in verts {
        positions.push(v.position[0]);
        positions.push(v.position[1]);
        positions.push(v.position[2]);
    }
    positions
}

/// Build the full RT scene geometry from the driving range config.
///
/// Ball reflections are computed analytically in the floor closest-hit shader
/// (no ball geometry in the TLAS). The trail is a round tube around
/// `trail_points` (oldest first), intersected procedurally by `rt.rint`.
pub fn build_scene_geometry(
    grid_config: &GridConfig,
    tee: &TeeBox,
    trail_points: &[glam::Vec3],
) -> Vec<RtGeometry> {
    let mut geometries = Vec::new();

    // Floor: tee box + fairway (non-rectangular compound surface)
    let floor_verts = cf_scene::grid::generate_floor_quad(grid_config, tee.half_size);
    geometries.push(RtGeometry {
        positions: grid_verts_to_positions(&floor_verts),
        primitive: RtPrimitive::Triangles,
        geom_type: GEOM_FLOOR,
        transform: glam::Mat4::IDENTITY,
    });

    // Tee box: filled quad (already at TEE_ELEVATION from generate_tee_fill)
    let tee_verts = generate_tee_fill(tee);
    let tee_box_transform = glam::Mat4::IDENTITY;
    geometries.push(RtGeometry {
        positions: grid_verts_to_positions(&tee_verts),
        primitive: RtPrimitive::Triangles,
        geom_type: GEOM_TEE_BOX,
        transform: tee_box_transform,
    });

    // Tee box border: cyan outline
    let border_verts = generate_tee_border(tee);
    geometries.push(RtGeometry {
        positions: grid_verts_to_positions(&border_verts),
        primitive: RtPrimitive::Triangles,
        geom_type: GEOM_TEE_BOX_BORDER,
        transform: tee_box_transform,
    });

    // Trail: the tube's axis, one segment per pair of neighbouring points.
    if trail_points.len() >= 2 {
        geometries.push(RtGeometry {
            positions: trail_points.iter().flat_map(|p| p.to_array()).collect(),
            primitive: RtPrimitive::Tube,
            geom_type: GEOM_TRAIL,
            transform: glam::Mat4::IDENTITY,
        });
    }

    geometries
}

/// Tracer fade value `t` of a trail point: 1 at `current_time`, 0 once the
/// point is `lifetime` old. The raster tracer (`generate_trail_glow`) fades by
/// the same value.
#[must_use]
pub fn trail_fade_t(point_time: f64, current_time: f64, lifetime: f64) -> f32 {
    (1.0 - (current_time - point_time) / lifetime).clamp(0.0, 1.0) as f32
}

/// The trail a flight's RT reflection is built from: exactly the points the
/// raster tracer still draws.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RtTrail {
    /// Points younger than the trail lifetime, oldest first.
    pub points: Vec<glam::Vec3>,
    /// Arc length of `points` in meters: the reflection fades over this
    /// distance from the ball.
    pub fade_dist: f32,
    /// Tracer fade value ([`trail_fade_t`]) at the oldest point: the
    /// reflection's fade value at the far end of `fade_dist`.
    pub tail_t: f32,
}

impl RtTrail {
    /// Keep the points of `trail` younger than `lifetime` at `current_time`.
    #[must_use]
    pub fn new(trail: &[TrailPoint], current_time: f64, lifetime: f64) -> Self {
        let kept: Vec<&TrailPoint> = trail
            .iter()
            .filter(|p| (current_time - p.time) < lifetime)
            .collect();
        let points: Vec<glam::Vec3> = kept.iter().map(|p| p.position).collect();
        let tail_t = kept
            .first()
            .map_or(0.0, |p| trail_fade_t(p.time, current_time, lifetime));
        Self {
            fade_dist: arc_length(&points),
            points,
            tail_t,
        }
    }
}

/// Total length of the polyline through `points`, in meters.
#[must_use]
pub fn arc_length(points: &[glam::Vec3]) -> f32 {
    points.windows(2).map(|w| (w[1] - w[0]).length()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(z: f32, time: f64) -> TrailPoint {
        TrailPoint {
            position: glam::Vec3::new(0.0, 0.0, z),
            time,
        }
    }

    #[test]
    fn rt_trail_keeps_points_the_tracer_draws() {
        // Points at t = 0, 1, 2, 3 s; at 4 s with a 3 s lifetime the first
        // two are too old.
        let trail = [
            point(0.0, 0.0),
            point(10.0, 1.0),
            point(25.0, 2.0),
            point(45.0, 3.0),
        ];
        let rt = RtTrail::new(&trail, 4.0, 3.0);
        assert_eq!(rt.points.len(), 2);
        assert_eq!(rt.points[0].z, 25.0);
        assert!((rt.fade_dist - 20.0).abs() < 1e-5);
        assert!((rt.tail_t - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn rt_trail_single_point() {
        let rt = RtTrail::new(&[point(0.0, 0.0), point(30.0, 2.5)], 4.0, 3.0);
        assert_eq!(rt.points, vec![glam::Vec3::new(0.0, 0.0, 30.0)]);
        assert_eq!(rt.fade_dist, 0.0);
        assert!((rt.tail_t - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rt_trail_empty_when_all_points_aged_out() {
        let rt = RtTrail::new(&[point(0.0, 0.0), point(5.0, 0.5)], 10.0, 3.0);
        assert_eq!(rt, RtTrail::default());
    }

    #[test]
    fn trail_is_a_tube_around_its_points() {
        let grid = GridConfig::default();
        let tee = TeeBox::default();
        let points = [
            glam::Vec3::new(0.0, 1.0, 2.0),
            glam::Vec3::new(3.0, 4.0, 5.0),
        ];
        let geometries = build_scene_geometry(&grid, &tee, &points);
        let trail = geometries.last().unwrap();
        assert_eq!(trail.geom_type, GEOM_TRAIL);
        assert_eq!(trail.primitive, RtPrimitive::Tube);
        assert_eq!(trail.positions, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(trail.primitive_count(), 1);
        let without = build_scene_geometry(&grid, &tee, &points[..1]);
        assert!(without.iter().all(|g| g.geom_type != GEOM_TRAIL));
    }

    #[test]
    fn trail_fade_t_clamps() {
        assert_eq!(trail_fade_t(5.0, 5.0, 3.0), 1.0);
        assert_eq!(trail_fade_t(0.0, 5.0, 3.0), 0.0);
        assert!((trail_fade_t(4.0, 5.0, 4.0) - 0.75).abs() < 1e-6);
    }
}
