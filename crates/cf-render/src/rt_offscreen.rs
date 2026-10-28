use cf_scene::grid::{GridConfig, GridVertex};
use cf_scene::tee::{TeeBox, generate_tee_border, generate_tee_fill};

use crate::rt_pipeline::{
    GEOM_FLOOR, GEOM_TEE_BOX, GEOM_TEE_BOX_BORDER, GEOM_TRAIL, GEOM_TRAIL_LINE, RtGeometry,
    RtPrimitive,
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
/// (no ball geometry in the TLAS). The tracer is `trail_tube`, the triangles
/// the raster pass draws (`generate_trail_tube`), and `trail_line`, the
/// segments it draws as the centre line (`generate_trail_line`).
pub fn build_scene_geometry(
    grid_config: &GridConfig,
    tee: &TeeBox,
    trail_tube: &[GridVertex],
    trail_line: &[GridVertex],
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

    // The tracer, vertex for vertex (position and fade): the tube, and its
    // centre line.
    let packed = |verts: &[GridVertex]| -> Vec<f32> {
        verts
            .iter()
            .flat_map(|v| {
                let [x, y, z] = v.position;
                [x, y, z, v.fade]
            })
            .collect()
    };
    if !trail_tube.is_empty() {
        geometries.push(RtGeometry {
            positions: packed(trail_tube),
            primitive: RtPrimitive::GlowTriangles,
            geom_type: GEOM_TRAIL,
            transform: glam::Mat4::IDENTITY,
        });
    }
    if !trail_line.is_empty() {
        geometries.push(RtGeometry {
            positions: packed(trail_line),
            primitive: RtPrimitive::LineSegments,
            geom_type: GEOM_TRAIL_LINE,
            transform: glam::Mat4::IDENTITY,
        });
    }

    geometries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tube() -> Vec<GridVertex> {
        let v = |x: f32, fade: f32| GridVertex {
            position: [x, 2.0, 3.0],
            fade,
        };
        vec![v(0.0, 0.1), v(1.0, 0.2), v(2.0, 0.3)]
    }

    #[test]
    fn trail_is_the_tube_vertex_for_vertex() {
        let grid = GridConfig::default();
        let tee = TeeBox::default();
        let geometries = build_scene_geometry(&grid, &tee, &tube(), &[]);
        let trail = geometries.last().unwrap();
        assert_eq!(trail.geom_type, GEOM_TRAIL);
        assert_eq!(trail.primitive, RtPrimitive::GlowTriangles);
        assert_eq!(
            trail.positions,
            vec![
                0.0, 2.0, 3.0, 0.1, //
                1.0, 2.0, 3.0, 0.2, //
                2.0, 2.0, 3.0, 0.3,
            ]
        );
        assert_eq!(trail.primitive_count(), 1);
        let without = build_scene_geometry(&grid, &tee, &[], &[]);
        assert!(without.iter().all(|g| g.geom_type != GEOM_TRAIL));
    }

    #[test]
    fn centre_line_is_the_segments_vertex_for_vertex() {
        let grid = GridConfig::default();
        let tee = TeeBox::default();
        let geometries = build_scene_geometry(&grid, &tee, &[], &tube()[..2]);
        let line = geometries.last().unwrap();
        assert_eq!(line.geom_type, GEOM_TRAIL_LINE);
        assert_eq!(line.primitive, RtPrimitive::LineSegments);
        assert_eq!(line.positions, vec![0.0, 2.0, 3.0, 0.1, 1.0, 2.0, 3.0, 0.2]);
        assert_eq!(line.primitive_count(), 1);
    }
}
