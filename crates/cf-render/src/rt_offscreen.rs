use cf_scene::grid::{GridConfig, GridVertex};
use cf_scene::tee::{TeeBox, generate_tee_border, generate_tee_fill};

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
/// (no ball geometry in the TLAS). The trail is `trail_tube`, the triangles
/// the raster pass draws as the tracer (`generate_trail_tube`).
pub fn build_scene_geometry(
    grid_config: &GridConfig,
    tee: &TeeBox,
    trail_tube: &[GridVertex],
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

    // Trail: the tracer tube, vertex for vertex (position and fade).
    if !trail_tube.is_empty() {
        geometries.push(RtGeometry {
            positions: trail_tube
                .iter()
                .flat_map(|v| {
                    let [x, y, z] = v.position;
                    [x, y, z, v.fade]
                })
                .collect(),
            primitive: RtPrimitive::GlowTriangles,
            geom_type: GEOM_TRAIL,
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
        let geometries = build_scene_geometry(&grid, &tee, &tube());
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
        let without = build_scene_geometry(&grid, &tee, &[]);
        assert!(without.iter().all(|g| g.geom_type != GEOM_TRAIL));
    }
}
