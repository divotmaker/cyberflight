use cf_scene::grid::{GridConfig, GridVertex};
use cf_scene::tee::{TeeBox, generate_tee_border, generate_tee_fill};

use crate::rt_pipeline::{GEOM_FLOOR, GEOM_TEE_BOX, GEOM_TEE_BOX_BORDER, RtGeometry};

// ── Scene geometry builders ──

/// Extract packed f32x3 positions from `GridVertex` data.
pub fn grid_verts_to_positions(verts: &[GridVertex]) -> Vec<f32> {
    let mut positions = Vec::with_capacity(verts.len() * 3);
    for v in verts {
        positions.push(v.position[0]);
        positions.push(v.position[1]);
        positions.push(v.position[2]);
    }
    positions
}

/// Build the static RT scene geometry from the driving range config: the
/// floor, the tee box and its border.
///
/// Ball reflections are computed analytically in the floor closest-hit shader
/// (no ball geometry in the TLAS). The tracer is not here: its structures are
/// rebuilt every frame from the frame's own vertex buffers.
pub fn build_scene_geometry(grid_config: &GridConfig, tee: &TeeBox) -> Vec<RtGeometry> {
    let mut geometries = Vec::new();

    // Floor: tee box + fairway (non-rectangular compound surface)
    let floor_verts = cf_scene::grid::generate_floor_quad(grid_config, tee.half_size);
    geometries.push(RtGeometry {
        positions: grid_verts_to_positions(&floor_verts),
        geom_type: GEOM_FLOOR,
        transform: glam::Mat4::IDENTITY,
    });

    // Tee box: filled quad (already at TEE_ELEVATION from generate_tee_fill)
    let tee_verts = generate_tee_fill(tee);
    let tee_box_transform = glam::Mat4::IDENTITY;
    geometries.push(RtGeometry {
        positions: grid_verts_to_positions(&tee_verts),
        geom_type: GEOM_TEE_BOX,
        transform: tee_box_transform,
    });

    // Tee box border: cyan outline
    let border_verts = generate_tee_border(tee);
    geometries.push(RtGeometry {
        positions: grid_verts_to_positions(&border_verts),
        geom_type: GEOM_TEE_BOX_BORDER,
        transform: tee_box_transform,
    });

    geometries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_static_scene_is_the_floor_and_the_tee_box() {
        let geometries = build_scene_geometry(&GridConfig::default(), &TeeBox::default());
        let types: Vec<u32> = geometries.iter().map(|g| g.geom_type).collect();
        assert_eq!(types, [GEOM_FLOOR, GEOM_TEE_BOX, GEOM_TEE_BOX_BORDER]);
        assert!(geometries.iter().all(|g| g.primitive_count() > 0));
    }

    #[test]
    fn positions_drop_the_fade() {
        let v = |x: f32, fade: f32| GridVertex {
            position: [x, 2.0, 3.0],
            fade,
        };
        assert_eq!(
            grid_verts_to_positions(&[v(0.0, 0.1), v(1.0, 0.2)]),
            [0.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn the_floor_follows_the_grid() {
        let tee = TeeBox::default();
        let grid = GridConfig {
            downrange: 200.0,
            ..GridConfig::default()
        };
        let floor = &build_scene_geometry(&grid, &tee)[0];
        let default_floor = &build_scene_geometry(&GridConfig::default(), &tee)[0];
        assert_ne!(floor.positions, default_floor.positions);
    }
}
