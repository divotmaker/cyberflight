use cf_scene::grid::GridVertex;
use cf_scene::scenery::Floor;
use cf_scene::tee::{TeeBox, generate_tee_border, generate_tee_fill};

use crate::rt_pipeline::{GEOM_FLOOR, GEOM_TEE_BOX, GEOM_TEE_BOX_BORDER, GEOM_TERRAIN, RtGeometry};

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

/// Build the static RT scene geometry: the reflective ground of `floor` (the
/// flat floor, or the terrain with its normals), the tee box and its border.
///
/// Ball reflections are computed analytically in the floor closest-hit shader
/// (no ball geometry in the TLAS). The tracer is not here: its structures are
/// rebuilt every frame from the frame's own vertex buffers.
pub fn build_scene_geometry(floor: &Floor, tee: &TeeBox) -> Vec<RtGeometry> {
    let mut geometries = Vec::new();

    match floor {
        Floor::Flat(grid_config) => {
            // Floor: tee box + fairway (non-rectangular compound surface)
            let floor_verts = cf_scene::grid::generate_floor_quad(grid_config, tee.half_size);
            geometries.push(RtGeometry {
                positions: grid_verts_to_positions(&floor_verts),
                geom_type: GEOM_FLOOR,
                normals: Vec::new(),
                transform: glam::Mat4::IDENTITY,
            });
        }
        Floor::Terrain(mesh) => geometries.push(RtGeometry {
            positions: mesh.positions.iter().flat_map(|p| p.to_array()).collect(),
            geom_type: GEOM_TERRAIN,
            normals: mesh.normals.iter().flat_map(|n| n.to_array()).collect(),
            transform: glam::Mat4::IDENTITY,
        }),
    }

    // Tee box: filled quad (already at TEE_ELEVATION from generate_tee_fill)
    let tee_verts = generate_tee_fill(tee);
    let tee_box_transform = glam::Mat4::IDENTITY;
    geometries.push(RtGeometry {
        positions: grid_verts_to_positions(&tee_verts),
        geom_type: GEOM_TEE_BOX,
        normals: Vec::new(),
        transform: tee_box_transform,
    });

    // Tee box border: cyan outline
    let border_verts = generate_tee_border(tee);
    geometries.push(RtGeometry {
        positions: grid_verts_to_positions(&border_verts),
        geom_type: GEOM_TEE_BOX_BORDER,
        normals: Vec::new(),
        transform: tee_box_transform,
    });

    geometries
}

#[cfg(test)]
mod tests {
    use super::*;
    use cf_scene::grid::GridConfig;
    use cf_scene::scenery::GroundMesh;
    use glam::Vec3;

    fn flat(grid: GridConfig) -> Floor {
        Floor::Flat(grid)
    }

    #[test]
    fn the_static_scene_is_the_floor_and_the_tee_box() {
        let geometries = build_scene_geometry(&flat(GridConfig::default()), &TeeBox::default());
        let types: Vec<u32> = geometries.iter().map(|g| g.geom_type).collect();
        assert_eq!(types, [GEOM_FLOOR, GEOM_TEE_BOX, GEOM_TEE_BOX_BORDER]);
        assert!(geometries.iter().all(|g| g.primitive_count() > 0));
        assert!(geometries.iter().all(|g| g.normals.is_empty()));
    }

    #[test]
    fn the_terrain_brings_a_normal_per_vertex() {
        let mesh = GroundMesh {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Z],
            normals: vec![Vec3::Y, Vec3::new(0.6, 0.8, 0.0), Vec3::Y],
        };
        let geometries = build_scene_geometry(&Floor::Terrain(mesh), &TeeBox::default());
        let terrain = &geometries[0];
        assert_eq!(terrain.geom_type, GEOM_TERRAIN);
        assert_eq!(terrain.primitive_count(), 1);
        assert_eq!(
            terrain.positions,
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            terrain.normals,
            [0.0, 1.0, 0.0, 0.6, 0.8, 0.0, 0.0, 1.0, 0.0]
        );
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
        let floor = &build_scene_geometry(&flat(grid), &tee)[0];
        let default_floor = &build_scene_geometry(&flat(GridConfig::default()), &tee)[0];
        assert_ne!(floor.positions, default_floor.positions);
    }
}
