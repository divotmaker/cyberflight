use cf_scene::grid::GridConfig;
use cf_scene::tee::{TeeBox, generate_tee_border, generate_tee_fill};

use crate::rt_pipeline::{GEOM_FLOOR, GEOM_TEE_BOX, GEOM_TEE_BOX_BORDER, GEOM_TRAIL, RtGeometry};

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
/// (no ball geometry in the TLAS). Trail reflections use ray-traced geometry.
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

    // Trail: cross-shaped ribbon (horizontal + vertical quads) for floor reflections.
    // Horizontal ribbon is visible from below (reflection rays); vertical adds side visibility.
    // Width matches raster glow visual extent (~4x ball radius with additive shells).
    if trail_points.len() >= 2 {
        let trail_positions = build_trail_cross(trail_points, tee.ball_radius * 6.0);
        if !trail_positions.is_empty() {
            geometries.push(RtGeometry {
                positions: trail_positions,
                geom_type: GEOM_TRAIL,
                transform: glam::Mat4::IDENTITY,
            });
        }
    }

    geometries
}

/// Build a cross-shaped ribbon along trail points for RT reflections.
///
/// Each segment produces two perpendicular quads (horizontal + vertical)
/// forming a cross section. The horizontal ribbon faces upward, making it
/// clearly visible to reflection rays bouncing off the floor. The vertical
/// ribbon adds side-view visibility.
fn build_trail_cross(points: &[glam::Vec3], radius: f32) -> Vec<f32> {
    let mut positions = Vec::new();

    for i in 0..points.len() - 1 {
        let p0 = points[i];
        let p1 = points[i + 1];
        let forward = (p1 - p0).normalize_or_zero();
        if forward.length_squared() < 0.001 {
            continue;
        }

        // Perpendicular directions for the cross section
        let up = glam::Vec3::Y;
        let right = forward.cross(up).normalize_or_zero();
        let actual_up = if right.length_squared() > 0.001 {
            right.cross(forward).normalize()
        } else {
            let r = forward.cross(glam::Vec3::X).normalize();
            r.cross(forward).normalize()
        };

        // Emit quad helper
        let mut emit_quad = |offset: glam::Vec3| {
            let a = p0 - offset;
            let b = p0 + offset;
            let c = p1 + offset;
            let d = p1 - offset;
            positions.extend_from_slice(&[a.x, a.y, a.z]);
            positions.extend_from_slice(&[b.x, b.y, b.z]);
            positions.extend_from_slice(&[c.x, c.y, c.z]);
            positions.extend_from_slice(&[a.x, a.y, a.z]);
            positions.extend_from_slice(&[c.x, c.y, c.z]);
            positions.extend_from_slice(&[d.x, d.y, d.z]);
        };

        // Horizontal ribbon (visible from below — floor reflections)
        emit_quad(right * radius);
        // Vertical ribbon (visible from side angles)
        emit_quad(actual_up * radius);
    }

    positions
}

/// Public wrapper for `trim_trail_from_ball` used by the hybrid offscreen renderer.
pub fn trim_trail_from_ball_pub(points: &[glam::Vec3], frac: f32) -> Vec<glam::Vec3> {
    trim_trail_from_ball(points, frac)
}

/// Keep only the most recent `frac` (0.0–1.0) of the trail by arc length.
///
/// Trail points go oldest-first (index 0) to newest (last = near ball).
/// We walk backwards from the ball end, accumulating arc length, and return
/// the suffix of points that fits within `frac` of total length.
fn trim_trail_from_ball(points: &[glam::Vec3], frac: f32) -> Vec<glam::Vec3> {
    if points.len() < 2 {
        return points.to_vec();
    }

    let total_len: f32 = points.windows(2).map(|w| (w[1] - w[0]).length()).sum();
    let budget = total_len * frac;

    let mut accum = 0.0_f32;
    let mut start = points.len() - 1;
    for i in (0..points.len() - 1).rev() {
        let seg = (points[i + 1] - points[i]).length();
        accum += seg;
        start = i;
        if accum >= budget {
            break;
        }
    }

    points[start..].to_vec()
}
