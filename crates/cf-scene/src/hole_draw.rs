//! A hole's geometry for drawing: the terrain mesh and the lines draped
//! over it, every vertex sampled from [`Hole::height`].
//!
//! The black mesh lies [`MESH_DROP_M`] under the surface, so the lines on
//! the surface never fight it for depth; the grid lies exactly on the
//! surface, and the outlines 1 cm above it.

use dimple::units::yards_to_meters;
use glam::Vec3;

use crate::grid::{GridConfig, GridVertex};
use crate::hole::{Bunker, Hole};

/// How far the terrain mesh lies under the surface (m).
pub const MESH_DROP_M: f64 = 0.15;
/// Cell and segment size of the mesh and the draped grid (yd).
pub const CELL_YD: f64 = 1.25;
/// Height of the outlines above the surface (m).
const LINE_LIFT_M: f64 = 0.01;
/// Segments in a bunker's rim.
pub const RIM_SEGMENTS: u32 = 64;
/// Segments in the green's ring.
pub const RING_SEGMENTS: u32 = 96;
/// Spacing of a bunker's cross-hatch (m).
const HATCH_SPACING_M: f64 = 2.0;
/// Radius of the cup (m).
pub const CUP_RADIUS_M: f64 = 0.054;
/// Height of the pin (m).
const PIN_HEIGHT_M: f64 = 2.13;
/// Segments in the cup's disc and rim.
const CUP_SEGMENTS: u32 = 24;

/// A vertex at scene (`x`, `y`, `z`) with `fade`.
fn vertex(x: f64, y: f64, z: f64, fade: f32) -> GridVertex {
    GridVertex {
        position: [x as f32, y as f32, z as f32],
        fade,
    }
}

/// A point `lift` m above the surface of `hole`, unfaded.
fn on(hole: &Hole, x: f64, z: f64, lift: f64) -> GridVertex {
    vertex(x, hole.height(x, z) + lift, z, 0.0)
}

/// Push a polyline of `points` as `LINE_LIST` pairs.
fn polyline(out: &mut Vec<GridVertex>, points: &[GridVertex]) {
    for pair in points.windows(2) {
        out.push(pair[0]);
        out.push(pair[1]);
    }
}

/// `n` + 1 evenly spaced values from `a` to `b`, `n` = at least one step
/// per `step`.
fn steps(a: f64, b: f64, step: f64) -> impl Iterator<Item = f64> {
    let n = ((b - a).abs() / step).ceil().max(1.0) as u32;
    (0..=n).map(move |i| a + (b - a) * f64::from(i) / f64::from(n))
}

/// The terrain mesh (`TRIANGLE_LIST`): the grid's lateral extent of `grid`,
/// from 5 yd behind the tee to its downrange end, in [`CELL_YD`] cells,
/// [`MESH_DROP_M`] under the surface. Unfaded.
#[must_use]
pub fn terrain_mesh(hole: &Hole, grid: &GridConfig) -> Vec<GridVertex> {
    let cell = yards_to_meters(CELL_YD);
    let lateral = yards_to_meters(f64::from(grid.lateral));
    let z0 = -yards_to_meters(5.0);
    let z1 = yards_to_meters(f64::from(grid.downrange));
    let nx = (2.0 * lateral / cell).round() as u32;
    let nz = ((z1 - z0) / cell).round() as u32;
    let x_at = |i: u32| -lateral + f64::from(i) * cell;
    let z_at = |j: u32| z0 + f64::from(j) * cell;
    let p = |i: u32, j: u32| {
        let (x, z) = (x_at(i), z_at(j));
        vertex(x, hole.height(x, z) - MESH_DROP_M, z, 0.0)
    };
    let mut out = Vec::with_capacity((6 * nx * nz) as usize);
    for j in 0..nz {
        for i in 0..nx {
            let (a, b, c, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
            out.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    out
}

/// The grid of `grid` draped over the hole: the same lines as
/// [`generate_grid_vertices`](crate::grid::generate_grid_vertices), cut
/// into [`CELL_YD`] segments that lie on the surface, with the same
/// distance fade.
#[must_use]
pub fn draped_grid(hole: &Hole, grid: &GridConfig) -> Vec<GridVertex> {
    let cell = yards_to_meters(CELL_YD);
    let spacing = yards_to_meters(f64::from(grid.spacing));
    let lateral = yards_to_meters(f64::from(grid.lateral));
    let downrange = yards_to_meters(f64::from(grid.downrange));
    let fade_dist = f64::from(grid.fade_distance_m());
    let point = |x: f64, z: f64| {
        let y = hole.height(x, z);
        let fade = (Vec3::new(x as f32, 0.0, z as f32).length() / fade_dist as f32).min(1.0);
        vertex(x, y, z, fade)
    };
    let mut out = Vec::new();
    let z_count = (downrange / spacing) as i32;
    for iz in 0..=z_count {
        let z = f64::from(iz) * spacing;
        let line: Vec<_> = steps(-lateral, lateral, cell)
            .map(|x| point(x, z))
            .collect();
        polyline(&mut out, &line);
    }
    let x_count = (lateral / spacing) as i32;
    for ix in -x_count..=x_count {
        let x = f64::from(ix) * spacing;
        let line: Vec<_> = steps(0.0, downrange, cell).map(|z| point(x, z)).collect();
        polyline(&mut out, &line);
    }
    out
}

/// The fairway's outline: up its left edge (+X) and back down its right
/// edge, a point every meter, 1 cm above the surface. Unfaded.
#[must_use]
pub fn fairway_outline(hole: &Hole) -> Vec<GridVertex> {
    let (Some(&(z0, _)), Some(&(z1, _))) = (hole.fairway.first(), hole.fairway.last()) else {
        return Vec::new();
    };
    let half = |z: f64| hole.fairway_half_width(z).unwrap_or(0.0);
    let mut points: Vec<_> = steps(z0, z1, 1.0)
        .map(|z| on(hole, half(z), z, LINE_LIFT_M))
        .collect();
    points.extend(steps(z1, z0, 1.0).map(|z| on(hole, -half(z), z, LINE_LIFT_M)));
    let mut out = Vec::new();
    polyline(&mut out, &points);
    out
}

/// A bunker's rim ([`RIM_SEGMENTS`] segments) and a cross-hatch every 2 m
/// in X and in Z, clipped to the rim, following the bowl with a point every
/// meter, 1 cm above the surface. Unfaded.
#[must_use]
pub fn bunker_lines(hole: &Hole, b: &Bunker) -> Vec<GridVertex> {
    let mut out = Vec::new();
    let rim: Vec<_> = (0..=RIM_SEGMENTS)
        .map(|k| {
            let a = f64::from(k) / f64::from(RIM_SEGMENTS) * std::f64::consts::TAU;
            on(
                hole,
                b.x + b.half_x * a.cos(),
                b.z + b.half_z * a.sin(),
                LINE_LIFT_M,
            )
        })
        .collect();
    polyline(&mut out, &rim);
    let hatch = |center: f64, half: f64| {
        let n = (half / HATCH_SPACING_M).floor() as i32;
        (-n..=n).map(move |k| center + f64::from(k) * HATCH_SPACING_M)
    };
    for x in hatch(b.x, b.half_x) {
        let u = (x - b.x) / b.half_x;
        let reach = b.half_z * (1.0 - u * u).max(0.0).sqrt();
        if reach > 0.0 {
            let line: Vec<_> = steps(b.z - reach, b.z + reach, 1.0)
                .map(|z| on(hole, x, z, LINE_LIFT_M))
                .collect();
            polyline(&mut out, &line);
        }
    }
    for z in hatch(b.z, b.half_z) {
        let v = (z - b.z) / b.half_z;
        let reach = b.half_x * (1.0 - v * v).max(0.0).sqrt();
        if reach > 0.0 {
            let line: Vec<_> = steps(b.x - reach, b.x + reach, 1.0)
                .map(|x| on(hole, x, z, LINE_LIFT_M))
                .collect();
            polyline(&mut out, &line);
        }
    }
    out
}

/// The green's edge: one ring of [`RING_SEGMENTS`] segments on the pad,
/// 1 cm above the surface. Unfaded.
#[must_use]
pub fn green_ring(hole: &Hole) -> Vec<GridVertex> {
    let (gx, gz) = hole.green_center();
    let r = f64::from(hole.green.radius_m());
    let ring: Vec<_> = (0..=RING_SEGMENTS)
        .map(|k| {
            let a = f64::from(k) / f64::from(RING_SEGMENTS) * std::f64::consts::TAU;
            on(hole, gx + r * a.cos(), gz + r * a.sin(), LINE_LIFT_M)
        })
        .collect();
    let mut out = Vec::new();
    polyline(&mut out, &ring);
    out
}

/// The cup as a disc (`TRIANGLE_LIST`), 4 mm above the surface. Unfaded.
#[must_use]
pub fn cup_disc(hole: &Hole) -> Vec<GridVertex> {
    let (cx, cz) = hole.cup;
    let center = on(hole, cx, cz, 0.004);
    let rim = |k: u32| {
        let a = f64::from(k) / f64::from(CUP_SEGMENTS) * std::f64::consts::TAU;
        on(
            hole,
            cx + CUP_RADIUS_M * a.cos(),
            cz + CUP_RADIUS_M * a.sin(),
            0.004,
        )
    };
    (0..CUP_SEGMENTS)
        .flat_map(|k| [center, rim(k), rim(k + 1)])
        .collect()
}

/// The cup's rim (5 mm above the surface), the pin and its flag's outline,
/// as `LINE_LIST` pairs. Unfaded.
#[must_use]
pub fn cup_and_pin(hole: &Hole) -> Vec<GridVertex> {
    let (cx, cz) = hole.cup;
    let mut out = Vec::new();
    let rim: Vec<_> = (0..=CUP_SEGMENTS)
        .map(|k| {
            let a = f64::from(k) / f64::from(CUP_SEGMENTS) * std::f64::consts::TAU;
            on(
                hole,
                cx + CUP_RADIUS_M * a.cos(),
                cz + CUP_RADIUS_M * a.sin(),
                0.005,
            )
        })
        .collect();
    polyline(&mut out, &rim);
    let base = hole.height(cx, cz);
    let top = base + PIN_HEIGHT_M;
    let pin = |y: f64, dx: f64| vertex(cx + dx, y, cz, 0.0);
    // The pin, then a 0.5 × 0.35 m triangular flag to the left (+X).
    polyline(&mut out, &[pin(base, 0.0), pin(top, 0.0)]);
    polyline(
        &mut out,
        &[pin(top, 0.0), pin(top - 0.175, 0.5), pin(top - 0.35, 0.0)],
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hole() -> Hole {
        Hole::no_1()
    }

    #[test]
    fn the_mesh_lies_under_the_surface() {
        let h = hole();
        let grid = GridConfig {
            spacing: 10.0,
            downrange: 450.0,
            lateral: 50.0,
        };
        let mesh = terrain_mesh(&h, &grid);
        assert_eq!(mesh.len(), 174_720);
        for v in mesh.iter().step_by(97) {
            let [x, y, z] = v.position.map(f64::from);
            let want = h.height(x, z) - MESH_DROP_M;
            assert!((y - want).abs() < 1e-3, "({x}, {z}): {y} vs {want}");
        }
    }

    #[test]
    fn the_draped_grid_lies_on_the_surface() {
        let h = hole();
        let grid = GridConfig {
            spacing: 10.0,
            downrange: 450.0,
            lateral: 50.0,
        };
        let lines = draped_grid(&h, &grid);
        assert_eq!(lines.len() % 2, 0);
        for v in lines.iter().step_by(31) {
            let [x, y, z] = v.position.map(f64::from);
            assert!((y - h.height(x, z)).abs() < 1e-3);
            assert!((0.0..=1.0).contains(&v.fade));
        }
    }

    #[test]
    fn the_green_ring_is_one_circle_on_the_pad() {
        let h = hole();
        let ring = green_ring(&h);
        assert_eq!(ring.len(), 2 * RING_SEGMENTS as usize);
        let (gx, gz) = h.green_center();
        for v in &ring {
            let [x, y, z] = v.position.map(f64::from);
            assert!(((x - gx).hypot(z - gz) - 13.0).abs() < 1e-3);
            assert!(y > h.height(x, z));
        }
    }

    #[test]
    fn bunker_lines_stay_inside_the_rim() {
        let h = hole();
        for b in &h.bunkers {
            for v in bunker_lines(&h, b) {
                let [x, _, z] = v.position.map(f64::from);
                assert!(b.rho2(x, z) < 1.0 + 1e-4, "({x}, {z})");
            }
        }
    }
}
