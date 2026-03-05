use std::f32::consts::TAU;

use glam::Vec3;

use crate::grid::GridVertex;

/// Tee box surface elevation above Y=0 (1mm) to prevent z-fighting with the floor.
pub const TEE_ELEVATION: f32 = 0.001;

/// Tee box configuration.
#[derive(Debug, Clone, Copy)]
pub struct TeeBox {
    /// Half-size of the tee box square in meters.
    pub half_size: f32,
    /// Border thickness in meters.
    pub border: f32,
    /// Ball visual radius in meters.
    pub ball_radius: f32,
}

impl Default for TeeBox {
    fn default() -> Self {
        Self {
            half_size: 2.5,
            border: 0.15,
            ball_radius: 0.021_335, // golf ball: 42.67mm diameter
        }
    }
}

/// Generate the opaque black fill quad for the tee box (2 triangles, 6 vertices).
///
/// This is drawn after the grid to occlude grid lines beneath the tee box.
#[must_use]
pub fn generate_tee_fill(tee: &TeeBox) -> Vec<GridVertex> {
    let h = tee.half_size;
    let y = TEE_ELEVATION;

    let corners = [
        Vec3::new(-h, y, -h), // 0: bottom-left
        Vec3::new(h, y, -h),  // 1: bottom-right
        Vec3::new(h, y, h),   // 2: top-right
        Vec3::new(-h, y, h),  // 3: top-left
    ];

    let v = |pos: Vec3| GridVertex {
        position: pos.into(),
        fade: 0.0,
    };

    vec![
        v(corners[0]),
        v(corners[1]),
        v(corners[2]),
        v(corners[0]),
        v(corners[2]),
        v(corners[3]),
    ]
}

/// Generate the cyan border frame for the tee box (4 quads = 24 vertices).
///
/// The border straddles the edge of the tee box square.
#[must_use]
pub fn generate_tee_border(tee: &TeeBox) -> Vec<GridVertex> {
    let inner = tee.half_size - tee.border / 2.0;
    let outer = tee.half_size + tee.border / 2.0;
    let y = TEE_ELEVATION;

    let v = |x: f32, z: f32| GridVertex {
        position: [x, y, z],
        fade: 0.0,
    };

    let mut verts = Vec::with_capacity(24);

    // Top strip (+Z edge)
    quad(&mut verts, v(-outer, inner), v(outer, inner), v(outer, outer), v(-outer, outer));
    // Bottom strip (-Z edge)
    quad(&mut verts, v(-outer, -outer), v(outer, -outer), v(outer, -inner), v(-outer, -inner));
    // Left strip (-X edge, between inner Z extents)
    quad(&mut verts, v(-outer, -inner), v(-inner, -inner), v(-inner, inner), v(-outer, inner));
    // Right strip (+X edge, between inner Z extents)
    quad(&mut verts, v(inner, -inner), v(outer, -inner), v(outer, inner), v(inner, inner));

    verts
}

/// Generate a UV sphere at an arbitrary center with a given radius.
///
/// `rings` = horizontal slices, `segments` = vertical slices.
#[must_use]
pub fn generate_ball_at(center: Vec3, radius: f32, rings: u32, segments: u32) -> Vec<GridVertex> {
    let v = |pos: Vec3| GridVertex {
        position: pos.into(),
        fade: 0.0,
    };

    let mut verts = Vec::with_capacity((rings * segments * 6) as usize);
    for ring in 0..rings {
        let theta0 = std::f32::consts::PI * ring as f32 / rings as f32;
        let theta1 = std::f32::consts::PI * (ring + 1) as f32 / rings as f32;
        let (sin0, cos0) = theta0.sin_cos();
        let (sin1, cos1) = theta1.sin_cos();

        for seg in 0..segments {
            let phi0 = TAU * seg as f32 / segments as f32;
            let phi1 = TAU * (seg + 1) as f32 / segments as f32;
            let (sp0, cp0) = phi0.sin_cos();
            let (sp1, cp1) = phi1.sin_cos();

            let p00 = center + Vec3::new(radius * sin0 * cp0, radius * cos0, radius * sin0 * sp0);
            let p10 = center + Vec3::new(radius * sin1 * cp0, radius * cos1, radius * sin1 * sp0);
            let p01 = center + Vec3::new(radius * sin0 * cp1, radius * cos0, radius * sin0 * sp1);
            let p11 = center + Vec3::new(radius * sin1 * cp1, radius * cos1, radius * sin1 * sp1);

            verts.push(v(p00));
            verts.push(v(p10));
            verts.push(v(p11));

            verts.push(v(p00));
            verts.push(v(p11));
            verts.push(v(p01));
        }
    }
    verts
}

/// Generate a UV sphere for the ball at the origin (sits on the raised tee box surface).
///
/// `rings` = horizontal slices, `segments` = vertical slices.
#[must_use]
pub fn generate_ball(tee: &TeeBox, rings: u32, segments: u32) -> Vec<GridVertex> {
    let center = Vec3::new(0.0, TEE_ELEVATION + tee.ball_radius, 0.0);
    generate_ball_at(center, tee.ball_radius, rings, segments)
}

/// Generate a smooth radial glow at an arbitrary center.
///
/// Many concentric spheres with constant per-shell fade. With additive
/// blending, pixels near the center are covered by many overlapping shells
/// (bright), while pixels at the edge are only covered by the largest
/// shells (dim), producing a smooth gradient.
#[must_use]
pub fn generate_ball_glow_at(
    center: Vec3,
    ball_radius: f32,
    rings: u32,
    segments: u32,
) -> Vec<GridVertex> {
    let num_shells: u32 = 20;
    let r_min = ball_radius;
    let r_max = ball_radius * 2.0;

    let verts_per_sphere = (rings * segments * 6) as usize;
    let mut out = Vec::with_capacity(num_shells as usize * verts_per_sphere);

    for shell in 0..num_shells {
        let t = shell as f32 / (num_shells - 1) as f32;
        // Steep exponential spacing: t^2 clusters shells heavily near the ball.
        let r = r_min * (r_max / r_min).powf(t * t);
        // Each shell contributes a small constant amount of light.
        // alpha = (1 - fade)^2;  we want alpha ≈ 0.08 per shell.
        // (1 - fade)^2 = 0.08 → fade ≈ 0.717
        let fade = GLOW_SHELL_FADE;

        let v = |pos: Vec3| GridVertex {
            position: pos.into(),
            fade,
        };

        for ring in 0..rings {
            let theta0 = std::f32::consts::PI * ring as f32 / rings as f32;
            let theta1 = std::f32::consts::PI * (ring + 1) as f32 / rings as f32;
            let (sin0, cos0) = theta0.sin_cos();
            let (sin1, cos1) = theta1.sin_cos();

            for seg in 0..segments {
                let phi0 = TAU * seg as f32 / segments as f32;
                let phi1 = TAU * (seg + 1) as f32 / segments as f32;
                let (sp0, cp0) = phi0.sin_cos();
                let (sp1, cp1) = phi1.sin_cos();

                let p00 = center + Vec3::new(r * sin0 * cp0, r * cos0, r * sin0 * sp0);
                let p10 = center + Vec3::new(r * sin1 * cp0, r * cos1, r * sin1 * sp0);
                let p01 = center + Vec3::new(r * sin0 * cp1, r * cos0, r * sin0 * sp1);
                let p11 = center + Vec3::new(r * sin1 * cp1, r * cos1, r * sin1 * sp1);

                out.push(v(p00));
                out.push(v(p10));
                out.push(v(p11));

                out.push(v(p00));
                out.push(v(p11));
                out.push(v(p01));
            }
        }
    }
    out
}

/// Per-shell fade value for glow: (1 - fade)^2 = 0.08 → fade ≈ 0.717.
pub const GLOW_SHELL_FADE: f32 = 1.0 - 0.282_843; // 1 - sqrt(0.08)

/// Generate a smooth radial glow around the ball at origin.
#[must_use]
pub fn generate_ball_glow(tee: &TeeBox, rings: u32, segments: u32) -> Vec<GridVertex> {
    let center = Vec3::new(0.0, TEE_ELEVATION + tee.ball_radius, 0.0);
    generate_ball_glow_at(center, tee.ball_radius, rings, segments)
}

fn quad(verts: &mut Vec<GridVertex>, bl: GridVertex, br: GridVertex, tr: GridVertex, tl: GridVertex) {
    verts.push(bl);
    verts.push(br);
    verts.push(tr);
    verts.push(bl);
    verts.push(tr);
    verts.push(tl);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tee_fill_is_two_triangles() {
        let verts = generate_tee_fill(&TeeBox::default());
        assert_eq!(verts.len(), 6);
    }

    #[test]
    fn tee_border_is_four_quads() {
        let verts = generate_tee_border(&TeeBox::default());
        assert_eq!(verts.len(), 24);
    }

    #[test]
    fn ball_vertex_count() {
        let verts = generate_ball(&TeeBox::default(), 12, 24);
        assert_eq!(verts.len(), (12 * 24 * 6) as usize);
    }

    #[test]
    fn tee_box_at_elevation() {
        let tee = TeeBox::default();
        let all: Vec<GridVertex> = generate_tee_fill(&tee)
            .into_iter()
            .chain(generate_tee_border(&tee))
            .collect();
        for v in &all {
            assert!(
                (v.position[1] - TEE_ELEVATION).abs() < 1e-6,
                "tee box vertex should be at TEE_ELEVATION, got {}",
                v.position[1]
            );
        }
    }

    #[test]
    fn ball_sits_on_tee() {
        let tee = TeeBox::default();
        let verts = generate_ball(&tee, 12, 24);
        let min_y = verts.iter().map(|v| v.position[1]).fold(f32::MAX, f32::min);
        let max_y = verts.iter().map(|v| v.position[1]).fold(f32::MIN, f32::max);
        // Bottom of sphere touches tee box surface, top at TEE_ELEVATION + 2*radius
        assert!((min_y - TEE_ELEVATION).abs() < 1e-4, "ball bottom should touch tee box, got {min_y}");
        assert!((max_y - (TEE_ELEVATION + 2.0 * tee.ball_radius)).abs() < 1e-4, "ball top");
    }

    #[test]
    fn tee_fill_fade_zero() {
        for v in &generate_tee_fill(&TeeBox::default()) {
            assert!((v.fade).abs() < 1e-6);
        }
    }
}
