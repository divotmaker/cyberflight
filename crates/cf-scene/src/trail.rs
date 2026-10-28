use glam::Vec3;

use crate::grid::GridVertex;
use crate::tee::GLOW_SHELL_FADE;

/// Trail lifetime the renderers use, in seconds: a trail point is fully
/// faded once it is this old.
pub const DEFAULT_TRAIL_LIFETIME: f64 = 3.0;

/// A trail sample with position and timestamp.
#[derive(Debug, Clone, Copy)]
pub struct TrailPoint {
    pub position: Vec3,
    pub time: f64,
}

/// Length of the tracer tail as a fraction of the elapsed time `t`: the tail
/// window is `[t - TRAIL_TAIL_FRACTION · t, t]`, clipped at time 0.
pub const TRAIL_TAIL_FRACTION: f64 = 1.2;

/// The tracer tail at time `t` from a ball's position `samples` (oldest
/// first, scene frame): the samples inside the tail window
/// `[t - TRAIL_TAIL_FRACTION · t, t]`, every `subsample`-th of them
/// (counted from the window's start; `0` counts as `1`), ending at `ball`.
/// `ball` is appended at time `t` unless the last kept sample lies within
/// 1 cm of it, so the trail always reaches the ball.
#[must_use]
pub fn tracer_tail(
    samples: impl IntoIterator<Item = TrailPoint>,
    t: f64,
    ball: Vec3,
    subsample: usize,
) -> Vec<TrailPoint> {
    let subsample = subsample.max(1);
    let tail_start = (t - t * TRAIL_TAIL_FRACTION).max(0.0);
    let mut tail: Vec<TrailPoint> = samples
        .into_iter()
        .filter(|p| p.time >= tail_start && p.time <= t)
        .enumerate()
        .filter(|(i, _)| i % subsample == 0)
        .map(|(_, p)| p)
        .collect();
    if tail
        .last()
        .is_none_or(|last| (last.position - ball).length() > 0.01)
    {
        tail.push(TrailPoint {
            position: ball,
            time: t,
        });
    }
    tail
}

/// Sides of the tracer tube's cross-section polygon.
pub const TRAIL_TUBE_SIDES: usize = 12;

/// Glow shells around the tracer core.
const GLOW_SHELLS: u32 = 8;

/// Outermost glow shell radius, in ball radii.
const GLOW_RADIUS_RATIO: f32 = 4.0;

/// Over this fraction of the tracer fade value, from the tail tip, the tube's
/// width grows from nothing to full.
const TIP_TAPER: f32 = 0.15;

/// How much wider the glow shells get at the tail than at the ball: the
/// radius is `1 + GLOW_TAIL_SPREAD · age` times its value at the ball.
const GLOW_TAIL_SPREAD: f32 = 0.5;

/// Rings of each shell's rounded end at the ball, from the tube's end to its
/// tip (a quarter circle of this many steps).
const CAP_STEPS: u32 = 3;

/// Fade value of the tube surface that carries `fade`'s share of light when
/// both of its faces are drawn.
///
/// A ray through a closed shell crosses the surface twice, so each face
/// carries half the alpha `(1 - fade)²` of the ribbon the shell stands for.
#[must_use]
fn face_fade(fade: f32) -> f32 {
    1.0 - (1.0 - fade) * std::f32::consts::FRAC_1_SQRT_2
}

/// One ring station along the tube: a trail point with the frame of the
/// plane its rings lie in.
struct Station {
    position: Vec3,
    tangent: Vec3,
    /// Two unit vectors across the tangent, perpendicular to each other.
    u: Vec3,
    v: Vec3,
    /// Tracer fade value: 1 at the newest time, 0 once `max_lifetime` old.
    t: f32,
}

/// The stations of `trail`: the averaged tangent at every point, and a frame
/// carried along the curve with the least rotation, so the rings do not twist.
fn stations(trail: &[TrailPoint], current_time: f64, max_lifetime: f64) -> Vec<Station> {
    let n = trail.len();
    let direction = |i: usize| (trail[i + 1].position - trail[i].position).normalize_or_zero();
    let mut out: Vec<Station> = Vec::with_capacity(n);
    for (i, point) in trail.iter().enumerate() {
        let before = if i > 0 { direction(i - 1) } else { Vec3::ZERO };
        let after = if i + 1 < n { direction(i) } else { Vec3::ZERO };
        let mut tangent = (before + after).normalize_or_zero();
        if tangent == Vec3::ZERO {
            tangent = if after != Vec3::ZERO { after } else { before };
        }
        if tangent == Vec3::ZERO {
            tangent = out.last().map_or(Vec3::Z, |s| s.tangent);
        }
        let reference = out.last().map_or_else(
            || {
                if tangent.y.abs() < 0.99 {
                    Vec3::Y
                } else {
                    Vec3::X
                }
            },
            |s| s.u,
        );
        let u = (reference - tangent * reference.dot(tangent)).normalize_or_zero();
        let u = if u == Vec3::ZERO {
            tangent.any_orthonormal_vector()
        } else {
            u
        };
        out.push(Station {
            position: point.position,
            tangent,
            u,
            v: tangent.cross(u),
            t: (1.0 - (current_time - point.time) / max_lifetime).clamp(0.0, 1.0) as f32,
        });
    }
    out
}

/// One layer of the tracer: the core or a glow shell.
struct Layer {
    /// Radius at the ball, in ball radii.
    radius: f32,
    /// `None` for the core; for a glow shell, its position among the shells
    /// in [0, 1] (0 innermost).
    shell: Option<f32>,
}

/// The tracer's layers, core first, then the glow shells from the inside out.
///
/// The shells use the ball glow's technique: few shells at tight radii with
/// exponential spacing, so they overlap heavily near the core for a smooth
/// continuous glow rather than discrete bands.
fn layers() -> impl Iterator<Item = Layer> {
    let core = Layer {
        radius: 1.0,
        shell: None,
    };
    let shells = (0..GLOW_SHELLS).map(|shell| {
        let st = shell as f32 / (GLOW_SHELLS - 1) as f32;
        Layer {
            radius: GLOW_RADIUS_RATIO.powf(st * st),
            shell: Some(st),
        }
    });
    std::iter::once(core).chain(shells)
}

impl Layer {
    /// Radius of this layer at a station of fade value `t`, for balls of
    /// radius `ball_radius`: tapered to a point at the tail tip, and for a
    /// shell wider toward the tail.
    fn radius_at(&self, t: f32, ball_radius: f32) -> f32 {
        let tip = (t / TIP_TAPER).min(1.0);
        let spread = if self.shell.is_some() {
            1.0 + GLOW_TAIL_SPREAD * (1.0 - t)
        } else {
            1.0
        };
        ball_radius * self.radius * spread * tip
    }

    /// Fade value of the ribbon this layer stands for at fade value `t`:
    /// `(1 - fade)²` is its alpha.
    fn ribbon_fade(&self, t: f32) -> f32 {
        match self.shell {
            // Solid at the ball (0), 0.7 at the tail.
            None => (1.0 - t) * 0.7,
            // Outer shells fade faster, so the glow narrows before it
            // disappears.
            Some(st) => {
                let boost = 1.0 + 2.0 * st;
                (GLOW_SHELL_FADE + (1.0 - GLOW_SHELL_FADE) * (1.0 - t) * boost).min(1.0)
            }
        }
    }

    /// Fade value of this layer's surface at fade value `t`.
    fn surface_fade(&self, t: f32) -> f32 {
        face_fade(self.ribbon_fade(t))
    }
}

/// Generate the tracer: a 3D tube along the trail whose cross-section
/// reproduces the ball glow aesthetic.
///
/// The tube is a bright core cylinder of radius `ball_radius` inside 8
/// concentric translucent glow shells out to four ball radii, with the ball
/// glow's shell technique: exponential radius spacing, one fade per shell,
/// drawn additively. The result looks like the glow stretched along the
/// trail, from any viewpoint: seen from the side its summed alpha across the
/// width is that of a flat ribbon of the same layers (each shell has two
/// faces, so each carries half the ribbon's alpha).
///
/// One ring of [`TRAIL_TUBE_SIDES`] vertices per layer sits at each trail
/// point, in the plane across the averaged tangent there, so the segments
/// join smoothly along the curve. The tube ends in a rounded cap at the ball
/// (the newest point) and tapers to nothing at the tail tip: the width grows
/// over the first 15 % of the fade value, and the shells widen up to 1.5×
/// toward the tail.
///
/// Only the "tail" behind the ball should be passed in — the caller controls
/// tracer length by choosing how many recent trail points to include.
/// The fade goes from ball-glow brightness at the newest point (last index)
/// to fully transparent at the oldest point (index 0), giving the tracer
/// bullet falloff. Segments whose points have both expired are left out.
///
/// `trail` points go from oldest (index 0) to newest (last index = near ball).
/// `ball_radius` must match the ball glow — layer widths are derived from it.
///
/// Returns a TRIANGLE_LIST that does not depend on the viewpoint: both
/// faces are drawn, and the ray-traced reflection is built from the same
/// triangles.
#[must_use]
pub fn generate_trail_tube(
    trail: &[TrailPoint],
    current_time: f64,
    max_lifetime: f64,
    ball_radius: f32,
) -> Vec<GridVertex> {
    let n = trail.len();
    if n < 2 {
        return Vec::new();
    }
    let stations = stations(trail, current_time, max_lifetime);

    let ring = |s: &Station, centre: Vec3, radius: f32| -> [Vec3; TRAIL_TUBE_SIDES] {
        std::array::from_fn(|k| {
            let (sin, cos) = (std::f32::consts::TAU * k as f32 / TRAIL_TUBE_SIDES as f32).sin_cos();
            centre + (s.u * cos + s.v * sin) * radius
        })
    };
    // A band of quads between two rings of equal fade at each end.
    let band = |out: &mut Vec<GridVertex>,
                a: &[Vec3; TRAIL_TUBE_SIDES],
                fade_a: f32,
                b: &[Vec3; TRAIL_TUBE_SIDES],
                fade_b: f32| {
        let v = |pos: Vec3, fade: f32| GridVertex {
            position: pos.into(),
            fade,
        };
        for k in 0..TRAIL_TUBE_SIDES {
            let k1 = (k + 1) % TRAIL_TUBE_SIDES;
            let corners = [
                [v(a[k], fade_a), v(b[k], fade_b), v(b[k1], fade_b)],
                [v(a[k], fade_a), v(b[k1], fade_b), v(a[k1], fade_a)],
            ];
            for tri in corners {
                // Rings that taper to a point leave triangles with no area.
                let same = |x: usize, y: usize| tri[x].position == tri[y].position;
                if !(same(0, 1) || same(1, 2) || same(0, 2)) {
                    out.extend(tri);
                }
            }
        }
    };

    let layer_count = GLOW_SHELLS as usize + 1;
    let mut out = Vec::with_capacity(layer_count * TRAIL_TUBE_SIDES * 6 * (n - 1));
    for layer in layers() {
        for pair in stations.windows(2) {
            let (s0, s1) = (&pair[0], &pair[1]);
            // Skip segments where both endpoints have fully expired.
            if s0.t <= 0.0 && s1.t <= 0.0 {
                continue;
            }
            band(
                &mut out,
                &ring(s0, s0.position, layer.radius_at(s0.t, ball_radius)),
                layer.surface_fade(s0.t),
                &ring(s1, s1.position, layer.radius_at(s1.t, ball_radius)),
                layer.surface_fade(s1.t),
            );
        }

        // Rounded cap at the ball (newest) end: a quarter circle of rings
        // that wraps the tube forward around its last point, so no flat edge
        // meets the ball glow.
        let last = &stations[n - 1];
        if last.t > 0.0 {
            let radius = layer.radius_at(last.t, ball_radius);
            let fade = layer.surface_fade(last.t);
            let mut prev = ring(last, last.position, radius);
            for step in 1..=CAP_STEPS {
                let angle = std::f32::consts::FRAC_PI_2 * step as f32 / CAP_STEPS as f32;
                let (sin, cos) = angle.sin_cos();
                // The last ring is the tip: a single point.
                let cos = if step == CAP_STEPS { 0.0 } else { cos };
                let next = ring(
                    last,
                    last.position + last.tangent * (radius * sin),
                    radius * cos,
                );
                band(&mut out, &prev, fade, &next, fade);
                prev = next;
            }
        }
    }
    out
}

/// Generate a LINE_LIST wireframe centerline for the trail.
///
/// Hardware LINE_LIST guarantees 1px minimum width at any distance — the trail
/// never disappears no matter how far away. Pairs with the glow ribbon for a
/// Tron-style bright core + soft halo look.
///
/// Returns `GridVertex` pairs (LINE_LIST topology, 2 verts per segment).
/// Fade is time-based: bright at the ball end, transparent at the tail.
#[must_use]
pub fn generate_trail_line(
    trail: &[TrailPoint],
    current_time: f64,
    max_lifetime: f64,
) -> Vec<GridVertex> {
    let n = trail.len();
    if n < 2 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity((n - 1) * 2);
    for i in 0..(n - 1) {
        let t0 = (1.0 - (current_time - trail[i].time) / max_lifetime).clamp(0.0, 1.0) as f32;
        let t1 = (1.0 - (current_time - trail[i + 1].time) / max_lifetime).clamp(0.0, 1.0) as f32;

        // Skip segments where both endpoints have fully expired.
        if t0 <= 0.0 && t1 <= 0.0 {
            continue;
        }

        // Fade: 0 at ball (full brightness), 1.0 at tail tip (invisible).
        // Matches the glow ribbon lifetime so the line disappears with it.
        let fade0 = 1.0 - t0;
        let fade1 = 1.0 - t1;

        out.push(GridVertex {
            position: trail[i].position.into(),
            fade: fade0,
        });
        out.push(GridVertex {
            position: trail[i + 1].position.into(),
            fade: fade1,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(n: usize) -> Vec<TrailPoint> {
        (0..n)
            .map(|i| TrailPoint {
                position: Vec3::new(0.0, 0.0, i as f32),
                time: i as f64 * 0.1,
            })
            .collect()
    }

    fn times(tail: &[TrailPoint]) -> Vec<f64> {
        tail.iter().map(|p| p.time).collect()
    }

    #[test]
    fn tracer_tail_keeps_the_window_and_ends_at_the_ball() {
        // At t = 0.5 the window starts at max(0.5 - 0.6, 0) = 0: samples 0..=5.
        let ball = Vec3::new(0.0, 0.0, 5.5);
        let tail = tracer_tail(samples(10), 0.5, ball, 1);
        assert_eq!(tail.len(), 7);
        assert_eq!(tail[5].time, samples(10)[5].time);
        assert_eq!(tail[6].position, ball);
        assert_eq!(tail[6].time, 0.5);
    }

    #[test]
    fn tracer_tail_subsamples_from_the_window_start() {
        let ball = Vec3::new(0.0, 0.0, 7.0);
        let all = samples(10);
        let tail = tracer_tail(all.clone(), 0.7, ball, 3);
        // Samples 0, 3, 6 of 0..=7, then the ball (sample 6 is 1 m away).
        assert_eq!(
            times(&tail),
            vec![all[0].time, all[3].time, all[6].time, 0.7]
        );
    }

    #[test]
    fn tracer_tail_does_not_repeat_a_ball_on_the_last_sample() {
        let all = samples(10);
        let tail = tracer_tail(all.clone(), all[4].time, all[4].position, 1);
        assert_eq!(tail.len(), 5);
        assert_eq!(tail[4].position, all[4].position);
    }

    #[test]
    fn tracer_tail_treats_subsample_zero_as_one() {
        let ball = Vec3::new(0.0, 0.0, 9.5);
        let all = samples(10);
        assert_eq!(
            times(&tracer_tail(all.clone(), 0.95, ball, 0)),
            times(&tracer_tail(all, 0.95, ball, 1))
        );
    }

    #[test]
    fn tracer_tail_without_samples_is_the_ball() {
        let ball = Vec3::new(1.0, 2.0, 3.0);
        let tail = tracer_tail([], 1.0, ball, 1);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].position, ball);
    }

    fn make_trail(n: usize) -> Vec<TrailPoint> {
        (0..n)
            .map(|i| TrailPoint {
                position: Vec3::new(0.0, (i as f32 * 0.5).sin() * 5.0, i as f32 * 5.0),
                time: i as f64 * 0.1,
            })
            .collect()
    }

    const R: f32 = 0.021;

    #[test]
    fn trail_tube_empty_on_single_point() {
        let trail = vec![TrailPoint {
            position: Vec3::ZERO,
            time: 0.0,
        }];
        assert!(generate_trail_tube(&trail, 0.0, 10.0, R).is_empty());
    }

    #[test]
    fn trail_tube_is_a_triangle_list() {
        let trail = make_trail(20);
        let current_time = trail.last().unwrap().time;
        let verts = generate_trail_tube(&trail, current_time, 10.0, R);
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 3, 0, "whole triangles");
    }

    #[test]
    fn trail_tube_fade_range() {
        let trail = make_trail(10);
        let current_time = trail.last().unwrap().time;
        for v in &generate_trail_tube(&trail, current_time, 10.0, R) {
            assert!(
                (0.0..=1.0).contains(&v.fade),
                "fade should be in [0, 1], got {}",
                v.fade
            );
        }
    }

    #[test]
    fn trail_tube_skips_expired_segments() {
        let trail = make_trail(10);
        // Every point is older than the lifetime.
        assert!(generate_trail_tube(&trail, 100.0, 1.0, R).is_empty());
    }

    #[test]
    fn trail_tube_survives_repeated_points() {
        let mut trail = make_trail(6);
        trail[3].position = trail[2].position;
        trail[4].position = trail[2].position;
        let verts = generate_trail_tube(&trail, 0.5, 10.0, R);
        assert!(!verts.is_empty());
        assert!(
            verts
                .iter()
                .all(|v| v.position.iter().all(|c| c.is_finite()))
        );
    }

    /// A straight tracer along +Z at the origin, a point every metre and
    /// 0.1 s, the newest at `t = 1` and the oldest at `t ≈ 0.03`.
    fn straight_tracer() -> (Vec<TrailPoint>, f64, f64) {
        let trail = (0..30)
            .map(|i| TrailPoint {
                position: Vec3::new(0.0, 0.0, i as f32),
                time: f64::from(i) * 0.1,
            })
            .collect();
        (trail, 2.9, 3.0)
    }

    /// Summed alpha `(1 - fade)²` of the triangles a ray down `-Y` crosses at
    /// (`x`, `z`), from above the axis.
    fn probe_alpha(verts: &[GridVertex], x: f32, z: f32) -> f32 {
        let origin = Vec3::new(x, 10.0, z);
        let dir = -Vec3::Y;
        let mut sum = 0.0;
        for tri in verts.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|i| Vec3::from(tri[i].position));
            // Moller-Trumbore.
            let (e1, e2) = (b - a, c - a);
            let p = dir.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-12 {
                continue;
            }
            let s = origin - a;
            let u = s.dot(p) / det;
            let q = s.cross(e1);
            let v = dir.dot(q) / det;
            if u < 0.0 || v < 0.0 || u + v > 1.0 {
                continue;
            }
            let fade = (1.0 - u - v) * tri[0].fade + u * tri[1].fade + v * tri[2].fade;
            sum += (1.0 - fade) * (1.0 - fade);
        }
        sum
    }

    /// The summed alpha across a flat ribbon of the tracer's layers, at lateral
    /// offset `d` from the axis where the fade value is `t`: a core of width
    /// `R` with alpha `(1 - 0.7 (1 - t))²`, and 8 shells of radius
    /// `R · 4^(i²/49) · (1 + 0.5 (1 - t))`, each with alpha
    /// `(1 - min(F + (1 - F) (1 - t) (1 + 2 i/7), 1))²` (`F = 1 - sqrt(0.08)`).
    fn ribbon_alpha(d: f32, t: f32) -> f32 {
        let age = 1.0 - t;
        let mut sum = 0.0;
        if d < R {
            sum += (1.0 - 0.7 * age).powi(2);
        }
        for i in 0..8 {
            let st = i as f32 / 7.0;
            let radius = R * 4.0_f32.powf(st * st) * (1.0 + 0.5 * age);
            let fade =
                (GLOW_SHELL_FADE + (1.0 - GLOW_SHELL_FADE) * age * (1.0 + 2.0 * st)).min(1.0);
            if d < radius {
                sum += (1.0 - fade).powi(2);
            }
        }
        sum
    }

    #[test]
    fn trail_tube_sums_to_the_ribbon_profile() {
        let (trail, now, lifetime) = straight_tracer();
        let verts = generate_trail_tube(&trail, now, lifetime, R);
        // Mid-trail, between points 15 and 16, off the triangle edges.
        let z = 15.37;
        let t = 1.0 - (now - 1.537) / lifetime;
        // Lateral offsets inside the core, between the core and the glow's
        // first shell, and between the layer radii further out.
        for d_ratio in [0.05, 1.1, 2.2, 4.2, 6.0] {
            let expected = ribbon_alpha(d_ratio * R, t as f32);
            let got = probe_alpha(&verts, d_ratio * R, z);
            assert!(
                (got - expected).abs() < 0.02 * expected.max(1.0),
                "d = {d_ratio} R: tube {got}, ribbon {expected}"
            );
        }
    }

    #[test]
    fn trail_tube_is_the_same_from_every_side() {
        let (trail, now, lifetime) = straight_tracer();
        let verts = generate_trail_tube(&trail, now, lifetime, R);
        // The same probe from the side, at the same offsets.
        let side = |x: f32, z: f32| {
            let rotated: Vec<GridVertex> = verts
                .iter()
                .map(|v| GridVertex {
                    position: [v.position[1], v.position[0], v.position[2]],
                    fade: v.fade,
                })
                .collect();
            probe_alpha(&rotated, x, z)
        };
        for d_ratio in [0.05, 1.1, 2.2, 4.2] {
            let up = probe_alpha(&verts, d_ratio * R, 15.37);
            let along = side(d_ratio * R, 15.37);
            assert!(
                (up - along).abs() < 0.03 * up.max(1.0),
                "d = {d_ratio} R: from above {up}, from the side {along}"
            );
        }
    }

    #[test]
    fn trail_tube_ends_in_a_cap_at_the_ball() {
        let (trail, now, lifetime) = straight_tracer();
        let verts = generate_trail_tube(&trail, now, lifetime, R);
        let last_z = 29.0;
        let reach = verts
            .iter()
            .map(|v| v.position[2])
            .fold(f32::NEG_INFINITY, f32::max);
        // The outermost shell, 4 radii at the ball, rounds off the end.
        assert!((reach - (last_z + 4.0 * R)).abs() < 1e-4, "reach {reach}");
        // The widest vertex is the outermost shell's, 4 radii from the axis.
        let width = verts
            .iter()
            .filter(|v| v.position[2] > 28.9)
            .map(|v| v.position[0].hypot(v.position[1]))
            .fold(0.0_f32, f32::max);
        assert!((width - 4.0 * R).abs() < 1e-4, "width {width}");
    }

    #[test]
    fn trail_tube_tapers_to_nothing_at_the_tail() {
        let (trail, _, lifetime) = straight_tracer();
        // The oldest point is exactly `lifetime` old: t = 0, no width.
        let verts = generate_trail_tube(&trail, 3.0, lifetime, R);
        let at_tip = verts
            .iter()
            .filter(|v| v.position[2] < 1e-6)
            .map(|v| v.position[0].hypot(v.position[1]))
            .fold(0.0_f32, f32::max);
        assert_eq!(at_tip, 0.0);
    }
}
