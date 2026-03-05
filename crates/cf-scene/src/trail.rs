use glam::Vec3;

use crate::grid::GridVertex;
use crate::tee::GLOW_SHELL_FADE;

/// Default trail lifetime in seconds — points older than this are pruned/fully faded.
pub const DEFAULT_TRAIL_LIFETIME: f64 = 3.0;

/// Safety cap: maximum trail points retained regardless of lifetime.
const MAX_TRAIL_POINTS: usize = 2000;

/// A trail sample with position and timestamp.
#[derive(Debug, Clone, Copy)]
pub struct TrailPoint {
    pub position: Vec3,
    pub time: f64,
}

/// A ball flight being displayed on the range.
#[derive(Debug, Clone)]
pub struct BallFlight {
    /// Trail points (world space, timestamped). Oldest entries at the front.
    trail: Vec<TrailPoint>,
    /// Current ball position (head of the trail).
    pub position: Vec3,
    /// Whether the ball is still in flight (vs. landed).
    pub active: bool,
    /// Flight start time (seconds since range start).
    pub start_time: f64,
    /// Trail color hint (index into color palette based on shot shape).
    pub color_index: u32,
}

impl BallFlight {
    /// Create a new ball flight starting at the tee box.
    #[must_use]
    pub fn new(start_time: f64) -> Self {
        Self {
            trail: Vec::with_capacity(512),
            position: Vec3::ZERO,
            active: true,
            start_time,
            color_index: 0,
        }
    }

    /// Update the ball position and append to trail.
    pub fn update_position(&mut self, pos: Vec3, time: f64) {
        self.position = pos;
        self.trail.push(TrailPoint { position: pos, time });

        // Safety cap
        if self.trail.len() > MAX_TRAIL_POINTS {
            let excess = self.trail.len() - MAX_TRAIL_POINTS;
            self.trail.drain(..excess);
        }
    }

    /// Remove trail points older than `max_lifetime` seconds before `current_time`.
    pub fn prune_expired(&mut self, current_time: f64, max_lifetime: f64) {
        let cutoff = current_time - max_lifetime;
        self.trail.retain(|p| p.time >= cutoff);
    }

    /// Mark the ball as landed.
    pub fn land(&mut self) {
        self.active = false;
    }

    /// Get trail points for rendering.
    #[must_use]
    pub fn trail_points(&self) -> &[TrailPoint] {
        &self.trail
    }

    /// Return positions of non-expired trail points.
    #[must_use]
    pub fn trail_positions_alive(&self, current_time: f64, max_lifetime: f64) -> Vec<Vec3> {
        let cutoff = current_time - max_lifetime;
        self.trail
            .iter()
            .filter(|p| p.time >= cutoff)
            .map(|p| p.position)
            .collect()
    }

    /// Number of trail points.
    #[must_use]
    pub fn trail_len(&self) -> usize {
        self.trail.len()
    }
}

/// Vertex for trail rendering (ribbon strip).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrailVertex {
    pub position: [f32; 3],
    /// Normalized age: 0.0 = newest (bright), 1.0 = oldest (faded).
    pub age: f32,
}

/// Generate trail vertices from a ball flight's trail points.
///
/// Returns vertices suitable for a line strip.
#[must_use]
pub fn generate_trail_vertices(flight: &BallFlight) -> Vec<TrailVertex> {
    let points = flight.trail_points();
    let len = points.len();
    if len == 0 {
        return Vec::new();
    }

    points
        .iter()
        .enumerate()
        .map(|(i, tp)| TrailVertex {
            position: tp.position.into(),
            age: 1.0 - (i as f32 / (len - 1).max(1) as f32),
        })
        .collect()
}

/// Generate a trail glow ribbon matching the ball glow aesthetic.
///
/// Uses the same shell technique as `generate_ball_glow_at` — same shell count,
/// exponential radius spacing, and per-shell fade. The result is a camera-facing
/// ribbon that looks like the ball glow stretched along the trail.
///
/// Only the "tail" behind the ball should be passed in — the caller controls
/// tracer length by choosing how many recent trail points to include.
/// The fade goes from ball-glow brightness at the newest point (last index)
/// to fully transparent at the oldest point (index 0), giving the tracer
/// bullet falloff.
///
/// `trail` points go from oldest (index 0) to newest (last index = near ball).
/// `camera_pos` orients the ribbon toward the viewer.
/// `ball_radius` must match the ball glow — shell widths are derived from it.
#[must_use]
pub fn generate_trail_glow(
    trail: &[TrailPoint],
    current_time: f64,
    max_lifetime: f64,
    camera_pos: Vec3,
    ball_radius: f32,
) -> Vec<GridVertex> {
    let n = trail.len();
    if n < 2 {
        return Vec::new();
    }

    // Few shells at tight radius with exponential spacing — shells overlap
    // heavily near the center for a smooth continuous glow, not discrete bands.
    let num_shells: u32 = 8;
    let r_min = ball_radius;
    let r_max = ball_radius * 4.0;

    let segments = n - 1;
    // +1 for the bright core ribbon
    let mut out = Vec::with_capacity((num_shells as usize + 1) * segments * 6);

    for i in 0..segments {
        let p0 = trail[i].position;
        let p1 = trail[i + 1].position;

        // Time-based normalized position: 0.0 = oldest (tail tip), 1.0 = newest (at ball).
        let t0 = (1.0 - (current_time - trail[i].time) / max_lifetime).clamp(0.0, 1.0) as f32;
        let t1 = (1.0 - (current_time - trail[i + 1].time) / max_lifetime).clamp(0.0, 1.0) as f32;

        // Tip taper: ramp width from 0 at tail tip to full over the first 15%.
        let tip0 = (t0 / 0.15).min(1.0);
        let tip1 = (t1 / 0.15).min(1.0);

        // Camera-facing ribbon: right = cross(tangent, to_camera)
        let tangent = (p1 - p0).normalize_or_zero();
        let mid = (p0 + p1) * 0.5;
        let to_camera = (camera_pos - mid).normalize_or_zero();
        let right = tangent.cross(to_camera).normalize_or_zero();

        if right.length_squared() < 0.001 {
            continue;
        }

        let v = |pos: Vec3, fade: f32| GridVertex {
            position: pos.into(),
            fade,
        };

        // ── Bright core ribbon ──
        // Solid magenta center (fade=0 at ball, ramping to 1 at tail).
        // Width = ball_radius, gives a hot bright line that reads as the tracer.
        let core_r = ball_radius;
        let core_w0 = core_r * tip0;
        let core_w1 = core_r * tip1;
        // Core uses low fade near ball (bright) ramping up toward tail.
        let core_fade0 = (1.0 - t0) * 0.7; // 0 at ball → 0.7 at tail
        let core_fade1 = (1.0 - t1) * 0.7;

        let a0 = p0 - right * core_w0;
        let b0 = p0 + right * core_w0;
        let a1 = p1 - right * core_w1;
        let b1 = p1 + right * core_w1;

        out.push(v(a0, core_fade0));
        out.push(v(b0, core_fade0));
        out.push(v(b1, core_fade1));
        out.push(v(a0, core_fade0));
        out.push(v(b1, core_fade1));
        out.push(v(a1, core_fade1));

        // ── Glow shells ──
        // Softer halo around the core, outer shells fade faster.
        for shell in 0..num_shells {
            let st = shell as f32 / (num_shells - 1) as f32;
            let shell_r = r_min * (r_max / r_min).powf(st * st);

            // Width grows slightly toward tail for comet shape.
            let age0 = 1.0 - t0;
            let age1 = 1.0 - t1;
            let spread0 = 1.0 + 0.5 * age0;
            let spread1 = 1.0 + 0.5 * age1;
            let w0 = shell_r * spread0 * tip0;
            let w1 = shell_r * spread1 * tip1;

            // Outer shells fade faster — glow narrows before disappearing.
            let shell_fade_boost = 1.0 + 2.0 * st;
            let fade0 = (GLOW_SHELL_FADE + (1.0 - GLOW_SHELL_FADE) * (1.0 - t0) * shell_fade_boost).min(1.0);
            let fade1 = (GLOW_SHELL_FADE + (1.0 - GLOW_SHELL_FADE) * (1.0 - t1) * shell_fade_boost).min(1.0);

            let a0 = p0 - right * w0;
            let b0 = p0 + right * w0;
            let a1 = p1 - right * w1;
            let b1 = p1 + right * w1;

            out.push(v(a0, fade0));
            out.push(v(b0, fade0));
            out.push(v(b1, fade1));
            out.push(v(a0, fade0));
            out.push(v(b1, fade1));
            out.push(v(a1, fade1));
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

        out.push(GridVertex { position: trail[i].position.into(), fade: fade0 });
        out.push(GridVertex { position: trail[i + 1].position.into(), fade: fade1 });
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trail_vertex_size() {
        assert_eq!(std::mem::size_of::<TrailVertex>(), 16);
    }

    #[test]
    fn new_flight_at_origin() {
        let flight = BallFlight::new(0.0);
        assert!(flight.active);
        assert_eq!(flight.trail_len(), 0);
    }

    #[test]
    fn update_appends_trail() {
        let mut flight = BallFlight::new(0.0);
        flight.update_position(Vec3::new(1.0, 1.0, 0.0), 0.1);
        flight.update_position(Vec3::new(2.0, 2.0, 0.0), 0.2);
        assert_eq!(flight.trail_len(), 2);
    }

    #[test]
    fn trail_trims_at_safety_cap() {
        let mut flight = BallFlight::new(0.0);
        for i in 0..2100 {
            flight.update_position(Vec3::new(i as f32, 0.0, 0.0), i as f64 * 0.001);
        }
        assert!(flight.trail_len() <= MAX_TRAIL_POINTS);
    }

    #[test]
    fn prune_expired_removes_old_points() {
        let mut flight = BallFlight::new(0.0);
        for i in 0..10 {
            flight.update_position(Vec3::new(i as f32, 0.0, 0.0), i as f64);
        }
        flight.prune_expired(9.0, 5.0); // cutoff = 4.0, keeps t=4..9
        assert_eq!(flight.trail_len(), 6);
    }

    #[test]
    fn trail_positions_alive_filters() {
        let mut flight = BallFlight::new(0.0);
        for i in 0..10 {
            flight.update_position(Vec3::new(i as f32, 0.0, 0.0), i as f64);
        }
        let alive = flight.trail_positions_alive(9.0, 5.0);
        // cutoff = 4.0, points at t=4..9 → 6 points
        assert_eq!(alive.len(), 6);
    }

    #[test]
    fn land_deactivates() {
        let mut flight = BallFlight::new(0.0);
        flight.land();
        assert!(!flight.active);
    }

    fn make_trail(n: usize) -> Vec<TrailPoint> {
        (0..n)
            .map(|i| TrailPoint {
                position: Vec3::new(0.0, (i as f32 * 0.5).sin() * 5.0, i as f32 * 5.0),
                time: i as f64 * 0.1,
            })
            .collect()
    }

    #[test]
    fn trail_glow_empty_on_single_point() {
        let trail = vec![TrailPoint { position: Vec3::ZERO, time: 0.0 }];
        let verts = generate_trail_glow(&trail, 0.0, 10.0, Vec3::new(0.0, 3.0, -10.0), 0.021);
        assert!(verts.is_empty());
    }

    #[test]
    fn trail_glow_generates_quads() {
        let trail = make_trail(20);
        let current_time = trail.last().unwrap().time;
        let camera_pos = Vec3::new(0.0, 3.0, -10.0);
        let verts = generate_trail_glow(&trail, current_time, 10.0, camera_pos, 0.021);
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 6, 0, "should be triangle pairs");
    }

    #[test]
    fn trail_glow_fade_range() {
        let trail = make_trail(10);
        let current_time = trail.last().unwrap().time;
        let verts = generate_trail_glow(&trail, current_time, 10.0, Vec3::new(0.0, 3.0, -10.0), 0.021);
        for v in &verts {
            assert!(
                v.fade >= 0.0 && v.fade <= 1.01,
                "fade should be in [0, 1], got {}",
                v.fade
            );
        }
    }

    #[test]
    fn trail_vertices_age_gradient() {
        let mut flight = BallFlight::new(0.0);
        for i in 0..10 {
            flight.update_position(Vec3::new(i as f32, 0.0, 0.0), i as f64 * 0.1);
        }
        let verts = generate_trail_vertices(&flight);
        assert_eq!(verts.len(), 10);
        // First vertex (oldest) should have age ~1.0, last (newest) should have age ~0.0
        assert!(verts[0].age > 0.9);
        assert!(verts[9].age < 0.1);
    }
}
