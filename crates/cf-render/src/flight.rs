//! Balls in flight: the per-frame input of both renderers and the raster
//! geometry built from it.

use glam::Vec3;

use cf_scene::grid::GridVertex;
use cf_scene::tee::{TeeBox, generate_ball_at, generate_ball_glow_at};
use cf_scene::trail::{
    DEFAULT_TRAIL_LIFETIME, TrailPoint, generate_trail_line, generate_trail_tube,
};

use crate::frame::FrameContent;

/// One ball's frame: position, tracer samples (oldest first) and time, in
/// scene space. `ball_pos` is dimple's reference point (a ball resting on the
/// ground has `y == 0`); renderers draw the sphere one ball radius above it.
pub struct FlightRenderData {
    pub ball_pos: Vec3,
    pub trail_points: Vec<TrailPoint>,
    pub current_time: f64,
}

/// Sphere tessellation of an in-flight ball, `(rings, segments)`, for its
/// fill and its glow.
pub(crate) const FLIGHT_BALL_MESH: (u32, u32) = (8, 16);

/// The tracer tube of `flight`: the triangles both the raster pass and the
/// ray-traced reflections draw. Empty without at least two trail points.
pub(crate) fn flight_trail_tube(flight: &FlightRenderData, tee: &TeeBox) -> Vec<GridVertex> {
    generate_trail_tube(
        &flight.trail_points,
        flight.current_time,
        DEFAULT_TRAIL_LIFETIME,
        tee.ball_radius,
    )
}

/// The tracer's centre line of `flight` (LINE_LIST): the segments both the
/// raster pass and the ray-traced reflections draw. Empty without at least
/// two trail points.
pub(crate) fn flight_trail_line(flight: &FlightRenderData) -> Vec<GridVertex> {
    generate_trail_line(
        &flight.trail_points,
        flight.current_time,
        DEFAULT_TRAIL_LIFETIME,
    )
}

/// Build the raster geometry of `flights` into `out`'s flight vertices: per
/// flight the ball and its glow, centred one ball radius above `ball_pos`,
/// and the trail's tube ([`flight_trail_tube`]) and centre line
/// ([`flight_trail_line`]). With no
/// flight, the ball is on the tee.
pub(crate) fn flight_vertices(flights: &[FlightRenderData], tee: &TeeBox, out: &mut FrameContent) {
    let (rings, segments) = FLIGHT_BALL_MESH;
    out.flight_lines.clear();
    out.flight_fills.clear();
    out.flight_glows.clear();
    out.ball_on_tee = flights.is_empty();

    for flight in flights {
        let centre = flight.ball_pos + Vec3::new(0.0, tee.ball_radius, 0.0);
        out.flight_fills
            .extend(generate_ball_at(centre, tee.ball_radius, rings, segments));
        out.flight_glows.extend(generate_ball_glow_at(
            centre,
            tee.ball_radius,
            rings,
            segments,
        ));
        out.flight_glows.extend(flight_trail_tube(flight, tee));
        out.flight_lines.extend(flight_trail_line(flight));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::as_bytes;

    fn trail() -> Vec<TrailPoint> {
        (0..20)
            .map(|i| TrailPoint {
                position: Vec3::new(0.0, i as f32, 2.0 * i as f32),
                time: f64::from(i) * 0.1,
            })
            .collect()
    }

    #[test]
    fn one_flight_draws_ball_glow_and_trail() {
        let tee = TeeBox::default();
        let flight = FlightRenderData {
            ball_pos: Vec3::new(0.0, 19.0, 38.0),
            trail_points: trail(),
            current_time: 1.9,
        };
        let mut out = FrameContent::default();
        flight_vertices(std::slice::from_ref(&flight), &tee, &mut out);

        let r = tee.ball_radius;
        let centre = Vec3::new(0.0, 19.0 + r, 38.0);
        let fills = generate_ball_at(centre, r, 8, 16);
        let mut glows = generate_ball_glow_at(centre, r, 8, 16);
        glows.extend(generate_trail_tube(
            &flight.trail_points,
            1.9,
            DEFAULT_TRAIL_LIFETIME,
            r,
        ));
        let lines = generate_trail_line(&flight.trail_points, 1.9, DEFAULT_TRAIL_LIFETIME);

        assert!(!out.ball_on_tee);
        assert!(!lines.is_empty());
        assert_eq!(as_bytes(&out.flight_fills), as_bytes(&fills));
        assert_eq!(as_bytes(&out.flight_glows), as_bytes(&glows));
        assert_eq!(as_bytes(&out.flight_lines), as_bytes(&lines));
    }

    #[test]
    fn single_point_trail_draws_no_trail() {
        let tee = TeeBox::default();
        let flight = FlightRenderData {
            ball_pos: Vec3::new(0.0, 1.0, 2.0),
            trail_points: trail()[..1].to_vec(),
            current_time: 0.0,
        };
        let mut out = FrameContent::default();
        flight_vertices(&[flight], &tee, &mut out);

        let ball_glow = generate_ball_glow_at(
            Vec3::new(0.0, 1.0 + tee.ball_radius, 2.0),
            tee.ball_radius,
            8,
            16,
        );
        assert!(out.flight_lines.is_empty());
        assert_eq!(as_bytes(&out.flight_glows), as_bytes(&ball_glow));
        assert!(!out.ball_on_tee);
    }

    #[test]
    fn no_flight_puts_the_ball_on_the_tee() {
        let tee = TeeBox::default();
        let mut out = FrameContent::default();
        let flight = FlightRenderData {
            ball_pos: Vec3::ONE,
            trail_points: trail(),
            current_time: 1.0,
        };
        flight_vertices(&[flight], &tee, &mut out);
        flight_vertices(&[], &tee, &mut out);
        assert!(out.ball_on_tee);
        assert!(out.flight_lines.is_empty());
        assert!(out.flight_fills.is_empty());
        assert!(out.flight_glows.is_empty());
    }
}
