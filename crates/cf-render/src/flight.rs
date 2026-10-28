//! Balls in flight: the per-frame input of both renderers and the raster
//! geometry built from it.

use glam::Vec3;

use cf_scene::tee::{TeeBox, generate_ball_at, generate_ball_glow_at};
use cf_scene::trail::{
    DEFAULT_TRAIL_LIFETIME, TrailPoint, generate_trail_glow, generate_trail_line,
};

use crate::frame::FrameContent;

/// Per-frame flight render data: ball position + trail points (scene space).
pub struct FlightRenderData {
    pub ball_pos: Vec3,
    pub trail_points: Vec<TrailPoint>,
    pub current_time: f64,
}

/// Sphere tessellation of an in-flight ball, `(rings, segments)`, for its
/// fill and its glow.
pub(crate) const FLIGHT_BALL_MESH: (u32, u32) = (8, 16);

/// Build the raster geometry of `flights` into `out`'s flight vertices: per
/// flight the ball, its glow, and the trail's glow ribbon (facing
/// `camera_pos`) and core line. With no flight, the ball is on the tee.
pub(crate) fn flight_vertices(
    flights: &[FlightRenderData],
    camera_pos: Vec3,
    tee: &TeeBox,
    out: &mut FrameContent,
) {
    let (rings, segments) = FLIGHT_BALL_MESH;
    out.flight_lines.clear();
    out.flight_fills.clear();
    out.flight_glows.clear();
    out.ball_on_tee = flights.is_empty();

    for flight in flights {
        out.flight_fills.extend(generate_ball_at(
            flight.ball_pos,
            tee.ball_radius,
            rings,
            segments,
        ));
        out.flight_glows.extend(generate_ball_glow_at(
            flight.ball_pos,
            tee.ball_radius,
            rings,
            segments,
        ));

        // Trail glow ribbon (with endcap) + centre line.
        if flight.trail_points.len() >= 2 {
            out.flight_glows.extend(generate_trail_glow(
                &flight.trail_points,
                flight.current_time,
                DEFAULT_TRAIL_LIFETIME,
                camera_pos,
                tee.ball_radius,
            ));
            out.flight_lines.extend(generate_trail_line(
                &flight.trail_points,
                flight.current_time,
                DEFAULT_TRAIL_LIFETIME,
            ));
        }
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
        let camera_pos = Vec3::new(0.0, 2.0, -5.0);
        let flight = FlightRenderData {
            ball_pos: Vec3::new(0.0, 19.0, 38.0),
            trail_points: trail(),
            current_time: 1.9,
        };
        let mut out = FrameContent::default();
        flight_vertices(std::slice::from_ref(&flight), camera_pos, &tee, &mut out);

        let r = tee.ball_radius;
        let fills = generate_ball_at(flight.ball_pos, r, 8, 16);
        let mut glows = generate_ball_glow_at(flight.ball_pos, r, 8, 16);
        glows.extend(generate_trail_glow(
            &flight.trail_points,
            1.9,
            DEFAULT_TRAIL_LIFETIME,
            camera_pos,
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
        flight_vertices(&[flight], Vec3::ZERO, &tee, &mut out);

        let ball_glow = generate_ball_glow_at(Vec3::new(0.0, 1.0, 2.0), tee.ball_radius, 8, 16);
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
        flight_vertices(&[flight], Vec3::ZERO, &tee, &mut out);
        flight_vertices(&[], Vec3::ZERO, &tee, &mut out);
        assert!(out.ball_on_tee);
        assert!(out.flight_lines.is_empty());
        assert!(out.flight_fills.is_empty());
        assert!(out.flight_glows.is_empty());
    }
}
