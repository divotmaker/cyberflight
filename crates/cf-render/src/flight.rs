//! Balls in flight: the per-frame input of both renderers and the raster
//! geometry built from it.

use glam::Vec3;

use cf_scene::grid::GridVertex;
use cf_scene::tee::{TEE_ELEVATION, TeeBox, generate_ball_at, generate_ball_glow_at};
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

/// The centre of the ball resting on `tee`'s tee box.
pub(crate) fn tee_ball_centre(tee: &TeeBox) -> Vec3 {
    Vec3::new(0.0, TEE_ELEVATION + tee.ball_radius, 0.0)
}

/// The tracer tube of `flight`: the triangles the raster pass and the
/// reflections draw. Empty without at least two trail points.
pub(crate) fn flight_trail_tube(flight: &FlightRenderData, tee: &TeeBox) -> Vec<GridVertex> {
    generate_trail_tube(
        &flight.trail_points,
        flight.current_time,
        DEFAULT_TRAIL_LIFETIME,
        tee.ball_radius,
    )
}

/// The tracer's centre line of `flight` (LINE_LIST): the segments the raster
/// pass and the reflections draw. Empty without at least two trail points.
pub(crate) fn flight_trail_line(flight: &FlightRenderData) -> Vec<GridVertex> {
    generate_trail_line(
        &flight.trail_points,
        flight.current_time,
        DEFAULT_TRAIL_LIFETIME,
    )
}

/// Build the raster geometry of `flights` (oldest first) into `out`'s flight
/// vertices: per flight the trail's tube ([`flight_trail_tube`]) and centre
/// line ([`flight_trail_line`]), and the ball and its glow, centred one ball
/// radius above `ball_pos`. With no flight, the ball is on the tee.
///
/// The newest (last) flight comes first, its tube and centre line leading
/// their buffers, so a buffer too small for every flight loses the oldest
/// tracers and never the newest. `out` notes the newest flight's tube and
/// centre line vertex counts and ball centre: the reflections show them.
pub(crate) fn flight_vertices(flights: &[FlightRenderData], tee: &TeeBox, out: &mut FrameContent) {
    let (rings, segments) = FLIGHT_BALL_MESH;
    out.flight_lines.clear();
    out.flight_fills.clear();
    out.flight_glows.clear();
    out.ball_on_tee = flights.is_empty();
    out.newest_tube = 0;
    out.newest_line = 0;
    out.ball_centre = tee_ball_centre(tee);

    for (i, flight) in flights.iter().rev().enumerate() {
        let centre = flight.ball_pos + Vec3::new(0.0, tee.ball_radius, 0.0);
        let tube = flight_trail_tube(flight, tee);
        let line = flight_trail_line(flight);
        if i == 0 {
            out.newest_tube = tube.len();
            out.newest_line = line.len();
            out.ball_centre = centre;
        }
        out.flight_glows.extend(tube);
        out.flight_glows.extend(generate_ball_glow_at(
            centre,
            tee.ball_radius,
            rings,
            segments,
        ));
        out.flight_lines.extend(line);
        out.flight_fills
            .extend(generate_ball_at(centre, tee.ball_radius, rings, segments));
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
        let mut glows = generate_trail_tube(&flight.trail_points, 1.9, DEFAULT_TRAIL_LIFETIME, r);
        glows.extend(generate_ball_glow_at(centre, r, 8, 16));
        let lines = generate_trail_line(&flight.trail_points, 1.9, DEFAULT_TRAIL_LIFETIME);

        assert!(!out.ball_on_tee);
        assert!(!lines.is_empty());
        assert_eq!(as_bytes(&out.flight_fills), as_bytes(&fills));
        assert_eq!(as_bytes(&out.flight_glows), as_bytes(&glows));
        assert_eq!(as_bytes(&out.flight_lines), as_bytes(&lines));
    }

    fn flight(z: f32, current_time: f64) -> FlightRenderData {
        FlightRenderData {
            ball_pos: Vec3::new(0.0, 19.0, z),
            trail_points: trail(),
            current_time,
        }
    }

    #[test]
    fn the_newest_flights_tracer_leads_the_buffers() {
        let tee = TeeBox::default();
        let flights = [flight(10.0, 1.5), flight(24.0, 1.7), flight(38.0, 1.9)];
        let mut out = FrameContent::default();
        flight_vertices(&flights, &tee, &mut out);

        let tube = |f: &FlightRenderData| flight_trail_tube(f, &tee);
        let line = flight_trail_line;
        let (newest, middle, oldest) = (&flights[2], &flights[1], &flights[0]);
        assert!(!tube(newest).is_empty() && !line(newest).is_empty());
        assert_eq!(out.newest_tube, tube(newest).len());
        assert_eq!(out.newest_line, line(newest).len());
        assert_eq!(out.flight_glows[..out.newest_tube], tube(newest)[..]);
        assert_eq!(out.flight_lines[..out.newest_line], line(newest)[..]);
        // The older flights follow, newest first.
        let ball = generate_ball_glow_at(
            newest.ball_pos + Vec3::new(0.0, tee.ball_radius, 0.0),
            tee.ball_radius,
            8,
            16,
        );
        let second = out.newest_tube + ball.len();
        assert_eq!(
            out.flight_glows[second..second + tube(middle).len()],
            tube(middle)[..]
        );
        assert_eq!(
            out.flight_lines[out.newest_line..out.newest_line + line(middle).len()],
            line(middle)[..]
        );
        assert_eq!(
            out.flight_lines[out.flight_lines.len() - line(oldest).len()..],
            line(oldest)[..]
        );
        assert_eq!(
            out.ball_centre,
            newest.ball_pos + Vec3::new(0.0, tee.ball_radius, 0.0)
        );
    }

    #[test]
    fn no_flight_has_no_tracer_and_the_ball_on_the_tee() {
        let tee = TeeBox::default();
        let mut out = FrameContent::default();
        let flight = FlightRenderData {
            ball_pos: Vec3::ONE,
            trail_points: trail(),
            current_time: 1.0,
        };
        flight_vertices(&[flight], &tee, &mut out);
        flight_vertices(&[], &tee, &mut out);
        assert_eq!((out.newest_tube, out.newest_line), (0, 0));
        assert_eq!(out.ball_centre, tee_ball_centre(&tee));
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
