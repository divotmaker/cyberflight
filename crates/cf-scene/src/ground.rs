//! The range floor as dimple ground.
//!
//! The range is level ground at Y = 0. It is fairway, except inside the
//! range's optional [`GreenConfig`], where it is green. Each point maps to
//! a dimple [`Surface`] preset, and [`RangeTerrain`] serves those to
//! dimple's ground physics, which samples the terrain under the ball at
//! every step: a ball that lands short of the green and runs onto it rolls
//! on as a green, and one that runs off it rolls out on fairway.

use dimple::aero::BallModel;
use dimple::environment::Environment;
use dimple::ground::{GroundPhysics, GroundSample, Surface, Terrain};
use dimple::trajectory::{ShotInput, ShotResult};
use glam::DVec3;

use crate::frame::physics_to_scene;
use crate::green::GreenConfig;

/// Surface at a scene-frame ground point (`x`, `z`) (see
/// [`frame`](crate::frame)): [`Surface::GREEN`] on `green`,
/// [`Surface::FAIRWAY`] elsewhere, and everywhere when `green` is `None`.
#[must_use]
pub fn surface_at(x: f64, z: f64, green: Option<GreenConfig>) -> Surface {
    if green.is_some_and(|g| g.contains(x, z)) {
        Surface::GREEN
    } else {
        Surface::FAIRWAY
    }
}

/// The range floor as a dimple [`Terrain`]: level ground with the surface
/// of [`surface_at`].
///
/// dimple samples it in its own frame; each sample is mapped to the scene
/// with [`physics_to_scene`] before the green lookup.
#[derive(Debug, Clone, Copy)]
pub struct RangeTerrain {
    green: Option<GreenConfig>,
}

impl RangeTerrain {
    /// The range floor with `green`, or all fairway for `None`.
    #[must_use]
    pub fn new(green: Option<GreenConfig>) -> Self {
        Self { green }
    }
}

impl Terrain for RangeTerrain {
    fn sample(&self, x: f64, z: f64) -> GroundSample {
        let scene = physics_to_scene(DVec3::new(x, 0.0, z));
        GroundSample {
            height: 0.0,
            normal: DVec3::Y,
            surface: surface_at(scene.x, scene.z, self.green),
        }
    }
}

/// Simulate a full shot on the range: flight, then impacts, hops, slide
/// and roll on the range floor with `green` (all fairway for `None`),
/// hopping through `env`'s air. The result is in dimple's frame; map
/// positions to the scene with [`physics_to_scene`].
#[must_use]
pub fn simulate_shot(
    input: &ShotInput,
    ball: &BallModel,
    env: &Environment,
    green: Option<GreenConfig>,
) -> ShotResult {
    let ground = GroundPhysics::new(RangeTerrain::new(green));
    dimple::trajectory::simulate_shot(input, ball, env, &ground)
}

#[cfg(test)]
mod tests {
    use dimple::trajectory::simulate_flight;

    use super::*;
    use crate::frame::screen_x;
    use crate::range::Range;

    fn shot(input: &ShotInput, green: Option<GreenConfig>) -> ShotResult {
        simulate_shot(input, &BallModel::TOUR, &Environment::SEA_LEVEL, green)
    }

    /// A green of `radius` m on the target line at the downrange distance
    /// of the dimple position `p`.
    fn green_at(p: DVec3, radius: f64) -> Option<GreenConfig> {
        Some(GreenConfig {
            distance_m: physics_to_scene(p).z as f32,
            diameter_m: (2.0 * radius) as f32,
        })
    }

    fn push() -> ShotInput {
        ShotInput {
            launch_azimuth_deg: 4.0,
            ..ShotInput::seven_iron()
        }
    }

    fn hook() -> ShotInput {
        ShotInput {
            launch_azimuth_deg: -1.0,
            sidespin_rpm: -900.0,
            ..ShotInput::seven_iron()
        }
    }

    fn landing(input: &ShotInput) -> DVec3 {
        let flight = simulate_flight(input, &BallModel::TOUR, &Environment::SEA_LEVEL);
        flight.points.last().expect("flight has points").pos
    }

    #[test]
    fn no_green_is_all_fairway() {
        let green = None;
        for z in (0..300).step_by(5) {
            for x in (-60..=60).step_by(5) {
                let s = surface_at(f64::from(x), f64::from(z), green);
                assert_eq!(s, Surface::FAIRWAY, "({x}, {z})");
            }
        }
    }

    #[test]
    fn default_range_plays_as_fairway() {
        let green = Range::new().green;
        assert_eq!(green, None);
        let result = shot(&ShotInput::seven_iron(), green);
        assert!(
            result
                .ground
                .contacts
                .iter()
                .all(|c| c.surface == Surface::FAIRWAY)
        );
    }

    #[test]
    fn configured_green_edge() {
        let green = Some(GreenConfig {
            distance_m: 100.0,
            diameter_m: 30.0,
        });
        assert_eq!(surface_at(0.0, 100.0, green), Surface::GREEN);
        assert_eq!(surface_at(14.99, 100.0, green), Surface::GREEN);
        assert_eq!(surface_at(-14.99, 100.0, green), Surface::GREEN);
        assert_eq!(surface_at(0.0, 85.01, green), Surface::GREEN);
        assert_eq!(surface_at(15.01, 100.0, green), Surface::FAIRWAY);
        assert_eq!(surface_at(0.0, 115.01, green), Surface::FAIRWAY);
        assert_eq!(surface_at(0.0, 0.0, green), Surface::FAIRWAY);
    }

    #[test]
    fn terrain_samples_in_dimple_frame() {
        // A 10 m green 100 m downrange.
        let terrain = RangeTerrain::new(Some(GreenConfig {
            distance_m: 100.0,
            diameter_m: 10.0,
        }));
        // dimple: x = downrange, z = right.
        let on = terrain.sample(100.0, 0.0);
        assert_eq!(on.surface, Surface::GREEN);
        assert_eq!(on.height, 0.0);
        assert_eq!(on.normal, DVec3::Y);
        assert_eq!(terrain.sample(100.0, 4.9).surface, Surface::GREEN);
        assert_eq!(terrain.sample(100.0, -5.1).surface, Surface::FAIRWAY);
        assert_eq!(terrain.sample(0.0, 100.0).surface, Surface::FAIRWAY);
        assert_eq!(
            RangeTerrain::new(None).sample(100.0, 0.0).surface,
            Surface::FAIRWAY
        );
    }

    #[test]
    fn green_reads_faster_than_fairway() {
        // Lower rolling resistance on the green: a putt-speed roll goes
        // further. The green is a tournament-speed stimp.
        let green = Surface::GREEN.stimp_feet();
        let fairway = Surface::FAIRWAY.stimp_feet();
        assert!(
            green > fairway,
            "green {green:.1} ft, fairway {fairway:.1} ft"
        );
        assert!((9.0..13.0).contains(&green), "green stimp {green:.1} ft");
    }

    #[test]
    fn fairway_runs_shorter_than_green() {
        // Fairway and green bounce alike; the fairway's higher rolling
        // resistance makes full shots finish shorter there.
        for input in [
            ShotInput::driver(),
            ShotInput::seven_iron(),
            ShotInput::wedge(),
        ] {
            let fairway = shot(&input, None);
            let green = shot(&input, green_at(landing(&input), 30.0));
            assert_eq!(green.ground.contacts[0].surface, Surface::GREEN);
            assert_eq!(fairway.ground.contacts[0].surface, Surface::FAIRWAY);
            assert!(
                (green.flight.carry_m - fairway.flight.carry_m).abs() < 1e-9,
                "the floor does not change the carry"
            );
            assert!(
                fairway.total_m < green.total_m,
                "fairway total {:.1} m, green {:.1} m",
                fairway.total_m,
                green.total_m
            );
        }
    }

    #[test]
    fn ball_rolls_off_a_green_onto_fairway() {
        // A small green under the landing: the driver's run carries it off
        // onto fairway, where it finishes.
        let input = ShotInput::driver();
        let green = green_at(landing(&input), 3.0);
        let result = shot(&input, green);
        assert_eq!(result.ground.contacts[0].surface, Surface::GREEN);
        let rest = physics_to_scene(result.final_pos);
        assert_eq!(surface_at(rest.x, rest.z, green), Surface::FAIRWAY);
    }

    #[test]
    fn pushed_shot_lands_right_of_the_target_line() {
        let land = landing(&push());
        assert!(
            land.z > 5.0,
            "pushed shot should land right: z={:.2}",
            land.z
        );
        assert!(physics_to_scene(land).x < -5.0);
    }

    #[test]
    fn hook_lands_left_of_the_target_line() {
        let land = landing(&hook());
        assert!(land.z < -5.0, "hook should land left: z={:.2}", land.z);
        let c = physics_to_scene(land);
        assert!(screen_x(c) < 0.0, "hook lands on the left of the screen");
    }

    // A green on the target line is symmetric about it, so this checks the
    // size of the lateral miss, not its side (the side is asserted above).
    #[test]
    fn green_catches_a_shot_only_if_wider_than_its_lateral_miss() {
        for input in [push(), hook()] {
            let land = landing(&input);
            let off = land.z.abs();
            let narrow = shot(&input, green_at(land, off - 2.0));
            let wide = shot(&input, green_at(land, off + 2.0));
            assert_eq!(narrow.ground.contacts[0].surface, Surface::FAIRWAY);
            assert_eq!(wide.ground.contacts[0].surface, Surface::GREEN);
        }
    }

    // LATERAL SIGN: a slice (positive azimuth and sidespin) finishes right
    // of the target line and is drawn on the right of the screen; a hook
    // mirrors it.
    #[test]
    fn slice_renders_on_the_right_of_the_range() {
        let slice = ShotInput {
            launch_azimuth_deg: 1.0,
            sidespin_rpm: 600.0,
            ..ShotInput::driver()
        };
        let hook = ShotInput {
            launch_azimuth_deg: -1.0,
            sidespin_rpm: -600.0,
            ..ShotInput::driver()
        };
        let green = Range::new().green;

        let sliced = shot(&slice, green);
        assert!(sliced.flight.lateral_m > 10.0, "slice lands right");
        assert!(sliced.final_pos.z > 10.0, "slice finishes right");
        let rest = physics_to_scene(sliced.final_pos);
        assert!(rest.x < -10.0, "scene X of a slice: {:.1}", rest.x);
        assert!(screen_x(rest) > 0.05, "slice on screen: {}", screen_x(rest));

        let hooked = shot(&hook, green);
        assert!((hooked.final_pos.z + sliced.final_pos.z).abs() < 1e-6);
        assert!(screen_x(physics_to_scene(hooked.final_pos)) < -0.05);
    }
}
