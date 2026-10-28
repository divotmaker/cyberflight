//! The range floor as dimple ground.
//!
//! The range is level ground at Y = 0. It is fairway by default; inside a
//! target's outer radius it is green. Each point maps to a dimple
//! [`Surface`] preset, and [`RangeTerrain`] serves those to dimple's ground
//! physics, which samples the terrain under the ball at every step: a ball
//! that lands short of a target and runs onto it rolls on as a green, and
//! one that runs off a target rolls out on fairway.

use dimple::aero::BallModel;
use dimple::environment::Environment;
use dimple::ground::{GroundPhysics, GroundSample, Surface, Terrain};
use dimple::trajectory::{ShotInput, ShotResult};
use glam::DVec3;

use crate::frame::physics_to_scene;
use crate::target::Target;

/// Surface at a scene-frame ground point (`x`, `z`), the frame of
/// [`Target::center`]: [`Surface::GREEN`] inside any target's outer radius,
/// [`Surface::FAIRWAY`] elsewhere. Targets are circles on the Y = 0 plane.
#[must_use]
pub fn surface_at(x: f64, z: f64, targets: &[Target]) -> Surface {
    let on_target = targets.iter().any(|t| {
        let dx = x - f64::from(t.center[0]);
        let dz = z - f64::from(t.center[2]);
        let r = f64::from(t.outer_radius_m());
        dx * dx + dz * dz <= r * r
    });
    if on_target {
        Surface::GREEN
    } else {
        Surface::FAIRWAY
    }
}

/// The range floor as a dimple [`Terrain`]: level ground with the surface
/// of [`surface_at`].
///
/// dimple samples it in its own frame; each sample is mapped to the scene
/// with [`physics_to_scene`] before the target lookup.
#[derive(Debug, Clone, Copy)]
pub struct RangeTerrain<'a> {
    targets: &'a [Target],
}

impl<'a> RangeTerrain<'a> {
    /// The range floor with greens at `targets`.
    #[must_use]
    pub fn new(targets: &'a [Target]) -> Self {
        Self { targets }
    }
}

impl Terrain for RangeTerrain<'_> {
    fn sample(&self, x: f64, z: f64) -> GroundSample {
        let scene = physics_to_scene(DVec3::new(x, 0.0, z));
        GroundSample {
            height: 0.0,
            normal: DVec3::Y,
            surface: surface_at(scene.x, scene.z, self.targets),
        }
    }
}

/// Simulate a full shot on the range: flight, then impacts, hops, slide
/// and roll on the range floor with greens at `targets`, hopping through
/// `env`'s air. The result is in dimple's frame; map positions to the scene
/// with [`physics_to_scene`].
#[must_use]
pub fn simulate_shot(
    input: &ShotInput,
    ball: &BallModel,
    env: &Environment,
    targets: &[Target],
) -> ShotResult {
    let ground = GroundPhysics::new(RangeTerrain::new(targets));
    dimple::trajectory::simulate_shot(input, ball, env, &ground)
}

#[cfg(test)]
mod tests {
    use dimple::trajectory::simulate_flight;

    use super::*;
    use crate::frame::screen_x;
    use crate::target::default_targets;

    fn shot(input: &ShotInput, targets: &[Target]) -> ShotResult {
        simulate_shot(input, &BallModel::TOUR, &Environment::SEA_LEVEL, targets)
    }

    /// A target of `radius` m centred on the scene point of the dimple
    /// position `p`.
    fn target_at(p: DVec3, radius: f32) -> Target {
        let c = physics_to_scene(p);
        Target {
            center: [c.x as f32, 0.0, c.z as f32],
            num_rings: 1,
            ring_spacing_m: radius,
            label: "T".to_owned(),
        }
    }

    fn landing(input: &ShotInput) -> DVec3 {
        let flight = simulate_flight(input, &BallModel::TOUR, &Environment::SEA_LEVEL);
        flight.points.last().expect("flight has points").pos
    }

    #[test]
    fn surface_at_origin_is_fairway() {
        let targets = default_targets();
        assert_eq!(surface_at(0.0, 0.0, &targets), Surface::FAIRWAY);
    }

    #[test]
    fn surface_at_target_center_is_green() {
        let targets = default_targets();
        // First target is at 45.72m downrange (50 yards)
        let t = &targets[0];
        assert!((t.center[2] - 45.72).abs() < 1e-4);
        assert_eq!(
            surface_at(f64::from(t.center[0]), f64::from(t.center[2]), &targets),
            Surface::GREEN
        );
    }

    #[test]
    fn surface_at_target_edge_is_green() {
        let targets = vec![Target::at_distance(100.0, "100")];
        // Just inside outer radius (5 rings × 5m = 25m), off to the side
        assert_eq!(surface_at(24.9, 100.0, &targets), Surface::GREEN);
    }

    #[test]
    fn surface_outside_target_is_fairway() {
        let targets = vec![Target::at_distance(100.0, "100")];
        // Just outside outer radius
        assert_eq!(surface_at(25.1, 100.0, &targets), Surface::FAIRWAY);
    }

    #[test]
    fn surface_at_off_centre_target_is_lateral_sensitive() {
        // 10m off the target line at scene +X, 100m downrange, 5m radius.
        let targets = vec![Target {
            center: [10.0, 0.0, 100.0],
            num_rings: 1,
            ring_spacing_m: 5.0,
            label: "R".to_owned(),
        }];
        assert_eq!(surface_at(10.0, 100.0, &targets), Surface::GREEN);
        assert_eq!(surface_at(-10.0, 100.0, &targets), Surface::FAIRWAY);
        assert_eq!(surface_at(100.0, 10.0, &targets), Surface::FAIRWAY);
    }

    #[test]
    fn terrain_samples_in_dimple_frame() {
        // 10m left of the target line (scene +X), 100m downrange.
        let targets = vec![Target {
            center: [10.0, 0.0, 100.0],
            num_rings: 1,
            ring_spacing_m: 5.0,
            label: "L".to_owned(),
        }];
        let terrain = RangeTerrain::new(&targets);
        // dimple: x = downrange, z = right, so the target is at z = −10.
        let on = terrain.sample(100.0, -10.0);
        assert_eq!(on.surface, Surface::GREEN);
        assert_eq!(on.height, 0.0);
        assert_eq!(on.normal, DVec3::Y);
        assert_eq!(terrain.sample(100.0, 10.0).surface, Surface::FAIRWAY);
        assert_eq!(terrain.sample(10.0, -100.0).surface, Surface::FAIRWAY);
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
            let fairway = shot(&input, &[]);
            let green = shot(&input, &[target_at(landing(&input), 30.0)]);
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
        let result = shot(&input, &[target_at(landing(&input), 3.0)]);
        assert_eq!(result.ground.contacts[0].surface, Surface::GREEN);
        let rest = physics_to_scene(result.final_pos);
        let targets = [target_at(landing(&input), 3.0)];
        assert_eq!(surface_at(rest.x, rest.z, &targets), Surface::FAIRWAY);
    }

    #[test]
    fn pushed_shot_lands_on_target_right_of_line() {
        let input = ShotInput {
            launch_azimuth_deg: 4.0,
            ..ShotInput::seven_iron()
        };
        let land = landing(&input);
        assert!(
            land.z > 5.0,
            "pushed shot should land right: z={:.2}",
            land.z
        );
        assert!(physics_to_scene(land).x < -5.0);

        let mirrored = DVec3::new(land.x, land.y, -land.z);
        let on_line = shot(&input, &[target_at(land, 3.0)]);
        let off_line = shot(&input, &[target_at(mirrored, 3.0)]);
        assert_eq!(on_line.ground.contacts[0].surface, Surface::GREEN);
        assert_eq!(off_line.ground.contacts[0].surface, Surface::FAIRWAY);
    }

    #[test]
    fn pulled_shot_lands_on_target_left_of_line() {
        let input = ShotInput {
            launch_azimuth_deg: -4.0,
            ..ShotInput::seven_iron()
        };
        let land = landing(&input);
        assert!(
            land.z < -5.0,
            "pulled shot should land left: z={:.2}",
            land.z
        );
        assert!(physics_to_scene(land).x > 5.0);

        let mirrored = DVec3::new(land.x, land.y, -land.z);
        let on_line = shot(&input, &[target_at(land, 3.0)]);
        let off_line = shot(&input, &[target_at(mirrored, 3.0)]);
        assert_eq!(on_line.ground.contacts[0].surface, Surface::GREEN);
        assert_eq!(off_line.ground.contacts[0].surface, Surface::FAIRWAY);
    }

    #[test]
    fn hook_lands_on_target_left_of_line() {
        let input = ShotInput {
            launch_azimuth_deg: -1.0,
            sidespin_rpm: -900.0,
            ..ShotInput::seven_iron()
        };
        let land = landing(&input);
        assert!(land.z < -5.0, "hook should land left: z={:.2}", land.z);
        let c = physics_to_scene(land);
        assert!(screen_x(c) < 0.0, "hook lands on the left of the screen");

        // A green centred on the landing, on scene +X (the golfer's left),
        // catches it; the same green mirrored to the right does not.
        let left = vec![Target {
            center: [c.x as f32, 0.0, c.z as f32],
            num_rings: 1,
            ring_spacing_m: 3.0,
            label: "L".to_owned(),
        }];
        let right = vec![Target {
            center: [-c.x as f32, 0.0, c.z as f32],
            ..left[0].clone()
        }];
        assert_eq!(
            shot(&input, &left).ground.contacts[0].surface,
            Surface::GREEN
        );
        assert_eq!(
            shot(&input, &right).ground.contacts[0].surface,
            Surface::FAIRWAY
        );
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
        let targets = default_targets();

        let sliced = shot(&slice, &targets);
        assert!(sliced.flight.lateral_m > 10.0, "slice lands right");
        assert!(sliced.final_pos.z > 10.0, "slice finishes right");
        let rest = physics_to_scene(sliced.final_pos);
        assert!(rest.x < -10.0, "scene X of a slice: {:.1}", rest.x);
        assert!(screen_x(rest) > 0.05, "slice on screen: {}", screen_x(rest));

        let hooked = shot(&hook, &targets);
        assert!((hooked.final_pos.z + sliced.final_pos.z).abs() < 1e-6);
        assert!(screen_x(physics_to_scene(hooked.final_pos)) < -0.05);
    }
}
