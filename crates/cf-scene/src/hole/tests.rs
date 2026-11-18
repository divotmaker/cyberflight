use dimple::ground::{GroundPoint, Termination};
use dimple::trajectory::{ShotInput, ShotResult, TrajectoryPoint};
use dimple::{AeroModel, Tour};
use glam::DVec3;

use super::*;
use crate::ground::simulate_shot;
use crate::range::Ground;

const TOL: f64 = 1e-3;

fn hole() -> Hole {
    Hole::no_1()
}

#[test]
fn heights_and_lies_at_known_points() {
    let h = hole();
    let table = [
        (0.0, 0.0, 0.0, Lie::Rough),
        (0.0, 57.5, -4.750, Lie::Rough),
        (0.0, 110.0, -9.500, Lie::Fairway),
        (16.0, 150.0, -9.883, Lie::Fairway),
        (16.1, 150.0, -9.883, Lie::Rough),
        (0.0, 205.0, -10.500, Lie::Fairway),
        (17.0, 212.0, -11.452, Lie::Sand),
        (-16.0, 224.0, -11.682, Lie::Sand),
        (0.0, 241.0, -10.955, Lie::Fairway),
        (0.0, 243.0, -10.797, Lie::Rough),
        // The green's edge, 1 cm in (the green is stored as f32).
        (0.0, 261.33, -8.760, Lie::Green),
        (0.0, 274.32, -8.500, Lie::Green),
        (0.0, 287.31, -8.240, Lie::Green),
        (12.99, 274.32, -8.500, Lie::Green),
        (13.1, 274.32, -8.500, Lie::Rough),
        (0.0, 320.0, -11.000, Lie::Rough),
        (0.0, 400.0, -11.000, Lie::Rough),
    ];
    for (x, z, y, lie) in table {
        let got = h.height(x, z);
        assert!(
            (got - y).abs() < TOL,
            "height at ({x}, {z}): {got:.4}, want {y}"
        );
        assert_eq!(h.lie_at(x, z), lie, "lie at ({x}, {z})");
    }
    // The middle of the hill is a 13.6 % slope.
    let n = h.normal(0.0, 57.5);
    let slope = (n.z / n.y).abs();
    assert!((slope - 0.136).abs() < 0.001, "slope {slope}");
}

#[test]
fn the_tee_pad_is_level() {
    let h = hole();
    for z in [-20.0, -5.0, 0.0, 3.0, 4.999] {
        for x in [-40.0, -5.0, 0.0, 5.0, 40.0] {
            assert_eq!(h.height(x, z), 0.0, "({x}, {z})");
            assert!((h.normal(x, z) - DVec3::Y).length() < 1e-9, "({x}, {z})");
        }
    }
}

#[test]
fn the_green_is_one_tilted_plane() {
    let h = hole();
    let want = DVec3::new(0.0, 1.0, -0.02).normalize();
    let (gx, gz) = h.green_center();
    for i in -12..=12 {
        for j in -12..=12 {
            let (x, z) = (gx + f64::from(i), gz + f64::from(j));
            if h.green.contains(x, z) {
                let n = h.normal(x, z);
                assert!((n - want).length() < 1e-6, "({x}, {z}): {n}");
            }
        }
    }
}

#[test]
fn the_normal_matches_the_height() {
    let h = hole();
    for z in (0..340).step_by(7) {
        for x in (-45..=45).step_by(4) {
            let (x, z) = (f64::from(x), f64::from(z));
            let n = h.normal(x, z);
            let y = h.height(x, z);
            // A 1 cm step along the tangent plane in x and in z.
            for (dx, dz) in [(0.01, 0.0), (0.0, 0.01)] {
                let plane = y - (n.x * dx + n.z * dz) / n.y;
                let got = h.height(x + dx, z + dz);
                assert!((got - plane).abs() < 1e-4, "({x}, {z}) step ({dx}, {dz})");
            }
        }
    }
}

#[test]
fn the_ground_is_continuous_across_every_seam() {
    let h = hole();
    let (gx, gz) = h.green_center();
    let level = f64::from(h.green.radius_m()) + h.green_pad.collar;
    let mut seams: Vec<(f64, f64, f64, f64)> = Vec::new(); // point, direction across
    for b in &h.bunkers {
        for k in 0..16 {
            let a = f64::from(k) * std::f64::consts::TAU / 16.0;
            let (c, s) = (a.cos(), a.sin());
            seams.push((b.x + b.half_x * c, b.z + b.half_z * s, c, s));
        }
    }
    for p in &h.ponds {
        let bank = 1.0 + p.bank / p.half_x.min(p.half_z);
        for scale in [1.0, bank] {
            for k in 0..16 {
                let a = f64::from(k) * std::f64::consts::TAU / 16.0;
                let (c, s) = (a.cos(), a.sin());
                seams.push((p.x + scale * p.half_x * c, p.z + scale * p.half_z * s, c, s));
            }
        }
    }
    for r in [level, level + h.green_pad.shoulder] {
        for k in 0..16 {
            let a = f64::from(k) * std::f64::consts::TAU / 16.0;
            let (c, s) = (a.cos(), a.sin());
            seams.push((gx + r * c, gz + r * s, c, s));
        }
    }
    for &(z, _) in &h.profile {
        seams.push((0.0, z, 0.0, 1.0));
        seams.push((-30.0, z, 0.0, 1.0));
    }
    for (x, z, dx, dz) in seams {
        let e = 0.5e-3;
        let (a, b) = ((x - e * dx, z - e * dz), (x + e * dx, z + e * dz));
        let dh = (h.height(a.0, a.1) - h.height(b.0, b.1)).abs();
        let dn = (h.normal(a.0, a.1) - h.normal(b.0, b.1)).length();
        assert!(dh < 1e-3, "height step {dh} at ({x}, {z})");
        assert!(dn < 0.01, "normal step {dn} at ({x}, {z})");
    }
}

#[test]
fn every_lie_but_sand_and_water_holds_a_ball_at_rest() {
    let h = hole();
    let mut z = -5.0;
    while z < 340.0 {
        let mut x = -45.0;
        while x <= 45.0 {
            let lie = h.lie_at(x, z);
            if lie != Lie::Sand && lie != Lie::Water {
                let n = h.normal(x, z);
                let sample = GroundSample {
                    height: h.height(x, z),
                    normal: scene_to_physics(n),
                    surface: lie.surface(),
                };
                assert!(sample.holds_at_rest(), "{lie:?} at ({x}, {z})");
            }
            x += 0.5;
        }
        z += 0.5;
    }
}

#[test]
fn terrain_samples_in_dimple_frame() {
    let h = hole();
    let terrain = HoleTerrain(&h);
    // dimple: x downrange, z right.
    let cup = terrain.sample(274.32, 0.0);
    assert!((cup.height - h.height(0.0, 274.32)).abs() < 1e-12);
    assert_eq!(cup.surface, Surface::GREEN);
    assert_eq!(terrain.sample(224.0, 16.0).surface, SAND);
    assert_eq!(terrain.sample(212.0, -17.0).surface, SAND);
    // The green tilts up toward the back: its dimple normal leans back
    // toward the tee (−x).
    assert!(cup.normal.x < 0.0);
    assert!((cup.normal.length() - 1.0).abs() < 1e-12);
}

#[test]
fn lies_map_to_their_surfaces() {
    assert_eq!(Lie::Fairway.surface(), Surface::FAIRWAY);
    assert_eq!(Lie::Rough.surface(), ROUGH);
    assert_eq!(Lie::Sand.surface(), SAND);
    assert_eq!(Lie::Green.surface(), Surface::GREEN);
    // Bunker 1 overlaps the fairway's edge: sand wins.
    let h = hole();
    assert!(h.fairway_half_width(212.0).is_some_and(|w| 12.0 <= w));
    assert_eq!(h.lie_at(12.0, 212.0), Lie::Sand);
}

#[test]
fn rough_runs_less_than_fairway_and_sand_does_not_hop() {
    let input = ShotInput::driver();
    let run = |s: &Surface| {
        let r = dimple::trajectory::simulate_shot(&input, &Tour::SEA_LEVEL, s);
        (r.total_m - r.flight.carry_m, r.ground.contacts.len())
    };
    let (fairway, _) = run(&Surface::FAIRWAY);
    let (rough, _) = run(&ROUGH);
    let (_, sand_contacts) = run(&SAND);
    assert!(
        rough < fairway,
        "rough {rough:.1} m, fairway {fairway:.1} m"
    );
    assert_eq!(sand_contacts, 1);
}

fn play(input: &ShotInput) -> (Hole, ShotResult) {
    let h = hole();
    let r = simulate_shot(input, &Tour::SEA_LEVEL, &Ground::Hole(h.clone()));
    (h, r)
}

/// First-contact lie, rest lie, rest point (scene), after checking the
/// ball came to rest on the ground.
fn outcome(input: &ShotInput) -> (Lie, Lie, DVec3, ShotResult) {
    let (h, r) = play(input);
    assert_eq!(r.ground.termination, Termination::Rest, "{input:?}");
    let first = physics_to_scene(r.ground.contacts[0].pos);
    let rest = physics_to_scene(r.ground.rest.pos);
    let ground = h.height(rest.x, rest.z);
    assert!(
        (rest.y - ground).abs() < 1e-3,
        "rest {rest} vs ground {ground:.4}"
    );
    (
        h.lie_at(first.x, first.z),
        h.lie_at(rest.x, rest.z),
        rest,
        r,
    )
}

fn drive(azimuth_deg: f64) -> ShotInput {
    ShotInput {
        ball_speed_mph: 140.0,
        launch_angle_deg: 13.0,
        launch_azimuth_deg: azimuth_deg,
        backspin_rpm: 3000.0,
        sidespin_rpm: 0.0,
    }
}

#[test]
fn a_drive_down_the_middle_finds_the_fairway() {
    let (first, rest, at, r) = outcome(&drive(0.0));
    assert_eq!((first, rest), (Lie::Fairway, Lie::Fairway));
    assert!((200.0..240.0).contains(&r.flight.carry_m));
    assert!(at.z > r.flight.carry_m);
}

#[test]
fn a_drive_at_the_right_bunker_stays_in_it() {
    let (first, rest, at, r) = outcome(&drive(4.2));
    assert_eq!((first, rest), (Lie::Sand, Lie::Sand));
    assert_eq!(r.ground.contacts.len(), 1);
    assert!(at.x < 0.0, "right of the line");
}

#[test]
fn a_drive_at_the_left_bunker_stays_in_it() {
    let (first, rest, at, _) = outcome(&drive(-4.6));
    assert_eq!((first, rest), (Lie::Sand, Lie::Sand));
    assert!(at.x > 0.0, "left of the line");
}

#[test]
fn a_slice_finishes_in_the_right_rough() {
    let slice = ShotInput {
        ball_speed_mph: 150.0,
        launch_angle_deg: 12.0,
        launch_azimuth_deg: 3.0,
        backspin_rpm: 2800.0,
        sidespin_rpm: 900.0,
    };
    let (_, rest, at, _) = outcome(&slice);
    assert_eq!(rest, Lie::Rough);
    assert!(at.x < -10.0, "right of the line: x {:.1}", at.x);
}

#[test]
fn the_preset_driver_reaches_the_green() {
    let (first, rest, _, _) = outcome(&ShotInput::driver());
    assert_eq!((first, rest), (Lie::Green, Lie::Green));
}

#[test]
fn the_preset_wedge_finishes_on_the_fairway() {
    let (_, rest, at, _) = outcome(&ShotInput::wedge());
    assert_eq!(rest, Lie::Fairway);
    assert!(at.z > 110.0, "z {:.1}", at.z);
}

#[test]
fn a_topped_shot_stops_on_the_hill() {
    let topped = ShotInput {
        ball_speed_mph: 80.0,
        launch_angle_deg: 2.0,
        launch_azimuth_deg: 0.0,
        backspin_rpm: 1000.0,
        sidespin_rpm: 0.0,
    };
    let (_, rest, at, _) = outcome(&topped);
    assert_eq!(rest, Lie::Rough);
    assert!((5.0..110.0).contains(&at.z), "z {:.1}", at.z);
    assert!(at.y < 0.0, "below the tee: y {:.2}", at.y);
}

const BALL_R: f64 = 0.021_335;
const G: f64 = 9.81;

/// A ground point at scene (x, z) moving downrange at `v` m/s.
fn gp(time: f64, x: f64, z: f64, v: f64, phase: dimple::ground::Phase) -> GroundPoint {
    let state = dimple::BallState {
        time,
        pos: scene_to_physics(DVec3::new(x, 0.0, z)),
        vel: Vec3::new(v, 0.0, 0.0),
        spin: Vec3::ZERO,
    };
    GroundPoint::new(&state, phase)
}

/// A straight roll at `v` m/s across the cup at (0, 10), `off` m to the
/// side, from 0.5 m short to 0.5 m past, in `phase`.
fn across(off: f64, v: f64, phase: dimple::ground::Phase) -> Option<f64> {
    let points = [gp(1.0, off, 9.5, v, phase), gp(2.0, off, 10.5, v, phase)];
    holed_at(&points, (0.0, 10.0), BALL_R, G)
}

#[test]
fn the_cup_takes_a_slow_enough_ball() {
    use dimple::ground::Phase::{Hop, Roll, Slide};
    let t = across(0.0, 1.2, Roll).expect("holed through the center");
    assert!((t - 1.5).abs() < 1e-9, "at the crossing: {t}");
    assert_eq!(across(0.0, 1.4, Roll), None);
    assert!(across(0.04, 0.7, Slide).is_some());
    assert_eq!(across(0.04, 0.9, Roll), None);
    assert_eq!(across(0.06, 0.1, Roll), None);
    assert_eq!(across(0.0, 0.5, Hop), None);
    // A ball that stops inside the cup's radius.
    let stops = [gp(1.0, 0.0, 9.9, 0.2, Roll), gp(2.0, 0.03, 10.0, 0.0, Roll)];
    assert!(holed_at(&stops, (0.0, 10.0), BALL_R, G).is_some());
}

/// A ball started at `v` m/s with no spin (it slides, then rolls) from 1 m
/// short of the cup on the green.
fn roll_at_the_cup(v: f64) -> (Hole, ShotResult) {
    let h = hole();
    let (cx, cz) = h.cup;
    let z = cz - 1.0;
    let start = dimple::BallState {
        time: 0.0,
        pos: scene_to_physics(DVec3::new(cx, h.height(cx, z) - 1e-6, z)),
        vel: Vec3::new(v, 0.0, 0.0),
        spin: Vec3::ZERO,
    };
    let physics = dimple::ground::GroundPhysics::new(HoleTerrain(&h));
    let r = dimple::trajectory::simulate_shot_from(&start, &Tour::SEA_LEVEL, &physics);
    (h, r)
}

fn holed(h: &Hole, r: &ShotResult) -> Option<f64> {
    use dimple::AeroModel;
    let model = Tour::SEA_LEVEL;
    holed_at(&r.ground.points, h.cup, model.radius_m(), model.gravity())
}

#[test]
fn a_roll_at_the_cup_drops() {
    let (h, r) = roll_at_the_cup(2.0);
    assert!(holed(&h, &r).is_some());
}

#[test]
fn a_fast_roll_runs_over_the_cup() {
    let (h, r) = roll_at_the_cup(3.0);
    assert_eq!(holed(&h, &r), None);
    assert!(physics_to_scene(r.ground.rest.pos).z > h.cup.1);
}

#[test]
fn a_slow_roll_stops_short() {
    let (h, r) = roll_at_the_cup(1.3);
    assert_eq!(holed(&h, &r), None);
    assert!(physics_to_scene(r.ground.rest.pos).z < h.cup.1);
}

#[test]
fn the_pond_is_a_bowl_under_a_level_surface() {
    let h = hole();
    let p = h.ponds[0];
    let level = h.water_level(&p);
    assert!((h.height(p.x, p.z) - (level - p.depth)).abs() < 1e-9);
    assert_eq!(h.lie_at(p.x, p.z), Lie::Water);
    for k in 0..32 {
        let a = f64::from(k) * std::f64::consts::TAU / 32.0;
        let (x, z) = (p.x + p.half_x * a.cos(), p.z + p.half_z * a.sin());
        assert!((h.height(x, z) - level).abs() < 1e-9, "shore at ({x}, {z})");
        let (ix, iz) = (p.x + 0.9 * (x - p.x), p.z + 0.9 * (z - p.z));
        assert!(h.height(ix, iz) < level, "under water at ({ix}, {iz})");
        assert_eq!(h.lie_at(ix, iz), Lie::Water);
    }
}

#[test]
fn the_pond_and_its_bank_stay_clear_of_bunkers_and_fairway() {
    let h = hole();
    for p in &h.ponds {
        let reach = 1.0 + p.bank / p.half_x.min(p.half_z);
        let mut z = p.z - reach * p.half_z;
        while z <= p.z + reach * p.half_z {
            let mut x = p.x - reach * p.half_x;
            while x <= p.x + reach * p.half_x {
                if p.rho2(x, z) < reach * reach {
                    assert!(!h.bunkers.iter().any(|b| b.contains(x, z)), "({x}, {z})");
                    let fairway = h.fairway_half_width(z).is_some_and(|w| x.abs() <= w);
                    assert!(!fairway, "fairway at ({x}, {z})");
                }
                x += 0.5;
            }
            z += 0.5;
        }
    }
}

/// The shot's timeline: flight points, then ground points.
fn path(r: &ShotResult) -> Vec<TrajectoryPoint> {
    let ground = r.ground.points.iter().map(|p| TrajectoryPoint {
        time: p.time,
        pos: p.pos,
        vel: p.vel,
    });
    r.flight.points.iter().copied().chain(ground).collect()
}

fn stop_of(h: &Hole, r: &ShotResult) -> Option<Cut> {
    let model = Tour::SEA_LEVEL;
    h.stop(
        &path(r),
        &r.ground.points,
        model.radius_m(),
        model.gravity(),
    )
}

#[test]
fn a_pulled_drive_ends_in_the_water() {
    let (h, r) = play(&ShotInput {
        ball_speed_mph: 120.0,
        ..drive(-9.0)
    });
    let cut = stop_of(&h, &r).expect("in the water");
    assert_eq!(cut.stop, Stop::Water);
    assert!(cut.time < r.ground.rest.time);
    let at = physics_to_scene(cut.pos);
    let p = h.ponds[0];
    assert!(p.contains(at.x, at.z), "({:.1}, {:.1})", at.x, at.z);
    assert!((at.y - h.water_level(&p)).abs() < 1e-9);
    assert_eq!(h.lie_at(at.x, at.z), Lie::Water);
}

#[test]
fn a_drive_that_carries_the_pond_stays_dry() {
    let (h, r) = play(&drive(-9.0));
    assert_eq!(stop_of(&h, &r), None);
    let first = physics_to_scene(r.ground.contacts[0].pos);
    assert!(first.z > h.ponds[0].z + h.ponds[0].half_z);
}

#[test]
fn a_ball_that_rolls_into_the_pond_ends_at_its_shore() {
    // Lands in the rough short and right of the pond and runs into it.
    let (h, r) = play(&ShotInput {
        ball_speed_mph: 110.0,
        ..drive(-8.0)
    });
    let first = physics_to_scene(r.ground.contacts[0].pos);
    assert_eq!(h.lie_at(first.x, first.z), Lie::Rough);
    let cut = stop_of(&h, &r).expect("in the water");
    assert_eq!(cut.stop, Stop::Water);
    assert!(cut.time > r.flight.flight_time);
}
