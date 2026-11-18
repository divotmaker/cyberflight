use dimple::Tour;
use dimple::ground::Termination;
use dimple::trajectory::{ShotInput, ShotResult};
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
fn every_lie_but_sand_holds_a_ball_at_rest() {
    let h = hole();
    let mut z = -5.0;
    while z < 340.0 {
        let mut x = -45.0;
        while x <= 45.0 {
            let lie = h.lie_at(x, z);
            if lie != Lie::Sand {
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
