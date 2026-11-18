//! The hole: a par 4 played from the range's tee, as one analytic ground.
//!
//! [`Hole`] is plain data: an elevation profile down the hole, a fairway,
//! bunkers, a green on a raised pad and a cup. [`Hole::height`] is the one
//! height function that the physics ([`HoleTerrain`]) and the drawing both
//! read; [`Hole::normal`] is its slope and [`Hole::lie_at`] the surface at a
//! point. All positions are in the scene frame (see [`frame`](crate::frame)),
//! in meters: the tee at the origin, +Z downrange, +X the golfer's left.
//!
//! The ground is smooth everywhere: smoothstep joins between profile knots
//! and across the green pad's shoulder, and a bowl `(1 − ρ²)³` for each
//! bunker, so heights and normals are continuous.

use dimple::Vec3;
use dimple::ground::{GroundSample, Restitution, Surface, Terrain};

use crate::frame::{physics_to_scene, scene_to_physics};
use crate::green::GreenConfig;

/// A bunker: an elliptical bowl of sand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bunker {
    /// Center, scene X (m).
    pub x: f64,
    /// Center, scene Z (m).
    pub z: f64,
    /// Half-axis along X (m).
    pub half_x: f64,
    /// Half-axis along Z (m).
    pub half_z: f64,
    /// Depth of the bowl at its center (m).
    pub depth: f64,
}

impl Bunker {
    /// `ρ²` of the scene point (`x`, `z`): below 1 inside the ellipse.
    #[must_use]
    pub fn rho2(&self, x: f64, z: f64) -> f64 {
        let u = (x - self.x) / self.half_x;
        let v = (z - self.z) / self.half_z;
        u * u + v * v
    }

    /// Whether the scene point (`x`, `z`) is in the bunker.
    #[must_use]
    pub fn contains(&self, x: f64, z: f64) -> bool {
        self.rho2(x, z) < 1.0
    }
}

/// The raised pad the green sits on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreenPad {
    /// Height of the pad above the profile at the green's center (m).
    pub height: f64,
    /// Slope of the pad along +Z (rise per meter; positive rises toward the
    /// back of the green).
    pub tilt: f64,
    /// Level width beyond the green's edge before the shoulder starts (m).
    pub collar: f64,
    /// Width of the smoothstep shoulder down to the profile (m).
    pub shoulder: f64,
}

/// A hole: ground shape, surfaces and cup.
#[derive(Debug, Clone, PartialEq)]
pub struct Hole {
    /// Elevation knots down the hole, ascending z: (z, y). Level at each
    /// knot, smoothstep between, constant before the first and after the
    /// last.
    pub profile: Vec<(f64, f64)>,
    /// Fairway half-width knots, ascending z: (z, half width), centered on
    /// x = 0. Linear between knots; no fairway before the first or after
    /// the last.
    pub fairway: Vec<(f64, f64)>,
    pub bunkers: Vec<Bunker>,
    /// The green: a circle on the target line.
    pub green: GreenConfig,
    pub green_pad: GreenPad,
    /// Cup center, scene (x, z).
    pub cup: (f64, f64),
}

/// The surface under a point of a hole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lie {
    Fairway,
    Rough,
    Sand,
    Green,
}

/// Rough: softer than fairway, with more rolling resistance. Chosen, not
/// measured.
pub const ROUGH: Surface = Surface {
    compliance: 1.15,
    mu_roll: 0.3,
    plow: 2.0,
    ..Surface::FAIRWAY
};

/// Sand: no bounce, a rigid flat contact that absorbs most of a landing,
/// and high friction. Chosen, not measured.
pub const SAND: Surface = Surface {
    restitution: Restitution::Constant(0.0),
    compliance: 0.0,
    absorption: 0.9,
    mu_slide: 0.40,
    mu_roll: 0.3,
    ..Surface::FAIRWAY
};

impl Lie {
    /// The dimple surface this lie plays as.
    #[must_use]
    pub fn surface(self) -> Surface {
        match self {
            Lie::Fairway => Surface::FAIRWAY,
            Lie::Rough => ROUGH,
            Lie::Sand => SAND,
            Lie::Green => Surface::GREEN,
        }
    }

    /// Upper-case label for the HUD and the log.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Lie::Fairway => "FAIRWAY",
            Lie::Rough => "ROUGH",
            Lie::Sand => "SAND",
            Lie::Green => "GREEN",
        }
    }
}

/// `3t² − 2t³` of `t` clamped to 0..1.
fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Step for the central differences of [`Hole::normal`] (m).
const NORMAL_EPS: f64 = 1e-3;

impl Hole {
    /// Hole 1: a straight 300 yd par 4 from an elevated tee, two fairway
    /// bunkers at drive distance and a raised green guarded by three more.
    #[must_use]
    pub fn no_1() -> Self {
        let bunker = |x, z, half_x, half_z, depth| Bunker {
            x,
            z,
            half_x,
            half_z,
            depth,
        };
        Self {
            profile: vec![(5.0, 0.0), (110.0, -9.5), (205.0, -10.5), (240.0, -11.0)],
            fairway: vec![
                (105.0, 0.0),
                (125.0, 16.0),
                (205.0, 16.0),
                (218.0, 11.0),
                (230.0, 13.0),
                (242.0, 0.0),
            ],
            bunkers: vec![
                bunker(17.0, 212.0, 6.0, 10.0, 0.9),
                bunker(-16.0, 224.0, 6.0, 9.0, 0.9),
                bunker(15.0, 258.0, 6.0, 5.0, 0.8),
                bunker(-19.0, 276.0, 5.0, 8.0, 0.8),
                bunker(9.0, 292.0, 7.0, 4.5, 0.7),
            ],
            green: GreenConfig {
                distance_m: 274.32,
                diameter_m: 26.0,
            },
            green_pad: GreenPad {
                height: 2.5,
                tilt: 0.02,
                collar: 4.0,
                shoulder: 18.0,
            },
            cup: (0.0, 274.32),
        }
    }

    /// Profile elevation at downrange distance `z`.
    fn profile_at(&self, z: f64) -> f64 {
        let Some(&(z0, y0)) = self.profile.first() else {
            return 0.0;
        };
        if z <= z0 {
            return y0;
        }
        for pair in self.profile.windows(2) {
            let ((za, ya), (zb, yb)) = (pair[0], pair[1]);
            if z <= zb {
                return ya + (yb - ya) * smoothstep((z - za) / (zb - za));
            }
        }
        self.profile.last().map_or(0.0, |&(_, y)| y)
    }

    /// Fairway half-width at downrange distance `z`, or `None` outside the
    /// fairway's knots.
    #[must_use]
    pub fn fairway_half_width(&self, z: f64) -> Option<f64> {
        self.fairway.windows(2).find_map(|pair| {
            let ((za, wa), (zb, wb)) = (pair[0], pair[1]);
            (za..=zb)
                .contains(&z)
                .then(|| wa + (wb - wa) * (z - za) / (zb - za))
        })
    }

    /// The green's center, scene (x, z).
    #[must_use]
    pub fn green_center(&self) -> (f64, f64) {
        let c = self.green.center_vec3();
        (f64::from(c.x), f64::from(c.z))
    }

    /// Ground height at the scene point (`x`, `z`).
    #[must_use]
    pub fn height(&self, x: f64, z: f64) -> f64 {
        let (gx, gz) = self.green_center();
        let pad = &self.green_pad;
        let d = (x - gx).hypot(z - gz);
        let level = f64::from(self.green.radius_m()) + pad.collar;
        let weight = 1.0 - smoothstep((d - level) / pad.shoulder);
        let mut h = self.profile_at(z) + weight * (pad.height + pad.tilt * (z - gz));
        for b in &self.bunkers {
            let rho2 = b.rho2(x, z);
            if rho2 < 1.0 {
                h -= b.depth * (1.0 - rho2).powi(3);
            }
        }
        h
    }

    /// Unit ground normal at the scene point (`x`, `z`), from central
    /// differences of [`height`](Self::height).
    #[must_use]
    pub fn normal(&self, x: f64, z: f64) -> glam::DVec3 {
        let e = NORMAL_EPS;
        let dx = (self.height(x + e, z) - self.height(x - e, z)) / (2.0 * e);
        let dz = (self.height(x, z + e) - self.height(x, z - e)) / (2.0 * e);
        glam::DVec3::new(-dx, 1.0, -dz).normalize()
    }

    /// The lie at the scene point (`x`, `z`). First match wins: green,
    /// sand, fairway, rough.
    #[must_use]
    pub fn lie_at(&self, x: f64, z: f64) -> Lie {
        if self.green.contains(x, z) {
            Lie::Green
        } else if self.bunkers.iter().any(|b| b.contains(x, z)) {
            Lie::Sand
        } else if self.fairway_half_width(z).is_some_and(|w| x.abs() <= w) {
            Lie::Fairway
        } else {
            Lie::Rough
        }
    }

    /// Horizontal distance from the scene point (`x`, `z`) to the cup (m).
    #[must_use]
    pub fn to_cup(&self, x: f64, z: f64) -> f64 {
        (x - self.cup.0).hypot(z - self.cup.1)
    }
}

/// A [`Hole`] as a dimple [`Terrain`]: each sample maps dimple's (x, z) to
/// the scene, reads the hole there and maps the normal back.
#[derive(Debug, Clone, Copy)]
pub struct HoleTerrain<'a>(pub &'a Hole);

impl Terrain for HoleTerrain<'_> {
    fn sample(&self, x: f64, z: f64) -> GroundSample {
        let scene = physics_to_scene(Vec3::new(x, 0.0, z));
        GroundSample {
            height: self.0.height(scene.x, scene.z),
            normal: scene_to_physics(self.0.normal(scene.x, scene.z)),
            surface: self.0.lie_at(scene.x, scene.z).surface(),
        }
    }
}

#[cfg(test)]
mod tests;
