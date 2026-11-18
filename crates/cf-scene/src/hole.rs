//! The hole: a par 4 played from the range's tee, as one analytic ground.
//!
//! [`Hole`] is plain data: an elevation profile down the hole, a fairway,
//! bunkers, a green on a raised pad and a cup. [`Hole::height`] is the one
//! height function that the physics ([`HoleTerrain`]) and the drawing both
//! read; [`Hole::normal`] is its slope and [`Hole::lie_at`] the surface at a
//! point. [`Hole::stop`] ends a shot early: in the cup or in the water. All positions are in the scene frame (see [`frame`](crate::frame)),
//! in meters: the tee at the origin, +Z downrange, +X the golfer's left.
//!
//! The ground is smooth everywhere: smoothstep joins between profile knots
//! and across the green pad's and each pond's shoulder, and a bowl
//! `(1 − ρ²)³` for each bunker and pond, so heights and normals are
//! continuous.

use dimple::Vec3;
use dimple::ground::{GroundPoint, GroundSample, Phase, Restitution, Surface, Terrain};
use dimple::trajectory::TrajectoryPoint;

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

/// A pond: an elliptical bowl filled with water. The ground around it is
/// leveled to the water's surface, so the shoreline is the ellipse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pond {
    /// Center, scene X (m).
    pub x: f64,
    /// Center, scene Z (m).
    pub z: f64,
    /// Half-axis along X (m).
    pub half_x: f64,
    /// Half-axis along Z (m).
    pub half_z: f64,
    /// Depth of the bowl under the surface at its center (m).
    pub depth: f64,
    /// Width of the smoothstep bank from the level shore back to the
    /// hole's ground, measured along the shorter half-axis (m).
    pub bank: f64,
}

impl Pond {
    /// `ρ²` of the scene point (`x`, `z`): below 1 inside the shoreline.
    #[must_use]
    pub fn rho2(&self, x: f64, z: f64) -> f64 {
        let u = (x - self.x) / self.half_x;
        let v = (z - self.z) / self.half_z;
        u * u + v * v
    }

    /// Whether the scene point (`x`, `z`) is in the water.
    #[must_use]
    pub fn contains(&self, x: f64, z: f64) -> bool {
        self.rho2(x, z) < 1.0
    }

    /// Weight of the level shore at `rho2`: 1 inside the shoreline, down to
    /// 0 across the bank.
    fn shore_weight(&self, rho2: f64) -> f64 {
        let past = (rho2.sqrt() - 1.0) * self.half_x.min(self.half_z);
        1.0 - smoothstep(past / self.bank)
    }
}

/// How a shot on a hole stops before dimple brings it to rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The ball dropped into the cup.
    Cup,
    /// The ball reached a pond's water.
    Water,
}

/// The end of a shot cut short by [`Hole::stop`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cut {
    /// Why the shot stops.
    pub stop: Stop,
    /// Shot time at which the ball stops (s).
    pub time: f64,
    /// Where the ball stops, dimple's frame (its center).
    pub pos: Vec3,
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
    pub ponds: Vec<Pond>,
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
    Water,
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
            // The bowl under the water: a shot ends at the surface (see
            // `Hole::stop`), so this only keeps dimple's ball quiet there.
            Lie::Water => SAND,
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
            Lie::Water => "WATER",
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
    /// bunkers at drive distance, a pond in the left rough short of them
    /// and a raised green guarded by three more bunkers.
    #[must_use]
    pub fn no_1() -> Self {
        let bunker = |x, z, half_x, half_z, depth| Bunker {
            x,
            z,
            half_x,
            half_z,
            depth,
        };
        let pond = |x, z, half_x, half_z, depth, bank| Pond {
            x,
            z,
            half_x,
            half_z,
            depth,
            bank,
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
            ponds: vec![pond(31.0, 176.0, 8.0, 24.0, 1.5, 5.0)],
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
        for p in &self.ponds {
            let rho2 = p.rho2(x, z);
            let shore = p.shore_weight(rho2);
            if shore > 0.0 {
                h += shore * (self.water_level(p) - h);
            }
            if rho2 < 1.0 {
                h -= p.depth * (1.0 - rho2).powi(3);
            }
        }
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

    /// Height of the water surface of `pond`: the profile's height at its
    /// center. The shore is level at this height and the bowl dips under
    /// it.
    #[must_use]
    pub fn water_level(&self, pond: &Pond) -> f64 {
        self.profile_at(pond.z)
    }

    /// The lie at the scene point (`x`, `z`). First match wins: green,
    /// water, sand, fairway, rough.
    #[must_use]
    pub fn lie_at(&self, x: f64, z: f64) -> Lie {
        if self.green.contains(x, z) {
            Lie::Green
        } else if self.ponds.iter().any(|p| p.contains(x, z)) {
            Lie::Water
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

impl Hole {
    /// Where the ball first reaches water along `path` (a shot's timeline
    /// in dimple's frame, flight then ground): the first point inside a
    /// pond's shoreline at or under its surface, interpolated to where the
    /// path crosses the surface. The time and the ball's center there, on
    /// the surface, in dimple's frame; `None` if it never does.
    #[must_use]
    pub fn water_entry(&self, path: &[TrajectoryPoint]) -> Option<(f64, Vec3)> {
        path.windows(2).find_map(|pair| {
            let (a, b) = (&pair[0], &pair[1]);
            let (pa, pb) = (physics_to_scene(a.pos), physics_to_scene(b.pos));
            self.ponds.iter().find_map(|pond| {
                let level = self.water_level(pond);
                if !pond.contains(pb.x, pb.z) || pb.y >= level {
                    return None;
                }
                let t = if pa.y > level {
                    (pa.y - level) / (pa.y - pb.y)
                } else {
                    1.0
                };
                let mut at = pa.lerp(pb, t);
                if !pond.contains(at.x, at.z) {
                    at = pb;
                }
                at.y = level;
                Some((a.time + (b.time - a.time) * t, scene_to_physics(at)))
            })
        })
    }

    /// Whether and where a shot stops before dimple brings it to rest:
    /// the first of the ball reaching water ([`water_entry`](Self::water_entry))
    /// and the ball dropping into the cup ([`holed_at`], at rest in the
    /// cup's center one ball radius under the surface). `path` is the
    /// shot's timeline and `points` its ground points, both in dimple's
    /// frame.
    #[must_use]
    pub fn stop(
        &self,
        path: &[TrajectoryPoint],
        points: &[GroundPoint],
        ball_radius: f64,
        gravity: f64,
    ) -> Option<Cut> {
        let water = self.water_entry(path).map(|(time, pos)| Cut {
            stop: Stop::Water,
            time,
            pos,
        });
        let cup = holed_at(points, self.cup, ball_radius, gravity).map(|time| {
            let (x, z) = self.cup;
            Cut {
                stop: Stop::Cup,
                time,
                pos: scene_to_physics(glam::DVec3::new(x, self.height(x, z) - ball_radius, z)),
            }
        });
        [water, cup]
            .into_iter()
            .flatten()
            .min_by(|a, b| a.time.total_cmp(&b.time))
    }
}

/// Radius of the cup (m): 4.25 in across.
pub const CUP_RADIUS_M: f64 = 0.054;

/// When the ball drops into the cup at scene (x, z) `cup`: the shot time,
/// or `None`.
///
/// `points` are dimple's ground points (its frame). Each segment the ball
/// slides or rolls along (never a hop) is checked at its closest approach
/// `d` to the cup's center, with the ball's horizontal speed `v` there. The
/// ball drops if `d < R` and `v ≤ (2·√(R² − d²) − r)·√(g / 2r)`, with `R`
/// the cup's radius and `r` = `ball_radius`: it falls its own radius before
/// it has crossed the chord of the cup under it. A ball that flies or hops
/// over the cup, lips out or is deflected is not modeled.
#[must_use]
pub fn holed_at(
    points: &[GroundPoint],
    cup: (f64, f64),
    ball_radius: f64,
    gravity: f64,
) -> Option<f64> {
    let r_cup = CUP_RADIUS_M;
    let fall = (gravity / (2.0 * ball_radius)).sqrt();
    points.windows(2).find_map(|pair| {
        let (a, b) = (&pair[0], &pair[1]);
        if a.phase == Phase::Hop {
            return None;
        }
        let (pa, pb) = (physics_to_scene(a.pos), physics_to_scene(b.pos));
        let (ax, az) = (pa.x - cup.0, pa.z - cup.1);
        let (sx, sz) = (pb.x - pa.x, pb.z - pa.z);
        let len2 = sx * sx + sz * sz;
        let t = if len2 > 0.0 {
            (-(ax * sx + az * sz) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let d = (ax + t * sx).hypot(az + t * sz);
        if d >= r_cup {
            return None;
        }
        let speed = |p: &GroundPoint| p.vel.x.hypot(p.vel.z);
        let v = speed(a) + (speed(b) - speed(a)) * t;
        let limit = (2.0 * (r_cup * r_cup - d * d).sqrt() - ball_radius) * fall;
        (v <= limit).then_some(a.time + (b.time - a.time) * t)
    })
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
