# Ball Flight

cyberflight's ball-flight physics (aerodynamics, RK4 flight, and the ground
phase: impacts, hops, slide and roll) comes from the **dimple** crate:

- Source: [github.com/divotmaker/dimple](https://github.com/divotmaker/dimple)
- Crate: [crates.io/crates/dimple](https://crates.io/crates/dimple)
- API docs: [docs.rs/dimple](https://docs.rs/dimple)

The aerodynamic and ground models, their coefficients, sources, validation
and known limitations are documented in dimple's `docs/AERO.md`. dimple's
ground model is beta: the run and total distances shown after landing are
not yet validated against measured data and may change between releases.

## How cyberflight uses it

- **Shot input.** `cf_scene::shot::ReceivedShot::from_frp` turns a
  flightrelay shot into dimple's `ShotInput`. It needs a ball speed and a
  launch elevation; a missing azimuth is a straight shot. Spin comes from
  the backspin/sidespin pair or from total spin and spin axis; a total
  spin without an axis counts as backspin. A shot without any spin flies
  with an assumed 2500 rpm of backspin.
  NaN or infinite launch values drop the shot. Values outside a physical
  envelope are clamped to it and logged: ball speed 0–250 mph, launch
  elevation −20° to 70°, azimuth ±60°, backspin and sidespin ±20 000 rpm.
- **Ball and air.** Shots fly with `BallModel::TOUR` in
  `Environment::SEA_LEVEL`; the hops between ground impacts fly through the
  same air.
- **Full shot.** `cf_scene::ground::simulate_shot` runs dimple's
  `simulate_shot` over the range floor: the flight, then impacts, hops,
  slide and roll until the ball stops. Carry, total and offline (the rest
  position's lateral offset) are dimple's results.
- **Range floor.** `cf_scene::ground::RangeTerrain` is a dimple `Terrain`
  wrapped in the default `GroundPhysics`: level ground at Y = 0,
  `Surface::GREEN` inside a target's outer radius and `Surface::FAIRWAY`
  everywhere else. dimple samples it under the ball at every step, so a
  ball that runs onto a target rolls on as a green and one that runs off it
  finishes on fairway.
- **Animation.** The rendered path is dimple's flight points followed by its
  ground points. Both carry absolute shot times, so the app plays the
  combined list back against the clock. A ball that dimple does not bring
  to rest (its time or impact limit ends the ground phase) stops moving
  8 s after landing at the latest.
- **Frames.** dimple uses +X downrange, +Y up, +Z right of the target line.
  The scene is right-handed with +Z downrange, +Y up and +X left of the
  target line: the range camera looks down +Z, which puts scene +X on the
  left of the screen. `cf_scene::frame::physics_to_scene` maps a dimple
  vector `(x, y, z)` to the scene vector `(−z, y, x)` and
  `scene_to_physics` maps back. Trajectory points go through the first;
  the terrain maps each dimple sample point to the scene with it before
  looking up the targets, which are in the scene frame. A slice (positive
  azimuth and sidespin) finishes right of the line and is drawn on the
  right of the screen.
