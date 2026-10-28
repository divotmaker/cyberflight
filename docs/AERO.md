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
  spin without an axis counts as backspin (less any measured sidespin). A
  shot without any spin flies with an assumed 2500 rpm of backspin; the
  HUD shows spin the device did not measure as `-`.
  NaN or infinite launch values drop the shot. Values outside a physical
  envelope are clamped to it and logged: ball speed up to 250 mph (a speed
  of zero or less drops the shot), launch elevation −20° to 70°, azimuth
  ±60°, backspin and sidespin ±20 000 rpm.
- **Ball and air.** Shots fly with `BallModel::TOUR` in
  `Environment::SEA_LEVEL`; the hops between ground impacts fly through the
  same air.
- **Full shot.** `cf_scene::ground::simulate_shot` runs dimple's
  `simulate_shot` over the range floor: the flight, then impacts, hops,
  slide and roll until the ball stops. Carry, total and offline (the rest
  position's lateral offset) are dimple's results.
- **Range floor.** `cf_scene::ground::RangeTerrain` is a dimple `Terrain`
  wrapped in the default `GroundPhysics`: level ground at Y = 0,
  `Surface::FAIRWAY` everywhere except on the range's optional green. The
  green (`cf_scene::green::GreenConfig`) is one circle on the target line,
  set by its distance downrange to the centre (`distance_m`) and its
  diameter (`diameter_m`); inside it the floor is `Surface::GREEN`, and its
  edge is drawn as one neon-green ring, the only circle on the range. dimple
  samples the floor under the ball at every step, so a ball that runs onto
  the green rolls on as a green and one that runs off it finishes on
  fairway. A new range has no green
  (`cf_scene::green::DEFAULT_GREEN` is `None`), so every shot plays on
  fairway; setting `Range::green` to a `GreenConfig` before the range is
  played and drawn adds one, e.g.
  `range.green = Some(GreenConfig { distance_m: 137.16, diameter_m: 30.0 })`.
  `DEFAULT_GREEN` is the value every new range starts with; changing it
  changes the default for all of them. `diameter_m` must be positive and
  finite.
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
  looking up the green, which is in the scene frame. A slice (positive
  azimuth and sidespin) finishes right of the line and is drawn on the
  right of the screen.
