# Ball Flight

cyberflight's ball-flight physics (aerodynamics, RK4 flight, bounce and
rollout) comes from the **dimple** crate:

- Source: [github.com/divotmaker/dimple](https://github.com/divotmaker/dimple)
- Crate: [crates.io/crates/dimple](https://crates.io/crates/dimple)
- API docs: [docs.rs/dimple](https://docs.rs/dimple)

The aerodynamic model, its coefficients, sources, validation against Trackman
data and known limitations are documented in dimple's `docs/AERO.md`.

## How cyberflight uses it

- **Ball and air.** Shots fly with `BallModel::TOUR` in
  `Environment::SEA_LEVEL`.
- **Surfaces.** The carry landing point picks the ground model for the whole
  bounce and rollout: inside a target's outer radius uses
  `BounceSurface::GREEN` / `RolloutSurface::GREEN`, anywhere else uses
  `FAIRWAY`.
- **Frames.** dimple trajectories use +X downrange, +Y up, +Z right of the
  target line. The scene uses +X right, +Y up, +Z downrange. `traj_to_scene`
  in `app/src/flight.rs` maps a trajectory point `(x, y, z)` to the scene
  point `(z, y, x)`. Targets and surface queries (`cf_scene::surface`) are in
  the scene frame.
