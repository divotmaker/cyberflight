# Golf Ball Aerodynamics Model

Aerodynamic model for golf ball flight simulation. Used by
[cyberflight](https://github.com/divotmaker/cyberflight) (cf-math crate) and
available for any project that needs realistic ball flight from launch monitor
data.

## Overview

The model computes drag coefficient (Cd) and lift coefficient (Cl) as functions
of Reynolds number and spin ratio, then integrates the trajectory using RK4 at
a 1ms fixed timestep. It handles altitude, temperature, and spin decay.

**Key inputs:** ball speed, launch angle, backspin RPM, sidespin RPM, altitude,
temperature.

**Key outputs:** carry distance, apex height, lateral deviation, flight time,
full trajectory points.

## Force Model

Three forces act on the ball at each timestep:

```
F_total = F_gravity + F_drag + F_magnus
```

- **Gravity:** constant -9.80665 m/s² in Y
- **Drag:** opposes velocity. Fd = -Cd × q × A / m × v̂
- **Magnus lift:** perpendicular to velocity, in the spin plane. Fl = Cl × q × A / m × (ω̂ × v̂)

Where q = ½ρv² (dynamic pressure), A = cross-sectional area, m = ball mass.

## Cd/Cl Model

The aerodynamic coefficients use a sigmoid transition at the dimpled-ball drag
crisis (~Re 75,000), with spin-dependent drag and exponential lift saturation.

### Drag coefficient

```
Cd = Cd_base(Re) + Cd_spin × SR

where:
  Cd_base = Cd_sub + (Cd_super - Cd_sub) × σ(Re)
  σ(Re)   = 1 / (1 + exp(-(Re - Re_crit) / Re_width))    [sigmoid]
  SR      = ω × R / V                                      [spin ratio]
  Re      = ρ × V × d / μ                                  [Reynolds number]
```

The base drag transitions from subcritical to supercritical at the drag crisis.
Spin adds drag linearly with spin ratio (asymmetric wake from spin-induced
pressure gradient).

### Lift coefficient

```
Cl = Cl_max(Re) × (1 - exp(-SR / SR_scale))

where:
  Cl_max = Cl_sub + (Cl_super - Cl_sub) × σ(Re)
```

Lift follows an exponential saturation curve in spin ratio. This means:
- **Low SR (driver, ~0.08):** Cl is well below the ceiling (~0.19)
- **High SR (iron, ~0.29):** Cl approaches the ceiling (~0.27)
- **Very high SR (wedge, ~0.46):** Cl is fully saturated (~0.29)

This spin-sensitive lift is a key feature of modern ball design — it rewards
the "high launch, low spin" fitting strategy that dominates modern equipment.

## Ball Models

### `BallModel::TOUR` — Modern Tour Ball (2020s)

| Parameter | Value | Role |
|---|---|---|
| mass_kg | 0.04593 | Ball mass (USGA max: 1.62 oz) |
| diameter_m | 0.04267 | Ball diameter (USGA min: 1.68 in) |
| cd_sub | 0.310 | Subcritical base drag |
| cd_super | 0.235 | Supercritical base drag |
| cd_spin | 0.12 | Spin-dependent drag increment |
| cl_sub | 0.050 | Subcritical lift ceiling |
| cl_super | 0.29 | Supercritical lift ceiling |
| sr_scale | 0.10 | Lift saturation rate |
| re_crit | 75,000 | Drag crisis Reynolds number |
| re_width | 125 | Sigmoid transition width |

Aerodynamic sources:
- Li, Tsubokura & Tsunoda (2017), *Flow Turb. Combust.* 99(3) — LES CFD, Cd/Cl at Re=1.1×10⁵
- Lyu, Kensrud, Smith & Tosaya (2018), *ISEA Proc.* — trajectory fit on 13 production balls
- Crabill (2019), *Sports Eng.* 22:1-9 — high-order CFD at Re=1.5×10⁵
- Bearman & Harvey (1976), *Aero. Q.* 27:112-122 — original wind tunnel study
- Trackman PGA Tour Averages (2024) — carry and descent angle validation

### Calibration process

The TOUR constants were calibrated against the 2024 Trackman PGA Tour Averages
(12 clubs, driver through PW, normalized to 25°C sea level). The primary
tuning targets are carry distance and descent angle for each club.

**Key tuning insight:** Descent angle accuracy requires high subcritical drag
(cd_sub=0.310) with a sharp drag crisis transition (re_width=125) at re_crit=75k.
At landing, irons decelerate into deeply subcritical Re (~56k) where high cd_sub
kills horizontal speed → steep 49-52° descent matching Trackman. Driver stays
near-supercritical at landing (Re≈78k) → moderate 38° descent.

The low cl_sub (0.050) is demanded by the descent angle data. At 50° descent,
Magnus force is 77% forward — even moderate subcritical lift (Cl=0.10) measurably
shallows descent for all irons. Ten experiments (grid search, 1-10) tested
structural model changes (split sigmoids, reverse Magnus, SR-dependent re_crit,
polynomial Cl(SR)) — all converge to cl_sub≈0.05 and re_width≈125. The gap vs
published wind tunnel Cl≈0.15 (1997 patent data) likely reflects differences
between 1990s and modern ball subcritical aerodynamics, and/or our 2D model
lacking spin axis precession.

### Fit quality

Validated against 2024 Trackman PGA Tour Averages (25°C sea level):

| Metric | Value |
|---|---|
| Carry RMSE | 3.8 yds |
| Carry max error | 5.5 yds (PW, -5.5y) |
| DA RMSE | 1.2° |
| DA mean error | -0.4° |

Per-club results:

| Club | Model | Target | Δ | DA° (model) | DA° (TM) | DA Δ |
|---|---:|---:|---:|---:|---:|---:|
| Driver | 277.9y | 282.0y | -4.1y | 38.0 | 39.0 | -1.0 |
| 3-Wood | 253.6y | 249.0y | +4.6y | 46.6 | 44.0 | +2.6 |
| 5-Wood | 241.5y | 236.0y | +5.5y | 48.5 | 48.0 | +0.5 |
| Hybrid | 229.2y | 231.0y | -1.8y | 48.3 | 49.0 | -0.7 |
| 3-Iron | 223.0y | 218.0y | +5.0y | 46.8 | 48.0 | -1.2 |
| 4-Iron | 213.2y | 209.0y | +4.2y | 47.2 | 49.0 | -1.8 |
| 5-Iron | 202.9y | 199.0y | +3.9y | 48.0 | 50.0 | -2.0 |
| 6-Iron | 191.5y | 188.0y | +3.5y | 50.1 | 50.0 | +0.1 |
| 7-Iron | 176.4y | 176.0y | +0.4y | 50.5 | 51.0 | -0.5 |
| 8-Iron | 165.0y | 164.0y | +1.0y | 50.9 | 51.0 | -0.1 |
| 9-Iron | 152.4y | 152.0y | +0.4y | 51.1 | 52.0 | -0.9 |
| PW | 136.5y | 142.0y | -5.5y | 52.1 | 52.0 | +0.1 |

### Known limitations

1. **PW carries ~6y cold.** Structural limit of single-sigmoid model — PW
   goes subcritical mid-flight, losing nearly all lift.
2. **Driver carries ~4y cold.** cd_super=0.235 is slightly high for driver's
   low-spin regime (SR≈0.07).
3. **Mid-irons 3-6y hot.** Our 2D "perfect straight shot" exceeds Trackman
   averages that include sidespin dispersion.
4. **Temperature/altitude sensitivity too low.** Sharp sigmoid (re_width=125)
   barely responds to density-driven Re shifts. Hot-cold delta ~4y vs
   published ~8y; Denver altitude gain ~1% vs published ~6%.
5. **Single ball model.** Real balls vary ±5% in drag across dimple designs.

### `BallModel::PATENT_1997` — 1990s Tour Ball

| Parameter | Value | Role |
|---|---|---|
| mass_kg | 0.04593 | Same physical ball (USGA limits unchanged) |
| diameter_m | 0.04267 | Same |
| cd_sub | 0.19 | Lower subcritical base |
| cd_super | 0.29 | **24% higher** supercritical base drag |
| cd_spin | 0.15 | Spin-dependent drag |
| cl_sub | 0.12 | Subcritical lift |
| cl_super | 0.285 | Similar lift ceiling |
| sr_scale | 0.01 | **Near-instant saturation** — Cl independent of SR |
| re_crit | 100,000 | Higher drag crisis point |
| re_width | 8,000 | Gradual transition |

Source: US6186002B1 (Quintavalla/Acushnet, filed 1997) — force-balance wind
tunnel measurements on production golf balls (Table 1, 7 data points). Model
fitted with RMSE: Cd=0.008, Cl=0.006.

Key differences from modern balls:
- **Base Cd ~24% higher** (0.29 vs 0.235) — dominant factor for carry gap
- **Cl saturates instantly** (sr_scale=0.01) — lift is the same for driver and
  wedge spin rates, unlike modern balls where Cl is spin-sensitive
- **Higher subcritical Cl** (0.12 vs 0.050) — more late-flight lift, gradual
  drag transition (re_width=8000 vs 125)

## Environment Model

Air density is computed from altitude and temperature using the barometric
formula for the troposphere:

```
ρ = ρ₀ × (1 - L×h/T₀)^(g×M/(R×L)) × (T_standard / T_actual)
```

Where ρ₀=1.225 kg/m³, L=0.0065 K/m, T₀=288.15 K. The temperature correction
scales by the ratio of standard to actual temperature at altitude (ideal gas
law, ρ ∝ 1/T at constant pressure).

Dynamic viscosity of air is constant at 1.81×10⁻⁵ Pa·s (adequate for 0–40°C).

## Spin Decay

Spin decays exponentially during flight:

```
ω(t) = ω₀ × exp(-λ × t)
```

Default λ = 0.04 s⁻¹ (~21% loss over 6 seconds). Published data:
- Trackman (2010): ~4%/s → λ=0.04 → 78% remaining at 6s
- Lyu et al. (2018): 4%/s trajectory fits
- Tutelman: τ=30s → λ=0.033 → 82% at 6s

## Implementation

The model lives in `cf-math/src/aero.rs`. Key types:

- **`BallModel`** — physical dimensions + aero coefficients. Presets: `TOUR`,
  `PATENT_1997`.
- **`AeroParams`** — per-shot spin state (axis, rate, decay). Created from
  backspin/sidespin RPM via `AeroParams::from_spin()`.
- **`Environment`** — altitude + temperature → air density.

Flight simulation: `simulate_flight(input, ball, env) -> FlightResult`

RK4 integration at 1ms timestep, spin decays each step, trajectory sampled
every 10ms. Stops when ball returns to ground level.

## Test Suite

170 tests across the workspace (94 in cf-math). Test categories:

- **Structural invariants** — energy conservation, origin start, axis normalization
- **Patent wind tunnel** — 7 Cd/Cl data points against PATENT_1997 model
- **Trackman carry** — 8 shots validated against 2024 PGA Tour Averages (±10-12y)
- **Descent angle** — iron DA within expected ranges
- **Altitude/temperature** — Denver gain, hot-cold delta, directional checks
- **Spin decay** — exponential decay rate against Trackman measurements
- **Era comparison** — TOUR vs PATENT_1997 carry gaps, Cd reduction, Cl sensitivity

Run: `cd cyberflight/cyberflight && cargo test --lib -p cf-math`
