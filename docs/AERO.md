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
crisis (~Re 70,000), with saturating spin-dependent drag and exponential lift
saturation.

### Drag coefficient

```
Cd = Cd_base(Re) + Cd_spin × SR / (1 + SR / SR_sat)

where:
  Cd_base = Cd_sub + (Cd_super - Cd_sub) × σ(Re)
  σ(Re)   = 1 / (1 + exp(-(Re - Re_crit) / Re_width))    [sigmoid]
  SR      = ω × R / V                                      [spin ratio]
  Re      = ρ × V × d / μ                                  [Reynolds number]
```

The base drag transitions from subcritical to supercritical at the drag crisis.
Spin adds drag through a saturating term: near-linear (slope Cd_spin) below
SR ≈ SR_sat, capped at Cd_spin × SR_sat at high spin ratio. Spin drag is
dominated by induced drag from Magnus lift, and since Cl saturates in SR, the
drag it induces saturates too. (Set SR_sat = ∞ for a purely linear term, as in
the PATENT_1997 model.)

### Lift coefficient

```
Cl = Cl_max(Re) × (1 - exp(-SR / SR_scale))

where:
  Cl_max = Cl_sub + (Cl_super - Cl_sub) × σ(Re)
```

Lift follows an exponential saturation curve in spin ratio. This means:
- **Low SR (driver, ~0.07):** Cl is well below the ceiling (~0.15)
- **High SR (iron, ~0.29):** Cl approaches the ceiling (~0.30)
- **Very high SR (wedge, ~0.45):** Cl is fully saturated (~0.32)

This spin-sensitive lift is a key feature of modern ball design — it rewards
the "high launch, low spin" fitting strategy that dominates modern equipment.

## Ball Models

### `BallModel::TOUR` — Modern Tour Ball (2020s)

| Parameter | Value | Role |
|---|---|---|
| mass_kg | 0.04593 | Ball mass (USGA max: 1.62 oz) |
| diameter_m | 0.04267 | Ball diameter (USGA min: 1.68 in) |
| cd_sub | 0.295 | Subcritical base drag |
| cd_super | 0.186 | Supercritical base drag (zero spin) |
| cd_spin | 0.85 | Spin-drag slope (saturating) |
| sr_sat | 0.25 | Spin-drag saturation constant (cap ≈ 0.21) |
| cl_sub | 0.203 | Subcritical lift ceiling |
| cl_super | 0.324 | Supercritical lift ceiling |
| sr_scale | 0.115 | Lift saturation rate |
| re_crit | 69,600 | Drag crisis Reynolds number |
| re_width | 106 | Sigmoid transition width |

Aerodynamic sources:
- Li, Tsubokura & Tsunoda (2017), *Flow Turb. Combust.* 99(3) — LES CFD, Cd/Cl at Re=1.1×10⁵
- Lyu, Kensrud, Smith & Tosaya (2018), *ISEA Proc.* — trajectory fit on 13 production balls
- Crabill (2019), *Sports Eng.* 22:1-9 — high-order CFD at Re=1.5×10⁵
- Bearman & Harvey (1976), *Aero. Q.* 27:112-122 — original wind tunnel study
- Trackman PGA Tour Averages (2024) — carry and descent angle validation

### Calibration

The TOUR constants are jointly calibrated against two references:

1. **2024 Trackman PGA Tour Averages** — 12 clubs, carry + descent angle,
   normalized to 25°C sea level. Real-world radar data; authoritative for
   descent angles.
2. **A 1,178-shot per-shot validation set** spanning 26–153 mph ball speed
   and 0.8k–14k rpm spin, fitting carry, peak height, descent angle, and
   offline for every shot — providing breadth (amateur speeds, sidespin and
   offline behavior) that tour averages can't.

The saturating spin-drag term is what lets both references fit at once:
steep iron descent angles (47–52°) come from heavy late-flight induced drag
on high-spin balls, while subcritical lift stays at a wind-tunnel-plausible
cl_sub=0.203 — which in turn produces correct sidespin curvature and
altitude response.

### Fit quality

Trackman 2024 (25°C sea level): carry RMSE **2.8 yds**, DA RMSE **1.3°**,
10 of 12 clubs within ±3y.
Per-shot validation set (1,178 shots): carry RMSE **3.2 yds**, peak
**2.6 ft**, descent **2.7°**, offline **1.5 yds**.

Per-club results:

| Club | Model | Target | Δ | DA° (model) | DA° (TM) | DA Δ |
|---|---:|---:|---:|---:|---:|---:|
| Driver | 279.6y | 282.0y | -2.4y | 39.7 | 39.0 | +0.7 |
| 3-Wood | 253.4y | 249.0y | +4.4y | 46.6 | 44.0 | +2.6 |
| 5-Wood | 238.7y | 236.0y | +2.7y | 48.6 | 48.0 | +0.6 |
| Hybrid | 226.0y | 231.0y | -5.0y | 48.3 | 49.0 | -0.7 |
| 3-Iron | 220.6y | 218.0y | +2.6y | 46.7 | 48.0 | -1.3 |
| 4-Iron | 210.3y | 209.0y | +1.3y | 46.9 | 49.0 | -2.1 |
| 5-Iron | 199.8y | 199.0y | +0.8y | 47.8 | 50.0 | -2.2 |
| 6-Iron | 188.1y | 188.0y | +0.1y | 49.9 | 50.0 | -0.1 |
| 7-Iron | 173.9y | 176.0y | -2.1y | 50.5 | 51.0 | -0.5 |
| 8-Iron | 163.5y | 164.0y | -0.5y | 50.9 | 51.0 | -0.1 |
| 9-Iron | 152.1y | 152.0y | +0.1y | 51.2 | 52.0 | -0.8 |
| PW | 137.3y | 142.0y | -4.7y | 52.2 | 52.0 | +0.2 |

Environmental response (density-only): Denver +6.9% driver carry (published
~6.1%), 90°F-vs-50°F delta ~9y (published ~8y).

### Known limitations

1. **PW carries ~5y cold, hybrid ~5y cold.** Residual structural limit of the
   single-sigmoid model.
2. **Fast low-spin shots (140+ mph, <3k rpm) run a few yards short** with
   slightly high peaks relative to the per-shot validation set.
3. **Ball COR vs temperature unmodeled** — only air density responds to
   temperature; cold-ball distance loss is understated.
4. **Single ball model.** Real balls vary ±5% in drag across dimple designs.

### `BallModel::PATENT_1997` — 1990s Tour Ball

| Parameter | Value | Role |
|---|---|---|
| mass_kg | 0.04593 | Same physical ball (USGA limits unchanged) |
| diameter_m | 0.04267 | Same |
| cd_sub | 0.19 | Lower subcritical base |
| cd_super | 0.29 | Much higher supercritical base drag |
| cd_spin | 0.15 | Spin-dependent drag |
| sr_sat | ∞ | Linear spin drag (no saturation in patent data) |
| cl_sub | 0.12 | Subcritical lift |
| cl_super | 0.285 | Similar lift ceiling |
| sr_scale | 0.01 | **Near-instant saturation** — Cl independent of SR |
| re_crit | 100,000 | Higher drag crisis point |
| re_width | 8,000 | Gradual transition |

Source: US6186002B1 (Quintavalla/Acushnet, filed 1997) — force-balance wind
tunnel measurements on production golf balls (Table 1, 7 data points). Model
fitted with RMSE: Cd=0.008, Cl=0.006.

Key differences from modern balls:
- **Base Cd much higher** (0.29 vs 0.186 + spin term; effective driver Cd
  0.30 vs 0.24) — dominant factor for carry gap
- **Cl saturates instantly** (sr_scale=0.01) — lift is the same for driver and
  wedge spin rates, unlike modern balls where Cl is spin-sensitive
- **Lower subcritical Cl** (0.12 vs 0.203) and a gradual drag crisis
  (re_width=8000 vs 106)

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

171 tests across the workspace (95 in cf-math). Test categories:

- **Structural invariants** — energy conservation, origin start, axis normalization
- **Patent wind tunnel** — 7 Cd/Cl data points against PATENT_1997 model
- **Trackman carry** — 8 shots validated against 2024 PGA Tour Averages (±10-12y)
- **Descent angle** — iron DA within expected ranges
- **Altitude/temperature** — Denver gain, hot-cold delta, directional checks
- **Spin decay** — exponential decay rate against Trackman measurements
- **Era comparison** — TOUR vs PATENT_1997 carry gaps, Cd reduction, Cl sensitivity

Run: `cd cyberflight/cyberflight && cargo test --lib -p cf-math`
