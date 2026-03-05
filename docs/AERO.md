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
crisis (~Re 100,000), with spin-dependent drag and exponential lift saturation.

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
| cd_sub | 0.206 | Subcritical base drag |
| cd_super | 0.205 | Supercritical base drag |
| cd_spin | 0.19 | Spin-dependent drag increment |
| cl_sub | 0.20 | Subcritical lift ceiling |
| cl_super | 0.29 | Supercritical lift ceiling |
| sr_scale | 0.08 | Lift saturation rate |
| re_crit | 100,000 | Drag crisis Reynolds number |
| re_width | 8,000 | Sigmoid transition width |

Sources:
- Li, Tsubokura & Tsunoda (2017), *Flow Turb. Combust.* 99(3) — LES CFD, Cd/Cl at Re=1.1×10⁵
- Lyu, Kensrud, Smith & Tosaya (2018), *ISEA Proc.* — trajectory fit on 13 production balls
- Jenkins, Arellano, Ross & Snell (2018), *World J. Mech.* 8(6) — wind tunnel
- Trackman PGA Tour Averages (2024) — carry distance validation
- PING Proving Grounds, "Unlocking Distance" — low/high launch validation

### `BallModel::PATENT_1997` — 1990s Tour Ball

| Parameter | Value | Role |
|---|---|---|
| mass_kg | 0.04593 | Same physical ball (USGA limits unchanged) |
| diameter_m | 0.04267 | Same |
| cd_sub | 0.19 | Lower subcritical base |
| cd_super | 0.29 | **40% higher** supercritical base drag |
| cd_spin | 0.15 | Lower spin-dependent drag |
| cl_sub | 0.12 | Lower subcritical lift |
| cl_super | 0.285 | Similar lift ceiling |
| sr_scale | 0.01 | **Near-instant saturation** — Cl independent of SR |
| re_crit | 100,000 | Same drag crisis point |
| re_width | 8,000 | Same transition width |

Source: US6186002B1 (Quintavalla/Acushnet, filed 1997) — force-balance wind
tunnel measurements on production golf balls (Table 1, 7 data points). Model
fitted with RMSE: Cd=0.008, Cl=0.006.

Key differences from modern balls:
- **Base Cd 40% higher** (0.29 vs 0.205) — dominant factor for carry gap
- **Cl saturates instantly** (sr_scale=0.01) — lift is the same for driver and
  wedge spin rates, unlike modern balls where Cl is spin-sensitive
- **Lower subcritical Cl** (0.12 vs 0.20) — less late-flight lift for
  high-trajectory shots

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

Default λ = 0.05 s⁻¹ (~26% loss over 6 seconds). Published data:
- Trackman (2010): ~4%/s → 78% remaining at 6s
- TrajectoWare Drive: ~3.3%/s → 81% at 6s
- Our model: exp(-0.05 × 6) = 74% at 6s

## Validation Results

### Carry Distance — TOUR model vs published data

All tests at sea level, 15°C (ISA standard).

| Shot | Speed | Launch | Spin | Model | Published | Δ | Δ% | Source |
|---|---|---|---|---|---|---|---|---|
| PGA Driver | 171 mph | 10.9° | 2686 rpm | 290.1 yd | 282 yd | +8.1 | +2.9% | Trackman PGA Tour 2024 |
| PING Low | 160 mph | 8.2° | 2994 rpm | 266.6 yd | 264 yd | +2.6 | +1.0% | PING Proving Grounds |
| PING High | 160 mph | 15.1° | 2179 rpm | 272.7 yd | 281 yd | -8.3 | -3.0% | PING Proving Grounds |
| PGA 7-iron | 123 mph | 16.3° | 7097 rpm | 183.8 yd | 176 yd | +7.8 | +4.4% | Trackman PGA Tour |
| PGA PW | 102 mph | 25.0° | 9300 rpm | 137.4 yd | 142 yd | -4.6 | -3.2% | Trackman PGA Tour |

All within ±5% of published measurements. Tolerance: ±10 yards.

### Environment validation

| Metric | Model | Published | Source |
|---|---|---|---|
| Denver altitude gain (5,280 ft) | 7.8% | ~6.1% | Titleist Learning Lab |
| Temperature: 90°F vs 50°F | 8.7 yd | ~8 yd | WeatherWorks |
| Spin remaining at 6s | 74.1% | ~78% | Trackman / tutelman.com |

### Patent wind tunnel validation — PATENT_1997 model

All 7 data points from US6186002B1 Table 1 pass within ±0.02 for both Cd and
Cl (except the V=250/spin=23 outlier at ±0.025 for Cd — the only point where
lower spin produces higher drag, likely measurement noise).

### Era comparison — same launch conditions, different ball technology

| Shot | Tour (2020s) | Patent (1990s) | Gap | Gap % |
|---|---|---|---|---|
| PGA Driver | 290.1 yd | 247.7 yd | +42.4 yd | +17.1% |
| PING Low | 266.6 yd | 228.9 yd | +37.7 yd | +16.5% |
| PING High | 272.7 yd | 236.1 yd | +36.6 yd | +15.5% |
| PGA 7-iron | 183.8 yd | 164.2 yd | +19.6 yd | +12.0% |
| PGA PW | 137.4 yd | 130.7 yd | +6.7 yd | +5.1% |

The era gap scales with ball speed: driver gains ~42 yards from ball tech,
wedge gains ~7. This is consistent with the USGA Distance Insights report
(2020), which attributes roughly 30 yards of driver carry increase since 1995
to ball technology.

The gap comes almost entirely from the ~30% reduction in base drag coefficient.
The lift model difference (spin-sensitive vs saturated) is secondary but
explains why the modern "high launch, low spin" driver fitting strategy works —
it only makes aerodynamic sense with modern balls that have low Cl at low SR.

## Implementation

The model lives in `cf-math/src/aero.rs` (cyberflight repo). Key types:

- **`BallModel`** — physical dimensions + aero coefficients. Presets: `TOUR`,
  `PATENT_1997`.
- **`AeroParams`** — per-shot spin state (axis, rate, decay). Created from
  backspin/sidespin RPM via `AeroParams::from_spin()`.
- **`Environment`** — altitude + temperature → air density.

Flight simulation: `simulate_flight(input, ball, env) -> FlightResult`

RK4 integration at 1ms timestep, spin decays each step, trajectory sampled
every 10ms. Stops when ball returns to ground level.

## Test Suite

112 tests total across the workspace (59 in cf-math). Test categories:

- **Structural invariants** — energy conservation, origin start, axis normalization
- **Patent wind tunnel** — 7 Cd/Cl data points against PATENT_1997 model
- **Modern carry** — 5 shots validated against Trackman/PING published data
- **Altitude/temperature** — Denver gain, hot-cold delta, directional checks
- **Spin decay** — exponential decay rate against Trackman measurements
- **Era comparison** — TOUR vs PATENT_1997 carry gaps, directional trends, Cl
  spin-sensitivity difference

Run: `cd cyberflight/cyberflight && cargo test --lib -p cf-math`
