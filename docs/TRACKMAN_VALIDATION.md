# Trackman Validation Data

Aerodynamic model validation against published Trackman PGA Tour Averages.

## Data Sources

### 2024 Trackman PGA Tour Averages

Source: [Trackman — Introducing Updated Tour Averages](https://www.trackman.com/blog/golf/introducing-updated-tour-averages) (March 2025).
Collected from 40+ PGA Tour and DP World Tour events, 200+ players.

**Normalization: 77°F (25°C), sea level, no wind.**

| Club | Club Spd (mph) | Attack Angle (°) | Ball Spd (mph) | Smash Factor | Launch (°) | Spin (rpm) | Apex (yds) | Descent (°) | Carry (yds) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Driver | 115 | -0.9 | 171 | 1.49 | 10.4 | 2545 | 35 | 39 | 282 |
| 3-Wood | 110 | -2.3 | 162 | 1.47 | 9.3 | 3663 | 32 | 44 | 249 |
| 5-Wood | 106 | -2.5 | 156 | 1.47 | 9.7 | 4322 | 33 | 48 | 236 |
| Hybrid | 102 | -2.4 | 149 | 1.47 | 10.2 | 4587 | 31 | 49 | 231 |
| 3-Iron | 100 | -2.5 | 145 | 1.46 | 10.3 | 4404 | 30 | 48 | 218 |
| 4-Iron | 98 | -2.9 | 140 | 1.44 | 10.8 | 4782 | 31 | 49 | 209 |
| 5-Iron | 96 | -3.4 | 135 | 1.41 | 11.9 | 5280 | 33 | 50 | 199 |
| 6-Iron | 94 | -3.7 | 130 | 1.39 | 14.0 | 6204 | 32 | 50 | 188 |
| 7-Iron | 92 | -3.9 | 123 | 1.34 | 16.1 | 7124 | 34 | 51 | 176 |
| 8-Iron | 89 | -4.2 | 118 | 1.33 | 17.8 | 8078 | 33 | 51 | 164 |
| 9-Iron | 87 | -4.3 | 112 | 1.29 | 20.0 | 8793 | 32 | 52 | 152 |
| PW | 84 | -4.7 | 104 | 1.24 | 23.7 | 9316 | 32 | 52 | 142 |

### Classic (2009-2014) Trackman PGA Tour Averages

Source: Trackman PGA Tour averages, widely republished (3Jack Golf Blog, GlobalGolf, GolfMagic). No descent angle data available.

| Club | Club Spd (mph) | Attack Angle (°) | Ball Spd (mph) | Launch (°) | Spin (rpm) | Apex (yds) | Carry (yds) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Driver | 112 | -1.3 | 165 | 11.2 | 2685 | 31 | 269 |
| 3-Wood | 107 | -2.9 | 158 | 9.2 | 3655 | 30 | 243 |
| 5-Wood | 103 | -3.3 | 152 | 9.4 | 4350 | 31 | 230 |
| Hybrid | 100 | -3.3 | 146 | 10.2 | 4437 | 29 | 225 |
| 3-Iron | 98 | -3.1 | 142 | 10.4 | 4630 | 27 | 212 |
| 4-Iron | 96 | -3.4 | 137 | 11.0 | 4836 | 28 | 203 |
| 5-Iron | 94 | -3.7 | 132 | 12.1 | 5361 | 31 | 194 |
| 6-Iron | 92 | -4.1 | 127 | 14.1 | 6231 | 30 | 183 |
| 7-Iron | 90 | -4.3 | 120 | 16.3 | 7097 | 32 | 172 |
| 8-Iron | 87 | -4.5 | 115 | 18.1 | 7998 | 31 | 160 |
| 9-Iron | 85 | -4.7 | 109 | 20.4 | 8647 | 30 | 148 |
| PW | 83 | -5.0 | 102 | 24.2 | 9304 | 29 | 136 |

### Key differences (classic → 2024)

| Metric | Classic | 2024 | Change |
|---|---:|---:|---|
| Driver ball speed | 165 mph | 171 mph | +6 |
| Driver spin | 2685 rpm | 2545 rpm | -140 |
| Driver carry | 269 yds | 282 yds | +13 |
| 7-Iron ball speed | 120 mph | 123 mph | +3 |
| 7-Iron carry | 172 yds | 176 yds | +4 |
| PW ball speed | 102 mph | 104 mph | +2 |
| PW carry | 136 yds | 142 yds | +6 |

## Aerodynamic References

Primary sources for the `BallModel::TOUR` drag and lift coefficients:

| Source | Method | Key findings |
|---|---|---|
| Li, Tsubokura & Tsunoda (2017), *Flow Turb. Combust.* 99(3) | LES CFD, Re=1.1×10⁵ | Cd=0.217 (static), Cd=0.240 (G=0.1), Cl=0.135 |
| Lyu, Kensrud, Smith & Tosaya (2018), *ISEA Proc.* 2(6):238 | Trajectory fit, 13 balls | Cd~0.20 avg at high Re, Cl 2nd-order polynomial in SR |
| Crabill (2019), *Sports Eng.* 22:1-9 | High-order CFD, Re=1.5×10⁵ | Cd=0.247 (static), Cd=0.256 (G=0.15), Cl=0.164 |
| Bearman & Harvey (1976), *Aero. Q.* 27:112-122 | Wind tunnel | Foundational Cd/Cl vs Re and spin for dimpled balls |
| US6186002B1 (Quintavalla/Acushnet 1997) | Wind tunnel, 7 conditions | Cd=0.22-0.31, Cl=0.15-0.28 (1990s-era balls) |
| Lyu, Kensrud & Smith (2020), *Sports Eng.* 23 | Trajectory fit | Reverse Magnus at 5×10⁴ < Re < 7×10⁴ |
| Alam et al. (2011), *Procedia Eng.* 13:226-231 | Wind tunnel | Cd min=0.195, varies up to 40% across dimple designs |

### Spin decay

| Source | Rate | λ (s⁻¹) | At 6s |
|---|---|---:|---|
| Trackman (Oct 2010 newsletter) | ~4%/s | 0.04 | ~78% remaining |
| Lyu et al. (2018) trajectory fits | 4%/s | 0.04 | ~78% remaining |
| Tutelman (tutelman.com) | τ=30s | 0.033 | ~82% remaining |
| James & Haake (2008), *Eng. of Sport 7* | — | — | Golf ~6.5× less decay than tennis |

## Model Fit Results

Grid search over `cd_super × cl_sub` at 25°C sea level (Trackman normalization).

### Current model: `cd_super=0.22, cl_sub=0.16, spin_decay=0.04`

| Dataset | RMSE |
|---|---:|
| 2024 Trackman (12 clubs) | 4.6 yds |
| Classic Trackman (12 clubs) | 3.3 yds |

### Grid search optimum: `cd_super=0.232, cl_sub=0.158`

| Dataset | RMSE |
|---|---:|
| 2024 Trackman (12 clubs) | 4.5 yds |
| Classic Trackman (12 clubs) | 3.2 yds |
| Combined | 3.9 yds |

### Per-club breakdown (grid search optimum, 25°C sea level)

#### 2024 Trackman

| Club | Model (yds) | Target (yds) | Delta | DA° (model) | DA° (TM) | DA Delta |
|---|---:|---:|---:|---:|---:|---:|
| Driver | 270.7 | 282.0 | -11.3 | 42.5 | 39.0 | +3.5 |
| 3-Wood | 253.9 | 249.0 | +4.9 | 43.3 | 44.0 | -0.7 |
| 5-Wood | 241.8 | 236.0 | +5.8 | 43.7 | 48.0 | -4.3 |
| Hybrid | 228.6 | 231.0 | -2.4 | 42.5 | 49.0 | -6.5 |
| 3-Iron | 221.6 | 218.0 | +3.6 | 40.8 | 48.0 | -7.2 |
| 4-Iron | 211.2 | 209.0 | +2.2 | 40.3 | 49.0 | -8.7 |
| 5-Iron | 200.9 | 199.0 | +1.9 | 40.5 | 50.0 | -9.5 |
| 6-Iron | 190.0 | 188.0 | +2.0 | 42.2 | 50.0 | -7.8 |
| 7-Iron | 174.9 | 176.0 | -1.1 | 42.6 | 51.0 | -8.4 |
| 8-Iron | 163.3 | 164.0 | -0.7 | 43.2 | 51.0 | -7.8 |
| 9-Iron | 151.2 | 152.0 | -0.8 | 43.7 | 52.0 | -8.3 |
| PW | 136.7 | 142.0 | -5.3 | 45.2 | 52.0 | -6.8 |

#### Classic Trackman

| Club | Model (yds) | Target (yds) | Delta |
|---|---:|---:|---:|
| Driver | 261.9 | 269.0 | -7.1 |
| 3-Wood | 246.5 | 243.0 | +3.5 |
| 5-Wood | 234.0 | 230.0 | +4.0 |
| Hybrid | 223.3 | 225.0 | -1.7 |
| 3-Iron | 215.1 | 212.0 | +3.1 |
| 4-Iron | 205.1 | 203.0 | +2.1 |
| 5-Iron | 194.7 | 194.0 | +0.7 |
| 6-Iron | 184.0 | 183.0 | +1.0 |
| 7-Iron | 169.1 | 172.0 | -2.9 |
| 8-Iron | 157.9 | 160.0 | -2.1 |
| 9-Iron | 146.1 | 148.0 | -1.9 |
| PW | 133.3 | 136.0 | -2.7 |

### Known limitations

1. **Descent angles are 7-9° too shallow for irons.** Trackman shows 48-52° for 3i-PW; our model produces 40-45°. The gap is worst for 5-iron (-9.5°) and smallest for 6-iron (-7.8°). This indicates insufficient late-flight lift (Magnus effect decays too fast or subcritical Cl is too low). Increasing `cl_sub` steepens descent but also increases carry — these two targets conflict with a fixed Cd/Cl model. Fixing this properly would require spin-ratio-dependent Cl (polynomial fit like Lyu et al. 2018) or Re-dependent Cl that evolves through the flight.

2. **Driver descent angle 3.5° too steep.** The opposite direction from irons. The driver's low spin ratio (SR≈0.07) means it stays supercritical throughout flight, and our model applies slightly too much lift relative to what Trackman measures. This divergence in opposite directions between driver and irons confirms a single Cl model cannot perfectly capture both regimes.

3. **2024 driver carries 11 yards cold.** The 2024 driver (171 mph, 10.4°, 2545 rpm, 282 yds) represents optimized modern launch conditions. Our model undershoots because the drag that correctly matches irons is too high for the driver's low spin ratio (SR=0.07). A single `cd_super` cannot perfectly fit both.

4. **2024 PW carries 5 yards cold.** The PW (104 mph, 23.7°, 9316 rpm) decelerates into subcritical Re early, where `cl_sub` dominates. Increasing `cl_sub` helps PW carry but worsens the descent angle gap.

5. **Single ball model.** Trackman calibrates per-ball aerodynamic profiles. Our model uses fixed Cd/Cl constants for all modern balls. Real balls vary by ±5% in drag across dimple designs (Alam et al. 2011).
