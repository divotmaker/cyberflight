/// Aerodynamic model experiments — focused on driver/iron DA balance.
///
/// Previous experiments found cd_sub=0.32 fixes iron descent angles but
/// makes driver 7.5° too steep. This experiment searches for the sweet spot
/// where re_crit/re_width keep driver supercritical while irons go subcritical.
use glam::DVec3;

const G: f64 = 9.80665;
const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
struct ModelParams {
    mass_kg: f64, diameter_m: f64,
    cd_sub: f64, cd_super: f64, cd_spin: f64,
    cl_sub: f64, cl_super: f64, sr_scale: f64,
    re_crit: f64, re_width: f64,
    spin_decay: f64,
}

impl ModelParams {
    fn radius(&self) -> f64 { self.diameter_m / 2.0 }
    fn area(&self) -> f64 { PI * self.radius() * self.radius() }

    fn baseline() -> Self {
        Self {
            mass_kg: 0.04593, diameter_m: 0.04267,
            cd_sub: 0.206, cd_super: 0.22, cd_spin: 0.19,
            cl_sub: 0.16, cl_super: 0.29, sr_scale: 0.08,
            re_crit: 100_000.0, re_width: 8_000.0,
            spin_decay: 0.04,
        }
    }

    fn cd_cl(&self, spin_rate: f64, speed: f64, rho: f64) -> (f64, f64) {
        const MU: f64 = 1.81e-5;
        if speed < 1e-10 { return (self.cd_sub, 0.0); }
        let re = rho * speed * self.diameter_m / MU;
        let sr = (spin_rate * self.radius() / speed).abs();
        let sigma = 1.0 / (1.0 + (-(re - self.re_crit) / self.re_width).exp());
        let cd_base = self.cd_sub + (self.cd_super - self.cd_sub) * sigma;
        let cd = cd_base + self.cd_spin * sr;
        let cl_max = self.cl_sub + (self.cl_super - self.cl_sub) * sigma;
        let cl = cl_max * (1.0 - (-sr / self.sr_scale).exp());
        (cd, cl)
    }

    fn acceleration(&self, vel: DVec3, spin_axis: DVec3, spin_rate: f64, rho: f64) -> DVec3 {
        let speed = vel.length();
        if speed < 1e-10 { return DVec3::new(0.0, -G, 0.0); }
        let vel_hat = vel / speed;
        let (cd, cl) = self.cd_cl(spin_rate, speed, rho);
        let q = 0.5 * rho * speed * speed;
        let drag = vel_hat * (-cd * q * self.area() / self.mass_kg);
        let magnus = if spin_rate.abs() > 1e-10 {
            let lift_dir = spin_axis.cross(vel_hat);
            let lift_mag = lift_dir.length();
            if lift_mag > 1e-10 { (lift_dir / lift_mag) * (cl * q * self.area() / self.mass_kg) }
            else { DVec3::ZERO }
        } else { DVec3::ZERO };
        DVec3::new(0.0, -G, 0.0) + drag + magnus
    }
}

struct SimResult { carry_yards: f64, descent_angle: f64, apex_yards: f64, flight_time: f64 }

fn simulate(params: &ModelParams, speed_mph: f64, launch_deg: f64, spin_rpm: f64, rho: f64) -> SimResult {
    let speed_mps = speed_mph * 0.44704;
    let launch_rad = launch_deg * PI / 180.0;
    let vx = speed_mps * launch_rad.cos();
    let vy = speed_mps * launch_rad.sin();
    let spin_axis = DVec3::Z;
    let mut spin_rate = spin_rpm * PI / 30.0;
    let dt = 0.001;
    let mut pos = DVec3::ZERO;
    let mut vel = DVec3::new(vx, vy, 0.0);
    let mut t = 0.0;
    let mut apex = 0.0_f64;
    let mut prev3 = DVec3::ZERO;
    let mut prev2 = DVec3::ZERO;
    let mut prev1 = DVec3::ZERO;
    let mut sc = 0u32;

    for _ in 1..=30_000 {
        let k1v = params.acceleration(vel, spin_axis, spin_rate, rho);
        let k1x = vel;
        let k2v = params.acceleration(vel + k1v*(dt/2.0), spin_axis, spin_rate, rho);
        let k2x = vel + k1v*(dt/2.0);
        let k3v = params.acceleration(vel + k2v*(dt/2.0), spin_axis, spin_rate, rho);
        let k3x = vel + k2v*(dt/2.0);
        let k4v = params.acceleration(vel + k3v*dt, spin_axis, spin_rate, rho);
        let k4x = vel + k3v*dt;
        pos = pos + (k1x + 2.0*k2x + 2.0*k3x + k4x)*(dt/6.0);
        vel = vel + (k1v + 2.0*k2v + 2.0*k3v + k4v)*(dt/6.0);
        t += dt;
        spin_rate *= (-params.spin_decay * dt).exp();
        apex = apex.max(pos.y);
        sc += 1;
        if sc % 10 == 0 { prev3 = prev2; prev2 = prev1; prev1 = pos; }
        if t > 0.1 && pos.y <= 0.0 { break; }
    }

    let dx = prev2.x - prev3.x;
    let dy = prev2.y - prev3.y;
    let h = (dx*dx + (prev2.z - prev3.z).powi(2)).sqrt();
    let da = if h > 1e-10 { (-dy/h).atan()*180.0/PI } else { 0.0 };
    let carry_m = (pos.x*pos.x + pos.z*pos.z).sqrt();

    SimResult { carry_yards: carry_m*1.09361, descent_angle: da, apex_yards: apex*1.09361, flight_time: t }
}

struct Target { label: &'static str, speed: f64, launch: f64, spin: f64, carry: f64, da: f64 }

fn modern_targets() -> Vec<Target> {
    vec![
        Target { label: "Driver",  speed: 171.0, launch: 10.4, spin: 2545.0, carry: 282.0, da: 39.0 },
        Target { label: "3-Wood",  speed: 162.0, launch:  9.3, spin: 3663.0, carry: 249.0, da: 44.0 },
        Target { label: "5-Wood",  speed: 156.0, launch:  9.7, spin: 4322.0, carry: 236.0, da: 48.0 },
        Target { label: "Hybrid",  speed: 149.0, launch: 10.2, spin: 4587.0, carry: 231.0, da: 49.0 },
        Target { label: "3-iron",  speed: 145.0, launch: 10.3, spin: 4404.0, carry: 218.0, da: 48.0 },
        Target { label: "4-iron",  speed: 140.0, launch: 10.8, spin: 4782.0, carry: 209.0, da: 49.0 },
        Target { label: "5-iron",  speed: 135.0, launch: 11.9, spin: 5280.0, carry: 199.0, da: 50.0 },
        Target { label: "6-iron",  speed: 130.0, launch: 14.0, spin: 6204.0, carry: 188.0, da: 50.0 },
        Target { label: "7-iron",  speed: 123.0, launch: 16.1, spin: 7124.0, carry: 176.0, da: 51.0 },
        Target { label: "8-iron",  speed: 118.0, launch: 17.8, spin: 8078.0, carry: 164.0, da: 51.0 },
        Target { label: "9-iron",  speed: 112.0, launch: 20.0, spin: 8793.0, carry: 152.0, da: 52.0 },
        Target { label: "PW",      speed: 104.0, launch: 23.7, spin: 9316.0, carry: 142.0, da: 52.0 },
    ]
}

fn classic_targets() -> Vec<Target> {
    vec![
        Target { label: "Driver",  speed: 165.0, launch: 11.2, spin: 2685.0, carry: 269.0, da: 0.0 },
        Target { label: "3-Wood",  speed: 158.0, launch:  9.2, spin: 3655.0, carry: 243.0, da: 0.0 },
        Target { label: "5-Wood",  speed: 152.0, launch:  9.4, spin: 4350.0, carry: 230.0, da: 0.0 },
        Target { label: "Hybrid",  speed: 146.0, launch: 10.2, spin: 4437.0, carry: 225.0, da: 0.0 },
        Target { label: "3-iron",  speed: 142.0, launch: 10.4, spin: 4630.0, carry: 212.0, da: 0.0 },
        Target { label: "4-iron",  speed: 137.0, launch: 11.0, spin: 4836.0, carry: 203.0, da: 0.0 },
        Target { label: "5-iron",  speed: 132.0, launch: 12.1, spin: 5361.0, carry: 194.0, da: 0.0 },
        Target { label: "6-iron",  speed: 127.0, launch: 14.1, spin: 6231.0, carry: 183.0, da: 0.0 },
        Target { label: "7-iron",  speed: 120.0, launch: 16.3, spin: 7097.0, carry: 172.0, da: 0.0 },
        Target { label: "8-iron",  speed: 115.0, launch: 18.1, spin: 7998.0, carry: 160.0, da: 0.0 },
        Target { label: "9-iron",  speed: 109.0, launch: 20.4, spin: 8647.0, carry: 148.0, da: 0.0 },
        Target { label: "PW",      speed: 102.0, launch: 24.2, spin: 9304.0, carry: 136.0, da: 0.0 },
    ]
}

struct ClubResult { carry_delta: f64, da_delta: f64 }

fn evaluate_per_club(params: &ModelParams, targets: &[Target], rho: f64) -> Vec<ClubResult> {
    targets.iter().map(|t| {
        let r = simulate(params, t.speed, t.launch, t.spin, rho);
        ClubResult {
            carry_delta: r.carry_yards - t.carry,
            da_delta: if t.da > 0.0 { r.descent_angle - t.da } else { 0.0 },
        }
    }).collect()
}

/// Balanced score: penalizes carry RMSE + DA RMSE + worst-club DA outlier
fn balanced_score(params: &ModelParams, modern: &[Target], classic: &[Target], rho: f64) -> f64 {
    let mc = evaluate_per_club(params, modern, rho);
    let cc = evaluate_per_club(params, classic, rho);

    let carry_m: f64 = (mc.iter().map(|r| r.carry_delta.powi(2)).sum::<f64>() / mc.len() as f64).sqrt();
    let carry_c: f64 = (cc.iter().map(|r| r.carry_delta.powi(2)).sum::<f64>() / cc.len() as f64).sqrt();
    let carry_combined = ((carry_m.powi(2) + carry_c.powi(2)) / 2.0).sqrt();

    let da_vals: Vec<f64> = mc.iter().zip(modern).filter(|(_, t)| t.da > 0.0).map(|(r, _)| r.da_delta).collect();
    let da_rmse = (da_vals.iter().map(|d| d*d).sum::<f64>() / da_vals.len() as f64).sqrt();
    let da_max = da_vals.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);

    // Balanced: carry + DA RMSE + penalty for worst single club DA
    carry_combined + 0.4 * da_rmse + 0.15 * da_max
}

fn print_full_bag(params: &ModelParams, targets: &[Target], rho: f64) {
    println!("  {:<12} {:>7} {:>7} {:>7}  {:>6} {:>6} {:>7}  {:>5} {:>5}",
        "Club", "Model", "Target", "Delta", "DA°", "TM DA°", "DA Δ", "Apex", "Time");
    println!("  {}", "-".repeat(76));
    for t in targets {
        let r = simulate(params, t.speed, t.launch, t.spin, rho);
        let da_d = if t.da > 0.0 { format!("{:>+6.1}", r.descent_angle - t.da) } else { "   n/a".into() };
        let tm = if t.da > 0.0 { format!("{:>5.0}", t.da) } else { "  n/a".into() };
        println!("  {:<12} {:>6.1}y {:>6.1}y {:>+6.1}y  {:>5.1} {} {}  {:>4.0}y {:>4.1}s",
            t.label, r.carry_yards, t.carry, r.carry_yards - t.carry,
            r.descent_angle, tm, da_d, r.apex_yards, r.flight_time);
    }
}

fn print_summary(label: &str, params: &ModelParams, modern: &[Target], classic: &[Target], rho: f64) {
    let mc = evaluate_per_club(params, modern, rho);
    let cc = evaluate_per_club(params, classic, rho);
    let carry_m = (mc.iter().map(|r| r.carry_delta.powi(2)).sum::<f64>() / mc.len() as f64).sqrt();
    let carry_c = (cc.iter().map(|r| r.carry_delta.powi(2)).sum::<f64>() / cc.len() as f64).sqrt();
    let da_vals: Vec<f64> = mc.iter().zip(modern).filter(|(_, t)| t.da > 0.0).map(|(r, _)| r.da_delta).collect();
    let da_rmse = (da_vals.iter().map(|d| d*d).sum::<f64>() / da_vals.len() as f64).sqrt();
    let da_mean = da_vals.iter().sum::<f64>() / da_vals.len() as f64;
    let da_max = da_vals.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    let s = balanced_score(params, modern, classic, rho);
    println!("  {label}:");
    println!("    2024 carry={:.1}y  classic carry={:.1}y", carry_m, carry_c);
    println!("    DA RMSE={:.1}°  DA mean={:+.1}°  DA max={:.1}°  score={:.2}", da_rmse, da_mean, da_max, s);
}

fn main() {
    let rho = 1.225 * (288.15 / (25.0 + 273.15));
    let modern = modern_targets();
    let classic = classic_targets();
    let baseline = ModelParams::baseline();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  BASELINE");
    println!("═══════════════════════════════════════════════════════════════\n");
    print_summary("Baseline", &baseline, &modern, &classic, rho);
    println!();
    print_full_bag(&baseline, &modern, rho);

    // Previous best (too steep driver)
    let prev_best = ModelParams {
        cd_sub: 0.32, cd_super: 0.23, cl_sub: 0.21,
        re_crit: 80_000.0, re_width: 4_000.0, ..baseline
    };
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  PREVIOUS BEST (driver too steep)");
    println!("═══════════════════════════════════════════════════════════════\n");
    print_summary("Prev best", &prev_best, &modern, &classic, rho);

    // ══════════════════════════════════════════════════════════════
    //  EXPERIMENT: 5D grid with balanced scoring
    //  cd_sub × cd_super × cl_sub × re_crit × re_width
    //  Focus: re_crit 85-120k to keep driver supercritical
    // ══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  5D GRID: cd_sub × cd_super × cl_sub × re_crit × re_width");
    println!("  Balanced scoring (carry + DA RMSE + DA outlier penalty)");
    println!("═══════════════════════════════════════════════════════════════\n");

    struct R { cd_sub: f64, cd_super: f64, cl_sub: f64, re_crit: f64, re_width: f64, score: f64 }

    let mut results: Vec<R> = Vec::new();
    let cd_subs = [0.24, 0.27, 0.30, 0.33, 0.36, 0.40, 0.45, 0.50];
    let cd_supers = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25];
    let cl_subs = [0.12, 0.15, 0.18, 0.21, 0.24, 0.27];
    let re_crits = [80_000.0, 85_000.0, 90_000.0, 95_000.0, 100_000.0, 110_000.0, 120_000.0];
    let re_widths = [3_000.0, 5_000.0, 8_000.0, 12_000.0, 16_000.0];

    let total = cd_subs.len() * cd_supers.len() * cl_subs.len() * re_crits.len() * re_widths.len();
    println!("  Scanning {} combinations...", total);

    for &cd_sub in &cd_subs {
        for &cd_super in &cd_supers {
            if cd_sub < cd_super { continue; } // subcritical should have more drag
            for &cl_sub in &cl_subs {
                for &re_crit in &re_crits {
                    for &re_width in &re_widths {
                        let p = ModelParams { cd_sub, cd_super, cl_sub, re_crit, re_width, ..baseline };
                        let s = balanced_score(&p, &modern, &classic, rho);
                        results.push(R { cd_sub, cd_super, cl_sub, re_crit, re_width, score: s });
                    }
                }
            }
        }
    }
    results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    println!("\n  Top 30:");
    println!("  {:>7} {:>7} {:>6} {:>7} {:>7} {:>7}",
        "cd_sub", "cd_sup", "cl_sb", "re_crt", "re_w", "score");
    println!("  {}", "-".repeat(46));
    for r in results.iter().take(30) {
        println!("  {:>7.3} {:>7.3} {:>6.2} {:>6.0}k {:>6.0}k {:>6.2}",
            r.cd_sub, r.cd_super, r.cl_sub, r.re_crit/1000.0, r.re_width/1000.0, r.score);
    }

    // ══════════════════════════════════════════════════════════════
    //  FINE GRID around top 3 coarse candidates
    // ══════════════════════════════════════════════════════════════
    let top3: Vec<(f64,f64,f64,f64,f64)> = results.iter().take(3)
        .map(|r| (r.cd_sub, r.cd_super, r.cl_sub, r.re_crit, r.re_width)).collect();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  FINE GRID around top 3");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut fine: Vec<R> = Vec::new();
    for (c_cd_sub, c_cd_sup, c_cl_sub, c_re_crit, c_re_width) in &top3 {
        for i in -3..=3 { for j in -3..=3 { for k in -3..=3 { for l in -2..=2 { for m in -2..=2 {
            let cd_sub = c_cd_sub + i as f64 * 0.01;
            let cd_super = c_cd_sup + j as f64 * 0.005;
            let cl_sub = c_cl_sub + k as f64 * 0.01;
            let re_crit = c_re_crit + l as f64 * 2_500.0;
            let re_width = c_re_width + m as f64 * 1_000.0;
            if cd_sub < 0.20 || cd_super < 0.15 || cl_sub < 0.05 || re_width < 1_000.0 { continue; }
            let p = ModelParams { cd_sub, cd_super, cl_sub, re_crit, re_width, ..baseline };
            let s = balanced_score(&p, &modern, &classic, rho);
            fine.push(R { cd_sub, cd_super, cl_sub, re_crit, re_width, score: s });
        }}}}}
    }
    fine.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    fine.dedup_by(|a, b| (a.cd_sub - b.cd_sub).abs() < 1e-6 && (a.cd_super - b.cd_super).abs() < 1e-6
        && (a.cl_sub - b.cl_sub).abs() < 1e-6 && (a.re_crit - b.re_crit).abs() < 1.0);

    println!("  Top 20:");
    println!("  {:>7} {:>7} {:>6} {:>7} {:>7} {:>7}",
        "cd_sub", "cd_sup", "cl_sb", "re_crt", "re_w", "score");
    println!("  {}", "-".repeat(46));
    for r in fine.iter().take(20) {
        println!("  {:>7.3} {:>7.3} {:>6.02} {:>6.0}k {:>6.0}k {:>6.2}",
            r.cd_sub, r.cd_super, r.cl_sub, r.re_crit/1000.0, r.re_width/1000.0, r.score);
    }

    // ══════════════════════════════════════════════════════════════
    //  Also vary cd_spin and sr_scale with the fine grid winner
    // ══════════════════════════════════════════════════════════════
    let fb = &fine[0];
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  cd_spin × sr_scale with best: cd_sub={:.3} cd_sup={:.3} cl_sub={:.2} re_crit={:.0}k re_w={:.0}k",
        fb.cd_sub, fb.cd_super, fb.cl_sub, fb.re_crit/1000.0, fb.re_width/1000.0);
    println!("═══════════════════════════════════════════════════════════════\n");

    struct RD { cd_spin: f64, sr_scale: f64, score: f64 }
    let mut dres: Vec<RD> = Vec::new();
    for &cd_spin in &[0.10, 0.13, 0.16, 0.19, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50] {
        for &sr_scale in &[0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15] {
            let p = ModelParams {
                cd_sub: fb.cd_sub, cd_super: fb.cd_super, cl_sub: fb.cl_sub,
                re_crit: fb.re_crit, re_width: fb.re_width,
                cd_spin, sr_scale, ..baseline
            };
            let s = balanced_score(&p, &modern, &classic, rho);
            dres.push(RD { cd_spin, sr_scale, score: s });
        }
    }
    dres.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    println!("  Top 15:");
    println!("  {:>8} {:>8} {:>8}", "cd_spin", "sr_scl", "score");
    println!("  {}", "-".repeat(28));
    for r in dres.iter().take(15) {
        println!("  {:>8.2} {:>8.02} {:>7.02}", r.cd_spin, r.sr_scale, r.score);
    }

    // ══════════════════════════════════════════════════════════════
    //  FINAL: Full bag for the overall winner
    // ══════════════════════════════════════════════════════════════
    let db = &dres[0];
    let winner = ModelParams {
        cd_sub: fb.cd_sub, cd_super: fb.cd_super, cl_sub: fb.cl_sub,
        re_crit: fb.re_crit, re_width: fb.re_width,
        cd_spin: db.cd_spin, sr_scale: db.sr_scale,
        ..baseline
    };

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  FINAL WINNER");
    println!("  cd_sub={:.3} cd_super={:.3} cd_spin={:.2} cl_sub={:.2} cl_super={:.2}",
        fb.cd_sub, fb.cd_super, db.cd_spin, fb.cl_sub, baseline.cl_super);
    println!("  sr_scale={:.02} re_crit={:.0} re_width={:.0}",
        db.sr_scale, fb.re_crit, fb.re_width);
    println!("═══════════════════════════════════════════════════════════════\n");

    print_summary("Baseline", &baseline, &modern, &classic, rho);
    println!();
    print_summary("Winner", &winner, &modern, &classic, rho);

    println!("\n  2024 Trackman:");
    print_full_bag(&winner, &modern, rho);
    println!("\n  Classic Trackman:");
    print_full_bag(&winner, &classic, rho);

    // Cd/Cl at key flight phases
    println!("\n  Cd/Cl at key flight phases (7-iron, winner):");
    println!("  {:>12} {:>8} {:>8} {:>8} {:>8} {:>8}", "Phase", "V(m/s)", "Re", "SR", "Cd", "Cl");
    println!("  {}", "-".repeat(56));
    let spin_7i = 7124.0 * PI / 30.0;
    for &(label, v, decay) in &[("Launch", 55.0, 1.0), ("Mid-flight", 40.0, 0.85),
        ("Late flight", 28.0, 0.75), ("Near landing", 20.0, 0.70)] {
        let spin = spin_7i * decay;
        let re = rho * v * winner.diameter_m / 1.81e-5;
        let sr = spin * winner.radius() / v;
        let (cd, cl) = winner.cd_cl(spin, v, rho);
        println!("  {:>12} {:>7.1} {:>7.0} {:>8.3} {:>8.3} {:>8.3}", label, v, re, sr, cd, cl);
    }
    println!("\n  Cd/Cl at key flight phases (Driver, winner):");
    println!("  {:>12} {:>8} {:>8} {:>8} {:>8} {:>8}", "Phase", "V(m/s)", "Re", "SR", "Cd", "Cl");
    println!("  {}", "-".repeat(56));
    let spin_dr = 2545.0 * PI / 30.0;
    for &(label, v, decay) in &[("Launch", 76.5, 1.0), ("Mid-flight", 55.0, 0.87),
        ("Late flight", 38.0, 0.78), ("Near landing", 28.0, 0.72)] {
        let spin = spin_dr * decay;
        let re = rho * v * winner.diameter_m / 1.81e-5;
        let sr = spin * winner.radius() / v;
        let (cd, cl) = winner.cd_cl(spin, v, rho);
        println!("  {:>12} {:>7.1} {:>7.0} {:>8.3} {:>8.3} {:>8.3}", label, v, re, sr, cd, cl);
    }

    // ══════════════════════════════════════════════════════════════
    //  COMPARISON TABLE: baseline vs prev_best vs winner
    // ══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");
    print_summary("Baseline (old)", &baseline, &modern, &classic, rho);
    println!();
    print_summary("Prev best (iron DA fixed, driver DA broken)", &prev_best, &modern, &classic, rho);
    println!();
    print_summary("New winner (balanced)", &winner, &modern, &classic, rho);
}
