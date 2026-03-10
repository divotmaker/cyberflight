/// Experiment 10: Polynomial Cl(SR) replacing exponential saturation.
///
/// Current:   Cl = cl_max(Re) × (1 - exp(-SR/sr_scale))     [saturates, never drops]
/// Proposed:  Cl = cl_max(Re) × clamp(c₁·SR - c₂·SR², 0, 1.2)  [can turn over at high SR]
///
/// Physics: Lyu et al. (2018) fit Cl = a₀ + a₁·SR + a₂·SR². The exponential
/// saturation was our approximation. The polynomial can peak at moderate SR
/// and drop at high SR — potentially matching wind tunnel cl_sub≈0.15 while
/// still giving PW less effective lift at high SR.
///
/// Approach: run BOTH models at multiple forced cl_sub levels and compare.
/// If polynomial + cl_sub=0.15 beats exponential + cl_sub=0.15, the polynomial
/// provides structural value.
use glam::DVec3;
use rayon::prelude::*;

const G: f64 = 9.80665;
const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
struct P {
    mass_kg: f64, diameter_m: f64,
    cd_sub: f64, cd_super: f64, cd_spin: f64,
    cl_sub: f64, cl_super: f64,
    re_crit: f64, re_width: f64, spin_decay: f64,
    // Cl(SR) model: if use_poly, use c1/c2. Otherwise use sr_scale (exponential).
    use_poly: bool,
    sr_scale: f64, // exponential: 1 - exp(-SR/sr_scale)
    c1: f64,       // polynomial: c1*SR - c2*SR²
    c2: f64,
}
impl P {
    fn r(&self) -> f64 { self.diameter_m / 2.0 }
    fn a(&self) -> f64 { PI * self.r() * self.r() }

    fn base() -> Self {
        Self {
            mass_kg: 0.04593, diameter_m: 0.04267,
            cd_sub: 0.310, cd_super: 0.235, cd_spin: 0.12,
            cl_sub: 0.050, cl_super: 0.29,
            re_crit: 75_000.0, re_width: 125.0, spin_decay: 0.04,
            use_poly: false, sr_scale: 0.10, c1: 0.0, c2: 0.0,
        }
    }

    fn cd_cl(&self, spin: f64, spd: f64, rho: f64) -> (f64, f64) {
        if spd < 1e-10 { return (self.cd_sub, 0.0) }
        let re = rho * spd * self.diameter_m / 1.81e-5;
        let sr = (spin * self.r() / spd).abs();
        let sigma = 1.0 / (1.0 + (-(re - self.re_crit) / self.re_width).exp());

        let cd = self.cd_sub + (self.cd_super - self.cd_sub) * sigma + self.cd_spin * sr;

        let cl_max = self.cl_sub + (self.cl_super - self.cl_sub) * sigma;
        let f_sr = if self.use_poly {
            (self.c1 * sr - self.c2 * sr * sr).clamp(0.0, 1.2)
        } else {
            1.0 - (-sr / self.sr_scale).exp()
        };
        let cl = cl_max * f_sr;
        (cd, cl)
    }

    fn acc(&self, vel: DVec3, sa: DVec3, spin: f64, rho: f64) -> DVec3 {
        let spd = vel.length();
        if spd < 1e-10 { return DVec3::new(0.0, -G, 0.0) }
        let vh = vel / spd;
        let (cd, cl) = self.cd_cl(spin, spd, rho);
        let q = 0.5 * rho * spd * spd;
        let drag = vh * (-cd * q * self.a() / self.mass_kg);
        let mag = if spin.abs() > 1e-10 {
            let ld = sa.cross(vh);
            let lm = ld.length();
            if lm > 1e-10 { (ld / lm) * (cl * q * self.a() / self.mass_kg) } else { DVec3::ZERO }
        } else { DVec3::ZERO };
        DVec3::new(0.0, -G, 0.0) + drag + mag
    }
}

fn sim(p: &P, mph: f64, ldeg: f64, rpm: f64, rho: f64) -> (f64, f64) {
    let v = mph * 0.44704;
    let lr = ldeg * PI / 180.0;
    let sa = DVec3::Z;
    let mut spin = rpm * PI / 30.0;
    let dt = 0.001;
    let mut pos = DVec3::ZERO;
    let mut vel = DVec3::new(v * lr.cos(), v * lr.sin(), 0.0);
    let mut t = 0.0;
    let (mut p3, mut p2, mut p1) = (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO);
    let mut sc = 0u32;
    for _ in 1..=30_000 {
        let k1v = p.acc(vel, sa, spin, rho);
        let k1x = vel;
        let k2v = p.acc(vel + k1v * (dt / 2.0), sa, spin, rho);
        let k2x = vel + k1v * (dt / 2.0);
        let k3v = p.acc(vel + k2v * (dt / 2.0), sa, spin, rho);
        let k3x = vel + k2v * (dt / 2.0);
        let k4v = p.acc(vel + k3v * dt, sa, spin, rho);
        let k4x = vel + k3v * dt;
        pos = pos + (k1x + 2.0 * k2x + 2.0 * k3x + k4x) * (dt / 6.0);
        vel = vel + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) * (dt / 6.0);
        t += dt;
        spin *= (-p.spin_decay * dt).exp();
        sc += 1;
        if sc % 10 == 0 { p3 = p2; p2 = p1; p1 = pos; }
        if t > 0.1 && pos.y <= 0.0 { break }
    }
    let dx = p2.x - p3.x;
    let dy = p2.y - p3.y;
    let h = (dx * dx + (p2.z - p3.z).powi(2)).sqrt();
    let da = if h > 1e-10 { (-dy / h).atan() * 180.0 / PI } else { 0.0 };
    ((pos.x * pos.x + pos.z * pos.z).sqrt() * 1.09361, da)
}

struct T { l: &'static str, s: f64, la: f64, sp: f64, c: f64, da: f64 }
fn trackman_2024() -> Vec<T> { vec![
    T { l: "Driver",  s: 171.0, la: 10.4, sp: 2545.0, c: 282.0, da: 39.0 },
    T { l: "3-Wood",  s: 162.0, la:  9.3, sp: 3663.0, c: 249.0, da: 44.0 },
    T { l: "5-Wood",  s: 156.0, la:  9.7, sp: 4322.0, c: 236.0, da: 48.0 },
    T { l: "Hybrid",  s: 149.0, la: 10.2, sp: 4587.0, c: 231.0, da: 49.0 },
    T { l: "3-iron",  s: 145.0, la: 10.3, sp: 4404.0, c: 218.0, da: 48.0 },
    T { l: "4-iron",  s: 140.0, la: 10.8, sp: 4782.0, c: 209.0, da: 49.0 },
    T { l: "5-iron",  s: 135.0, la: 11.9, sp: 5280.0, c: 199.0, da: 50.0 },
    T { l: "6-iron",  s: 130.0, la: 14.0, sp: 6204.0, c: 188.0, da: 50.0 },
    T { l: "7-iron",  s: 123.0, la: 16.1, sp: 7124.0, c: 176.0, da: 51.0 },
    T { l: "8-iron",  s: 118.0, la: 17.8, sp: 8078.0, c: 164.0, da: 51.0 },
    T { l: "9-iron",  s: 112.0, la: 20.0, sp: 8793.0, c: 152.0, da: 52.0 },
    T { l: "PW",      s: 104.0, la: 23.7, sp: 9316.0, c: 142.0, da: 52.0 },
]}

fn score(p: &P, mt: &[T], rho: f64) -> f64 {
    let mut csq = 0.0;
    let mut cmax = 0.0_f64;
    let mut dsq = 0.0;
    let mut dmax = 0.0_f64;
    let mut dn = 0;
    for t in mt {
        let (cy, da) = sim(p, t.s, t.la, t.sp, rho);
        let e = cy - t.c;
        csq += e * e;
        cmax = cmax.max(e.abs());
        if t.da > 0.0 { let de = da - t.da; dsq += de * de; dmax = dmax.max(de.abs()); dn += 1; }
    }
    let carry = (csq / mt.len() as f64).sqrt();
    let da_rmse = if dn > 0 { (dsq / dn as f64).sqrt() } else { 0.0 };
    let co = if cmax > 5.0 { (cmax - 5.0).powi(2) } else { 0.0 };
    let do_ = if dmax > 3.0 { (dmax - 3.0).powi(2) } else { 0.0 };
    carry + 0.5 * da_rmse + 0.5 * co + 0.3 * do_
}

fn rho_at_c(temp_c: f64) -> f64 { 1.225 * (288.15 / (temp_c + 273.15)) }
fn rho_denver() -> f64 {
    let t = 288.15 - 0.0065 * 1609.0;
    let p = 101325.0 * (t / 288.15_f64).powf(5.2561);
    p / (287.05 * t)
}

fn main() {
    let rho = rho_at_c(25.0);
    let mt = trackman_2024();
    let base = P::base();

    println!("Experiment 10: Polynomial Cl(SR) vs Exponential Cl(SR)");
    println!("=======================================================");
    println!("  Polynomial: Cl = cl_max(Re) × clamp(c₁·SR - c₂·SR², 0, 1.2)");
    println!("  Exponential: Cl = cl_max(Re) × (1 - exp(-SR/sr_scale))");
    println!("  Using {} threads, ρ(25°C)={:.4} kg/m³\n", rayon::current_num_threads(), rho);

    let mt_r = &mt;

    // ═══════════════════════════════════════════════════════════════
    //  PART A: Exponential model at various fixed cl_sub levels
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PART A: Best EXPONENTIAL model at each cl_sub level");
    println!("═══════════════════════════════════════════════════════════════\n");

    for &cl_sub in &[0.05, 0.10, 0.15, 0.20] {
        let mut combos: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
        for &cd_sub in &[0.22, 0.26, 0.30, 0.34, 0.38, 0.42] {
            for &cd_spin in &[0.08, 0.12, 0.16, 0.20] {
                for &sr_sc in &[0.06, 0.08, 0.10, 0.12, 0.14] {
                    for &re_c in &[70e3, 75e3, 80e3] {
                        for &re_w in &[62.5, 125.0, 500.0, 2000.0, 5000.0, 10000.0] {
                            combos.push((cd_sub, cd_spin, sr_sc, re_c, re_w));
                        }
                    }
                }
            }
        }
        let best: f64 = combos.par_iter().map(|&(cd_sub, cd_spin, sr_sc, re_c, re_w)| {
            let p = P { cd_sub, cd_spin, cl_sub, sr_scale: sr_sc, re_crit: re_c, re_width: re_w,
                use_poly: false, ..base };
            score(&p, mt_r, rho)
        }).reduce(|| f64::MAX, f64::min);

        // Find the actual winner params
        let winner = combos.par_iter().map(|&(cd_sub, cd_spin, sr_sc, re_c, re_w)| {
            let p = P { cd_sub, cd_spin, cl_sub, sr_scale: sr_sc, re_crit: re_c, re_width: re_w,
                use_poly: false, ..base };
            let s = score(&p, mt_r, rho);
            (s, cd_sub, cd_spin, sr_sc, re_c, re_w)
        }).reduce(|| (f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0),
            |a, b| if a.0 < b.0 { a } else { b });

        let wp = P { cd_sub: winner.1, cd_spin: winner.2, cl_sub, sr_scale: winner.3,
            re_crit: winner.4, re_width: winner.5, use_poly: false, ..base };
        let mut csq = 0.0; let mut dsq = 0.0; let mut dn = 0;
        for t in &mt {
            let (cy, da) = sim(&wp, t.s, t.la, t.sp, rho);
            csq += (cy - t.c).powi(2);
            if t.da > 0.0 { dsq += (da - t.da).powi(2); dn += 1; }
        }
        println!("  cl_sub={:.02}  score={:.2}  carry_rmse={:.1}y  da_rmse={:.1}°  [cd_sub={:.02} cd_spin={:.02} sr_sc={:.02} re_c={:.0}k re_w={:.0}]",
            cl_sub, best, (csq / mt.len() as f64).sqrt(), (dsq / dn as f64).sqrt(),
            winner.1, winner.2, winner.3, winner.4 / 1e3, winner.5);
    }

    // ═══════════════════════════════════════════════════════════════
    //  PART B: Polynomial model at various fixed cl_sub levels
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  PART B: Best POLYNOMIAL model at each cl_sub level");
    println!("═══════════════════════════════════════════════════════════════\n");

    for &cl_sub in &[0.05, 0.10, 0.15, 0.20] {
        type C = (f64, f64, f64, f64, f64, f64);
        let mut combos: Vec<C> = Vec::new();
        for &cd_sub in &[0.22, 0.26, 0.30, 0.34, 0.38, 0.42] {
            for &cd_spin in &[0.08, 0.12, 0.16, 0.20] {
                for &c1 in &[3.0, 5.0, 7.0, 10.0, 14.0] {
                    for &c2 in &[0.0, 2.0, 5.0, 10.0, 15.0, 25.0] {
                        for &re_c in &[70e3, 75e3, 80e3] {
                            for &re_w in &[62.5, 125.0, 500.0, 2000.0, 5000.0, 10000.0] {
                                combos.push((cd_sub, cd_spin, c1, c2, re_c, re_w));
                            }
                        }
                    }
                }
            }
        }

        let winner = combos.par_iter().map(|&(cd_sub, cd_spin, c1, c2, re_c, re_w)| {
            let p = P { cd_sub, cd_spin, cl_sub, c1, c2, re_crit: re_c, re_width: re_w,
                use_poly: true, ..base };
            let s = score(&p, mt_r, rho);
            (s, cd_sub, cd_spin, c1, c2, re_c, re_w)
        }).reduce(|| (f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            |a, b| if a.0 < b.0 { a } else { b });

        let wp = P { cd_sub: winner.1, cd_spin: winner.2, cl_sub, c1: winner.3, c2: winner.4,
            re_crit: winner.5, re_width: winner.6, use_poly: true, ..base };
        let mut csq = 0.0; let mut dsq = 0.0; let mut dn = 0;
        for t in &mt {
            let (cy, da) = sim(&wp, t.s, t.la, t.sp, rho);
            csq += (cy - t.c).powi(2);
            if t.da > 0.0 { dsq += (da - t.da).powi(2); dn += 1; }
        }
        // Show f(SR) at key SR values
        let f_driver = (winner.3 * 0.08 - winner.4 * 0.0064).clamp(0.0, 1.2);
        let f_7iron = (winner.3 * 0.20 - winner.4 * 0.04).clamp(0.0, 1.2);
        let f_pw = (winner.3 * 0.50 - winner.4 * 0.25).clamp(0.0, 1.2);
        println!("  cl_sub={:.02}  score={:.2}  carry={:.1}y  da={:.1}°  [cd_sub={:.02} cd_spin={:.02} c1={:.0} c2={:.0} re_c={:.0}k re_w={:.0}]",
            cl_sub, winner.0, (csq / mt.len() as f64).sqrt(), (dsq / dn as f64).sqrt(),
            winner.1, winner.2, winner.3, winner.4, winner.5 / 1e3, winner.6);
        println!("          f(SR): driver(0.08)={:.2}  7i(0.20)={:.2}  PW(0.50)={:.2}",
            f_driver, f_7iron, f_pw);
    }

    // ═══════════════════════════════════════════════════════════════
    //  PART C: Unconstrained polynomial — what does optimizer choose?
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  PART C: Unconstrained polynomial search");
    println!("═══════════════════════════════════════════════════════════════\n");

    type C7 = (f64, f64, f64, f64, f64, f64, f64);
    let mut combos: Vec<C7> = Vec::new();
    for &cd_sub in &[0.22, 0.26, 0.30, 0.34, 0.38] {
        for &cd_spin in &[0.08, 0.12, 0.16, 0.20] {
            for &cl_sub in &[0.04, 0.08, 0.12, 0.16, 0.20] {
                for &c1 in &[3.0, 5.0, 7.0, 10.0, 14.0] {
                    for &c2 in &[0.0, 2.0, 5.0, 10.0, 15.0, 25.0] {
                        for &re_c in &[70e3, 75e3, 80e3] {
                            for &re_w in &[62.5, 125.0, 500.0, 2000.0, 5000.0] {
                                combos.push((cd_sub, cd_spin, cl_sub, c1, c2, re_c, re_w));
                            }
                        }
                    }
                }
            }
        }
    }
    println!("  {} combos", combos.len());

    #[derive(Clone)]
    struct R { cd_sub: f64, cd_spin: f64, cl_sub: f64, c1: f64, c2: f64, re_c: f64, re_w: f64, sc: f64 }

    let mut results: Vec<R> = combos.par_iter().map(|&(cd_sub, cd_spin, cl_sub, c1, c2, re_c, re_w)| {
        let p = P { cd_sub, cd_spin, cl_sub, c1, c2, re_crit: re_c, re_width: re_w,
            use_poly: true, ..base };
        let s = score(&p, mt_r, rho);
        R { cd_sub, cd_spin, cl_sub, c1, c2, re_c, re_w, sc: s }
    }).collect();
    results.sort_by(|a, b| a.sc.partial_cmp(&b.sc).unwrap());

    println!("\n  Top 20:");
    println!("  {:>5} {:>5} {:>5} {:>4} {:>4} {:>5} {:>6}  {:>6}",
        "cd_sb", "cd_sn", "cl_s", "c1", "c2", "re_c", "re_w", "score");
    println!("  {}", "-".repeat(52));
    for r in results.iter().take(20) {
        println!("  {:>5.02} {:>5.02} {:>5.02} {:>4.0} {:>4.0} {:>3.0}k {:>6.0}  {:>6.02}",
            r.cd_sub, r.cd_spin, r.cl_sub, r.c1, r.c2, r.re_c / 1e3, r.re_w, r.sc);
    }

    // ── Fine grid around top 3 ──
    let top3: Vec<R> = results.iter().take(3).cloned().collect();
    println!("\n  Fine grid around top 3...");
    type C9 = (f64, f64, f64, f64, f64, f64, f64, f64, f64);
    let mut fine: Vec<C9> = Vec::new();
    for r in &top3 {
        for &dcs in &[-0.02, -0.01, 0.0, 0.01, 0.02] {
            let cd_sub = r.cd_sub + dcs;
            if cd_sub < 0.15 { continue }
            for &dcsn in &[-0.02, 0.0, 0.02] {
                let cd_spin = r.cd_spin + dcsn;
                if cd_spin < 0.06 { continue }
                for &dcl in &[-0.02, -0.01, 0.0, 0.01, 0.02] {
                    let cl_sub = r.cl_sub + dcl;
                    if cl_sub < 0.02 { continue }
                    for &cd_sup in &[0.225, 0.230, 0.235, 0.240] {
                        for &dc1 in &[-1.0, 0.0, 1.0, 2.0] {
                            let c1 = r.c1 + dc1;
                            if c1 < 1.0 { continue }
                            for &dc2 in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
                                let c2 = r.c2 + dc2;
                                if c2 < 0.0 { continue }
                                for &re_c in &[r.re_c - 2500.0, r.re_c, r.re_c + 2500.0] {
                                    if re_c < 60e3 { continue }
                                    for &re_w in &[(r.re_w * 0.5).max(30.0), r.re_w, r.re_w * 2.0] {
                                        fine.push((cd_sub, cd_sup, cd_spin, cl_sub, c1, c2, re_c, re_w, 0.0));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fine.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    fine.dedup();
    println!("  {} fine combos", fine.len());

    #[derive(Clone)]
    struct R2 { cd_sub: f64, cd_sup: f64, cd_spin: f64, cl_sub: f64, c1: f64, c2: f64, re_c: f64, re_w: f64, sc: f64 }

    let mut fine_results: Vec<R2> = fine.par_iter().map(|&(cd_sub, cd_sup, cd_spin, cl_sub, c1, c2, re_c, re_w, _)| {
        let p = P { cd_sub, cd_super: cd_sup, cd_spin, cl_sub, c1, c2, re_crit: re_c, re_width: re_w,
            use_poly: true, ..base };
        let s = score(&p, mt_r, rho);
        R2 { cd_sub, cd_sup, cd_spin, cl_sub, c1, c2, re_c, re_w, sc: s }
    }).collect();
    fine_results.sort_by(|a, b| a.sc.partial_cmp(&b.sc).unwrap());

    println!("\n  Top 20 (fine):");
    println!("  {:>5} {:>5} {:>5} {:>5} {:>4} {:>4} {:>5} {:>6}  {:>6}",
        "cd_sb", "cd_sp", "cd_sn", "cl_s", "c1", "c2", "re_c", "re_w", "score");
    println!("  {}", "-".repeat(58));
    for r in fine_results.iter().take(20) {
        println!("  {:>5.03} {:>5.03} {:>5.02} {:>5.03} {:>4.0} {:>4.0} {:>3.0}k {:>6.0}  {:>6.03}",
            r.cd_sub, r.cd_sup, r.cd_spin, r.cl_sub, r.c1, r.c2, r.re_c / 1e3, r.re_w, r.sc);
    }

    // ── Winner analysis ──
    let w = &fine_results[0];
    let winner = P {
        cd_sub: w.cd_sub, cd_super: w.cd_sup, cd_spin: w.cd_spin,
        cl_sub: w.cl_sub, cl_super: 0.29, c1: w.c1, c2: w.c2,
        re_crit: w.re_c, re_width: w.re_w, use_poly: true, ..base
    };

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  WINNER (Polynomial Cl):");
    println!("    cd_sub={:.3}  cd_super={:.3}  cd_spin={:.02}",
        w.cd_sub, w.cd_sup, w.cd_spin);
    println!("    cl_sub={:.03}  cl_super=0.29  c1={:.1}  c2={:.1}",
        w.cl_sub, w.c1, w.c2);
    println!("    re_crit={:.0}  re_width={:.0}  score={:.3}",
        w.re_c, w.re_w, w.sc);
    // Show f(SR) at key points
    let peak_sr = if w.c2 > 0.0 { w.c1 / (2.0 * w.c2) } else { f64::INFINITY };
    println!("    f(SR) peak at SR={:.2}", peak_sr);
    for &(lb, sr) in &[("Driver", 0.08), ("7i-flight", 0.20), ("7i-land", 0.45), ("PW-land", 0.60), ("PW-deep", 0.80)] {
        let f = (w.c1 * sr - w.c2 * sr * sr).clamp(0.0, 1.2);
        println!("    f({:<10} SR={:.2}) = {:.3}", lb, sr, f);
    }
    println!("═══════════════════════════════════════════════════════════════\n");

    // Per-club breakdown
    println!("  2024 Trackman (25°C sea level):");
    println!("  {:<12} {:>7} {:>7} {:>7}  {:>6} {:>6} {:>7}",
        "Club", "Model", "Target", "Delta", "DA°", "TM DA°", "DA Δ");
    println!("  {}", "-".repeat(60));

    let mut csq = 0.0; let mut cmax = 0.0_f64;
    let mut dsq = 0.0; let mut dsum = 0.0; let mut dn = 0;
    for t in &mt {
        let (cy, da) = sim(&winner, t.s, t.la, t.sp, rho);
        let ce = cy - t.c; csq += ce * ce; cmax = cmax.max(ce.abs());
        let dd = if t.da > 0.0 { let de = da - t.da; dsq += de * de; dsum += de; dn += 1; format!("{:>+6.1}", de) }
                 else { "   n/a".into() };
        let tm = if t.da > 0.0 { format!("{:>5.0}", t.da) } else { "  n/a".into() };
        println!("  {:<12} {:>6.1}y {:>6.1}y {:>+6.1}y  {:>5.1} {} {}",
            t.l, cy, t.c, ce, da, tm, dd);
    }
    let carry_rmse = (csq / mt.len() as f64).sqrt();
    let da_rmse = (dsq / dn as f64).sqrt();
    let da_mean = dsum / dn as f64;
    println!("\n  Carry RMSE={:.1}y  Max={:.1}y  DA RMSE={:.1}°  DA mean={:+.1}°\n",
        carry_rmse, cmax, da_rmse, da_mean);

    // Temperature & altitude
    let rho_50 = rho_at_c(10.0);
    let rho_90 = rho_at_c(32.22);
    let t7 = &mt[8];
    let (c50, _) = sim(&winner, t7.s, t7.la, t7.sp, rho_50);
    let (c90, _) = sim(&winner, t7.s, t7.la, t7.sp, rho_90);
    let v2 = P { use_poly: false, sr_scale: 0.10, ..base };
    let (v50, _) = sim(&v2, t7.s, t7.la, t7.sp, rho_50);
    let (v90, _) = sim(&v2, t7.s, t7.la, t7.sp, rho_90);
    println!("  Temp sensitivity (7i, 50°F vs 90°F):");
    println!("    Poly: {:.1}y → {:.1}y  Δ={:.1}y  (target: ~8y)", c50, c90, c90 - c50);
    println!("    v2:   {:.1}y → {:.1}y  Δ={:.1}y", v50, v90, v90 - v50);

    let rho_den = rho_denver();
    let td = &mt[0];
    let (csl, _) = sim(&winner, td.s, td.la, td.sp, rho);
    let (cden, _) = sim(&winner, td.s, td.la, td.sp, rho_den);
    let pct = (cden - csl) / csl * 100.0;
    let (vsl, _) = sim(&v2, td.s, td.la, td.sp, rho);
    let (vden, _) = sim(&v2, td.s, td.la, td.sp, rho_den);
    let vpct = (vden - vsl) / vsl * 100.0;
    println!("\n  Alt sensitivity (driver, sea level vs Denver):");
    println!("    Poly: {:.1}y → {:.1}y  Δ={:+.1}y  ({:.1}%)  (target: ~6%)", csl, cden, cden - csl, pct);
    println!("    v2:   {:.1}y → {:.1}y  Δ={:+.1}y  ({:.1}%)", vsl, vden, vden - vsl, vpct);

    // ── Summary comparison ──
    println!("\n  ╔══════════════════════════════════════════════════════════╗");
    println!("  ║  Summary                                               ║");
    println!("  ╠══════════════╦══════════╦══════════╦══════════╦═════════╣");
    println!("  ║ Metric       ║    v2    ║  Poly    ║ Published║ Better? ║");
    println!("  ╠══════════════╬══════════╬══════════╬══════════╬═════════╣");
    println!("  ║ cl_sub       ║  0.050   ║  {:.03}   ║  ~0.15   ║  {}║",
        w.cl_sub, if w.cl_sub > 0.060 { "  ✓  " } else { "     " });
    println!("  ║ re_width     ║    125   ║  {:>6.0}  ║ 5k-40k  ║  {}║",
        w.re_w, if w.re_w > 200.0 { "  ✓  " } else { "     " });
    println!("  ║ temp Δ (y)   ║  {:>5.1}   ║  {:>5.1}   ║   ~8     ║  {}║",
        v90 - v50, c90 - c50, if (c90 - c50) > 5.0 { "  ✓  " } else { "     " });
    println!("  ║ alt gain %   ║  {:>5.1}   ║  {:>5.1}   ║   ~6%    ║  {}║",
        vpct, pct, if pct > 2.0 { "  ✓  " } else { "     " });
    println!("  ║ carry RMSE   ║    3.8   ║  {:>5.1}   ║   ≤4     ║  {}║",
        carry_rmse, if carry_rmse <= 4.0 { "  ✓  " } else { "     " });
    println!("  ║ DA RMSE      ║    1.2   ║  {:>5.1}   ║   ≤2°    ║  {}║",
        da_rmse, if da_rmse <= 2.0 { "  ✓  " } else { "     " });
    println!("  ╚══════════════╩══════════╩══════════╩══════════╩═════════╝");
}
