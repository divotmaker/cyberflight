use cf_math::aero::BallModel;
use cf_math::environment::Environment;
use cf_math::trajectory::{ShotInput, simulate_flight};

struct Target {
    label: &'static str,
    speed: f64,
    launch: f64,
    spin: f64,
    carry: f64,
    land_angle: f64, // descent angle (degrees), 0 = not available
}

fn rmse(ball: &BallModel, env: &Environment, targets: &[Target]) -> f64 {
    let sum: f64 = targets
        .iter()
        .map(|t| {
            let input = ShotInput {
                ball_speed_mph: t.speed,
                launch_angle_deg: t.launch,
                launch_azimuth_deg: 0.0,
                backspin_rpm: t.spin,
                sidespin_rpm: 0.0,
            };
            let r = simulate_flight(&input, ball, env);
            (r.carry_yards - t.carry).powi(2)
        })
        .sum();
    (sum / targets.len() as f64).sqrt()
}

fn full_bag(ball: &BallModel, env: &Environment, targets: &[Target]) {
    println!(
        "  {:<12} {:>7} {:>7} {:>7}  {:>6} {:>6} {:>7}",
        "Club", "Model", "Target", "Delta", "DA°", "TM DA°", "DA Δ"
    );
    println!("  {}", "-".repeat(60));
    for t in targets {
        let input = ShotInput {
            ball_speed_mph: t.speed,
            launch_angle_deg: t.launch,
            launch_azimuth_deg: 0.0,
            backspin_rpm: t.spin,
            sidespin_rpm: 0.0,
        };
        let r = simulate_flight(&input, ball, env);
        // Compute descent angle from 3rd-to-last and 2nd-to-last points.
        // The LAST point is ground-clamped (y<=0), so using it produces NaN/garbage.
        let da = if r.points.len() >= 3 {
            let p1 = &r.points[r.points.len() - 3];
            let p2 = &r.points[r.points.len() - 2];
            let dx = p2.pos.x - p1.pos.x;
            let dy = p2.pos.y - p1.pos.y;
            let horiz = (dx * dx + (p2.pos.z - p1.pos.z).powi(2)).sqrt();
            (-dy / horiz).atan().to_degrees()
        } else {
            0.0
        };
        let da_delta = if t.land_angle > 0.0 {
            format!("{:>+6.1}", da - t.land_angle)
        } else {
            "   n/a".to_string()
        };
        let tm_da = if t.land_angle > 0.0 {
            format!("{:>5.0}", t.land_angle)
        } else {
            "  n/a".to_string()
        };
        println!(
            "  {:<12} {:>6.1}y {:>6.1}y {:>+6.1}y  {:>5.1} {} {}",
            t.label, r.carry_yards, t.carry, r.carry_yards - t.carry, da, tm_da, da_delta
        );
    }
}

fn main() {
    // Trackman normalizes to 77°F (25°C), sea level
    let env = Environment {
        altitude_m: 0.0,
        temperature_c: 25.0,
    };

    // Complete 2024 Trackman PGA Tour Averages — all 12 clubs, consistent dataset
    // Source: trackman.com/blog/golf/introducing-updated-tour-averages (March 2025)
    // 40+ PGA Tour & DP World Tour events, 200+ players
    let modern: Vec<Target> = vec![
        Target { label: "Driver",  speed: 171.0, launch: 10.4, spin: 2545.0, carry: 282.0, land_angle: 39.0 },
        Target { label: "3-Wood",  speed: 162.0, launch:  9.3, spin: 3663.0, carry: 249.0, land_angle: 44.0 },
        Target { label: "5-Wood",  speed: 156.0, launch:  9.7, spin: 4322.0, carry: 236.0, land_angle: 48.0 },
        Target { label: "Hybrid",  speed: 149.0, launch: 10.2, spin: 4587.0, carry: 231.0, land_angle: 49.0 },
        Target { label: "3-iron",  speed: 145.0, launch: 10.3, spin: 4404.0, carry: 218.0, land_angle: 48.0 },
        Target { label: "4-iron",  speed: 140.0, launch: 10.8, spin: 4782.0, carry: 209.0, land_angle: 49.0 },
        Target { label: "5-iron",  speed: 135.0, launch: 11.9, spin: 5280.0, carry: 199.0, land_angle: 50.0 },
        Target { label: "6-iron",  speed: 130.0, launch: 14.0, spin: 6204.0, carry: 188.0, land_angle: 50.0 },
        Target { label: "7-iron",  speed: 123.0, launch: 16.1, spin: 7124.0, carry: 176.0, land_angle: 51.0 },
        Target { label: "8-iron",  speed: 118.0, launch: 17.8, spin: 8078.0, carry: 164.0, land_angle: 51.0 },
        Target { label: "9-iron",  speed: 112.0, launch: 20.0, spin: 8793.0, carry: 152.0, land_angle: 52.0 },
        Target { label: "PW",      speed: 104.0, launch: 23.7, spin: 9316.0, carry: 142.0, land_angle: 52.0 },
    ];

    // Classic (2009-2014) Trackman PGA Tour Averages — full bag, self-consistent
    let classic: Vec<Target> = vec![
        Target { label: "Driver",  speed: 165.0, launch: 11.2, spin: 2685.0, carry: 269.0, land_angle: 0.0 },
        Target { label: "3-Wood",  speed: 158.0, launch:  9.2, spin: 3655.0, carry: 243.0, land_angle: 0.0 },
        Target { label: "5-Wood",  speed: 152.0, launch:  9.4, spin: 4350.0, carry: 230.0, land_angle: 0.0 },
        Target { label: "Hybrid",  speed: 146.0, launch: 10.2, spin: 4437.0, carry: 225.0, land_angle: 0.0 },
        Target { label: "3-iron",  speed: 142.0, launch: 10.4, spin: 4630.0, carry: 212.0, land_angle: 0.0 },
        Target { label: "4-iron",  speed: 137.0, launch: 11.0, spin: 4836.0, carry: 203.0, land_angle: 0.0 },
        Target { label: "5-iron",  speed: 132.0, launch: 12.1, spin: 5361.0, carry: 194.0, land_angle: 0.0 },
        Target { label: "6-iron",  speed: 127.0, launch: 14.1, spin: 6231.0, carry: 183.0, land_angle: 0.0 },
        Target { label: "7-iron",  speed: 120.0, launch: 16.3, spin: 7097.0, carry: 172.0, land_angle: 0.0 },
        Target { label: "8-iron",  speed: 115.0, launch: 18.1, spin: 7998.0, carry: 160.0, land_angle: 0.0 },
        Target { label: "9-iron",  speed: 109.0, launch: 20.4, spin: 8647.0, carry: 148.0, land_angle: 0.0 },
        Target { label: "PW",      speed: 102.0, launch: 24.2, spin: 9304.0, carry: 136.0, land_angle: 0.0 },
    ];

    // ══════════════════════════════════════════════════════════════
    //  PASS 1: Coarse grid
    // ══════════════════════════════════════════════════════════════

    let cd_range: Vec<f64> = (0..=12).map(|i| 0.200 + i as f64 * 0.005).collect();
    let cl_range: Vec<f64> = (0..=14).map(|i| 0.08 + i as f64 * 0.01).collect();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  PASS 1: Coarse grid ({} × {} = {} combos)", cd_range.len(), cl_range.len(), cd_range.len() * cl_range.len());
    println!("  Now using COMPLETE 2024 dataset (12 clubs, consistent inputs)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut results: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for &cd in &cd_range {
        for &cl in &cl_range {
            let ball = BallModel { cd_super: cd, cl_sub: cl, ..BallModel::TOUR };
            let rc = rmse(&ball, &env, &classic);
            let rm = rmse(&ball, &env, &modern);
            let combined = ((rc * rc + rm * rm) / 2.0).sqrt(); // equal weight now
            results.push((cd, cl, rc, rm, combined));
        }
    }

    // Sort by 2024 RMSE (primary — this is the dataset we trust most)
    results.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
    println!("Top 15 by 2024 RMSE (12-club, consistent inputs):");
    println!("  {:>10} {:>8} {:>10} {:>10} {:>10}", "cd_super", "cl_sub", "classic", "2024", "combined");
    println!("  {}", "-".repeat(52));
    for (cd, cl, rc, rm, comb) in results.iter().take(15) {
        println!("  {:>10.3} {:>8.2} {:>9.1}y {:>9.1}y {:>9.1}y", cd, cl, rc, rm, comb);
    }

    // Sort by classic RMSE
    results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("\nTop 15 by classic RMSE:");
    println!("  {:>10} {:>8} {:>10} {:>10} {:>10}", "cd_super", "cl_sub", "classic", "2024", "combined");
    println!("  {}", "-".repeat(52));
    for (cd, cl, rc, rm, comb) in results.iter().take(15) {
        println!("  {:>10.3} {:>8.02} {:>9.1}y {:>9.1}y {:>9.1}y", cd, cl, rc, rm, comb);
    }

    // Sort by combined
    results.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
    println!("\nTop 15 by combined RMSE (equal weight classic + 2024):");
    println!("  {:>10} {:>8} {:>10} {:>10} {:>10}", "cd_super", "cl_sub", "classic", "2024", "combined");
    println!("  {}", "-".repeat(52));
    for (cd, cl, rc, rm, comb) in results.iter().take(15) {
        println!("  {:>10.3} {:>8.02} {:>9.1}y {:>9.1}y {:>9.1}y", cd, cl, rc, rm, comb);
    }

    // ══════════════════════════════════════════════════════════════
    //  PASS 2: Fine grid around top 3 combined candidates
    // ══════════════════════════════════════════════════════════════

    let top3: Vec<(f64, f64)> = results.iter().take(3).map(|(cd, cl, _, _, _)| (*cd, *cl)).collect();

    let mut fine_results: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for (center_cd, center_cl) in &top3 {
        for i in -5..=5 {
            for j in -5..=5 {
                let cd = center_cd + i as f64 * 0.001;
                let cl = center_cl + j as f64 * 0.002;
                if cd < 0.18 || cl < 0.05 { continue; }
                let ball = BallModel { cd_super: cd, cl_sub: cl, ..BallModel::TOUR };
                let rc = rmse(&ball, &env, &classic);
                let rm = rmse(&ball, &env, &modern);
                let combined = ((rc * rc + rm * rm) / 2.0).sqrt();
                fine_results.push((cd, cl, rc, rm, combined));
            }
        }
    }

    fine_results.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
    fine_results.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  PASS 2: Fine grid (step 0.001 cd, 0.002 cl)");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("  Top 15 by combined RMSE:");
    println!("  {:>10} {:>8} {:>10} {:>10} {:>10}", "cd_super", "cl_sub", "classic", "2024", "combined");
    println!("  {}", "-".repeat(52));
    for (cd, cl, rc, rm, comb) in fine_results.iter().take(15) {
        println!("  {:>10.3} {:>8.3} {:>9.1}y {:>9.1}y {:>9.1}y", cd, cl, rc, rm, comb);
    }

    // ══════════════════════════════════════════════════════════════
    //  Full bag for top 3 (with descent angles!)
    // ══════════════════════════════════════════════════════════════

    for (i, (cd, cl, rc, rm, comb)) in fine_results.iter().take(3).enumerate() {
        let ball = BallModel { cd_super: *cd, cl_sub: *cl, ..BallModel::TOUR };
        println!("\n═══════════════════════════════════════════════════════════════");
        println!(
            "  #{}: cd_super={cd:.3}, cl_sub={cl:.3}  (classic={rc:.1}, 2024={rm:.1}, combined={comb:.1})",
            i + 1
        );
        println!("═══════════════════════════════════════════════════════════════\n");

        println!("  2024 Trackman (consistent dataset):");
        full_bag(&ball, &env, &modern);
        println!("\n  Classic Trackman:");
        full_bag(&ball, &env, &classic);
    }
}
