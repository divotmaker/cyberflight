use std::path::Path;

use image::RgbaImage;

use crate::compare::ssim_compare;

/// Default SSIM threshold for golden comparison.
pub const DEFAULT_THRESHOLD: f64 = 0.97;

/// Assert that a rendered image matches its golden reference.
///
/// If `UPDATE_GOLDENS=1` is set, the golden image is created/updated instead.
/// On failure, saves `_FAIL.png` and `_DIFF.png` alongside the golden.
///
/// # Panics
/// Panics if the comparison fails (SSIM below threshold).
pub fn assert_golden(rendered: &RgbaImage, golden_path: &Path, threshold: f64) {
    let update = std::env::var("UPDATE_GOLDENS").is_ok_and(|v| v == "1");

    if update || !golden_path.exists() {
        rendered.save(golden_path).expect("failed to save golden");
        if update {
            eprintln!("Updated golden: {}", golden_path.display());
        } else {
            eprintln!("Created golden: {}", golden_path.display());
        }
        return;
    }

    let golden = image::open(golden_path)
        .expect("failed to load golden image")
        .to_rgba8();

    let result = ssim_compare(rendered, &golden, threshold);

    if !result.passed {
        // Save failure artifacts
        let stem = golden_path.with_extension("");
        let fail_path = format!("{}_FAIL.png", stem.display());
        rendered.save(&fail_path).expect("failed to save fail image");
        eprintln!(
            "Golden comparison FAILED: SSIM {:.4} < {:.4}\n  golden: {}\n  actual: {}",
            result.ssim,
            threshold,
            golden_path.display(),
            fail_path,
        );
        panic!(
            "SSIM {:.4} below threshold {:.4} for {}",
            result.ssim,
            threshold,
            golden_path.display()
        );
    }
}
