//! Image and directory comparison.

use std::fs;
use std::path::{Path, PathBuf};

use image::{Rgb, RgbImage, RgbaImage};

/// Default minimum similarity for an image that is not pixel-identical.
pub const DEFAULT_SSIM_MIN: f64 = 0.999;

/// Gain applied to per-channel differences in diff images, so that
/// one-step differences are visible.
const DIFF_GAIN: u16 = 32;

/// Errors that stop a directory comparison.
#[derive(Debug, thiserror::Error)]
pub enum CompareError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{}: the diff directory must differ from both input directories", .0.display())]
    DiffDirIsInput(PathBuf),
    #[error("{path}: {source}")]
    Image {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
}

/// Options for [`compare_dirs`].
#[derive(Debug, Clone)]
pub struct CompareOptions {
    /// Minimum similarity score (0..=1) for an image that differs in pixels.
    pub ssim_min: f64,
    /// Write an amplified diff image here for every image that differs.
    pub diff_dir: Option<PathBuf>,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self {
            ssim_min: DEFAULT_SSIM_MIN,
            diff_dir: None,
        }
    }
}

/// Outcome for one baseline image.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The two files are byte-for-byte equal.
    Identical,
    /// The files differ but decode to the same pixels.
    PixelIdentical,
    /// Pixels differ, similarity at or above the threshold.
    Similar { ssim: f64, changed_pixels: u64 },
    /// Pixels differ, similarity below the threshold.
    Fail { ssim: f64, changed_pixels: u64 },
    /// The images have different dimensions.
    SizeMismatch {
        baseline: (u32, u32),
        candidate: (u32, u32),
    },
    /// The candidate directory has no image of this name.
    Missing,
    /// One of the two files could not be read or decoded.
    Unreadable(String),
}

impl Verdict {
    /// Whether this verdict passes the comparison.
    #[must_use]
    pub fn passed(&self) -> bool {
        matches!(
            self,
            Self::Identical | Self::PixelIdentical | Self::Similar { .. }
        )
    }
}

/// One baseline image and its verdict.
#[derive(Debug, Clone)]
pub struct Entry {
    /// File name, relative to the compared directories.
    pub name: String,
    pub verdict: Verdict,
}

/// Result of comparing two screenshot directories.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// One entry per baseline PNG, sorted by name.
    pub entries: Vec<Entry>,
    /// PNGs in the candidate directory that the baseline does not have.
    pub extra: Vec<String>,
}

impl Report {
    /// Whether every baseline image passed. Extra candidate images do not
    /// fail the comparison.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.entries.iter().all(|e| e.verdict.passed())
    }

    /// Number of entries that failed.
    #[must_use]
    pub fn failures(&self) -> usize {
        self.entries.iter().filter(|e| !e.verdict.passed()).count()
    }
}

/// Compare two decoded images.
///
/// Returns [`Verdict::PixelIdentical`] when every RGBA value matches. Otherwise
/// the score is `image-compare`'s hybrid similarity on the RGB channels (MSSIM
/// on luma, RMS on chroma; 1.0 = identical), and the changed-pixel count
/// covers all four channels.
#[must_use]
pub fn compare_images(baseline: &RgbaImage, candidate: &RgbaImage, ssim_min: f64) -> Verdict {
    if baseline.dimensions() != candidate.dimensions() {
        return Verdict::SizeMismatch {
            baseline: baseline.dimensions(),
            candidate: candidate.dimensions(),
        };
    }
    let changed_pixels = baseline
        .pixels()
        .zip(candidate.pixels())
        .filter(|(a, b)| a != b)
        .count() as u64;
    if changed_pixels == 0 {
        return Verdict::PixelIdentical;
    }
    let ssim = match image_compare::rgb_hybrid_compare(&to_rgb(baseline), &to_rgb(candidate)) {
        Ok(similarity) => similarity.score,
        Err(e) => return Verdict::Unreadable(e.to_string()),
    };
    if ssim >= ssim_min {
        Verdict::Similar {
            ssim,
            changed_pixels,
        }
    } else {
        Verdict::Fail {
            ssim,
            changed_pixels,
        }
    }
}

/// Absolute per-channel RGB difference, amplified so small changes are
/// visible. Unchanged pixels are black. Both images must have the same size.
#[must_use]
pub fn diff_image(baseline: &RgbaImage, candidate: &RgbaImage) -> RgbImage {
    let (w, h) = baseline.dimensions();
    let mut out = RgbImage::new(w, h);
    for ((a, b), d) in baseline
        .pixels()
        .zip(candidate.pixels())
        .zip(out.pixels_mut())
    {
        let ch = |i: usize| {
            let diff = u16::from(a[i].abs_diff(b[i]));
            u8::try_from((diff * DIFF_GAIN).min(255)).unwrap_or(u8::MAX)
        };
        // Alpha-only changes show as grey, so they do not vanish.
        let alpha = ch(3);
        *d = Rgb([ch(0).max(alpha), ch(1).max(alpha), ch(2).max(alpha)]);
    }
    out
}

/// Compare every PNG in `baseline_dir` with the file of the same name in
/// `candidate_dir`.
///
/// A file that is byte-identical is [`Verdict::Identical`] without being
/// decoded. Files that differ are decoded and compared with
/// [`compare_images`]. With [`CompareOptions::diff_dir`] set, a diff image
/// (see [`diff_image`]) is written for every same-size pair whose pixels
/// differ.
///
/// # Errors
/// Returns [`CompareError`] if a directory cannot be listed, the diff
/// directory cannot be created or is one of the input directories, or a diff
/// image cannot be written. Problems
/// with a single image are reported as [`Verdict::Unreadable`].
pub fn compare_dirs(
    baseline_dir: &Path,
    candidate_dir: &Path,
    options: &CompareOptions,
) -> Result<Report, CompareError> {
    let baseline_names = png_names(baseline_dir)?;
    let candidate_names = png_names(candidate_dir)?;
    if let Some(dir) = &options.diff_dir {
        fs::create_dir_all(dir).map_err(|source| CompareError::Io {
            path: dir.clone(),
            source,
        })?;
        let canonical = |p: &Path| {
            fs::canonicalize(p).map_err(|source| CompareError::Io {
                path: p.to_path_buf(),
                source,
            })
        };
        let diff = canonical(dir)?;
        if diff == canonical(baseline_dir)? || diff == canonical(candidate_dir)? {
            return Err(CompareError::DiffDirIsInput(dir.clone()));
        }
    }

    let mut report = Report {
        extra: candidate_names
            .iter()
            .filter(|n| !baseline_names.contains(n))
            .cloned()
            .collect(),
        ..Report::default()
    };

    for name in baseline_names {
        let verdict = if candidate_names.contains(&name) {
            compare_files(
                &baseline_dir.join(&name),
                &candidate_dir.join(&name),
                &name,
                options,
            )?
        } else {
            Verdict::Missing
        };
        report.entries.push(Entry { name, verdict });
    }
    Ok(report)
}

fn compare_files(
    baseline: &Path,
    candidate: &Path,
    name: &str,
    options: &CompareOptions,
) -> Result<Verdict, CompareError> {
    let bytes = |p: &Path| fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
    let (a, b) = match (bytes(baseline), bytes(candidate)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return Ok(Verdict::Unreadable(e)),
    };
    if a == b {
        return Ok(Verdict::Identical);
    }
    let decode = |data: &[u8], p: &Path| {
        image::load_from_memory(data)
            .map(|img| img.to_rgba8())
            .map_err(|e| format!("{}: {e}", p.display()))
    };
    let (a, b) = match (decode(&a, baseline), decode(&b, candidate)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return Ok(Verdict::Unreadable(e)),
    };
    let verdict = compare_images(&a, &b, options.ssim_min);
    if let (Some(dir), Verdict::Similar { .. } | Verdict::Fail { .. }) =
        (&options.diff_dir, &verdict)
    {
        let path = dir.join(name);
        diff_image(&a, &b)
            .save(&path)
            .map_err(|source| CompareError::Image { path, source })?;
    }
    Ok(verdict)
}

/// Sorted names of the `.png` files directly inside `dir`.
fn png_names(dir: &Path) -> Result<Vec<String>, CompareError> {
    let io_err = |source| CompareError::Io {
        path: dir.to_path_buf(),
        source,
    };
    let mut names = Vec::new();
    for entry in fs::read_dir(dir).map_err(io_err)? {
        let entry = entry.map_err(io_err)?;
        let path = entry.path();
        let is_png = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png"));
        if is_png && path.is_file() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                names.push(name.to_owned());
            }
        }
    }
    names.sort();
    Ok(names)
}

fn to_rgb(img: &RgbaImage) -> RgbImage {
    image::DynamicImage::ImageRgba8(img.clone()).to_rgb8()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    /// A 32×32 test card: dark background, a bright diagonal and a block,
    /// so SSIM has structure to work with.
    fn card() -> RgbaImage {
        RgbaImage::from_fn(32, 32, |x, y| {
            if x == y || (8..16).contains(&x) && (20..28).contains(&y) {
                Rgba([0, 255, 255, 255])
            } else {
                Rgba([4, 4, 8, 255])
            }
        })
    }

    #[test]
    fn identical_pixels_are_pixel_identical() {
        assert_eq!(
            compare_images(&card(), &card(), DEFAULT_SSIM_MIN),
            Verdict::PixelIdentical
        );
    }

    #[test]
    fn one_step_change_passes_with_its_pixel_count() {
        let mut b = card();
        b.put_pixel(3, 5, Rgba([5, 4, 8, 255]));
        b.put_pixel(30, 1, Rgba([4, 4, 9, 255]));
        match compare_images(&card(), &b, 0.99) {
            Verdict::Similar {
                ssim,
                changed_pixels,
            } => {
                assert_eq!(changed_pixels, 2);
                assert!(ssim > 0.99 && ssim < 1.0 + 1e-9, "ssim {ssim}");
            }
            v => panic!("expected Similar, got {v:?}"),
        }
    }

    #[test]
    fn structural_change_fails() {
        let mut b = card();
        for x in 0..32 {
            for y in 0..16 {
                b.put_pixel(x, y, Rgba([255, 0, 255, 255]));
            }
        }
        match compare_images(&card(), &b, DEFAULT_SSIM_MIN) {
            Verdict::Fail {
                ssim,
                changed_pixels,
            } => {
                assert_eq!(changed_pixels, 32 * 16);
                assert!(ssim < DEFAULT_SSIM_MIN, "ssim {ssim}");
            }
            v => panic!("expected Fail, got {v:?}"),
        }
    }

    #[test]
    fn alpha_only_change_counts_as_changed() {
        let mut b = card();
        b.put_pixel(0, 31, Rgba([4, 4, 8, 128]));
        let v = compare_images(&card(), &b, DEFAULT_SSIM_MIN);
        assert!(
            matches!(
                v,
                Verdict::Similar {
                    changed_pixels: 1,
                    ..
                }
            ),
            "{v:?}"
        );
        assert_eq!(*diff_image(&card(), &b).get_pixel(0, 31), Rgb([255; 3]));
    }

    #[test]
    fn size_mismatch_is_reported() {
        let b = RgbaImage::new(16, 32);
        assert_eq!(
            compare_images(&card(), &b, DEFAULT_SSIM_MIN),
            Verdict::SizeMismatch {
                baseline: (32, 32),
                candidate: (16, 32)
            }
        );
        assert!(
            !Verdict::SizeMismatch {
                baseline: (1, 1),
                candidate: (2, 2)
            }
            .passed()
        );
    }

    #[test]
    fn diff_image_amplifies_and_blacks_out_unchanged() {
        let mut b = card();
        b.put_pixel(3, 5, Rgba([6, 4, 8, 255]));
        let d = diff_image(&card(), &b);
        assert_eq!(*d.get_pixel(3, 5), Rgb([2 * 32, 0, 0]));
        assert_eq!(*d.get_pixel(0, 0), Rgb([0, 0, 0]));
    }

    /// A path under the system temp dir, unique to this process and `tag`,
    /// with nothing at it.
    fn temp_path(tag: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("cf-test-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        path
    }

    /// A fresh, empty directory (see [`temp_path`]).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = temp_path(tag);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn directory_report_covers_every_case() {
        let base = temp_dir("base");
        let cand = temp_dir("cand");
        let diffs = temp_path("diff");

        card().save(base.join("same.png")).unwrap();
        fs::copy(base.join("same.png"), cand.join("same.png")).unwrap();

        let mut tweaked = card();
        tweaked.put_pixel(1, 2, Rgba([5, 4, 8, 255]));
        card().save(base.join("tweaked.png")).unwrap();
        tweaked.save(cand.join("tweaked.png")).unwrap();

        card().save(base.join("gone.png")).unwrap();
        card().save(cand.join("new.png")).unwrap();
        fs::write(base.join("notes.txt"), "not an image").unwrap();

        let options = CompareOptions {
            ssim_min: 0.99,
            diff_dir: Some(diffs.clone()),
        };
        let report = compare_dirs(&base, &cand, &options).unwrap();

        let verdicts: Vec<_> = report
            .entries
            .iter()
            .map(|e| (e.name.as_str(), &e.verdict))
            .collect();
        assert_eq!(verdicts.len(), 3, "{verdicts:?}");
        assert_eq!(verdicts[0], ("gone.png", &Verdict::Missing));
        assert_eq!(verdicts[1], ("same.png", &Verdict::Identical));
        assert_eq!(verdicts[2].0, "tweaked.png");
        assert!(matches!(
            verdicts[2].1,
            Verdict::Similar {
                changed_pixels: 1,
                ..
            }
        ));
        assert_eq!(report.extra, vec!["new.png".to_owned()]);
        assert!(!report.passed(), "a missing image fails");
        assert_eq!(report.failures(), 1);
        assert!(diffs.join("tweaked.png").is_file());
        assert!(!diffs.join("same.png").exists());

        for d in [base, cand, diffs] {
            let _ = fs::remove_dir_all(d);
        }
    }

    #[test]
    fn diff_dir_equal_to_an_input_is_refused() {
        let base = temp_dir("diffin-base");
        let cand = temp_dir("diffin-cand");
        card().save(base.join("a.png")).unwrap();
        let mut tweaked = card();
        tweaked.put_pixel(1, 2, Rgba([5, 4, 8, 255]));
        tweaked.save(cand.join("a.png")).unwrap();
        let before = fs::read(cand.join("a.png")).unwrap();

        for diff_dir in [&base, &cand] {
            let options = CompareOptions {
                diff_dir: Some(diff_dir.clone()),
                ..CompareOptions::default()
            };
            let err = compare_dirs(&base, &cand, &options).unwrap_err();
            assert!(matches!(err, CompareError::DiffDirIsInput(_)), "{err}");
        }
        assert_eq!(fs::read(cand.join("a.png")).unwrap(), before);

        for d in [base, cand] {
            let _ = fs::remove_dir_all(d);
        }
    }

    #[test]
    fn reencoded_png_is_pixel_identical() {
        let base = temp_dir("reenc-base");
        let cand = temp_dir("reenc-cand");
        card().save(base.join("a.png")).unwrap();
        // Same pixels, different file bytes: RGBA vs RGB encoding.
        image::DynamicImage::ImageRgba8(card())
            .to_rgb8()
            .save(cand.join("a.png"))
            .unwrap();
        let report = compare_dirs(&base, &cand, &CompareOptions::default()).unwrap();
        assert_eq!(report.entries[0].verdict, Verdict::PixelIdentical);
        assert!(report.passed());
        for d in [base, cand] {
            let _ = fs::remove_dir_all(d);
        }
    }

    #[test]
    fn missing_directory_is_an_error() {
        let err = compare_dirs(
            Path::new("/nonexistent/cf-test-baseline"),
            Path::new("."),
            &CompareOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, CompareError::Io { .. }));
    }
}
