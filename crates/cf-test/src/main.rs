//! `cf-test`: compare two directories of cyberflight screenshots.
//!
//! ```text
//! cf-test compare <baseline_dir> <candidate_dir> [--ssim-min <score>] [--diff <dir>]
//! ```
//!
//! Prints one line per baseline PNG and exits with status 0 when every image
//! passes, 1 when any image fails or is missing, and 2 on a usage or I/O
//! error.

use std::path::PathBuf;
use std::process::ExitCode;

use cf_test::{CompareOptions, DEFAULT_SSIM_MIN, Report, Verdict, compare_dirs};

fn usage() -> String {
    format!(
        "\
usage: cf-test compare <baseline_dir> <candidate_dir> [--ssim-min <score>] [--diff <dir>]

For every PNG in <baseline_dir>, compares the file of the same name in
<candidate_dir>: byte-identical, pixel-identical, similarity >= --ssim-min
(default {DEFAULT_SSIM_MIN}), or FAIL. Missing images fail; extra candidate
images are listed but do not fail. --diff writes an amplified difference
image for every image whose pixels differ."
    )
}

struct Args {
    baseline: PathBuf,
    candidate: PathBuf,
    options: CompareOptions,
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    match args.next().as_deref() {
        Some("compare") => {}
        Some(other) => return Err(format!("unknown command '{other}'")),
        None => return Err("missing command".to_owned()),
    }
    let mut dirs = Vec::new();
    let mut options = CompareOptions::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ssim-min" => {
                let value = args.next().ok_or("--ssim-min needs a value")?;
                let score: f64 = value
                    .parse()
                    .map_err(|_| format!("--ssim-min: '{value}' is not a number"))?;
                if !(0.0..=1.0).contains(&score) {
                    return Err(format!("--ssim-min: {score} is outside 0..=1"));
                }
                options.ssim_min = score;
            }
            "--diff" => {
                options.diff_dir = Some(args.next().ok_or("--diff needs a directory")?.into());
            }
            flag if flag.starts_with('-') => return Err(format!("unknown option '{flag}'")),
            _ => dirs.push(PathBuf::from(arg)),
        }
    }
    let [baseline, candidate]: [PathBuf; 2] = dirs
        .try_into()
        .map_err(|_| "expected <baseline_dir> and <candidate_dir>".to_owned())?;
    Ok(Args {
        baseline,
        candidate,
        options,
    })
}

fn describe(verdict: &Verdict, ssim_min: f64) -> String {
    match verdict {
        Verdict::Identical => "byte-identical".to_owned(),
        Verdict::PixelIdentical => "pixel-identical".to_owned(),
        Verdict::Similar {
            ssim,
            changed_pixels,
        } => format!("similar   ssim {ssim:.6} >= {ssim_min}, {changed_pixels} px changed"),
        Verdict::Fail {
            ssim,
            changed_pixels,
        } => format!("FAIL      ssim {ssim:.6} < {ssim_min}, {changed_pixels} px changed"),
        Verdict::SizeMismatch {
            baseline,
            candidate,
        } => format!(
            "FAIL      size {}x{} vs {}x{}",
            baseline.0, baseline.1, candidate.0, candidate.1
        ),
        Verdict::Missing => "FAIL      missing from candidate".to_owned(),
        Verdict::Unreadable(e) => format!("FAIL      {e}"),
    }
}

fn print_report(report: &Report, ssim_min: f64) {
    let width = report
        .entries
        .iter()
        .map(|e| e.name.len())
        .max()
        .unwrap_or(0);
    for entry in &report.entries {
        println!(
            "{:width$}  {}",
            entry.name,
            describe(&entry.verdict, ssim_min)
        );
    }
    for name in &report.extra {
        println!("{name:width$}  extra (not in baseline)");
    }
    let total = report.entries.len();
    let failed = report.failures();
    if failed == 0 {
        println!("PASS: {total} of {total} images match");
    } else {
        println!("FAIL: {failed} of {total} images do not match");
    }
}

fn main() -> ExitCode {
    if std::env::args().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        return ExitCode::SUCCESS;
    }
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("cf-test: {e}\n\n{}", usage());
            return ExitCode::from(2);
        }
    };
    match compare_dirs(&args.baseline, &args.candidate, &args.options) {
        Ok(report) => {
            print_report(&report, args.options.ssim_min);
            if report.passed() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("cf-test: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_args(args.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn parses_compare_with_options() {
        let args = parse(&["compare", "a", "b", "--ssim-min", "0.99", "--diff", "d"]).unwrap();
        assert_eq!(args.baseline, PathBuf::from("a"));
        assert_eq!(args.candidate, PathBuf::from("b"));
        assert!((args.options.ssim_min - 0.99).abs() < 1e-12);
        assert_eq!(args.options.diff_dir, Some(PathBuf::from("d")));
    }

    #[test]
    fn defaults_without_options() {
        let args = parse(&["compare", "a", "b"]).unwrap();
        assert!((args.options.ssim_min - DEFAULT_SSIM_MIN).abs() < 1e-12);
        assert_eq!(args.options.diff_dir, None);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(&[]).is_err());
        assert!(parse(&["diff", "a", "b"]).is_err());
        assert!(parse(&["compare", "a"]).is_err());
        assert!(parse(&["compare", "a", "b", "c"]).is_err());
        assert!(parse(&["compare", "a", "b", "--ssim-min"]).is_err());
        assert!(parse(&["compare", "a", "b", "--ssim-min", "x"]).is_err());
        assert!(parse(&["compare", "a", "b", "--ssim-min", "1.5"]).is_err());
        assert!(parse(&["compare", "a", "b", "--verbose"]).is_err());
    }
}
