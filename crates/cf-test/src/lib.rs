//! Screenshot comparison for cyberflight.
//!
//! Compares two directories of PNG screenshots (a baseline and a candidate,
//! e.g. two runs of the `screenshot` example) image by image. See
//! [`compare_dirs`] for the rules and the `cf-test` binary for the CLI.

pub mod compare;

pub use compare::{
    CompareError, CompareOptions, DEFAULT_SSIM_MIN, Entry, Report, Verdict, compare_dirs,
    compare_images, diff_image,
};
