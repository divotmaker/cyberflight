use cf_render::readback::FrameBuffers;
use image::RgbaImage;

/// Result of an SSIM comparison.
#[derive(Debug)]
pub struct CompareResult {
    pub ssim: f64,
    pub passed: bool,
}

/// Convert a readback framebuffer to an image::RgbaImage.
#[must_use]
pub fn framebuffer_to_image(fb: &FrameBuffers) -> RgbaImage {
    RgbaImage::from_raw(fb.width, fb.height, fb.color.clone())
        .expect("framebuffer dimensions match color data length")
}

/// Compare two images using SSIM.
#[must_use]
pub fn ssim_compare(a: &RgbaImage, b: &RgbaImage, threshold: f64) -> CompareResult {
    let result = image_compare::rgba_hybrid_compare(a, b)
        .expect("images must have same dimensions for SSIM comparison");
    let ssim = result.score;
    CompareResult {
        ssim,
        passed: ssim >= threshold,
    }
}
