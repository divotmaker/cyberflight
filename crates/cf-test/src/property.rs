use cf_render::readback::FrameBuffers;

/// Assert that the frame is not entirely black.
///
/// # Panics
/// Panics if all pixels are black (RGB all zero).
pub fn assert_not_all_black(fb: &FrameBuffers) {
    let has_content = fb.color.chunks_exact(4).any(|px| px[0] > 0 || px[1] > 0 || px[2] > 0);
    assert!(has_content, "frame is entirely black — nothing was rendered");
}

/// Assert that the background is predominantly dark.
///
/// At least `min_fraction` of pixels should have all RGB channels below `threshold`.
///
/// # Panics
/// Panics if fewer than `min_fraction` pixels are dark.
pub fn assert_dark_background(fb: &FrameBuffers, threshold: u8, min_fraction: f64) {
    let total = (fb.width * fb.height) as usize;
    let dark_count = fb
        .color
        .chunks_exact(4)
        .filter(|px| px[0] < threshold && px[1] < threshold && px[2] < threshold)
        .count();
    let fraction = dark_count as f64 / total as f64;
    assert!(
        fraction >= min_fraction,
        "background not dark enough: {:.1}% dark pixels (need {:.1}%)",
        fraction * 100.0,
        min_fraction * 100.0
    );
}

/// Assert that bright pixels exist in the frame (neon elements are rendering).
///
/// At least `min_count` pixels should have any RGB channel above `threshold`.
///
/// # Panics
/// Panics if fewer than `min_count` bright pixels exist.
pub fn assert_has_bright_pixels(fb: &FrameBuffers, threshold: u8, min_count: usize) {
    let bright_count = fb
        .color
        .chunks_exact(4)
        .filter(|px| px[0] > threshold || px[1] > threshold || px[2] > threshold)
        .count();
    assert!(
        bright_count >= min_count,
        "not enough bright pixels: {bright_count} (need {min_count})"
    );
}

/// Assert that a specific screen region contains pixels of approximately the expected color.
///
/// Region is defined by pixel coordinates (x0, y0, x1, y1).
/// Color match uses per-channel tolerance.
///
/// # Panics
/// Panics if no pixels in the region match the expected color.
pub fn assert_region_has_color(
    fb: &FrameBuffers,
    region: [u32; 4], // [x0, y0, x1, y1]
    expected: [u8; 3], // [R, G, B]
    tolerance: u8,
) {
    let [x0, y0, x1, y1] = region;
    let [expected_r, expected_g, expected_b] = expected;
    let mut found = false;
    for y in y0..y1.min(fb.height) {
        for x in x0..x1.min(fb.width) {
            let px = fb.pixel(x, y);
            if (px[0] as i16 - expected_r as i16).unsigned_abs() <= tolerance as u16
                && (px[1] as i16 - expected_g as i16).unsigned_abs() <= tolerance as u16
                && (px[2] as i16 - expected_b as i16).unsigned_abs() <= tolerance as u16
            {
                found = true;
                break;
            }
        }
        if found {
            break;
        }
    }
    assert!(
        found,
        "no pixels matching RGB({expected_r},{expected_g},{expected_b}) +/-{tolerance} in region ({x0},{y0})-({x1},{y1})"
    );
}
