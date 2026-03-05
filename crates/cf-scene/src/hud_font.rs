//! Vector font for HUD text rendering.
//!
//! Each glyph is defined as line segments in a 5×7 grid (x: 0–4, y: 0–6).
//! (0,0) is top-left. Characters are rendered as LINE_LIST vertices using
//! the existing grid pipeline with an orthographic projection.

use crate::grid::GridVertex;

/// Returns line segments for a glyph as `[x1, y1, x2, y2]` in a 5×7 grid.
#[must_use]
fn glyph_segments(ch: char) -> &'static [[f32; 4]] {
    match ch {
        // Digits
        '0' => &[
            [0., 0., 4., 0.], [4., 0., 4., 6.],
            [4., 6., 0., 6.], [0., 6., 0., 0.],
            [1., 5., 3., 1.],
        ],
        '1' => &[
            [1., 1., 2., 0.], [2., 0., 2., 6.],
            [1., 6., 3., 6.],
        ],
        '2' => &[
            [0., 0., 4., 0.], [4., 0., 4., 3.],
            [4., 3., 0., 3.], [0., 3., 0., 6.],
            [0., 6., 4., 6.],
        ],
        '3' => &[
            [0., 0., 4., 0.], [4., 0., 4., 6.],
            [4., 6., 0., 6.], [1., 3., 4., 3.],
        ],
        '4' => &[
            [0., 0., 0., 3.], [0., 3., 4., 3.],
            [4., 0., 4., 6.],
        ],
        '5' => &[
            [4., 0., 0., 0.], [0., 0., 0., 3.],
            [0., 3., 4., 3.], [4., 3., 4., 6.],
            [4., 6., 0., 6.],
        ],
        '6' => &[
            [4., 0., 0., 0.], [0., 0., 0., 6.],
            [0., 6., 4., 6.], [4., 6., 4., 3.],
            [4., 3., 0., 3.],
        ],
        '7' => &[
            [0., 0., 4., 0.], [4., 0., 1., 6.],
        ],
        '8' => &[
            [0., 0., 4., 0.], [4., 0., 4., 6.],
            [4., 6., 0., 6.], [0., 6., 0., 0.],
            [0., 3., 4., 3.],
        ],
        '9' => &[
            [4., 3., 0., 3.], [0., 3., 0., 0.],
            [0., 0., 4., 0.], [4., 0., 4., 6.],
            [4., 6., 0., 6.],
        ],

        // Letters
        'A' => &[
            [0., 6., 0., 2.], [0., 2., 2., 0.],
            [2., 0., 4., 2.], [4., 2., 4., 6.],
            [0., 3., 4., 3.],
        ],
        'B' => &[
            [0., 0., 0., 6.], [0., 0., 3., 0.],
            [3., 0., 4., 1.], [4., 1., 3., 3.],
            [3., 3., 0., 3.], [3., 3., 4., 4.5],
            [4., 4.5, 3., 6.], [3., 6., 0., 6.],
        ],
        'C' => &[
            [4., 0., 0., 0.], [0., 0., 0., 6.],
            [0., 6., 4., 6.],
        ],
        'D' => &[
            [0., 0., 0., 6.], [0., 0., 3., 0.],
            [3., 0., 4., 2.], [4., 2., 4., 4.],
            [4., 4., 3., 6.], [3., 6., 0., 6.],
        ],
        'E' => &[
            [4., 0., 0., 0.], [0., 0., 0., 6.],
            [0., 6., 4., 6.], [0., 3., 3., 3.],
        ],
        'F' => &[
            [4., 0., 0., 0.], [0., 0., 0., 6.],
            [0., 3., 3., 3.],
        ],
        'G' => &[
            [4., 1., 4., 0.], [4., 0., 0., 0.],
            [0., 0., 0., 6.], [0., 6., 4., 6.],
            [4., 6., 4., 3.], [4., 3., 2., 3.],
        ],
        'H' => &[
            [0., 0., 0., 6.], [4., 0., 4., 6.],
            [0., 3., 4., 3.],
        ],
        'I' => &[
            [1., 0., 3., 0.], [2., 0., 2., 6.],
            [1., 6., 3., 6.],
        ],
        'J' => &[
            [1., 0., 4., 0.], [4., 0., 4., 6.],
            [4., 6., 1., 6.], [1., 6., 0., 5.],
        ],
        'K' => &[
            [0., 0., 0., 6.], [4., 0., 0., 3.],
            [0., 3., 4., 6.],
        ],
        'L' => &[
            [0., 0., 0., 6.], [0., 6., 4., 6.],
        ],
        'M' => &[
            [0., 6., 0., 0.], [0., 0., 2., 3.],
            [2., 3., 4., 0.], [4., 0., 4., 6.],
        ],
        'N' => &[
            [0., 6., 0., 0.], [0., 0., 4., 6.],
            [4., 6., 4., 0.],
        ],
        'O' => &[
            [0., 0., 4., 0.], [4., 0., 4., 6.],
            [4., 6., 0., 6.], [0., 6., 0., 0.],
        ],
        'P' => &[
            [0., 6., 0., 0.], [0., 0., 4., 0.],
            [4., 0., 4., 3.], [4., 3., 0., 3.],
        ],
        'Q' => &[
            [0., 0., 4., 0.], [4., 0., 4., 5.],
            [4., 5., 3., 6.], [3., 6., 0., 6.],
            [0., 6., 0., 0.], [3., 5., 4.5, 6.5],
        ],
        'R' => &[
            [0., 6., 0., 0.], [0., 0., 4., 0.],
            [4., 0., 4., 3.], [4., 3., 0., 3.],
            [2., 3., 4., 6.],
        ],
        'S' => &[
            [4., 0., 0., 0.], [0., 0., 0., 3.],
            [0., 3., 4., 3.], [4., 3., 4., 6.],
            [4., 6., 0., 6.],
        ],
        'T' => &[
            [0., 0., 4., 0.], [2., 0., 2., 6.],
        ],
        'U' => &[
            [0., 0., 0., 6.], [0., 6., 4., 6.],
            [4., 6., 4., 0.],
        ],
        'V' => &[
            [0., 0., 2., 6.], [2., 6., 4., 0.],
        ],
        'W' => &[
            [0., 0., 1., 6.], [1., 6., 2., 3.],
            [2., 3., 3., 6.], [3., 6., 4., 0.],
        ],
        'X' => &[
            [0., 0., 4., 6.], [4., 0., 0., 6.],
        ],
        'Y' => &[
            [0., 0., 2., 3.], [4., 0., 2., 3.],
            [2., 3., 2., 6.],
        ],
        'Z' => &[
            [0., 0., 4., 0.], [4., 0., 0., 6.],
            [0., 6., 4., 6.],
        ],

        // Symbols
        '.' => &[[1.5, 5.5, 2.5, 5.5], [2.5, 5.5, 2.5, 6.], [2.5, 6., 1.5, 6.], [1.5, 6., 1.5, 5.5]],
        ':' => &[[2., 1., 2., 2.], [2., 4., 2., 5.]],
        '-' => &[[1., 3., 3., 3.]],
        '+' => &[[1., 3., 3., 3.], [2., 2., 2., 4.]],
        '/' => &[[0., 6., 4., 0.]],

        _ => &[], // space and unknown
    }
}

/// Compute text width in pixels.
#[must_use]
pub fn text_width(text: &str, char_height: f32) -> f32 {
    let scale = char_height / 7.0;
    let advance = 5.0 * scale + scale; // char_width + gap
    text.len() as f32 * advance
}

/// Build LINE_LIST vertices for a text string.
///
/// `x`, `y` are the top-left pixel position. `char_height` is in pixels.
/// `fade` controls brightness (0.0 = full bright, higher = dimmer).
#[must_use]
pub fn build_text(text: &str, x: f32, y: f32, char_height: f32, fade: f32) -> Vec<GridVertex> {
    let scale = char_height / 7.0;
    let advance = 5.0 * scale + scale;

    let mut verts = Vec::new();
    let mut cursor_x = x;

    for ch in text.chars() {
        let segs = glyph_segments(ch.to_ascii_uppercase());
        for &[x1, y1, x2, y2] in segs {
            verts.push(GridVertex {
                position: [cursor_x + x1 * scale, y + y1 * scale, 0.0],
                fade,
            });
            verts.push(GridVertex {
                position: [cursor_x + x2 * scale, y + y2 * scale, 0.0],
                fade,
            });
        }
        cursor_x += advance;
    }

    verts
}

/// Build a horizontal line (LINE_LIST: 2 vertices).
#[must_use]
pub fn build_hline(x1: f32, x2: f32, y: f32, fade: f32) -> [GridVertex; 2] {
    [
        GridVertex { position: [x1, y, 0.0], fade },
        GridVertex { position: [x2, y, 0.0], fade },
    ]
}

/// Build a vertical line (LINE_LIST: 2 vertices).
#[must_use]
pub fn build_vline(x: f32, y1: f32, y2: f32, fade: f32) -> [GridVertex; 2] {
    [
        GridVertex { position: [x, y1, 0.0], fade },
        GridVertex { position: [x, y2, 0.0], fade },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_digits_have_segments() {
        for ch in '0'..='9' {
            assert!(!glyph_segments(ch).is_empty(), "digit '{ch}' missing segments");
        }
    }

    #[test]
    fn all_letters_have_segments() {
        for ch in 'A'..='Z' {
            assert!(!glyph_segments(ch).is_empty(), "letter '{ch}' missing segments");
        }
    }

    #[test]
    fn space_has_no_segments() {
        assert!(glyph_segments(' ').is_empty());
    }

    #[test]
    fn build_text_produces_line_pairs() {
        let verts = build_text("HI", 0.0, 0.0, 14.0, 0.0);
        assert_eq!(verts.len() % 2, 0, "LINE_LIST needs even vertex count");
        assert!(!verts.is_empty());
    }

    #[test]
    fn text_width_matches_char_count() {
        let w3 = text_width("ABC", 14.0);
        let w6 = text_width("ABCDEF", 14.0);
        assert!((w6 - 2.0 * w3).abs() < 0.01, "width should scale linearly with char count");
    }

    #[test]
    fn empty_text_has_no_vertices() {
        let verts = build_text("", 0.0, 0.0, 14.0, 0.0);
        assert!(verts.is_empty());
    }

    #[test]
    fn glyph_coords_in_bounds() {
        for ch in ('0'..='9').chain('A'..='Z') {
            for &[x1, y1, x2, y2] in glyph_segments(ch) {
                assert!(
                    x1 >= -0.5 && x1 <= 5.0 && y1 >= -0.5 && y1 <= 7.0
                    && x2 >= -0.5 && x2 <= 5.0 && y2 >= -0.5 && y2 <= 7.0,
                    "glyph '{ch}' has out-of-bounds segment: [{x1},{y1},{x2},{y2}]"
                );
            }
        }
    }
}
