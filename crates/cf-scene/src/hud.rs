use serde::{Deserialize, Serialize};

use crate::grid::GridVertex;
use crate::hud_font;

/// Screen-space rectangle for HUD panel positioning.
/// Coordinates are normalized: (0,0) = top-left, (1,1) = bottom-right.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// A HUD text field with a label and a value binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextField {
    pub label: String,
    pub unit: String,
    pub decimals: u8,
}

/// A HUD panel that displays a collection of stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Panel {
    pub rect: Rect,
    pub title: String,
    pub fields: Vec<TextField>,
}

/// Complete HUD layout for the driving range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HudLayout {
    pub panels: Vec<Panel>,
}

impl HudLayout {
    /// Default driving range HUD layout.
    #[must_use]
    pub fn driving_range() -> Self {
        Self {
            panels: vec![
                // Left panel: last shot stats
                Panel {
                    rect: Rect {
                        x: 0.01,
                        y: 0.15,
                        w: 0.18,
                        h: 0.50,
                    },
                    title: "Last Shot".to_owned(),
                    fields: vec![
                        TextField {
                            label: "Carry".to_owned(),
                            unit: "yds".to_owned(),
                            decimals: 1,
                        },
                        TextField {
                            label: "Apex".to_owned(),
                            unit: "ft".to_owned(),
                            decimals: 0,
                        },
                        TextField {
                            label: "Lateral".to_owned(),
                            unit: "yds".to_owned(),
                            decimals: 1,
                        },
                        TextField {
                            label: "Descent".to_owned(),
                            unit: "deg".to_owned(),
                            decimals: 1,
                        },
                        TextField {
                            label: "Flight".to_owned(),
                            unit: "s".to_owned(),
                            decimals: 1,
                        },
                    ],
                },
                // Right panel: session averages
                Panel {
                    rect: Rect {
                        x: 0.81,
                        y: 0.15,
                        w: 0.18,
                        h: 0.40,
                    },
                    title: "Session".to_owned(),
                    fields: vec![
                        TextField {
                            label: "Shots".to_owned(),
                            unit: "".to_owned(),
                            decimals: 0,
                        },
                        TextField {
                            label: "Avg Carry".to_owned(),
                            unit: "yds".to_owned(),
                            decimals: 1,
                        },
                        TextField {
                            label: "Carry SD".to_owned(),
                            unit: "yds".to_owned(),
                            decimals: 1,
                        },
                        TextField {
                            label: "Lateral SD".to_owned(),
                            unit: "yds".to_owned(),
                            decimals: 1,
                        },
                    ],
                },
                // Top bar: mode and shot count
                Panel {
                    rect: Rect {
                        x: 0.30,
                        y: 0.01,
                        w: 0.40,
                        h: 0.06,
                    },
                    title: "Free Practice".to_owned(),
                    fields: Vec::new(),
                },
            ],
        }
    }
}

// ── Telemetry HUD ──────────────────────────────────────────────────────

/// Imperial vs metric display units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitSystem {
    /// Distance in yards, height in feet, speed in MPH.
    Imperial,
    /// Everything in meters, speed in KPH.
    Metric,
}

/// Shot telemetry data for HUD display.
#[derive(Debug, Clone)]
pub struct ShotTelemetry {
    pub club_name: String,
    pub club_speed_mph: f64,
    pub ball_speed_mph: f64,
    pub launch_angle_deg: f64,
    pub launch_azimuth_deg: f64,
    pub backspin_rpm: f64,
    pub sidespin_rpm: f64,
    pub apex_m: f64,
    pub carry_yards: f64,
    pub lateral_m: f64,
    pub flight_time_s: f64,
    pub elapsed_s: f64,
    pub in_flight: bool,
}

/// Pre-built HUD geometry for one frame.
pub struct HudGeometry {
    /// LINE_LIST vertices for text and decorative lines.
    pub lines: Vec<GridVertex>,
    /// TRIANGLE_LIST vertices for panel background.
    pub fills: Vec<GridVertex>,
}

// Layout constants (pixels at 1080p).
const CHAR_HEIGHT: f32 = 16.0;
const LINE_HEIGHT: f32 = 22.0;
const SECTION_GAP: f32 = 14.0;
const MARGIN_LEFT: f32 = 24.0;
const MARGIN_TOP: f32 = 40.0;
const LABEL_X: f32 = 36.0;
const PANEL_FRAC: f32 = 0.20;
const LABEL_FADE: f32 = 0.35;
const VALUE_FADE: f32 = 0.0;
const HEADER_FADE: f32 = 0.0;
const DECOR_FADE: f32 = 0.55;

/// Build HUD geometry for the telemetry overlay.
///
/// Returns line vertices (for text + decorations) and fill vertices (panel background).
/// Designed for the left 20% of the screen with a telemetry/rocket-monitoring feel.
#[must_use]
pub fn build_hud(
    telemetry: Option<&ShotTelemetry>,
    units: UnitSystem,
    screen_w: f32,
    screen_h: f32,
) -> HudGeometry {
    let panel_w = screen_w * PANEL_FRAC;
    let mut lines: Vec<GridVertex> = Vec::with_capacity(4096);

    let fills = build_panel_fill(panel_w, screen_h);

    // Vertical separator at panel edge
    let sep = hud_font::build_vline(panel_w, 0.0, screen_h, DECOR_FADE);
    lines.extend_from_slice(&sep);

    let mut y = MARGIN_TOP;

    if let Some(t) = telemetry {
        // Unit labels and conversions
        let (spd_label, spd_factor) = match units {
            UnitSystem::Imperial => ("MPH", 1.0),
            UnitSystem::Metric => ("KPH", 1.60934),
        };
        let fmt_dist = |yards: f64| match units {
            UnitSystem::Imperial => format!("{:.0} YDS", yards),
            UnitSystem::Metric => format!("{:.0} M", yards * 0.9144),
        };
        let fmt_height = |meters: f64| match units {
            UnitSystem::Imperial => format!("{:.1} FT", meters / 0.3048),
            UnitSystem::Metric => format!("{:.1} M", meters),
        };

        // ── CLUB section
        y = emit_section_header(&mut lines, "CLUB", y, panel_w);
        y = emit_row(&mut lines, "TYPE", &t.club_name, y);
        y = emit_row(
            &mut lines,
            "SPEED",
            &format!("{:.0} {spd_label}", t.club_speed_mph * spd_factor),
            y,
        );
        y += SECTION_GAP;

        // ── IMPACT section
        y = emit_section_header(&mut lines, "IMPACT", y, panel_w);
        y = emit_row(
            &mut lines,
            "BALL SPD",
            &format!("{:.1} {spd_label}", t.ball_speed_mph * spd_factor),
            y,
        );
        y = emit_row(&mut lines, "VLA", &format!("{:.1}", t.launch_angle_deg), y);
        y = emit_row(&mut lines, "HLA", &format!("{:.1}", t.launch_azimuth_deg), y);
        y = emit_row(&mut lines, "BACK", &format!("{:.0} RPM", t.backspin_rpm), y);
        y = emit_row(&mut lines, "SIDE", &format!("{:.0} RPM", t.sidespin_rpm), y);
        y += SECTION_GAP;

        // ── RESULT section
        y = emit_section_header(&mut lines, "RESULT", y, panel_w);
        y = emit_row(&mut lines, "APEX", &fmt_height(t.apex_m), y);
        y = emit_row(&mut lines, "CARRY", &fmt_dist(t.carry_yards), y);
        y = emit_row(&mut lines, "TOTAL", &fmt_dist(t.carry_yards), y);
        y = emit_row(&mut lines, "LATERAL", &fmt_height(t.lateral_m), y);
        y = emit_row(&mut lines, "TIME", &format!("{:.1} S", t.elapsed_s), y);
        y += SECTION_GAP;

        // Status line
        let status = if t.in_flight { "IN FLIGHT" } else { "LANDED" };
        let status_fade = if t.in_flight { 0.0 } else { LABEL_FADE };
        lines.extend(hud_font::build_text(status, MARGIN_LEFT, y, CHAR_HEIGHT, status_fade));
    } else {
        // No telemetry — show standby
        lines.extend(hud_font::build_text("STANDBY", MARGIN_LEFT, y, CHAR_HEIGHT, LABEL_FADE));
    }

    HudGeometry { lines, fills }
}

fn emit_section_header(lines: &mut Vec<GridVertex>, title: &str, y: f32, panel_w: f32) -> f32 {
    lines.extend(hud_font::build_text(title, MARGIN_LEFT, y, CHAR_HEIGHT, HEADER_FADE));
    // Decorative line from end of title text to near panel edge
    let line_start = MARGIN_LEFT + hud_font::text_width(title, CHAR_HEIGHT) + 8.0;
    let line_end = panel_w - 16.0;
    if line_end > line_start + 10.0 {
        let sep = hud_font::build_hline(line_start, line_end, y + CHAR_HEIGHT * 0.5, DECOR_FADE);
        lines.extend_from_slice(&sep);
    }
    y + LINE_HEIGHT
}

fn emit_row(lines: &mut Vec<GridVertex>, label: &str, value: &str, y: f32) -> f32 {
    lines.extend(hud_font::build_text(label, LABEL_X, y, CHAR_HEIGHT, LABEL_FADE));
    // Right-align value: compute value width, position so it ends near the panel margin
    let panel_right = 384.0 - 16.0; // approximate at 1920 width
    let value_w = hud_font::text_width(value, CHAR_HEIGHT);
    let value_x = (panel_right - value_w).max(LABEL_X + hud_font::text_width(label, CHAR_HEIGHT) + 16.0);
    lines.extend(hud_font::build_text(value, value_x, y, CHAR_HEIGHT, VALUE_FADE));
    y + LINE_HEIGHT
}

fn build_panel_fill(panel_w: f32, screen_h: f32) -> Vec<GridVertex> {
    let v = |x: f32, y: f32| GridVertex {
        position: [x, y, 0.0],
        fade: 0.0,
    };
    vec![
        v(0.0, 0.0), v(panel_w, 0.0), v(panel_w, screen_h),
        v(0.0, 0.0), v(panel_w, screen_h), v(0.0, screen_h),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_has_panels() {
        let layout = HudLayout::driving_range();
        assert_eq!(layout.panels.len(), 3);
    }

    #[test]
    fn panels_within_screen() {
        let layout = HudLayout::driving_range();
        for panel in &layout.panels {
            assert!(panel.rect.x >= 0.0 && panel.rect.x <= 1.0);
            assert!(panel.rect.y >= 0.0 && panel.rect.y <= 1.0);
            assert!(panel.rect.x + panel.rect.w <= 1.01);
            assert!(panel.rect.y + panel.rect.h <= 1.01);
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let layout = HudLayout::driving_range();
        let json = serde_json::to_string(&layout).unwrap();
        let layout2: HudLayout = serde_json::from_str(&json).unwrap();
        assert_eq!(layout.panels.len(), layout2.panels.len());
    }

    #[test]
    fn build_hud_with_telemetry() {
        let t = ShotTelemetry {
            club_name: "7I".to_owned(),
            club_speed_mph: 90.0,
            ball_speed_mph: 132.0,
            launch_angle_deg: 16.0,
            launch_azimuth_deg: -1.2,
            backspin_rpm: 7000.0,
            sidespin_rpm: -300.0,
            apex_m: 28.3,
            carry_yards: 159.0,
            lateral_m: -4.2,
            flight_time_s: 5.8,
            elapsed_s: 3.2,
            in_flight: true,
        };
        let geom = build_hud(Some(&t), UnitSystem::Imperial, 1920.0, 1080.0);
        assert!(!geom.lines.is_empty(), "should have text lines");
        assert_eq!(geom.lines.len() % 2, 0, "LINE_LIST needs even count");
        assert_eq!(geom.fills.len(), 6, "panel is 2 triangles");
    }

    #[test]
    fn build_hud_without_telemetry() {
        let geom = build_hud(None, UnitSystem::Imperial, 1920.0, 1080.0);
        assert!(!geom.lines.is_empty(), "should show standby text");
    }

    #[test]
    fn panel_fill_covers_left_fifth() {
        let geom = build_hud(None, UnitSystem::Imperial, 1920.0, 1080.0);
        let max_x = geom.fills.iter().map(|v| v.position[0]).fold(0.0_f32, f32::max);
        assert!((max_x - 384.0).abs() < 1.0, "panel should be 20% of 1920 = 384, got {max_x}");
    }
}
