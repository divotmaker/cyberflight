use std::collections::HashMap;

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
///
/// Distance fields (`carry_yards`, `total_yards`, `downrange_yards`) are live —
/// they increase with the ball animation each frame, not final calculated values.
#[derive(Debug, Clone)]
pub struct ShotTelemetry {
    pub club_speed_mph: f64,
    pub ball_speed_mph: f64,
    pub smash_factor: Option<f64>,
    pub launch_angle_deg: f64,
    pub launch_azimuth_deg: f64,
    pub backspin_rpm: f64,
    pub sidespin_rpm: f64,
    /// Club path (degrees, negative = left of target for RH).
    pub club_path_deg: Option<f64>,
    /// Attack angle (degrees, negative = descending blow).
    pub attack_angle_deg: Option<f64>,
    /// Face angle at impact (degrees, positive = open for RH).
    pub face_angle_deg: Option<f64>,
    /// Dynamic loft at impact (degrees).
    pub dynamic_loft_deg: Option<f64>,
    pub apex_m: f64,
    /// Live carry distance (Euclidean XZ from tee), updated each frame.
    pub carry_yards: f64,
    /// Carry distance reported by the launch monitor, if available.
    pub lm_carry_yards: Option<f64>,
    /// Live total distance (Euclidean XZ from tee), updated each frame.
    pub total_yards: f64,
    /// Live downrange distance (Z-axis only), updated each frame.
    pub downrange_yards: f64,
    pub lateral_m: f64,
    /// Carry flight time only (excludes bounce/rollout).
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

// Layout constants (reference values at 1080p, scaled by screen height).
const REF_HEIGHT: f32 = 1080.0;
const REF_CHAR_HEIGHT: f32 = 16.0;
const REF_LINE_HEIGHT: f32 = 22.0;
const REF_SECTION_GAP: f32 = 14.0;
const REF_MARGIN_LEFT: f32 = 24.0;
const REF_MARGIN_TOP: f32 = 40.0;
const REF_LABEL_X: f32 = 36.0;
const PANEL_FRAC: f32 = 0.20;
const LABEL_FADE: f32 = 0.35;
const VALUE_FADE: f32 = 0.0;
const HEADER_FADE: f32 = 0.0;
const DECOR_FADE: f32 = 0.55;

/// Scaled layout dimensions for the current screen height.
struct HudScale {
    char_height: f32,
    line_height: f32,
    section_gap: f32,
    margin_left: f32,
    margin_top: f32,
    label_x: f32,
}

impl HudScale {
    fn new(screen_h: f32) -> Self {
        let s = screen_h / REF_HEIGHT;
        Self {
            char_height: REF_CHAR_HEIGHT * s,
            line_height: REF_LINE_HEIGHT * s,
            section_gap: REF_SECTION_GAP * s,
            margin_left: REF_MARGIN_LEFT * s,
            margin_top: REF_MARGIN_TOP * s,
            label_x: REF_LABEL_X * s,
        }
    }
}

/// Build HUD geometry for the telemetry overlay.
///
/// Returns line vertices (for text + decorations) and fill vertices (panel background).
/// Designed for the left 20% of the screen with a telemetry/rocket-monitoring feel.
///
/// `lm_states`: per-source armed state map.
/// `connected`: whether the flighthook connection is alive.
pub fn build_hud(
    telemetry: Option<&ShotTelemetry>,
    units: UnitSystem,
    chase_active: bool,
    lm_states: &HashMap<String, bool>,
    connected: bool,
    screen_w: f32,
    screen_h: f32,
) -> HudGeometry {
    let sc = HudScale::new(screen_h);
    let panel_w = screen_w * PANEL_FRAC;
    let panel_margin = 16.0 * screen_h / REF_HEIGHT;
    let mut lines: Vec<GridVertex> = Vec::with_capacity(4096);

    let mut fills = build_panel_fill(0.0, panel_w, screen_h);
    // Right panel fill is hidden when chase camera is active (chase cam has priority).
    if !chase_active {
        fills.extend_from_slice(&build_panel_fill(screen_w - panel_w, screen_w, screen_h));
    }

    // Vertical separator at panel edge
    let sep = hud_font::build_vline(panel_w, 0.0, screen_h, DECOR_FADE);
    lines.extend_from_slice(&sep);

    // Vertical separator at left edge of right panel
    let right_sep_x = screen_w * (1.0 - PANEL_FRAC);
    let right_sep = hud_font::build_vline(right_sep_x, 0.0, screen_h, DECOR_FADE);
    lines.extend_from_slice(&right_sep);

    let mut y = sc.margin_top;

    if let Some(t) = telemetry {
        // Unit labels and conversions
        let (spd_label, spd_factor) = match units {
            UnitSystem::Imperial => ("MPH", 1.0),
            UnitSystem::Metric => ("KPH", 1.60934),
        };
        let fmt_dist = |yards: f64| match units {
            UnitSystem::Imperial => format!("{:.1} YDS", yards),
            UnitSystem::Metric => format!("{:.1} M", yards * 0.9144),
        };
        let fmt_height = |meters: f64| match units {
            UnitSystem::Imperial => format!("{:.1} FT", meters / 0.3048),
            UnitSystem::Metric => format!("{:.1} M", meters),
        };

        // ── CLUB section
        y = emit_section_header(&mut lines, "CLUB", y, panel_w, &sc);
        y = emit_row(&mut lines, "SPEED",
            &format!("{:.0} {spd_label}", t.club_speed_mph * spd_factor), y, panel_w, &sc);
        if let Some(v) = t.club_path_deg {
            y = emit_row(&mut lines, "PATH", &format!("{v:.1}"), y, panel_w, &sc);
        }
        if let Some(v) = t.attack_angle_deg {
            y = emit_row(&mut lines, "AOA", &format!("{v:.1}"), y, panel_w, &sc);
        }
        if let Some(v) = t.face_angle_deg {
            y = emit_row(&mut lines, "FACE", &format!("{v:.1}"), y, panel_w, &sc);
        }
        if let Some(v) = t.dynamic_loft_deg {
            y = emit_row(&mut lines, "D LOFT", &format!("{v:.1}"), y, panel_w, &sc);
        }
        y += sc.section_gap;

        // ── LAUNCH section
        y = emit_section_header(&mut lines, "LAUNCH", y, panel_w, &sc);
        y = emit_row(&mut lines, "SPEED",
            &format!("{:.1} {spd_label}", t.ball_speed_mph * spd_factor), y, panel_w, &sc);
        if let Some(sf) = t.smash_factor {
            y = emit_row(&mut lines, "SMASH", &format!("{sf:.2}"), y, panel_w, &sc);
        }
        y = emit_row(&mut lines, "VLA", &format!("{:.1}", t.launch_angle_deg), y, panel_w, &sc);
        y = emit_row(&mut lines, "HLA", &format!("{:.1}", t.launch_azimuth_deg), y, panel_w, &sc);
        y = emit_row(&mut lines, "BACK", &format!("{:.0} RPM", t.backspin_rpm), y, panel_w, &sc);
        y = emit_row(&mut lines, "SIDE", &format!("{:.0} RPM", t.sidespin_rpm), y, panel_w, &sc);
        y += sc.section_gap;

        // ── FLIGHT section
        y = emit_section_header(&mut lines, "FLIGHT", y, panel_w, &sc);
        y = emit_row(&mut lines, "APEX", &fmt_height(t.apex_m), y, panel_w, &sc);
        y = emit_row(&mut lines, "CARRY", &fmt_dist(t.carry_yards), y, panel_w, &sc);
        if let Some(lm_carry) = t.lm_carry_yards {
            let carry_complete = t.elapsed_s >= t.flight_time_s;
            let lm_text = if carry_complete {
                fmt_dist(lm_carry)
            } else {
                "-".to_owned()
            };
            y = emit_row(&mut lines, "CARRY(LM)", &lm_text, y, panel_w, &sc);
        }
        y = emit_row(&mut lines, "TIME", &format!("{:.1} S", t.elapsed_s), y, panel_w, &sc);
        y += sc.section_gap;

        // ── RESULT section
        y = emit_section_header(&mut lines, "RESULT", y, panel_w, &sc);
        y = emit_row(&mut lines, "TOTAL", &fmt_dist(t.total_yards), y, panel_w, &sc);
        y = emit_row(&mut lines, "DOWNRANGE", &fmt_dist(t.downrange_yards), y, panel_w, &sc);
        y = emit_row(&mut lines, "LATERAL", &fmt_height(t.lateral_m), y, panel_w, &sc);
        let _ = y;
    } else {
        // No telemetry — show standby
        lines.extend(hud_font::build_text("SEND IT", sc.margin_left, y, sc.char_height, LABEL_FADE));
    }

    // ── SYSTEM section (bottom-aligned in left panel)
    {
        let mut rows: Vec<(&str, &str, f32, f32)> = Vec::new(); // (label, value, label_fade, value_fade)

        if !connected {
            // Negative fade = red alert color (see grid.frag).
            rows.push(("LAUNCH MONITOR", "-", -1.0, -1.0));
        } else if lm_states.is_empty() {
            rows.push(("LAUNCH MONITOR", "UNKNOWN", LABEL_FADE, LABEL_FADE));
        } else {
            let multi = lm_states.len() > 1;
            for (source, &armed) in lm_states {
                let label = if multi { source.as_str() } else { "LAUNCH MONITOR" };
                let (status, vfade) = if armed { ("ARMED", 0.0) } else { ("STANDBY", LABEL_FADE) };
                rows.push((label, status, LABEL_FADE, vfade));
            }
        }

        let num_rows = rows.len() as f32;
        let header_y = screen_h - sc.margin_top - (num_rows * sc.line_height) - sc.line_height;
        emit_section_header(&mut lines, "SYSTEM", header_y, panel_w, &sc);

        let mut row_y = header_y + sc.line_height;
        for (label, value, label_fade, value_fade) in &rows {
            lines.extend(hud_font::build_text(label, sc.label_x, row_y, sc.char_height, *label_fade));
            let value_x = panel_w - panel_margin - hud_font::text_width(value, sc.char_height);
            lines.extend(hud_font::build_text(value, value_x, row_y, sc.char_height, *value_fade));
            row_y += sc.line_height;
        }
    }

    HudGeometry { lines, fills }
}

fn emit_section_header(lines: &mut Vec<GridVertex>, title: &str, y: f32, panel_w: f32, sc: &HudScale) -> f32 {
    lines.extend(hud_font::build_text(title, sc.margin_left, y, sc.char_height, HEADER_FADE));
    let panel_margin = 16.0 * sc.line_height / REF_LINE_HEIGHT;
    let line_start = sc.margin_left + hud_font::text_width(title, sc.char_height) + 8.0 * sc.char_height / REF_CHAR_HEIGHT;
    let line_end = panel_w - panel_margin;
    if line_end > line_start + 10.0 {
        let sep = hud_font::build_hline(line_start, line_end, y + sc.char_height * 0.5, DECOR_FADE);
        lines.extend_from_slice(&sep);
    }
    y + sc.line_height
}

fn emit_row(lines: &mut Vec<GridVertex>, label: &str, value: &str, y: f32, panel_w: f32, sc: &HudScale) -> f32 {
    lines.extend(hud_font::build_text(label, sc.label_x, y, sc.char_height, LABEL_FADE));
    let panel_margin = 16.0 * sc.line_height / REF_LINE_HEIGHT;
    let panel_right = panel_w - panel_margin;
    let value_w = hud_font::text_width(value, sc.char_height);
    let min_x = sc.label_x + hud_font::text_width(label, sc.char_height) + panel_margin;
    let value_x = (panel_right - value_w).max(min_x);
    lines.extend(hud_font::build_text(value, value_x, y, sc.char_height, VALUE_FADE));
    y + sc.line_height
}

fn build_panel_fill(x_left: f32, x_right: f32, screen_h: f32) -> Vec<GridVertex> {
    let v = |x: f32, y: f32| GridVertex {
        position: [x, y, 0.0],
        fade: 0.0,
    };
    vec![
        v(x_left, 0.0), v(x_right, 0.0), v(x_right, screen_h),
        v(x_left, 0.0), v(x_right, screen_h), v(x_left, screen_h),
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
            club_speed_mph: 90.0,
            ball_speed_mph: 132.0,
            smash_factor: Some(1.47),
            launch_angle_deg: 16.0,
            launch_azimuth_deg: -1.2,
            backspin_rpm: 7000.0,
            sidespin_rpm: -300.0,
            club_path_deg: Some(-2.1),
            attack_angle_deg: Some(-4.5),
            face_angle_deg: Some(0.3),
            dynamic_loft_deg: Some(22.8),
            apex_m: 28.3,
            carry_yards: 159.0,
            lm_carry_yards: None,
            total_yards: 172.0,
            downrange_yards: 170.5,
            lateral_m: -4.2,
            flight_time_s: 5.8,
            elapsed_s: 3.2,
            in_flight: true,
        };
        let geom = build_hud(Some(&t), UnitSystem::Imperial, false, &HashMap::new(), true, 1920.0, 1080.0);
        assert!(!geom.lines.is_empty(), "should have text lines");
        assert_eq!(geom.lines.len() % 2, 0, "LINE_LIST needs even count");
        assert_eq!(geom.fills.len(), 12, "2 panels × 2 triangles each");
    }

    #[test]
    fn build_hud_without_telemetry() {
        let geom = build_hud(None, UnitSystem::Imperial, false, &HashMap::new(), true, 1920.0, 1080.0);
        assert!(!geom.lines.is_empty(), "should show standby text");
    }

    #[test]
    fn panel_fills_cover_both_sides() {
        let geom = build_hud(None, UnitSystem::Imperial, false, &HashMap::new(), true, 1920.0, 1080.0);
        // Left panel: 0..384, right panel: 1536..1920
        let left_fills = &geom.fills[..6];
        let right_fills = &geom.fills[6..12];
        let left_max_x = left_fills.iter().map(|v| v.position[0]).fold(0.0_f32, f32::max);
        let right_min_x = right_fills.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min);
        assert!((left_max_x - 384.0).abs() < 1.0, "left panel edge should be 384, got {left_max_x}");
        assert!((right_min_x - 1536.0).abs() < 1.0, "right panel edge should be 1536, got {right_min_x}");
    }
}
