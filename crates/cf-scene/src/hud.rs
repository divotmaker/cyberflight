use std::collections::HashMap;

use crate::grid::GridVertex;
use crate::hud_font;

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
    /// Club speed, when the device measured it.
    pub club_speed_mph: Option<f64>,
    pub ball_speed_mph: f64,
    pub smash_factor: Option<f64>,
    pub launch_angle_deg: f64,
    pub launch_azimuth_deg: f64,
    /// Backspin, when the device measured it.
    pub backspin_rpm: Option<f64>,
    /// Sidespin (positive = right), when the device measured it.
    pub sidespin_rpm: Option<f64>,
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
    /// Live offset from the target line (positive = right), updated each frame.
    pub lateral_yards: f64,
    /// Carry flight time only (excludes bounce/rollout).
    pub flight_time_s: f64,
    /// Time since launch, counting until the ball comes to rest.
    pub elapsed_s: f64,
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

/// One line of the telemetry panel.
#[derive(Debug, Clone, PartialEq)]
enum HudRow {
    /// Section title with a rule.
    Header(&'static str),
    /// Label and right-aligned value.
    Value(&'static str, String),
    /// Space between sections.
    Gap,
}

/// Value text for a measurement the device may not have sent: `-` when absent.
fn or_dash(value: Option<f64>, fmt: impl Fn(f64) -> String) -> String {
    value.map_or_else(|| "-".to_owned(), fmt)
}

/// The telemetry panel's sections and rows for `t`.
fn telemetry_rows(t: &ShotTelemetry, units: UnitSystem) -> Vec<HudRow> {
    use HudRow::{Gap, Header, Value};

    let (spd_label, spd_factor) = match units {
        UnitSystem::Imperial => ("MPH", 1.0),
        UnitSystem::Metric => ("KPH", 1.60934),
    };
    let fmt_dist = |yards: f64| match units {
        UnitSystem::Imperial => format!("{yards:.1} YDS"),
        UnitSystem::Metric => format!("{:.1} M", yards * 0.9144),
    };
    let fmt_height = |meters: f64| match units {
        UnitSystem::Imperial => format!("{:.1} FT", meters / 0.3048),
        UnitSystem::Metric => format!("{meters:.1} M"),
    };
    // Offset from the target line: `R`/`L` and the distance, unsigned.
    let fmt_lateral = |yards: f64| {
        let text = fmt_dist(yards.abs());
        if text.starts_with("0.0 ") {
            text
        } else if yards > 0.0 {
            format!("R {text}")
        } else {
            format!("L {text}")
        }
    };
    let fmt_rpm = |rpm: f64| format!("{rpm:.0} RPM");

    let mut rows = vec![
        Header("CLUB"),
        Value(
            "SPEED",
            or_dash(t.club_speed_mph, |v| {
                format!("{:.0} {spd_label}", v * spd_factor)
            }),
        ),
    ];
    let club_angles = [
        ("PATH", t.club_path_deg),
        ("AOA", t.attack_angle_deg),
        ("FACE", t.face_angle_deg),
        ("D LOFT", t.dynamic_loft_deg),
    ];
    for (label, value) in club_angles {
        if let Some(v) = value {
            rows.push(Value(label, format!("{v:.1}")));
        }
    }

    rows.extend([
        Gap,
        Header("LAUNCH"),
        Value(
            "SPEED",
            format!("{:.1} {spd_label}", t.ball_speed_mph * spd_factor),
        ),
    ]);
    if let Some(sf) = t.smash_factor {
        rows.push(Value("SMASH", format!("{sf:.2}")));
    }
    rows.extend([
        Value("VLA", format!("{:.1}", t.launch_angle_deg)),
        Value("HLA", format!("{:.1}", t.launch_azimuth_deg)),
        Value("BACK", or_dash(t.backspin_rpm, fmt_rpm)),
        Value("SIDE", or_dash(t.sidespin_rpm, fmt_rpm)),
        Gap,
        Header("FLIGHT"),
        Value("APEX", fmt_height(t.apex_m)),
        Value("CARRY", fmt_dist(t.carry_yards)),
    ]);
    if let Some(lm_carry) = t.lm_carry_yards {
        let carry_complete = t.elapsed_s >= t.flight_time_s;
        let lm_text = if carry_complete {
            fmt_dist(lm_carry)
        } else {
            "-".to_owned()
        };
        rows.push(Value("CARRY(LM)", lm_text));
    }
    rows.extend([
        Value("TIME", format!("{:.1} S", t.elapsed_s)),
        Gap,
        Header("RESULT"),
        Value("TOTAL", fmt_dist(t.total_yards)),
        Value("DOWNRANGE", fmt_dist(t.downrange_yards)),
        Value("LATERAL", fmt_lateral(t.lateral_yards)),
    ]);
    rows
}

/// One row of the SYSTEM section.
#[derive(Debug, Clone, PartialEq)]
struct SystemRow<'a> {
    label: &'a str,
    value: &'a str,
    label_fade: f32,
    value_fade: f32,
}

/// SYSTEM section rows: the connection, or each launch monitor's ready
/// state, sorted by device name.
fn system_rows(lm_states: &HashMap<String, bool>, connected: bool) -> Vec<SystemRow<'_>> {
    let row = |label, value, label_fade, value_fade| SystemRow {
        label,
        value,
        label_fade,
        value_fade,
    };
    if !connected {
        // Negative fade = red alert color (see grid.frag).
        return vec![row("LAUNCH MONITOR", "-", -1.0, -1.0)];
    }
    if lm_states.is_empty() {
        return vec![row("LAUNCH MONITOR", "UNKNOWN", LABEL_FADE, LABEL_FADE)];
    }
    let multi = lm_states.len() > 1;
    let mut devices: Vec<(&String, &bool)> = lm_states.iter().collect();
    devices.sort_unstable_by_key(|&(name, _)| name);
    devices
        .into_iter()
        .map(|(source, &armed)| {
            let label = if multi {
                source.as_str()
            } else {
                "LAUNCH MONITOR"
            };
            if armed {
                row(label, "READY", LABEL_FADE, 0.0)
            } else {
                row(label, "WAITING", LABEL_FADE, LABEL_FADE)
            }
        })
        .collect()
}

/// Build HUD geometry for the telemetry overlay.
///
/// Returns line vertices (for text + decorations) and fill vertices (panel background).
/// Designed for the left 20% of the screen with a telemetry/rocket-monitoring feel.
///
/// `lm_states`: per-source ready state map.
/// `connected`: whether the FRP connection is alive.
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
        for row in telemetry_rows(t, units) {
            y = match row {
                HudRow::Header(title) => emit_section_header(&mut lines, title, y, panel_w, &sc),
                HudRow::Value(label, value) => emit_row(&mut lines, label, &value, y, panel_w, &sc),
                HudRow::Gap => y + sc.section_gap,
            };
        }
    } else {
        // No telemetry — show standby
        lines.extend(hud_font::build_text(
            "SEND IT",
            sc.margin_left,
            y,
            sc.char_height,
            LABEL_FADE,
        ));
    }

    // ── SYSTEM section (bottom-aligned in left panel)
    {
        let rows = system_rows(lm_states, connected);

        let num_rows = rows.len() as f32;
        let header_y = screen_h - sc.margin_top - (num_rows * sc.line_height) - sc.line_height;
        emit_section_header(&mut lines, "SYSTEM", header_y, panel_w, &sc);

        let mut row_y = header_y + sc.line_height;
        for row in &rows {
            lines.extend(hud_font::build_text(
                row.label,
                sc.label_x,
                row_y,
                sc.char_height,
                row.label_fade,
            ));
            let value_x = panel_w - panel_margin - hud_font::text_width(row.value, sc.char_height);
            lines.extend(hud_font::build_text(
                row.value,
                value_x,
                row_y,
                sc.char_height,
                row.value_fade,
            ));
            row_y += sc.line_height;
        }
    }

    HudGeometry { lines, fills }
}

fn emit_section_header(
    lines: &mut Vec<GridVertex>,
    title: &str,
    y: f32,
    panel_w: f32,
    sc: &HudScale,
) -> f32 {
    lines.extend(hud_font::build_text(
        title,
        sc.margin_left,
        y,
        sc.char_height,
        HEADER_FADE,
    ));
    let panel_margin = 16.0 * sc.line_height / REF_LINE_HEIGHT;
    let line_start = sc.margin_left
        + hud_font::text_width(title, sc.char_height)
        + 8.0 * sc.char_height / REF_CHAR_HEIGHT;
    let line_end = panel_w - panel_margin;
    if line_end > line_start + 10.0 {
        let sep = hud_font::build_hline(line_start, line_end, y + sc.char_height * 0.5, DECOR_FADE);
        lines.extend_from_slice(&sep);
    }
    y + sc.line_height
}

fn emit_row(
    lines: &mut Vec<GridVertex>,
    label: &str,
    value: &str,
    y: f32,
    panel_w: f32,
    sc: &HudScale,
) -> f32 {
    lines.extend(hud_font::build_text(
        label,
        sc.label_x,
        y,
        sc.char_height,
        LABEL_FADE,
    ));
    let panel_margin = 16.0 * sc.line_height / REF_LINE_HEIGHT;
    let panel_right = panel_w - panel_margin;
    let value_w = hud_font::text_width(value, sc.char_height);
    let min_x = sc.label_x + hud_font::text_width(label, sc.char_height) + panel_margin;
    let value_x = (panel_right - value_w).max(min_x);
    lines.extend(hud_font::build_text(
        value,
        value_x,
        y,
        sc.char_height,
        VALUE_FADE,
    ));
    y + sc.line_height
}

fn build_panel_fill(x_left: f32, x_right: f32, screen_h: f32) -> Vec<GridVertex> {
    let v = |x: f32, y: f32| GridVertex {
        position: [x, y, 0.0],
        fade: 0.0,
    };
    vec![
        v(x_left, 0.0),
        v(x_right, 0.0),
        v(x_right, screen_h),
        v(x_left, 0.0),
        v(x_right, screen_h),
        v(x_left, screen_h),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ShotTelemetry {
        ShotTelemetry {
            club_speed_mph: Some(90.0),
            ball_speed_mph: 132.0,
            smash_factor: Some(1.47),
            launch_angle_deg: 16.0,
            launch_azimuth_deg: -1.2,
            backspin_rpm: Some(7000.0),
            sidespin_rpm: Some(-300.0),
            club_path_deg: Some(-2.1),
            attack_angle_deg: Some(-4.5),
            face_angle_deg: Some(0.3),
            dynamic_loft_deg: Some(22.8),
            apex_m: 28.3,
            carry_yards: 159.0,
            lm_carry_yards: None,
            total_yards: 172.0,
            downrange_yards: 170.5,
            lateral_yards: -4.6,
            flight_time_s: 5.8,
            elapsed_s: 3.2,
        }
    }

    /// The value shown for `label` in `section`.
    fn value(rows: &[HudRow], section: &str, label: &str) -> String {
        let start = rows
            .iter()
            .position(|r| matches!(r, HudRow::Header(h) if *h == section))
            .expect("section");
        rows[start + 1..]
            .iter()
            .take_while(|r| !matches!(r, HudRow::Header(_)))
            .find_map(|r| match r {
                HudRow::Value(l, v) if *l == label => Some(v.clone()),
                _ => None,
            })
            .expect("row")
    }

    #[test]
    fn telemetry_values() {
        let rows = telemetry_rows(&sample(), UnitSystem::Imperial);
        assert_eq!(value(&rows, "CLUB", "SPEED"), "90 MPH");
        assert_eq!(value(&rows, "CLUB", "PATH"), "-2.1");
        assert_eq!(value(&rows, "LAUNCH", "SPEED"), "132.0 MPH");
        assert_eq!(value(&rows, "LAUNCH", "SMASH"), "1.47");
        assert_eq!(value(&rows, "LAUNCH", "HLA"), "-1.2");
        assert_eq!(value(&rows, "LAUNCH", "BACK"), "7000 RPM");
        assert_eq!(value(&rows, "LAUNCH", "SIDE"), "-300 RPM");
        assert_eq!(value(&rows, "FLIGHT", "APEX"), "92.8 FT");
        assert_eq!(value(&rows, "FLIGHT", "CARRY"), "159.0 YDS");
        assert_eq!(value(&rows, "FLIGHT", "TIME"), "3.2 S");
        assert_eq!(value(&rows, "RESULT", "TOTAL"), "172.0 YDS");
        assert_eq!(value(&rows, "RESULT", "LATERAL"), "L 4.6 YDS");
    }

    #[test]
    fn unmeasured_values_show_a_dash() {
        let t = ShotTelemetry {
            club_speed_mph: None,
            backspin_rpm: None,
            sidespin_rpm: None,
            club_path_deg: None,
            ..sample()
        };
        let rows = telemetry_rows(&t, UnitSystem::Imperial);
        assert_eq!(value(&rows, "CLUB", "SPEED"), "-");
        assert_eq!(value(&rows, "LAUNCH", "BACK"), "-");
        assert_eq!(value(&rows, "LAUNCH", "SIDE"), "-");
        assert!(rows.iter().all(|r| !matches!(r, HudRow::Value("PATH", _))));
    }

    #[test]
    fn lateral_is_yards_left_or_right() {
        let lateral = |yards: f64, units| {
            let t = ShotTelemetry {
                lateral_yards: yards,
                ..sample()
            };
            value(&telemetry_rows(&t, units), "RESULT", "LATERAL")
        };
        assert_eq!(lateral(4.24, UnitSystem::Imperial), "R 4.2 YDS");
        assert_eq!(lateral(-12.0, UnitSystem::Imperial), "L 12.0 YDS");
        assert_eq!(lateral(0.01, UnitSystem::Imperial), "0.0 YDS");
        assert_eq!(lateral(-0.04, UnitSystem::Imperial), "0.0 YDS");
        assert_eq!(lateral(10.0, UnitSystem::Metric), "R 9.1 M");
    }

    #[test]
    fn lm_carry_waits_for_landing() {
        let flying = ShotTelemetry {
            lm_carry_yards: Some(161.2),
            ..sample()
        };
        let rows = telemetry_rows(&flying, UnitSystem::Imperial);
        assert_eq!(value(&rows, "FLIGHT", "CARRY(LM)"), "-");
        let landed = ShotTelemetry {
            elapsed_s: 7.5,
            ..flying
        };
        let rows = telemetry_rows(&landed, UnitSystem::Imperial);
        assert_eq!(value(&rows, "FLIGHT", "CARRY(LM)"), "161.2 YDS");
        assert_eq!(value(&rows, "FLIGHT", "TIME"), "7.5 S");
    }

    #[test]
    fn system_rows_are_sorted_by_device() {
        let lm = HashMap::from([
            ("square.0".to_owned(), false),
            ("mevo.0".to_owned(), true),
            ("r10.0".to_owned(), true),
            ("gc3.0".to_owned(), false),
        ]);
        let labels: Vec<_> = system_rows(&lm, true).iter().map(|r| r.label).collect();
        assert_eq!(labels, ["gc3.0", "mevo.0", "r10.0", "square.0"]);
        let values: Vec<_> = system_rows(&lm, true).iter().map(|r| r.value).collect();
        assert_eq!(values, ["WAITING", "READY", "READY", "WAITING"]);
    }

    #[test]
    fn system_rows_for_one_device_or_none() {
        let one = HashMap::from([("mevo.0".to_owned(), true)]);
        let rows = system_rows(&one, true);
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].label, rows[0].value), ("LAUNCH MONITOR", "READY"));
        let offline = system_rows(&one, false);
        assert_eq!(offline[0].value, "-");
        assert!(offline[0].label_fade < 0.0, "alert colour");
        assert_eq!(system_rows(&HashMap::new(), true)[0].value, "UNKNOWN");
    }

    #[test]
    fn build_hud_with_telemetry() {
        let t = sample();
        let geom = build_hud(
            Some(&t),
            UnitSystem::Imperial,
            false,
            &HashMap::new(),
            true,
            1920.0,
            1080.0,
        );
        assert!(!geom.lines.is_empty(), "should have text lines");
        assert_eq!(geom.lines.len() % 2, 0, "LINE_LIST needs even count");
        assert_eq!(geom.fills.len(), 12, "2 panels × 2 triangles each");
    }

    #[test]
    fn build_hud_without_telemetry() {
        let geom = build_hud(
            None,
            UnitSystem::Imperial,
            false,
            &HashMap::new(),
            true,
            1920.0,
            1080.0,
        );
        assert!(!geom.lines.is_empty(), "should show standby text");
    }

    #[test]
    fn panel_fills_cover_both_sides() {
        let geom = build_hud(
            None,
            UnitSystem::Imperial,
            false,
            &HashMap::new(),
            true,
            1920.0,
            1080.0,
        );
        // Left panel: 0..384, right panel: 1536..1920
        let left_fills = &geom.fills[..6];
        let right_fills = &geom.fills[6..12];
        let left_max_x = left_fills
            .iter()
            .map(|v| v.position[0])
            .fold(0.0_f32, f32::max);
        let right_min_x = right_fills
            .iter()
            .map(|v| v.position[0])
            .fold(f32::MAX, f32::min);
        assert!(
            (left_max_x - 384.0).abs() < 1.0,
            "left panel edge should be 384, got {left_max_x}"
        );
        assert!(
            (right_min_x - 1536.0).abs() < 1.0,
            "right panel edge should be 1536, got {right_min_x}"
        );
    }
}
