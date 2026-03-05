use cf_math::units::Unit;
use glam::Vec3;

/// Vertex for grid line rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GridVertex {
    pub position: [f32; 3],
    /// Normalized distance from origin (0.0 = near, 1.0 = far edge).
    /// Used for distance fade in the fragment shader.
    pub fade: f32,
}

/// Configuration for the neon ground grid.
#[derive(Debug, Clone, Copy)]
pub struct GridConfig {
    /// Display unit (yards or meters).
    pub unit: Unit,
    /// Grid line spacing in display units.
    pub spacing: f32,
    /// Downrange extent in display units (+Z direction, starts at 0).
    pub downrange: f32,
    /// Lateral half-width in display units (+/- X direction).
    pub lateral: f32,
}

impl GridConfig {
    /// Driving range grid for the given unit system.
    ///
    /// Yards: 60 wide (±30 yd) × 450 long, 10-yard grid.
    /// Meters: 60 wide (±30 m) × 400 long, 10-meter grid.
    ///
    /// Width is a multiple of spacing so grid lines align with the edges
    /// and a centerline runs down X=0.
    #[must_use]
    pub fn driving_range(unit: Unit) -> Self {
        match unit {
            Unit::Yards => Self {
                unit,
                spacing: 10.0,
                downrange: 450.0,
                lateral: 30.0,
            },
            Unit::Meters => Self {
                unit,
                spacing: 10.0,
                downrange: 400.0,
                lateral: 30.0,
            },
        }
    }

    fn spacing_m(&self) -> f32 {
        self.unit.to_meters(f64::from(self.spacing)) as f32
    }

    fn downrange_m(&self) -> f32 {
        self.unit.to_meters(f64::from(self.downrange)) as f32
    }

    fn lateral_m(&self) -> f32 {
        self.unit.to_meters(f64::from(self.lateral)) as f32
    }

    /// World-space clip bounds for the grid floor: [min_x, min_z, max_x, max_z].
    #[must_use]
    pub fn clip_bounds(&self) -> [f32; 4] {
        let lat = self.lateral_m();
        let dr = self.downrange_m();
        [-lat, 0.0, lat, dr]
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self::driving_range(Unit::Yards)
    }
}

/// Generate grid line vertices for the driving range floor.
///
/// Returns vertex pairs (each pair is one line segment).
/// The grid sits on the Y=0 plane, Z=0 at the tee box extending downrange (+Z).
/// X is lateral (±lateral). All positions are in meters.
#[must_use]
pub fn generate_grid_vertices(config: &GridConfig) -> Vec<GridVertex> {
    let downrange = config.downrange_m();
    let lateral = config.lateral_m();
    let spacing = config.spacing_m();

    let max_dist = (downrange * downrange + lateral * lateral).sqrt();
    let mut verts = Vec::new();

    // Lines parallel to X axis (lateral lines, at Z intervals along downrange)
    let z_count = (downrange / spacing) as i32;
    for iz in 0..=z_count {
        let z = iz as f32 * spacing;
        let start = Vec3::new(-lateral, 0.0, z);
        let end = Vec3::new(lateral, 0.0, z);

        verts.push(GridVertex {
            position: start.into(),
            fade: start.length() / max_dist,
        });
        verts.push(GridVertex {
            position: end.into(),
            fade: end.length() / max_dist,
        });
    }

    // Lines parallel to Z axis (downrange lines, at X intervals across lateral)
    let x_count = (lateral / spacing) as i32;
    for ix in -x_count..=x_count {
        let x = ix as f32 * spacing;
        let start = Vec3::new(x, 0.0, 0.0);
        let end = Vec3::new(x, 0.0, downrange);

        verts.push(GridVertex {
            position: start.into(),
            fade: start.length() / max_dist,
        });
        verts.push(GridVertex {
            position: end.into(),
            fade: end.length() / max_dist,
        });
    }

    verts
}

/// Generate the reflective floor surface at Y=0 (TRIANGLE_LIST).
///
/// The floor is NOT a single rectangle — it's the union of:
/// - The tee box area (extends behind Z=0 within the tee box bounds)
/// - The fairway (Z≥0, full lateral extent)
///
/// Decomposed into non-overlapping quads to avoid RT ray intersection ambiguity:
/// ```text
///         -lat          -th    +th          +lat
///           |            |      |            |
///    dr   ──┼────────────┼──────┼────────────┤
///           │  left wing │center│ right wing │
///           │            │above │            │
///    th   ──┼────────────┼──────┼────────────┤
///           │ left wing  │ TEE  │ right wing │
///     0   ──┼────────────┼──BOX─┼────────────┤
///           :            │      │            :
///   -th     :            └──────┘            :
/// ```
///
/// `tee_half_size` is the tee box half-extent in meters.
#[must_use]
pub fn generate_floor_quad(config: &GridConfig, tee_half_size: f32) -> Vec<GridVertex> {
    let lat = config.lateral_m();
    let dr = config.downrange_m();
    let th = tee_half_size;
    let v = |x: f32, z: f32| GridVertex {
        position: [x, 0.0, z],
        fade: 0.0,
    };

    let mut verts = Vec::with_capacity(24);

    // 1. Tee box: X ∈ [-th, +th], Z ∈ [-th, +th]
    verts.extend_from_slice(&[
        v(-th, -th), v(th, -th), v(th, th),
        v(-th, -th), v(th, th), v(-th, th),
    ]);

    // 2. Left wing: X ∈ [-lat, -th], Z ∈ [0, dr]
    verts.extend_from_slice(&[
        v(-lat, 0.0), v(-th, 0.0), v(-th, dr),
        v(-lat, 0.0), v(-th, dr), v(-lat, dr),
    ]);

    // 3. Center above tee box: X ∈ [-th, +th], Z ∈ [th, dr]
    verts.extend_from_slice(&[
        v(-th, th), v(th, th), v(th, dr),
        v(-th, th), v(th, dr), v(-th, dr),
    ]);

    // 4. Right wing: X ∈ [+th, +lat], Z ∈ [0, dr]
    verts.extend_from_slice(&[
        v(th, 0.0), v(lat, 0.0), v(lat, dr),
        v(th, 0.0), v(lat, dr), v(th, dr),
    ]);

    verts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_vertex_size() {
        assert_eq!(std::mem::size_of::<GridVertex>(), 16);
    }

    #[test]
    fn default_grid_has_lines() {
        let verts = generate_grid_vertices(&GridConfig::default());
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 2, 0);
    }

    #[test]
    fn grid_on_ground_plane() {
        let verts = generate_grid_vertices(&GridConfig::default());
        for v in &verts {
            assert!(
                (v.position[1]).abs() < 1e-6,
                "grid vertex should be on Y=0 plane"
            );
        }
    }

    #[test]
    fn fade_values_normalized() {
        let verts = generate_grid_vertices(&GridConfig::default());
        for v in &verts {
            assert!(
                v.fade >= 0.0 && v.fade <= 1.01,
                "fade should be in [0, 1], got {}",
                v.fade
            );
        }
    }

    #[test]
    fn yards_grid_dimensions() {
        let config = GridConfig::driving_range(Unit::Yards);
        let verts = generate_grid_vertices(&config);
        let max_x = verts.iter().map(|v| v.position[0]).fold(0.0_f32, f32::max);
        let max_z = verts.iter().map(|v| v.position[2]).fold(0.0_f32, f32::max);
        // 30 yards ≈ 27.43m (lateral half-width, +X)
        assert!((max_x - 27.43).abs() < 1.0, "max_x={max_x}");
        // 450 yards ≈ 411.48m (downrange, +Z)
        assert!((max_z - 411.48).abs() < 1.0, "max_z={max_z}");
    }

    #[test]
    fn meters_grid_dimensions() {
        let config = GridConfig::driving_range(Unit::Meters);
        let verts = generate_grid_vertices(&config);
        let max_x = verts.iter().map(|v| v.position[0]).fold(0.0_f32, f32::max);
        let max_z = verts.iter().map(|v| v.position[2]).fold(0.0_f32, f32::max);
        assert!((max_x - 30.0).abs() < 1.0, "max_x={max_x}");
        assert!((max_z - 400.0).abs() < 1.0, "max_z={max_z}");
    }
}
