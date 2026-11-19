//! The static scene as colored layers: what a range draws under the tee
//! box, the ball and the flights.
//!
//! [`Range::scenery`](crate::range::Range::scenery) builds it; the renderer
//! draws the fill layers first (triangles, depth test and write), then the
//! line layers (1 px lines), each in its color, with the pipelines it uses
//! for the grid.

use glam::{Vec3, Vec4};

use crate::grid::{GridConfig, GridVertex};

/// One draw: vertices in one color.
#[derive(Debug, Clone)]
pub struct Layer {
    pub color: Vec4,
    pub vertices: Vec<GridVertex>,
}

/// A range's static scene.
#[derive(Debug, Clone)]
pub struct Scenery {
    /// `TRIANGLE_LIST` layers, drawn first.
    pub fills: Vec<Layer>,
    /// `LINE_LIST` layers, 1 px, drawn after the fills.
    pub lines: Vec<Layer>,
    /// The reflective ground the reflections are drawn on. `None` draws no
    /// reflections.
    pub floor: Option<Floor>,
}

/// The reflective ground of a range.
#[derive(Debug, Clone)]
pub enum Floor {
    /// The plane Y = 0 under the grid of this config and the tee box. Both
    /// reflection paths draw on it: ray-traced and mirrored.
    Flat(GridConfig),
    /// Shaped ground. Only the ray tracer reflects on it; without ray
    /// tracing it draws no reflections.
    Terrain(GroundMesh),
}

/// A `TRIANGLE_LIST` with a unit normal per vertex: the surface's normal at
/// that vertex, which the ray tracer interpolates across each triangle.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroundMesh {
    pub positions: Vec<Vec3>,
    /// One per position.
    pub normals: Vec<Vec3>,
}
