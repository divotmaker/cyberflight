//! The static scene as colored layers: what a range draws under the tee
//! box, the ball and the flights.
//!
//! [`Range::scenery`](crate::range::Range::scenery) builds it; the renderer
//! draws the fill layers first (triangles, depth test and write), then the
//! line layers (1 px lines), each in its color, with the pipelines it uses
//! for the grid.

use glam::Vec4;

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
    /// The flat reflective floor, when the ground is one: the grid whose
    /// floor the reflections are drawn on. `None` draws no reflections.
    pub floor: Option<GridConfig>,
}
