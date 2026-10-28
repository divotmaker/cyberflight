/// Rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// Traditional rasterization with no reflections.
    Rasterized,
    /// Hardware ray tracing with true reflections off the floor surface.
    /// Requires VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure.
    RayTraced,
}

/// Whether an offscreen renderer ray-traces floor reflections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtMode {
    /// Never. The renderer's device is created without ray-tracing extensions.
    Off,
    /// When the GPU supports it and setup succeeds; otherwise raster.
    Auto,
}
