/// Rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// The floor reflects the tracer and the ball by mirroring them across
    /// the floor plane and rasterizing them. Works on any Vulkan device.
    Rasterized,
    /// Hardware ray tracing with true reflections off the floor surface.
    /// Requires VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure.
    /// Looks the same as `Rasterized`: both shade the floor with the same code.
    RayTraced,
}

/// Whether an offscreen renderer ray-traces floor reflections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtMode {
    /// Never. The renderer's device is created without ray-tracing extensions,
    /// and the floor's reflections are mirrored.
    Off,
    /// When the GPU supports it and setup succeeds; otherwise the floor's
    /// reflections are mirrored.
    Auto,
}
