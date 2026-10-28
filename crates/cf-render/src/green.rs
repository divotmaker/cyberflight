//! The edge of the range's optional green: the one circle on the range
//! floor, drawn by both the windowed and the offscreen renderer with the
//! grid's line pipeline. With no green nothing is drawn.

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use cf_scene::color;
use cf_scene::green::{GreenConfig, generate_green_edge};
use cf_scene::grid::{GridConfig, GridVertex};

use crate::error::RenderError;
use crate::gpu::{HostBuffer, push_constants};
use crate::pipeline::{GridPipeline, GridPushConstants};

/// Edge vertices of `green`: one ring, or none when there is no green.
fn edge_vertices(green: Option<&GreenConfig>, grid: &GridConfig) -> Vec<GridVertex> {
    green.map_or_else(Vec::new, |green| generate_green_edge(green, grid))
}

/// Uploaded edge geometry of a range's green.
pub(crate) struct GreenEdge {
    buffer: HostBuffer,
    count: u32,
}

impl GreenEdge {
    /// Upload the edge of `green`, faded like `grid`. `None` when there is
    /// no green.
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        green: Option<&GreenConfig>,
        grid: &GridConfig,
    ) -> Result<Option<Self>, RenderError> {
        let verts = edge_vertices(green, grid);
        if verts.is_empty() {
            return Ok(None);
        }
        let buffer = HostBuffer::with_data(
            device,
            allocator,
            &verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "green edge",
        )?;
        Ok(Some(Self {
            buffer,
            count: verts.len() as u32,
        }))
    }

    /// Record the edge draw: 1 px lines in [`color::GREEN_EDGE`]. Binds
    /// `pipeline` itself, so any pipeline may be bound before the call.
    ///
    /// # Safety
    /// Must be called within an active render pass compatible with
    /// `pipeline`, with viewport and scissor set.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        pipeline: &GridPipeline,
        view_proj: [[f32; 4]; 4],
    ) {
        let pc = GridPushConstants {
            view_proj,
            color: color::GREEN_EDGE.into(),
        };
        // SAFETY: Caller guarantees an active render pass; the buffer lives
        // until `destroy`, and the push constants match the pipeline layout.
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_set_line_width(cb, 1.0);
            push_constants(
                device,
                cb,
                pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc,
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[self.buffer.handle()], &[0]);
            device.cmd_draw(cb, self.count, 1, 0, 0);
        }
    }

    /// Free the buffer.
    ///
    /// # Safety
    /// The GPU must no longer use the buffer (wait idle first).
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: Caller guarantees the buffer is idle.
        unsafe { self.buffer.destroy(device, allocator) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cf_scene::green::RING_SEGMENTS;
    use cf_scene::range::Range;
    use glam::Vec3;

    #[test]
    fn default_range_draws_no_circle() {
        let range = Range::new();
        assert!(edge_vertices(range.green.as_ref(), &range.grid).is_empty());
    }

    #[test]
    fn a_green_draws_exactly_one_circle() {
        let mut range = Range::new();
        let green = GreenConfig {
            distance_m: 137.16,
            diameter_m: 30.0,
        };
        range.green = Some(green);
        let verts = edge_vertices(range.green.as_ref(), &range.grid);
        assert_eq!(verts.len(), 2 * RING_SEGMENTS as usize);
        let center = green.center_vec3();
        for v in &verts {
            let p = Vec3::from(v.position);
            let r = Vec3::new(p.x - center.x, 0.0, p.z - center.z).length();
            assert!((r - green.radius_m()).abs() < 1e-3, "radius {r}");
        }
    }
}
