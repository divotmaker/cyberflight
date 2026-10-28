//! Range rings: the targets' concentric neon circles and the green's edge
//! on the range floor, drawn by both the windowed and the offscreen
//! renderer with the grid's line pipeline.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

use cf_scene::color;
use cf_scene::green::{GreenConfig, generate_green_edge};
use cf_scene::grid::{GridConfig, GridVertex};
use cf_scene::target::{Target, generate_target_rings};

use crate::error::RenderError;
use crate::pipeline::{GridPipeline, GridPushConstants};

/// Creates a host-visible vertex buffer holding `vertices`.
pub(crate) type CreateVertexBuffer = fn(
    &ash::Device,
    &mut Allocator,
    &[GridVertex],
) -> Result<(vk::Buffer, Allocation), RenderError>;

/// Uploaded ring geometry for a range's targets and green.
pub(crate) struct TargetRings {
    buffer: vk::Buffer,
    allocation: Allocation,
    count: u32,
}

impl TargetRings {
    /// Upload the rings of `targets` and the edge of `green`, faded like
    /// `grid`. `None` when there is nothing to draw.
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        create: CreateVertexBuffer,
        targets: &[Target],
        green: Option<&GreenConfig>,
        grid: &GridConfig,
    ) -> Result<Option<Self>, RenderError> {
        let mut verts = generate_target_rings(targets, grid);
        if let Some(green) = green {
            verts.extend(generate_green_edge(green, grid));
        }
        if verts.is_empty() {
            return Ok(None);
        }
        let (buffer, allocation) = create(device, allocator, &verts)?;
        Ok(Some(Self {
            buffer,
            allocation,
            count: verts.len() as u32,
        }))
    }

    /// Record the ring draw: 1 px lines in [`color::TARGET_RING`]. Binds
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
            color: color::TARGET_RING.into(),
        };
        // SAFETY: Caller guarantees an active render pass; the buffer lives
        // until `destroy`, and the push constants match the pipeline layout.
        unsafe {
            let pc_bytes: &[u8] = std::slice::from_raw_parts(
                std::ptr::from_ref(&pc).cast::<u8>(),
                std::mem::size_of::<GridPushConstants>(),
            );
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_set_line_width(cb, 1.0);
            device.cmd_push_constants(
                cb,
                pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                pc_bytes,
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[self.buffer], &[0]);
            device.cmd_draw(cb, self.count, 1, 0, 0);
        }
    }

    /// Free the buffer.
    ///
    /// # Safety
    /// The GPU must no longer use the buffer (wait idle first).
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: Caller guarantees the buffer is idle.
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        let _ = allocator.free(self.allocation);
    }
}
