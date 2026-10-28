//! Per-frame geometry: what `update_*` builds on the CPU and the GPU buffers
//! a frame is drawn from.

use ash::vk;
use glam::Vec3;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use cf_scene::color;
use cf_scene::grid::GridVertex;
use cf_scene::tee::TeeBox;

use crate::error::RenderError;
use crate::flight::tee_ball_centre;
use crate::gpu::{HostBuffer, push_constants};
use crate::mirror::MirrorTracer;
use crate::pipeline::{GridPushConstants, ScenePipelines};

/// The next frame's CPU-side content, built by `update_*` and uploaded when
/// the frame is recorded.
pub(crate) struct FrameContent {
    /// Trail core lines (LINE_LIST).
    pub(crate) flight_lines: Vec<GridVertex>,
    /// In-flight balls (TRIANGLE_LIST, black fill).
    pub(crate) flight_fills: Vec<GridVertex>,
    /// Ball and trail glow shells (TRIANGLE_LIST, additive).
    pub(crate) flight_glows: Vec<GridVertex>,
    /// HUD text and decoration lines, in pixels.
    pub(crate) hud_lines: Vec<GridVertex>,
    /// HUD panel fills, in pixels.
    pub(crate) hud_fills: Vec<GridVertex>,
    /// No ball is in flight: draw the ball on the tee.
    pub(crate) ball_on_tee: bool,
    /// Vertices of the newest flight's tracer tube: the first ones of
    /// `flight_glows`, so a buffer that cannot hold everything keeps the
    /// newest tracer. Zero without a flight.
    pub(crate) newest_tube: usize,
    /// Vertices of the newest flight's tracer centre line: the first ones of
    /// `flight_lines`. Zero without a flight.
    pub(crate) newest_line: usize,
    /// Centre of the newest flight's ball, or of the ball on the tee.
    pub(crate) ball_centre: Vec3,
}

impl Default for FrameContent {
    fn default() -> Self {
        Self {
            flight_lines: Vec::new(),
            flight_fills: Vec::new(),
            flight_glows: Vec::new(),
            hud_lines: Vec::new(),
            hud_fills: Vec::new(),
            ball_on_tee: true,
            newest_tube: 0,
            newest_line: 0,
            ball_centre: tee_ball_centre(&TeeBox::default()),
        }
    }
}

const FLIGHT_LINES: usize = 0;
const FLIGHT_FILLS: usize = 1;
const FLIGHT_GLOWS: usize = 2;
const HUD_LINES: usize = 3;
const HUD_FILLS: usize = 4;
const SLOTS: usize = 5;

/// How many of the first `wanted` vertices of a buffer lie within the first
/// `count` vertices that were uploaded.
fn uploaded(wanted: usize, count: u32) -> u32 {
    u32::try_from(wanted).map_or(count, |wanted| wanted.min(count))
}

/// Capacity in bytes and allocation name of each buffer, by slot.
/// Each holds 16-byte `GridVertex`es: 8,192 flight lines, 16,384 flight
/// fills, 524,288 flight glows, 8,192 HUD lines, 256 HUD fills.
const BUFFERS: [(vk::DeviceSize, &str); SLOTS] = [
    (128 * 1024, "flight lines"),
    (256 * 1024, "flight fills"),
    (8 * 1024 * 1024, "flight glows"),
    (128 * 1024, "hud lines"),
    (4 * 1024, "hud fills"),
];

/// The GPU buffers one frame is drawn from.
pub(crate) struct FrameResources {
    buffers: [HostBuffer; SLOTS],
    counts: [u32; SLOTS],
    ball_on_tee: bool,
    /// Vertices of the newest flight's tube at the start of the glow buffer,
    /// and of its centre line at the start of the line buffer, as far as
    /// they were uploaded.
    newest_tube: u32,
    newest_line: u32,
}

impl FrameResources {
    /// Allocate the mapped buffers, empty, with the ball on the tee.
    ///
    /// # Errors
    /// An allocation fails; the buffers already allocated are freed first.
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> Result<Self, RenderError> {
        let mut made = Vec::with_capacity(SLOTS);
        for (size, name) in BUFFERS {
            match HostBuffer::new(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
                name,
            ) {
                Ok(buffer) => made.push(buffer),
                Err(e) => {
                    for buffer in made {
                        // SAFETY: Just created; no GPU work has used it.
                        unsafe { buffer.destroy(device, allocator) };
                    }
                    return Err(e);
                }
            }
        }
        match <[HostBuffer; SLOTS]>::try_from(made) {
            Ok(buffers) => Ok(Self {
                buffers,
                counts: [0; SLOTS],
                ball_on_tee: true,
                newest_tube: 0,
                newest_line: 0,
            }),
            Err(made) => {
                for buffer in made {
                    // SAFETY: Just created; no GPU work has used it.
                    unsafe { buffer.destroy(device, allocator) };
                }
                Err(RenderError::Allocator("frame buffers: wrong count".into()))
            }
        }
    }

    /// Copy `content` into the buffers. Each buffer keeps the leading
    /// vertices that fit (see [`HostBuffer::write`]).
    ///
    /// The GPU must have finished the frames that read these buffers;
    /// otherwise it may draw a mix of old and new vertices.
    pub(crate) fn upload(&mut self, content: &FrameContent) {
        let data: [&[GridVertex]; SLOTS] = [
            &content.flight_lines,
            &content.flight_fills,
            &content.flight_glows,
            &content.hud_lines,
            &content.hud_fills,
        ];
        for ((buffer, count), data) in self.buffers.iter_mut().zip(&mut self.counts).zip(data) {
            *count = buffer.write(data);
        }
        self.ball_on_tee = content.ball_on_tee;
        self.newest_tube = uploaded(content.newest_tube, self.counts[FLIGHT_GLOWS]);
        self.newest_line = uploaded(content.newest_line, self.counts[FLIGHT_LINES]);
    }

    /// The newest flight's tracer as uploaded: the buffers and vertex counts
    /// the mirror draws.
    pub(crate) fn newest_tracer(&self) -> MirrorTracer {
        MirrorTracer {
            tube_buffer: self.buffers[FLIGHT_GLOWS].handle(),
            tube: self.newest_tube,
            line_buffer: self.buffers[FLIGHT_LINES].handle(),
            line: self.newest_line,
        }
    }

    /// No ball is in flight in the uploaded content.
    pub(crate) fn ball_on_tee(&self) -> bool {
        self.ball_on_tee
    }

    /// Draw the flight geometry: trail core lines `trail_line_width` px wide,
    /// then the glow shells, then the balls; each only when it has vertices.
    ///
    /// # Safety
    /// Inside a scene pass of `pipes`, with viewport and scissor set.
    pub(crate) unsafe fn draw_flights(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        pipes: &ScenePipelines,
        view_proj: [[f32; 4]; 4],
        trail_line_width: f32,
    ) {
        let pc_magenta = GridPushConstants {
            view_proj,
            color: color::MAGENTA.into(),
        };
        let pc_black = GridPushConstants {
            view_proj,
            color: [0.0, 0.0, 0.0, 1.0],
        };
        // SAFETY: The caller guarantees an active scene pass; the buffers live
        // until `destroy`, and the push constants match the layout.
        unsafe {
            // Trail core (LINE_LIST, magenta).
            if self.counts[FLIGHT_LINES] > 0 {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipes.line);
                device.cmd_set_line_width(cb, trail_line_width);
                self.draw(device, cb, pipes.layout, FLIGHT_LINES, &pc_magenta);
            }
            // Ball glow + trail glow (additive, magenta).
            if self.counts[FLIGHT_GLOWS] > 0 {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipes.glow);
                self.draw(device, cb, pipes.layout, FLIGHT_GLOWS, &pc_magenta);
            }
            // Ball fill (black sphere, occludes the glow centre).
            if self.counts[FLIGHT_FILLS] > 0 {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipes.fill);
                self.draw(device, cb, pipes.layout, FLIGHT_FILLS, &pc_black);
            }
        }
    }

    /// Draw the HUD in pixel coordinates of `extent`: the translucent panel
    /// fills with `fill`, then the white lines 1 px wide with `line`; each
    /// only when it has vertices.
    ///
    /// # Safety
    /// Inside a render pass compatible with `fill` and `line`, whose layout
    /// is `layout`, with viewport and scissor set.
    pub(crate) unsafe fn draw_hud(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        fill: vk::Pipeline,
        line: vk::Pipeline,
        layout: vk::PipelineLayout,
        extent: vk::Extent2D,
    ) {
        if self.counts[HUD_FILLS] == 0 && self.counts[HUD_LINES] == 0 {
            return;
        }
        let w = extent.width as f32;
        let h = extent.height as f32;
        let hud_proj = glam::Mat4::orthographic_rh(0.0, w, 0.0, h, -1.0, 1.0).to_cols_array_2d();
        // SAFETY: The caller guarantees an active, compatible render pass; the
        // buffers live until `destroy`, and the push constants match the layout.
        unsafe {
            if self.counts[HUD_FILLS] > 0 {
                let pc_hud_bg = GridPushConstants {
                    view_proj: hud_proj,
                    color: [0.0, 0.0, 0.0, 0.8],
                };
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, fill);
                self.draw(device, cb, layout, HUD_FILLS, &pc_hud_bg);
            }
            if self.counts[HUD_LINES] > 0 {
                let pc_hud_text = GridPushConstants {
                    view_proj: hud_proj,
                    color: [1.0, 1.0, 1.0, 1.0],
                };
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, line);
                device.cmd_set_line_width(cb, 1.0);
                self.draw(device, cb, layout, HUD_LINES, &pc_hud_text);
            }
        }
    }

    /// Bind the buffer of `slot`, push `pc` and draw all of its vertices.
    ///
    /// # Safety
    /// Inside a render pass with a pipeline of `layout` bound.
    unsafe fn draw(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        slot: usize,
        pc: &GridPushConstants,
    ) {
        // SAFETY: The caller guarantees the render pass and pipeline state.
        unsafe {
            device.cmd_bind_vertex_buffers(cb, 0, &[self.buffers[slot].handle()], &[0]);
            push_constants(
                device,
                cb,
                layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                pc,
            );
            device.cmd_draw(cb, self.counts[slot], 1, 0, 0);
        }
    }

    /// Free the buffers.
    ///
    /// # Safety
    /// No pending GPU work uses them; `device` and `allocator` are the ones
    /// they were created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        for buffer in self.buffers {
            // SAFETY: The caller guarantees the buffers are unused.
            unsafe { buffer.destroy(device, allocator) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_tracer_is_clipped_to_what_was_uploaded() {
        assert_eq!(uploaded(20, 100), 20);
        assert_eq!(uploaded(20, 15), 15);
        assert_eq!(uploaded(0, 0), 0);
        assert_eq!(uploaded(usize::MAX, 7), 7);
    }
}
