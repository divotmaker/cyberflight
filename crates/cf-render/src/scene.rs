//! The static scene both renderers draw: grid, optional green edge, tee box
//! and the ball on the tee, followed by the frame's flights.

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use cf_scene::color;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex, generate_grid_vertices};
use cf_scene::tee::{
    TeeBox, generate_ball, generate_ball_glow, generate_tee_border, generate_tee_fill,
};

use crate::error::RenderError;
use crate::frame::FrameResources;
use crate::gpu::{HostBuffer, push_constants};
use crate::green::GreenEdge;
use crate::pipeline::{GridPushConstants, ScenePipelines};

/// Uploaded static scene geometry.
pub(crate) struct SceneGeometry {
    grid: HostBuffer,
    grid_count: u32,
    /// Tee fill, tee border and ball, in that order.
    tee: HostBuffer,
    tee_fill_count: u32,
    tee_border_count: u32,
    tee_ball_count: u32,
    tee_glow: HostBuffer,
    tee_glow_count: u32,
    /// `None` when there is no green.
    green_edge: Option<GreenEdge>,
    /// Width of every trail core line: 2 px where wide lines work, else 1 px.
    trail_line_width: f32,
}

impl SceneGeometry {
    /// Upload the grid of `grid`, the tee box of `tee` and the ball on the
    /// tee. No green. Trail cores are drawn 2 px wide when `wide_lines` is
    /// enabled on the device.
    ///
    /// # Errors
    /// An upload fails; what was uploaded is freed first.
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        grid: &GridConfig,
        tee: &TeeBox,
        wide_lines: bool,
    ) -> Result<Self, RenderError> {
        let grid_verts = generate_grid_vertices(grid);
        let tee_fill = generate_tee_fill(tee);
        let tee_border = generate_tee_border(tee);
        let ball = generate_ball(tee, 12, 24);
        let glow = generate_ball_glow(tee, 16, 32);
        let tee_fill_count = tee_fill.len() as u32;
        let tee_border_count = tee_border.len() as u32;
        let tee_ball_count = ball.len() as u32;
        let mut tee_verts = tee_fill;
        tee_verts.extend(tee_border);
        tee_verts.extend(ball);

        let [grid_buffer, tee_buffer, glow_buffer] = upload_all(
            device,
            allocator,
            [
                (&grid_verts, "grid"),
                (&tee_verts, "tee and ball"),
                (&glow, "ball glow"),
            ],
        )?;
        Ok(Self {
            grid: grid_buffer,
            grid_count: grid_verts.len() as u32,
            tee: tee_buffer,
            tee_fill_count,
            tee_border_count,
            tee_ball_count,
            tee_glow: glow_buffer,
            tee_glow_count: glow.len() as u32,
            green_edge: None,
            trail_line_width: if wide_lines { 2.0 } else { 1.0 },
        })
    }

    /// Draw the edge of `green` as one ring on the range floor, faded like
    /// `grid`; `None` draws nothing. Replaces any green set before, waiting
    /// for the device to go idle before it frees the old edge.
    ///
    /// # Errors
    /// The wait or the upload fails.
    pub(crate) fn set_green(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        green: Option<&GreenConfig>,
        grid: &GridConfig,
    ) -> Result<(), RenderError> {
        if let Some(old) = self.green_edge.take() {
            // SAFETY: Waiting idle so the old edge buffer is unused.
            unsafe {
                device.device_wait_idle().map_err(RenderError::Vulkan)?;
                old.destroy(device, allocator);
            }
        }
        self.green_edge = GreenEdge::new(device, allocator, green, grid)?;
        Ok(())
    }

    /// Draw the scene: grid, green edge, tee box; then, with the ball on
    /// the tee, its glow and the ball; then `frame`'s flights.
    ///
    /// # Safety
    /// Inside a scene pass of `pipes`, with viewport and scissor set.
    pub(crate) unsafe fn draw(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        pipes: &ScenePipelines,
        frame: &FrameResources,
        view_proj: [[f32; 4]; 4],
    ) {
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let pc_cyan = GridPushConstants {
            view_proj,
            color: color::CYAN.into(),
        };
        let pc_black = GridPushConstants {
            view_proj,
            color: [0.0, 0.0, 0.0, 1.0],
        };
        let pc_magenta = GridPushConstants {
            view_proj,
            color: color::MAGENTA.into(),
        };

        // SAFETY: The caller guarantees an active scene pass of `pipes`; the
        // buffers live until `destroy`, and the push constants match the layout.
        unsafe {
            // Grid lines (LINE_LIST, cyan, 1 px)
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipes.line);
            device.cmd_set_line_width(cb, 1.0);
            push_constants(device, cb, pipes.layout, stages, &pc_cyan);
            device.cmd_bind_vertex_buffers(cb, 0, &[self.grid.handle()], &[0]);
            device.cmd_draw(cb, self.grid_count, 1, 0, 0);

            // Green edge (LINE_LIST, neon green, 1 px)
            if let Some(edge) = &self.green_edge {
                edge.record(device, cb, pipes, view_proj);
            }

            // Tee box fill (black quad — masks grid lines under the tee box)
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipes.fill);
            device.cmd_bind_vertex_buffers(cb, 0, &[self.tee.handle()], &[0]);
            push_constants(device, cb, pipes.layout, stages, &pc_black);
            device.cmd_draw(cb, self.tee_fill_count, 1, 0, 0);

            // Tee box border (cyan filled quads)
            push_constants(device, cb, pipes.layout, stages, &pc_cyan);
            device.cmd_draw(cb, self.tee_border_count, 1, self.tee_fill_count, 0);

            if frame.ball_on_tee() {
                // Ball glow (additive, magenta — before the ball so the ball
                // occludes its centre)
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipes.glow);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.tee_glow.handle()], &[0]);
                push_constants(device, cb, pipes.layout, stages, &pc_magenta);
                device.cmd_draw(cb, self.tee_glow_count, 1, 0, 0);

                // Ball (black sphere — occludes the glow centre, revealing the outline)
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipes.fill);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.tee.handle()], &[0]);
                push_constants(device, cb, pipes.layout, stages, &pc_black);
                device.cmd_draw(
                    cb,
                    self.tee_ball_count,
                    1,
                    self.tee_fill_count + self.tee_border_count,
                    0,
                );
            }

            // In-flight trails, glows and balls (none while the ball is on the tee).
            frame.draw_flights(device, cb, pipes, view_proj, self.trail_line_width);
        }
    }

    /// Free every buffer, including the green edge.
    ///
    /// # Safety
    /// No pending GPU work uses them; `device` and `allocator` are the ones
    /// they were created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees nothing uses the buffers.
        unsafe {
            if let Some(edge) = self.green_edge {
                edge.destroy(device, allocator);
            }
            self.tee_glow.destroy(device, allocator);
            self.tee.destroy(device, allocator);
            self.grid.destroy(device, allocator);
        }
    }
}

/// Upload each `(vertices, name)` into its own vertex buffer; on failure,
/// free those already uploaded.
fn upload_all<const N: usize>(
    device: &ash::Device,
    allocator: &mut Allocator,
    sets: [(&[GridVertex], &'static str); N],
) -> Result<[HostBuffer; N], RenderError> {
    let mut made = Vec::with_capacity(N);
    for (verts, name) in sets {
        match HostBuffer::with_data(
            device,
            allocator,
            verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            name,
        ) {
            Ok(buffer) => made.push(buffer),
            Err(e) => {
                free_all(device, allocator, made);
                return Err(e);
            }
        }
    }
    <[HostBuffer; N]>::try_from(made).map_err(|made| {
        free_all(device, allocator, made);
        RenderError::Allocator("scene buffers: wrong count".into())
    })
}

/// Free buffers no GPU work has used yet.
fn free_all(device: &ash::Device, allocator: &mut Allocator, buffers: Vec<HostBuffer>) {
    for buffer in buffers {
        // SAFETY: Just created; no GPU work has used it.
        unsafe { buffer.destroy(device, allocator) };
    }
}
