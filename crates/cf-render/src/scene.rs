//! The static scene both renderers draw: the range's scenery layers, the
//! tee box and the ball on the tee, followed by the frame's flights.

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use cf_scene::color;
use cf_scene::grid::GridVertex;
use cf_scene::scenery::{Layer, Scenery};
use cf_scene::tee::{
    TeeBox, generate_ball, generate_ball_glow, generate_tee_border, generate_tee_fill,
};

use crate::error::RenderError;
use crate::frame::FrameResources;
use crate::gpu::{HostBuffer, push_constants};
use crate::pipeline::{GridPushConstants, ScenePipelines};

/// One kind of scenery layer (fills or lines) in one vertex buffer: each
/// layer's color, first vertex and vertex count.
struct LayerSet {
    /// `None` when every layer of the set is empty.
    buffer: Option<HostBuffer>,
    draws: Vec<([f32; 4], u32, u32)>,
}

impl LayerSet {
    /// Upload `layers` into one buffer, skipping empty ones.
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        layers: &[Layer],
        name: &'static str,
    ) -> Result<Self, RenderError> {
        let mut verts: Vec<GridVertex> = Vec::new();
        let mut draws = Vec::new();
        for layer in layers.iter().filter(|l| !l.vertices.is_empty()) {
            draws.push((
                layer.color.into(),
                verts.len() as u32,
                layer.vertices.len() as u32,
            ));
            verts.extend_from_slice(&layer.vertices);
        }
        let buffer = if verts.is_empty() {
            None
        } else {
            Some(HostBuffer::with_data(
                device,
                allocator,
                &verts,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                name,
            )?)
        };
        Ok(Self { buffer, draws })
    }

    /// Draw every layer with `pipeline`, each in its color.
    ///
    /// # Safety
    /// Inside a scene pass of `pipes`, with viewport and scissor set.
    unsafe fn draw(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        pipes: &ScenePipelines,
        pipeline: vk::Pipeline,
        view_proj: [[f32; 4]; 4],
    ) {
        let Some(buffer) = &self.buffer else {
            return;
        };
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        // SAFETY: The caller guarantees an active scene pass of `pipes`; the
        // buffer lives until `destroy`, and the push constants match the layout.
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            if pipeline == pipes.line {
                device.cmd_set_line_width(cb, 1.0);
            }
            device.cmd_bind_vertex_buffers(cb, 0, &[buffer.handle()], &[0]);
            for &(color, first, count) in &self.draws {
                push_constants(
                    device,
                    cb,
                    pipes.layout,
                    stages,
                    &GridPushConstants { view_proj, color },
                );
                device.cmd_draw(cb, count, 1, first, 0);
            }
        }
    }

    /// Free the buffer.
    ///
    /// # Safety
    /// No pending GPU work uses it.
    unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(buffer) = self.buffer {
            // SAFETY: The caller guarantees nothing uses the buffer.
            unsafe { buffer.destroy(device, allocator) };
        }
    }
}

/// Uploaded static scene geometry.
pub(crate) struct SceneGeometry {
    fills: LayerSet,
    lines: LayerSet,
    /// Tee fill, tee border and ball, in that order.
    tee: HostBuffer,
    tee_fill_count: u32,
    tee_border_count: u32,
    tee_ball_count: u32,
    tee_glow: HostBuffer,
    tee_glow_count: u32,
    /// Width of every trail core line: 2 px where wide lines work, else 1 px.
    trail_line_width: f32,
}

impl SceneGeometry {
    /// Upload the layers of `scenery`, the tee box of `tee` and the ball on
    /// the tee. Trail cores are drawn 2 px wide when `wide_lines` is enabled
    /// on the device.
    ///
    /// # Errors
    /// An upload fails; what was uploaded is freed first.
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        scenery: &Scenery,
        tee: &TeeBox,
        wide_lines: bool,
    ) -> Result<Self, RenderError> {
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

        let fills = LayerSet::new(device, allocator, &scenery.fills, "scenery fills")?;
        let lines = match LayerSet::new(device, allocator, &scenery.lines, "scenery lines") {
            Ok(lines) => lines,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used it.
                unsafe { fills.destroy(device, allocator) };
                return Err(e);
            }
        };
        let [tee_buffer, glow_buffer] = match upload_all(
            device,
            allocator,
            [(&tee_verts, "tee and ball"), (&glow, "ball glow")],
        ) {
            Ok(buffers) => buffers,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe {
                    fills.destroy(device, allocator);
                    lines.destroy(device, allocator);
                }
                return Err(e);
            }
        };
        Ok(Self {
            fills,
            lines,
            tee: tee_buffer,
            tee_fill_count,
            tee_border_count,
            tee_ball_count,
            tee_glow: glow_buffer,
            tee_glow_count: glow.len() as u32,
            trail_line_width: if wide_lines { 2.0 } else { 1.0 },
        })
    }

    /// Draw the scene: scenery fills, scenery lines, tee box; then, with the ball on
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
            // Scenery: fills (TRIANGLE_LIST, depth write), then lines
            // (LINE_LIST, 1 px), each layer in its color.
            self.fills.draw(device, cb, pipes, pipes.fill, view_proj);
            self.lines.draw(device, cb, pipes, pipes.line, view_proj);

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
                // occludes its center)
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipes.glow);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.tee_glow.handle()], &[0]);
                push_constants(device, cb, pipes.layout, stages, &pc_magenta);
                device.cmd_draw(cb, self.tee_glow_count, 1, 0, 0);

                // Ball (black sphere — occludes the glow center, revealing the outline)
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

    /// Free every buffer.
    ///
    /// # Safety
    /// No pending GPU work uses them; `device` and `allocator` are the ones
    /// they were created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees nothing uses the buffers.
        unsafe {
            self.tee_glow.destroy(device, allocator);
            self.tee.destroy(device, allocator);
            self.lines.destroy(device, allocator);
            self.fills.destroy(device, allocator);
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
