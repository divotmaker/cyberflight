use std::mem::ManuallyDrop;

use ash::vk;

use cf_scene::camera::Camera;
use cf_scene::color;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex, generate_grid_vertices};
use cf_scene::tee::{
    TeeBox, generate_ball, generate_ball_at, generate_ball_glow, generate_ball_glow_at,
    generate_tee_border, generate_tee_fill,
};
use cf_scene::trail::{TrailPoint, generate_trail_glow, generate_trail_line};

use crate::composite::CompositePass;
use crate::context::{GpuConfig, GpuContext};
use crate::error::RenderError;
use crate::flight::flight_vertices;
use crate::frame::{FrameContent, FrameResources};
use crate::framebuffer::OffscreenTarget;
use crate::gpu::{HostBuffer, push_constants};
use crate::green::GreenEdge;
use crate::pipeline::{GridPushConstants, ScenePipelines};
use crate::readback::FrameBuffers;
use crate::recorder::{ChaseViewport, view_proj};
use crate::render::FlightRenderData;
use crate::rt_offscreen::{RtTrail, build_scene_geometry};
use crate::rt_pipeline::{RtPipeline, RtPushConstants};

/// Offscreen renderer for headless PNG generation.
pub struct OffscreenRenderer {
    gpu: GpuContext,
    pipeline: ManuallyDrop<ScenePipelines>,
    target: Option<OffscreenTarget>,
    vertex_buffer: ManuallyDrop<HostBuffer>,
    vertex_count: u32,
    /// Filled geometry: tee box fill + tee box border + ball.
    fill_buffer: ManuallyDrop<HostBuffer>,
    tee_fill_count: u32,
    tee_border_count: u32,
    ball_count: u32,
    glow_buffer: ManuallyDrop<HostBuffer>,
    glow_count: u32,
    /// Trail centerline (LINE_LIST, always visible regardless of viewing angle).
    trail_line_buffer: Option<HostBuffer>,
    trail_line_count: u32,
    /// Next frame's flights and HUD, uploaded into `frame` by `render`.
    content: FrameContent,
    /// Per-frame flight and HUD buffers.
    frame: ManuallyDrop<FrameResources>,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    // Optional RT pipeline + composite pass (when RT hardware available).
    rt_pipeline: Option<RtPipeline>,
    composite: Option<CompositePass>,
    /// Ball center position for RT push constants (xyz).
    rt_ball_center: glam::Vec3,
    /// RT trail fade distance (arc length of the reflected trail, meters).
    rt_trail_fade_dist: f32,
    /// Tracer fade value at the reflected trail's oldest point.
    rt_trail_tail_t: f32,
    /// Optional chase camera for split viewport (right 20%).
    chase_camera: Option<Camera>,

    /// Edge of the range's green, if any (see [`OffscreenRenderer::set_green`]).
    green_edge: Option<GreenEdge>,
}

/// Offscreen image format (RGBA8 sRGB for correct gamma in PNG output).
const OFFSCREEN_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

impl OffscreenRenderer {
    /// Create a headless renderer for the given grid configuration.
    /// Enables raytracing if hardware supports it.
    pub fn new(width: u32, height: u32, grid_config: &GridConfig) -> Result<Self, RenderError> {
        let config = GpuConfig {
            validation: cfg!(debug_assertions),
            enable_raytracing: true,
        };

        let mut gpu = GpuContext::new_headless(config)?;
        let rt_available = gpu.rt_supported;

        let pipeline = ScenePipelines::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            pipeline.scene_pass,
            pipeline.hud_pass,
        )?;

        let grid_verts = generate_grid_vertices(grid_config);
        let vertex_count = grid_verts.len() as u32;
        let vertex_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &grid_verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "grid",
        )?;

        // Tee box + ball geometry (all TRIANGLE_LIST, packed into one buffer)
        let tee = TeeBox::default();
        let tee_fill_verts = generate_tee_fill(&tee);
        let tee_border_verts = generate_tee_border(&tee);
        let ball_verts = generate_ball(&tee, 12, 24);
        let tee_fill_count = tee_fill_verts.len() as u32;
        let tee_border_count = tee_border_verts.len() as u32;
        let ball_count = ball_verts.len() as u32;

        let mut fill_verts = tee_fill_verts;
        fill_verts.extend(tee_border_verts);
        fill_verts.extend(ball_verts);
        let fill_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &fill_verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "tee and ball",
        )?;

        let glow_verts = generate_ball_glow(&tee, 16, 32);
        let glow_count = glow_verts.len() as u32;
        let glow_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &glow_verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "ball glow",
        )?;

        let frame = FrameResources::new(&gpu.device, &mut gpu.allocator)?;

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Build RT pipeline + composite pass if hardware supports it.
        let ball_center = glam::Vec3::new(0.0, tee.ball_radius, 0.0);
        let geometries = build_scene_geometry(grid_config, &tee, &[]);
        let (rt_pipeline, composite) = if rt_available {
            let rtp = RtPipeline::new(&mut gpu, width, height, &geometries)?;
            let composite = CompositePass::new(&gpu.device, OFFSCREEN_FORMAT, rtp.storage.view())?;
            (Some(rtp), Some(composite))
        } else {
            (None, None)
        };

        Ok(Self {
            gpu,
            pipeline: ManuallyDrop::new(pipeline),
            target: Some(target),
            vertex_buffer: ManuallyDrop::new(vertex_buffer),
            vertex_count,
            fill_buffer: ManuallyDrop::new(fill_buffer),
            tee_fill_count,
            tee_border_count,
            ball_count,
            glow_buffer: ManuallyDrop::new(glow_buffer),
            glow_count,
            trail_line_buffer: None,
            trail_line_count: 0,
            content: FrameContent::default(),
            frame: ManuallyDrop::new(frame),
            command_buffer,
            fence,

            rt_pipeline,
            composite,
            rt_ball_center: ball_center,
            rt_trail_fade_dist: 0.0,
            rt_trail_tail_t: 0.0,
            chase_camera: None,

            green_edge: None,
        })
    }

    /// Create a headless renderer with a ball in flight and a tracer trail.
    ///
    /// `ball_pos` is the current ball center (bottom of sphere touches this Y - radius).
    /// `trail_points` go from oldest (index 0) to newest (last = near ball_pos).
    /// `camera` is used to orient the trail glow ribbon toward the viewer.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flight(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        ball_pos: glam::Vec3,
        trail_points: &[TrailPoint],
        current_time: f64,
        max_lifetime: f64,
        camera: &Camera,
    ) -> Result<Self, RenderError> {
        let config = GpuConfig {
            validation: cfg!(debug_assertions),
            enable_raytracing: false,
        };

        let mut gpu = GpuContext::new_headless(config)?;

        let pipeline = ScenePipelines::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            pipeline.scene_pass,
            pipeline.hud_pass,
        )?;

        // Grid (same as default)
        let grid_verts = generate_grid_vertices(grid_config);
        let vertex_count = grid_verts.len() as u32;
        let vertex_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &grid_verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "grid",
        )?;

        // Tee box + ball at flight position
        let tee = TeeBox::default();
        let tee_fill_verts = generate_tee_fill(&tee);
        let tee_border_verts = generate_tee_border(&tee);
        let ball_center = ball_pos + glam::Vec3::new(0.0, tee.ball_radius, 0.0);
        let ball_verts = generate_ball_at(ball_center, tee.ball_radius, 12, 24);
        let tee_fill_count = tee_fill_verts.len() as u32;
        let tee_border_count = tee_border_verts.len() as u32;
        let ball_count = ball_verts.len() as u32;

        let mut fill_verts = tee_fill_verts;
        fill_verts.extend(tee_border_verts);
        fill_verts.extend(ball_verts);
        let fill_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &fill_verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "tee and ball",
        )?;

        // Ball glow sphere + trail glow ribbon (with endcap to bridge the junction).
        let mut glow_verts = generate_ball_glow_at(ball_center, tee.ball_radius, 16, 32);
        glow_verts.extend(generate_trail_glow(
            trail_points,
            current_time,
            max_lifetime,
            camera.position,
            tee.ball_radius,
        ));
        let glow_count = glow_verts.len() as u32;
        let glow_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &glow_verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "ball glow",
        )?;

        // Trail centerline (LINE_LIST — always visible regardless of viewing angle).
        let trail_line_verts = generate_trail_line(trail_points, current_time, max_lifetime);
        let (trail_line_buffer, trail_line_count) = if trail_line_verts.is_empty() {
            (None, 0)
        } else {
            let buffer = HostBuffer::with_data(
                &gpu.device,
                &mut gpu.allocator,
                &trail_line_verts,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "trail line",
            )?;
            (Some(buffer), trail_line_verts.len() as u32)
        };

        let frame = FrameResources::new(&gpu.device, &mut gpu.allocator)?;

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        Ok(Self {
            gpu,
            pipeline: ManuallyDrop::new(pipeline),
            target: Some(target),
            vertex_buffer: ManuallyDrop::new(vertex_buffer),
            vertex_count,
            fill_buffer: ManuallyDrop::new(fill_buffer),
            tee_fill_count,
            tee_border_count,
            ball_count,
            glow_buffer: ManuallyDrop::new(glow_buffer),
            glow_count,
            trail_line_buffer,
            trail_line_count,
            content: FrameContent::default(),
            frame: ManuallyDrop::new(frame),
            command_buffer,
            fence,

            rt_pipeline: None,
            composite: None,
            rt_ball_center: ball_center,
            rt_trail_fade_dist: 0.0,
            rt_trail_tail_t: 0.0,
            chase_camera: None,

            green_edge: None,
        })
    }

    /// Create a headless renderer with RT reflection compositing for the static range.
    ///
    /// Falls back to raster-only if RT hardware is not available.
    pub fn new_rt(width: u32, height: u32, grid_config: &GridConfig) -> Result<Self, RenderError> {
        let tee = TeeBox::default();
        let ball_center = glam::Vec3::new(0.0, cf_scene::tee::TEE_ELEVATION + tee.ball_radius, 0.0);
        let geometries = build_scene_geometry(grid_config, &tee, &[]);
        Self::create_rt(
            width,
            height,
            grid_config,
            &tee,
            ball_center,
            &geometries,
            None,
            &RtTrail::default(),
        )
    }

    /// Create a headless renderer with RT reflection compositing and a ball in flight.
    ///
    /// Falls back to raster-only if RT hardware is not available.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flight_rt(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        ball_pos: glam::Vec3,
        trail_points: &[TrailPoint],
        current_time: f64,
        max_lifetime: f64,
        camera: &Camera,
    ) -> Result<Self, RenderError> {
        let tee = TeeBox::default();
        let ball_center = ball_pos + glam::Vec3::new(0.0, tee.ball_radius, 0.0);

        // Reflect the part of the trail the raster tracer draws.
        let trail = RtTrail::new(trail_points, current_time, max_lifetime);
        let geometries = build_scene_geometry(grid_config, &tee, &trail.points);

        Self::create_rt(
            width,
            height,
            grid_config,
            &tee,
            ball_center,
            &geometries,
            Some((trail_points, current_time, max_lifetime, camera)),
            &trail,
        )
    }

    /// Internal RT constructor shared by `new_rt` and `new_with_flight_rt`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn create_rt(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        tee: &TeeBox,
        ball_center: glam::Vec3,
        geometries: &[crate::rt_pipeline::RtGeometry],
        flight: Option<(&[TrailPoint], f64, f64, &Camera)>,
        trail: &RtTrail,
    ) -> Result<Self, RenderError> {
        let config = GpuConfig {
            validation: cfg!(debug_assertions),
            enable_raytracing: true,
        };

        let mut gpu = GpuContext::new_headless(config)?;

        // Determine if RT is actually available.
        let rt_available = gpu.rt_supported;

        let pipeline = ScenePipelines::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            pipeline.scene_pass,
            pipeline.hud_pass,
        )?;

        // Grid vertices
        let grid_verts = generate_grid_vertices(grid_config);
        let vertex_count = grid_verts.len() as u32;
        let vertex_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &grid_verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "grid",
        )?;

        // Tee box + ball geometry
        let (
            tee_fill_count,
            tee_border_count,
            ball_count,
            fill_buffer,
            glow_count,
            glow_buffer,
            trail_line_buffer,
            trail_line_count,
        ) = if let Some((trail_points, current_time, max_lifetime, camera)) = flight {
            let tee_fill_verts = generate_tee_fill(tee);
            let tee_border_verts = generate_tee_border(tee);
            let ball_verts = generate_ball_at(ball_center, tee.ball_radius, 12, 24);
            let tfc = tee_fill_verts.len() as u32;
            let tbc = tee_border_verts.len() as u32;
            let bc = ball_verts.len() as u32;
            let mut fv = tee_fill_verts;
            fv.extend(tee_border_verts);
            fv.extend(ball_verts);
            let fb = HostBuffer::with_data(
                &gpu.device,
                &mut gpu.allocator,
                &fv,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "tee and ball",
            )?;

            let mut gv = generate_ball_glow_at(ball_center, tee.ball_radius, 16, 32);
            gv.extend(generate_trail_glow(
                trail_points,
                current_time,
                max_lifetime,
                camera.position,
                tee.ball_radius,
            ));
            let gc = gv.len() as u32;
            let gb = HostBuffer::with_data(
                &gpu.device,
                &mut gpu.allocator,
                &gv,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "ball glow",
            )?;

            // Trail centerline (LINE_LIST).
            let tlv = generate_trail_line(trail_points, current_time, max_lifetime);
            let (tlb, tlc) = if tlv.is_empty() {
                (None, 0u32)
            } else {
                let b = HostBuffer::with_data(
                    &gpu.device,
                    &mut gpu.allocator,
                    &tlv,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    "trail line",
                )?;
                (Some(b), tlv.len() as u32)
            };

            (tfc, tbc, bc, fb, gc, gb, tlb, tlc)
        } else {
            let tee_fill_verts = generate_tee_fill(tee);
            let tee_border_verts = generate_tee_border(tee);
            let ball_verts = generate_ball(tee, 12, 24);
            let tfc = tee_fill_verts.len() as u32;
            let tbc = tee_border_verts.len() as u32;
            let bc = ball_verts.len() as u32;
            let mut fv = tee_fill_verts;
            fv.extend(tee_border_verts);
            fv.extend(ball_verts);
            let fb = HostBuffer::with_data(
                &gpu.device,
                &mut gpu.allocator,
                &fv,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "tee and ball",
            )?;

            let gv = generate_ball_glow(tee, 16, 32);
            let gc = gv.len() as u32;
            let gb = HostBuffer::with_data(
                &gpu.device,
                &mut gpu.allocator,
                &gv,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "ball glow",
            )?;

            (tfc, tbc, bc, fb, gc, gb, None, 0u32)
        };

        let frame = FrameResources::new(&gpu.device, &mut gpu.allocator)?;

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Build RT pipeline + composite pass if hardware supports it.
        let (rt_pipeline, composite) = if rt_available {
            let rtp = RtPipeline::new(&mut gpu, width, height, geometries)?;
            let composite = CompositePass::new(&gpu.device, OFFSCREEN_FORMAT, rtp.storage.view())?;
            (Some(rtp), Some(composite))
        } else {
            (None, None)
        };

        Ok(Self {
            gpu,
            pipeline: ManuallyDrop::new(pipeline),
            target: Some(target),
            vertex_buffer: ManuallyDrop::new(vertex_buffer),
            vertex_count,
            fill_buffer: ManuallyDrop::new(fill_buffer),
            tee_fill_count,
            tee_border_count,
            ball_count,
            glow_buffer: ManuallyDrop::new(glow_buffer),
            glow_count,
            trail_line_buffer,
            trail_line_count,
            content: FrameContent::default(),
            frame: ManuallyDrop::new(frame),
            command_buffer,
            fence,

            rt_pipeline,
            composite,
            rt_ball_center: ball_center,
            rt_trail_fade_dist: trail.fade_dist,
            rt_trail_tail_t: trail.tail_t,
            chase_camera: None,

            green_edge: None,
        })
    }

    /// Draw all scene geometry (grid, green edge, tee box, ball, glow) with
    /// the given view-projection.
    ///
    /// Assumes viewport and scissor are already set, and we're inside a render pass.
    ///
    /// # Safety
    /// Must be called within an active render pass.
    unsafe fn draw_scene_geometry(&self, view_proj: [[f32; 4]; 4]) {
        // SAFETY: Caller guarantees we're inside an active render pass with valid
        // command buffer. All buffers and pipelines are created during construction.
        unsafe {
            let device = &self.gpu.device;
            let cb = self.command_buffer;

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

            // Grid lines (LINE_LIST, cyan)
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.line);
            device.cmd_set_line_width(cb, 1.0);
            push_constants(
                device,
                cb,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc_cyan,
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer.handle()], &[0]);
            device.cmd_draw(cb, self.vertex_count, 1, 0, 0);

            // Green edge (LINE_LIST, neon green)
            if let Some(edge) = &self.green_edge {
                edge.record(device, cb, &self.pipeline, view_proj);
            }

            // Tee box fill (black)
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.fill);
            device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer.handle()], &[0]);
            push_constants(
                device,
                cb,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc_black,
            );
            device.cmd_draw(cb, self.tee_fill_count, 1, 0, 0);

            // Tee box border (cyan)
            push_constants(
                device,
                cb,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc_cyan,
            );
            device.cmd_draw(cb, self.tee_border_count, 1, self.tee_fill_count, 0);

            if self.frame.ball_on_tee() {
                // Static ball: trail + glow + ball from construction-time buffers.
                if let Some(trail_buf) = &self.trail_line_buffer {
                    device.cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline.line,
                    );
                    device.cmd_set_line_width(cb, 1.0);
                    device.cmd_bind_vertex_buffers(cb, 0, &[trail_buf.handle()], &[0]);
                    push_constants(
                        device,
                        cb,
                        self.pipeline.layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        &pc_magenta,
                    );
                    device.cmd_draw(cb, self.trail_line_count, 1, 0, 0);
                }

                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.glow);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.glow_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc_magenta,
                );
                device.cmd_draw(cb, self.glow_count, 1, 0, 0);

                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.fill);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc_black,
                );
                device.cmd_draw(
                    cb,
                    self.ball_count,
                    1,
                    self.tee_fill_count + self.tee_border_count,
                    0,
                );
            } else {
                // In-flight trails, glows and balls.
                self.frame
                    .draw_flights(device, cb, &self.pipeline, view_proj, 1.0);
            }
        } // unsafe
    }

    /// Whether this renderer has RT reflection compositing enabled.
    #[must_use]
    pub fn has_rt(&self) -> bool {
        self.rt_pipeline.is_some()
    }

    /// Draw the edge of `green` as one ring on the range floor, faded like
    /// `grid`; `None` draws nothing. Replaces any green set before.
    pub fn set_green(
        &mut self,
        green: Option<&GreenConfig>,
        grid: &GridConfig,
    ) -> Result<(), RenderError> {
        if let Some(old) = self.green_edge.take() {
            // SAFETY: Waiting idle so the old edge buffer is no longer in use.
            unsafe {
                self.gpu
                    .device
                    .device_wait_idle()
                    .map_err(RenderError::Vulkan)?;
                old.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }
        }
        self.green_edge = GreenEdge::new(&self.gpu.device, &mut self.gpu.allocator, green, grid)?;
        Ok(())
    }

    /// Set or clear the chase camera for the right 20% split viewport.
    ///
    /// Pass `Some(camera)` while a ball is being followed, `None` otherwise.
    pub fn set_chase_camera(&mut self, camera: Option<Camera>) {
        self.chase_camera = camera;
    }

    /// Set the HUD overlay geometry (pixel coordinates) for the next
    /// `render`; the same as [`OffscreenRenderer::update_hud`].
    ///
    /// # Errors
    /// None; the HUD buffers always exist.
    pub fn set_hud(
        &mut self,
        lines: &[GridVertex],
        fills: &[GridVertex],
    ) -> Result<(), RenderError> {
        self.update_hud(lines, fills);
        Ok(())
    }

    /// Does nothing: the per-frame flight and HUD buffers always exist.
    ///
    /// # Errors
    /// None.
    pub fn init_streaming(&mut self) -> Result<(), RenderError> {
        Ok(())
    }

    /// Update dynamic flight geometry for the next `render`, and rebuild the
    /// RT scene from the newest flight.
    pub fn update_flight(&mut self, flights: &[FlightRenderData], camera: &Camera) {
        use cf_scene::trail::DEFAULT_TRAIL_LIFETIME;

        let tee = TeeBox::default();
        flight_vertices(flights, camera.position, &tee, &mut self.content);

        // Rebuild RT TLAS with updated ball/trail positions. Reflections
        // follow the newest flight, the one the HUD and chase camera show,
        // and reflect the part of its trail the raster tracer still draws.
        if let Some(ref mut rt_pipeline) = self.rt_pipeline {
            let newest = flights.last();
            let trail = newest.map_or_else(RtTrail::default, |f| {
                RtTrail::new(&f.trail_points, f.current_time, DEFAULT_TRAIL_LIFETIME)
            });

            if let Some(newest) = newest {
                self.rt_ball_center = newest.ball_pos;
                self.rt_trail_tail_t = trail.tail_t;
            }
            self.rt_trail_fade_dist = trail.fade_dist;

            let grid_config = cf_scene::grid::GridConfig::default();
            let geometries = build_scene_geometry(&grid_config, &tee, &trail.points);
            if let Err(e) = rt_pipeline.update_scene(&mut self.gpu, &geometries) {
                eprintln!("RT scene update failed: {e}");
            }
        }
    }

    /// Update HUD overlay geometry (pixel coordinates) for the next `render`.
    pub fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.content.hud_lines.clear();
        self.content.hud_lines.extend_from_slice(lines);
        self.content.hud_fills.clear();
        self.content.hud_fills.extend_from_slice(fills);
    }

    /// Update HUD overlay geometry for the next `render`; the same as
    /// [`OffscreenRenderer::update_hud`].
    pub fn update_hud_streaming(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.update_hud(lines, fills);
    }

    /// Render the grid and read back the result as RGBA8 pixels.
    ///
    /// When RT is enabled, renders the raster scene, traces RT reflection
    /// rays, and composites them onto the rasterized output.
    pub fn render(&mut self, camera: &Camera) -> Result<FrameBuffers, RenderError> {
        self.frame.upload(&self.content);
        let target = self.target.as_ref().expect("target exists");
        let device = &self.gpu.device;
        let width = target.width;
        let height = target.height;
        let extent = vk::Extent2D { width, height };
        let rt_active = self.rt_pipeline.is_some();

        // Record command buffer
        // SAFETY: Recording commands into a valid command buffer.
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(RenderError::Vulkan)?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(RenderError::Vulkan)?;

            // Begin render pass
            let clear_color = [
                color::BACKGROUND.x,
                color::BACKGROUND.y,
                color::BACKGROUND.z,
                color::BACKGROUND.w,
            ];
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: clear_color,
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.pipeline.scene_pass)
                .framebuffer(target.scene)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(
                self.command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.line,
            );
            device.cmd_set_line_width(self.command_buffer, 1.0);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);

            // Main scene
            let aspect = width as f32 / height as f32;
            self.draw_scene_geometry(view_proj(camera, aspect));

            // Chase camera: right 20% viewport
            if let Some(chase) = &self.chase_camera.clone() {
                let chase_vp = ChaseViewport::new(width);

                let clear_attachments = [
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        color_attachment: 0,
                        clear_value: vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [
                                    color::BACKGROUND.x,
                                    color::BACKGROUND.y,
                                    color::BACKGROUND.z,
                                    color::BACKGROUND.w,
                                ],
                            },
                        },
                    },
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        color_attachment: 0,
                        clear_value: vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    },
                ];
                let clear_rect = vk::ClearRect {
                    rect: chase_vp.rect(height),
                    base_array_layer: 0,
                    layer_count: 1,
                };
                device.cmd_clear_attachments(
                    self.command_buffer,
                    &clear_attachments,
                    &[clear_rect],
                );

                device.cmd_set_viewport(self.command_buffer, 0, &[chase_vp.viewport(height)]);
                device.cmd_set_scissor(self.command_buffer, 0, &[chase_vp.rect(height)]);

                self.draw_scene_geometry(view_proj(chase, chase_vp.aspect(height)));

                // Restore full viewport for HUD
                device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
                device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
            }

            // HUD overlay (2D, pixel coordinates)
            self.frame.draw_hud(
                device,
                self.command_buffer,
                self.pipeline.fill,
                self.pipeline.line,
                self.pipeline.layout,
                extent,
            );

            device.cmd_end_render_pass(self.command_buffer);

            // ── RT reflection compositing ──
            if rt_active {
                self.record_rt_composite(camera, width, height, extent, target.overlay);
            }

            // The HUD pass ends the frame and leaves the image in
            // TRANSFER_SRC_OPTIMAL. The HUD itself is drawn in the scene pass.
            let hud_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.pipeline.hud_pass)
                .framebuffer(target.overlay)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                });
            device.cmd_begin_render_pass(
                self.command_buffer,
                &hud_pass_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_end_render_pass(self.command_buffer);

            // Barrier: HUD pass output -> transfer read
            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(target.color.image())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Copy image to staging buffer
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });

            device.cmd_copy_image_to_buffer(
                self.command_buffer,
                target.color.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target.staging.handle(),
                &[region],
            );

            device
                .end_command_buffer(self.command_buffer)
                .map_err(RenderError::Vulkan)?;
        }

        // Submit and wait
        let command_buffers = [self.command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        // SAFETY: Submitting and waiting on a valid fence.
        unsafe {
            device
                .reset_fences(&[self.fence])
                .map_err(RenderError::Vulkan)?;
            device
                .queue_submit(self.gpu.graphics_queue, &[submit_info], self.fence)
                .map_err(RenderError::Vulkan)?;
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(RenderError::Vulkan)?;
        }

        // Read back pixels from staging buffer
        let pixel_count = (width * height * 4) as usize;
        let pixels = target
            .staging
            .bytes()
            .get(..pixel_count)
            .ok_or_else(|| RenderError::Allocator("staging buffer smaller than the frame".into()))?
            .to_vec();

        Ok(FrameBuffers {
            color: pixels,
            width,
            height,
        })
    }

    /// Record RT trace + composite pass commands.
    ///
    /// After the main raster render pass ends at `COLOR_ATTACHMENT_OPTIMAL`:
    /// 1. Transition RT storage image: UNDEFINED -> GENERAL
    /// 2. Trace RT reflection rays
    /// 3. Transition RT storage image: GENERAL -> SHADER_READ_ONLY_OPTIMAL
    /// 4. Composite pass into `overlay` (LOAD, additive blend, stays in
    ///    `COLOR_ATTACHMENT_OPTIMAL`)
    ///
    /// # Safety
    /// The command buffer must be in the recording state, outside any render pass.
    unsafe fn record_rt_composite(
        &self,
        camera: &Camera,
        width: u32,
        height: u32,
        extent: vk::Extent2D,
        overlay: vk::Framebuffer,
    ) {
        // SAFETY: The command buffer is in the recording state, outside any render pass.
        // All Vulkan handles are valid (created in the constructor, not yet destroyed).
        unsafe {
            let device = &self.gpu.device;
            let rt = self.rt_pipeline.as_ref().expect("RT pipeline present");

            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            // Build RT push constants (full-frame dispatch, no viewport offset).
            let pc = RtPushConstants::new(
                camera,
                width as f32 / height as f32,
                [0.0, 0.0],
                self.rt_ball_center,
                self.rt_trail_fade_dist,
                self.rt_trail_tail_t,
            );

            // Step 1: Transition RT storage image: UNDEFINED -> GENERAL
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(rt.storage.image())
                    .subresource_range(subresource_range)],
            );

            // Step 2: Trace RT reflection rays.
            rt.record_trace(device, self.command_buffer, &pc, width, height);

            // Step 3: Transition RT storage image: GENERAL -> SHADER_READ_ONLY_OPTIMAL
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(rt.storage.image())
                    .subresource_range(subresource_range)],
            );

            // Step 4: Composite pass (LOAD, additive blend fullscreen triangle)
            // into the overlay framebuffer.
            if let Some(composite) = &self.composite {
                composite.record(device, self.command_buffer, overlay, extent);
            }
        } // unsafe
    }

    fn allocate_command_buffer(gpu: &GpuContext) -> Result<vk::CommandBuffer, RenderError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(gpu.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: Allocating a command buffer.
        let cbs = unsafe {
            gpu.device
                .allocate_command_buffers(&alloc_info)
                .map_err(RenderError::Vulkan)?
        };

        Ok(cbs[0])
    }
}

impl Drop for OffscreenRenderer {
    fn drop(&mut self) {
        unsafe {
            self.gpu.device.device_wait_idle().ok();

            if let Some(edge) = self.green_edge.take() {
                edge.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            self.gpu.device.destroy_fence(self.fence, None);

            let device = &self.gpu.device;
            let allocator = &mut self.gpu.allocator;
            // SAFETY: Each `ManuallyDrop` buffer is taken once, here in `Drop`,
            // and never used after.
            ManuallyDrop::take(&mut self.vertex_buffer).destroy(device, allocator);
            ManuallyDrop::take(&mut self.fill_buffer).destroy(device, allocator);
            ManuallyDrop::take(&mut self.glow_buffer).destroy(device, allocator);
            if let Some(buffer) = self.trail_line_buffer.take() {
                buffer.destroy(device, allocator);
            }
            ManuallyDrop::take(&mut self.frame).destroy(device, allocator);

            if let Some(composite) = self.composite.take() {
                composite.destroy(&self.gpu.device);
            }

            if let Some(rtp) = self.rt_pipeline.take() {
                rtp.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            if let Some(target) = self.target.take() {
                target.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            // SAFETY: Taken once, here in `Drop`, and never used after.
            ManuallyDrop::take(&mut self.pipeline).destroy(&self.gpu.device);
        }
    }
}
