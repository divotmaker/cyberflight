use std::mem::ManuallyDrop;

use ash::vk;
use gpu_allocator::MemoryLocation;

use cf_scene::camera::Camera;
use cf_scene::color;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex, generate_grid_vertices};
use cf_scene::tee::{
    TeeBox, generate_ball, generate_ball_at, generate_ball_glow, generate_ball_glow_at,
    generate_tee_border, generate_tee_fill,
};
use cf_scene::trail::{TrailPoint, generate_trail_glow, generate_trail_line};

use crate::context::{GpuConfig, GpuContext};
use crate::error::RenderError;
use crate::framebuffer::OffscreenTarget;
use crate::gpu::{HostBuffer, load_shader, push_constants};
use crate::green::GreenEdge;
use crate::pipeline::{GridPipeline, GridPushConstants};
use crate::readback::FrameBuffers;
use crate::recorder::{ChaseViewport, view_proj};
use crate::render::FlightRenderData;
use crate::rt_offscreen::{RtTrail, build_scene_geometry};
use crate::rt_pipeline::{RtPipeline, RtPushConstants};

/// Offscreen renderer for headless PNG generation.
pub struct OffscreenRenderer {
    gpu: GpuContext,
    pipeline: GridPipeline,
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
    // Optional HUD overlay.
    hud_line_buffer: Option<HostBuffer>,
    hud_line_count: u32,
    hud_fill_buffer: Option<HostBuffer>,
    hud_fill_count: u32,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    // Optional RT pipeline + composite pass (when RT hardware available).
    rt_pipeline: Option<RtPipeline>,
    composite_render_pass: Option<vk::RenderPass>,
    composite_framebuffer: Option<vk::Framebuffer>,
    composite_pipeline: Option<vk::Pipeline>,
    composite_pipeline_layout: Option<vk::PipelineLayout>,
    composite_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    composite_descriptor_pool: Option<vk::DescriptorPool>,
    composite_descriptor_set: Option<vk::DescriptorSet>,
    composite_sampler: Option<vk::Sampler>,
    composite_vert_module: Option<vk::ShaderModule>,
    composite_frag_module: Option<vk::ShaderModule>,
    /// Ball center position for RT push constants (xyz).
    rt_ball_center: glam::Vec3,
    /// RT trail fade distance (arc length of the reflected trail, meters).
    rt_trail_fade_dist: f32,
    /// Tracer fade value at the reflected trail's oldest point.
    rt_trail_tail_t: f32,
    /// Optional chase camera for split viewport (right 20%).
    chase_camera: Option<Camera>,

    // Dynamic per-frame flight geometry (pre-allocated for streaming use).
    flight_line_buffer: Option<HostBuffer>,
    flight_line_count: u32,
    flight_fill_buffer: Option<HostBuffer>,
    flight_fill_count: u32,
    flight_glow_buffer: Option<HostBuffer>,
    flight_glow_count: u32,
    /// When true, draw the static ball on tee. When false, draw from dynamic flight buffers.
    ball_on_tee_box: bool,
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

        // When RT is available, raster pass leaves image in COLOR_ATTACHMENT_OPTIMAL
        // so the composite pass can follow. Otherwise go straight to TRANSFER_SRC.
        let raster_final_layout = if rt_available {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };

        let pipeline = GridPipeline::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            raster_final_layout,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            pipeline.render_pass,
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
        let (
            rt_pipeline,
            composite_render_pass,
            composite_framebuffer,
            composite_pipeline,
            composite_pipeline_layout,
            composite_descriptor_set_layout,
            composite_descriptor_pool,
            composite_descriptor_set,
            composite_sampler,
            composite_vert_module,
            composite_frag_module,
        ) = if rt_available {
            let rtp = RtPipeline::new(&mut gpu, width, height, &geometries)?;
            let composite = Self::create_composite_pass(
                &gpu.device,
                OFFSCREEN_FORMAT,
                target.color.view(),
                rtp.storage.view(),
                width,
                height,
            )?;
            (
                Some(rtp),
                Some(composite.0),
                Some(composite.1),
                Some(composite.2),
                Some(composite.3),
                Some(composite.4),
                Some(composite.5),
                Some(composite.6),
                Some(composite.7),
                Some(composite.8),
                Some(composite.9),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None,
            )
        };

        Ok(Self {
            gpu,
            pipeline,
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
            hud_line_buffer: None,
            hud_line_count: 0,
            hud_fill_buffer: None,
            hud_fill_count: 0,
            command_buffer,
            fence,

            rt_pipeline,
            composite_render_pass,
            composite_framebuffer,
            composite_pipeline,
            composite_pipeline_layout,
            composite_descriptor_set_layout,
            composite_descriptor_pool,
            composite_descriptor_set,
            composite_sampler,
            composite_vert_module,
            composite_frag_module,
            rt_ball_center: ball_center,
            rt_trail_fade_dist: 0.0,
            rt_trail_tail_t: 0.0,
            chase_camera: None,

            flight_line_buffer: None,
            flight_line_count: 0,
            flight_fill_buffer: None,
            flight_fill_count: 0,
            flight_glow_buffer: None,
            flight_glow_count: 0,
            ball_on_tee_box: true,
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

        let pipeline = GridPipeline::new(
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
            pipeline.render_pass,
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
            pipeline,
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
            hud_line_buffer: None,
            hud_line_count: 0,
            hud_fill_buffer: None,
            hud_fill_count: 0,
            command_buffer,
            fence,

            rt_pipeline: None,
            composite_render_pass: None,
            composite_framebuffer: None,
            composite_pipeline: None,
            composite_pipeline_layout: None,
            composite_descriptor_set_layout: None,
            composite_descriptor_pool: None,
            composite_descriptor_set: None,
            composite_sampler: None,
            composite_vert_module: None,
            composite_frag_module: None,
            rt_ball_center: ball_center,
            rt_trail_fade_dist: 0.0,
            rt_trail_tail_t: 0.0,
            chase_camera: None,

            flight_line_buffer: None,
            flight_line_count: 0,
            flight_fill_buffer: None,
            flight_fill_count: 0,
            flight_glow_buffer: None,
            flight_glow_count: 0,
            ball_on_tee_box: true,
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

        // Choose raster final_layout based on whether RT composite follows.
        let raster_final_layout = if rt_available {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };

        let pipeline = GridPipeline::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            raster_final_layout,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            pipeline.render_pass,
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

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Build RT pipeline + composite pass if hardware supports it.
        let (
            rt_pipeline,
            composite_render_pass,
            composite_framebuffer,
            composite_pipeline,
            composite_pipeline_layout,
            composite_descriptor_set_layout,
            composite_descriptor_pool,
            composite_descriptor_set,
            composite_sampler,
            composite_vert_module,
            composite_frag_module,
        ) = if rt_available {
            let rtp = RtPipeline::new(&mut gpu, width, height, geometries)?;

            let composite = Self::create_composite_pass(
                &gpu.device,
                OFFSCREEN_FORMAT,
                target.color.view(),
                rtp.storage.view(),
                width,
                height,
            )?;

            (
                Some(rtp),
                Some(composite.0),
                Some(composite.1),
                Some(composite.2),
                Some(composite.3),
                Some(composite.4),
                Some(composite.5),
                Some(composite.6),
                Some(composite.7),
                Some(composite.8),
                Some(composite.9),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None,
            )
        };

        Ok(Self {
            gpu,
            pipeline,
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
            hud_line_buffer: None,
            hud_line_count: 0,
            hud_fill_buffer: None,
            hud_fill_count: 0,
            command_buffer,
            fence,

            rt_pipeline,
            composite_render_pass,
            composite_framebuffer,
            composite_pipeline,
            composite_pipeline_layout,
            composite_descriptor_set_layout,
            composite_descriptor_pool,
            composite_descriptor_set,
            composite_sampler,
            composite_vert_module,
            composite_frag_module,
            rt_ball_center: ball_center,
            rt_trail_fade_dist: trail.fade_dist,
            rt_trail_tail_t: trail.tail_t,
            chase_camera: None,

            flight_line_buffer: None,
            flight_line_count: 0,
            flight_fill_buffer: None,
            flight_fill_count: 0,
            flight_glow_buffer: None,
            flight_glow_count: 0,
            ball_on_tee_box: true,
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
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            device.cmd_set_line_width(cb, 1.0);
            push_constants(
                device,
                cb,
                self.pipeline.pipeline_layout,
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
            device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.fill_pipeline,
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer.handle()], &[0]);
            push_constants(
                device,
                cb,
                self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc_black,
            );
            device.cmd_draw(cb, self.tee_fill_count, 1, 0, 0);

            // Tee box border (cyan)
            push_constants(
                device,
                cb,
                self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc_cyan,
            );
            device.cmd_draw(cb, self.tee_border_count, 1, self.tee_fill_count, 0);

            if self.ball_on_tee_box {
                // Static ball: trail + glow + ball from construction-time buffers.
                if let Some(trail_buf) = &self.trail_line_buffer {
                    device.cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline.pipeline,
                    );
                    device.cmd_set_line_width(cb, 1.0);
                    device.cmd_bind_vertex_buffers(cb, 0, &[trail_buf.handle()], &[0]);
                    push_constants(
                        device,
                        cb,
                        self.pipeline.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        &pc_magenta,
                    );
                    device.cmd_draw(cb, self.trail_line_count, 1, 0, 0);
                }

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.glow_pipeline,
                );
                device.cmd_bind_vertex_buffers(cb, 0, &[self.glow_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc_magenta,
                );
                device.cmd_draw(cb, self.glow_count, 1, 0, 0);

                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.fill_pipeline,
                );
                device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.pipeline_layout,
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
                // Dynamic flight geometry from pre-allocated streaming buffers.
                if let Some(flight_line_buf) = &self.flight_line_buffer {
                    if self.flight_line_count > 0 {
                        device.cmd_bind_pipeline(
                            cb,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.pipeline,
                        );
                        device.cmd_set_line_width(cb, 1.0);
                        device.cmd_bind_vertex_buffers(cb, 0, &[flight_line_buf.handle()], &[0]);
                        push_constants(
                            device,
                            cb,
                            self.pipeline.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            &pc_magenta,
                        );
                        device.cmd_draw(cb, self.flight_line_count, 1, 0, 0);
                    }
                }
                if let Some(flight_glow_buf) = &self.flight_glow_buffer {
                    if self.flight_glow_count > 0 {
                        device.cmd_bind_pipeline(
                            cb,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.glow_pipeline,
                        );
                        device.cmd_bind_vertex_buffers(cb, 0, &[flight_glow_buf.handle()], &[0]);
                        push_constants(
                            device,
                            cb,
                            self.pipeline.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            &pc_magenta,
                        );
                        device.cmd_draw(cb, self.flight_glow_count, 1, 0, 0);
                    }
                }
                if let Some(flight_fill_buf) = &self.flight_fill_buffer {
                    if self.flight_fill_count > 0 {
                        device.cmd_bind_pipeline(
                            cb,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.fill_pipeline,
                        );
                        device.cmd_bind_vertex_buffers(cb, 0, &[flight_fill_buf.handle()], &[0]);
                        push_constants(
                            device,
                            cb,
                            self.pipeline.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            &pc_black,
                        );
                        device.cmd_draw(cb, self.flight_fill_count, 1, 0, 0);
                    }
                }
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

    /// Add HUD overlay geometry (call before `render()`). Replaces the HUD
    /// buffers it is given new geometry for.
    pub fn set_hud(
        &mut self,
        lines: &[GridVertex],
        fills: &[GridVertex],
    ) -> Result<(), RenderError> {
        if !lines.is_empty() {
            free_slot(&mut self.gpu, &mut self.hud_line_buffer)?;
            self.hud_line_buffer = Some(HostBuffer::with_data(
                &self.gpu.device,
                &mut self.gpu.allocator,
                lines,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "hud lines",
            )?);
            self.hud_line_count = lines.len() as u32;
        }
        if !fills.is_empty() {
            free_slot(&mut self.gpu, &mut self.hud_fill_buffer)?;
            self.hud_fill_buffer = Some(HostBuffer::with_data(
                &self.gpu.device,
                &mut self.gpu.allocator,
                fills,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "hud fill",
            )?);
            self.hud_fill_count = fills.len() as u32;
        }
        Ok(())
    }

    /// Allocate pre-sized dynamic buffers for streaming use, replacing any
    /// flight or HUD buffers allocated before.
    ///
    /// Call once after construction to enable `update_flight()` and `update_hud_streaming()`.
    /// Without this, the renderer only supports static geometry set at construction time.
    pub fn init_streaming(&mut self) -> Result<(), RenderError> {
        // Flight line: trail centerlines, ~2000 points × 2 verts × 16 bytes = ~64 KB
        // Flight fill: up to 5 in-flight balls × 1728 verts = ~140 KB
        // Flight glow: up to 5 balls × (ball glow + trail glow) ≈ 6 MB
        // HUD: line and fill buffers sized like the windowed renderer's.
        let gpu = &mut self.gpu;
        for (slot, capacity, name) in [
            (
                &mut self.flight_line_buffer,
                128 * 1024,
                "stream flight line",
            ),
            (
                &mut self.flight_fill_buffer,
                256 * 1024,
                "stream flight fill",
            ),
            (
                &mut self.flight_glow_buffer,
                8 * 1024 * 1024,
                "stream flight glow",
            ),
            (&mut self.hud_line_buffer, 128 * 1024, "stream hud line"),
            (&mut self.hud_fill_buffer, 4 * 1024, "stream hud fill"),
        ] {
            free_slot(gpu, slot)?;
            *slot = Some(HostBuffer::new(
                &gpu.device,
                &mut gpu.allocator,
                capacity,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
                name,
            )?);
        }
        Ok(())
    }

    /// Update dynamic flight geometry for this frame (streaming mode).
    ///
    /// Requires `init_streaming()` to have been called first.
    pub fn update_flight(&mut self, flights: &[FlightRenderData], camera: &Camera) {
        use cf_scene::tee::{generate_ball_at, generate_ball_glow_at};
        use cf_scene::trail::{DEFAULT_TRAIL_LIFETIME, generate_trail_glow, generate_trail_line};

        self.ball_on_tee_box = flights.is_empty();

        let tee = TeeBox::default();
        let mut line_verts: Vec<GridVertex> = Vec::new();
        let mut fill_verts: Vec<GridVertex> = Vec::new();
        let mut glow_verts: Vec<GridVertex> = Vec::new();

        for flight in flights {
            fill_verts.extend(generate_ball_at(flight.ball_pos, tee.ball_radius, 8, 16));
            glow_verts.extend(generate_ball_glow_at(
                flight.ball_pos,
                tee.ball_radius,
                8,
                16,
            ));

            if flight.trail_points.len() >= 2 {
                glow_verts.extend(generate_trail_glow(
                    &flight.trail_points,
                    flight.current_time,
                    DEFAULT_TRAIL_LIFETIME,
                    camera.position,
                    tee.ball_radius,
                ));
                line_verts.extend(generate_trail_line(
                    &flight.trail_points,
                    flight.current_time,
                    DEFAULT_TRAIL_LIFETIME,
                ));
            }
        }

        if let Some(buffer) = &mut self.flight_line_buffer {
            self.flight_line_count = buffer.write(&line_verts);
        }
        if let Some(buffer) = &mut self.flight_fill_buffer {
            self.flight_fill_count = buffer.write(&fill_verts);
        }
        if let Some(buffer) = &mut self.flight_glow_buffer {
            self.flight_glow_count = buffer.write(&glow_verts);
        }

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

    /// Update HUD overlay geometry for this frame (streaming mode).
    ///
    /// Requires `init_streaming()` to have been called first.
    pub fn update_hud_streaming(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        if let Some(buffer) = &mut self.hud_line_buffer {
            self.hud_line_count = buffer.write(lines);
        }
        if let Some(buffer) = &mut self.hud_fill_buffer {
            self.hud_fill_count = buffer.write(fills);
        }
    }

    /// Render the grid and read back the result as RGBA8 pixels.
    ///
    /// When RT is enabled, renders the raster scene, traces RT reflection
    /// rays, and composites them onto the rasterized output.
    pub fn render(&self, camera: &Camera) -> Result<FrameBuffers, RenderError> {
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
                .render_pass(self.pipeline.render_pass)
                .framebuffer(target.framebuffer)
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
                self.pipeline.pipeline,
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
            if self.hud_fill_count > 0 || self.hud_line_count > 0 {
                let w = width as f32;
                let h = height as f32;
                let hud_proj = glam::Mat4::orthographic_rh(0.0, w, 0.0, h, -1.0, 1.0);

                if let Some(hud_fill_buf) = &self.hud_fill_buffer {
                    if self.hud_fill_count > 0 {
                        let pc_hud_bg = GridPushConstants {
                            view_proj: hud_proj.to_cols_array_2d(),
                            color: [0.0, 0.0, 0.0, 0.8],
                        };
                        device.cmd_bind_pipeline(
                            self.command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.fill_pipeline,
                        );
                        device.cmd_bind_vertex_buffers(
                            self.command_buffer,
                            0,
                            &[hud_fill_buf.handle()],
                            &[0],
                        );
                        push_constants(
                            device,
                            self.command_buffer,
                            self.pipeline.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            &pc_hud_bg,
                        );
                        device.cmd_draw(self.command_buffer, self.hud_fill_count, 1, 0, 0);
                    }
                }

                if let Some(hud_line_buf) = &self.hud_line_buffer {
                    if self.hud_line_count > 0 {
                        let pc_hud_text = GridPushConstants {
                            view_proj: hud_proj.to_cols_array_2d(),
                            color: [1.0, 1.0, 1.0, 1.0],
                        };
                        device.cmd_bind_pipeline(
                            self.command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.pipeline,
                        );
                        device.cmd_set_line_width(self.command_buffer, 1.0);
                        device.cmd_bind_vertex_buffers(
                            self.command_buffer,
                            0,
                            &[hud_line_buf.handle()],
                            &[0],
                        );
                        push_constants(
                            device,
                            self.command_buffer,
                            self.pipeline.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            &pc_hud_text,
                        );
                        device.cmd_draw(self.command_buffer, self.hud_line_count, 1, 0, 0);
                    }
                }
            }

            device.cmd_end_render_pass(self.command_buffer);

            // ── RT reflection compositing ──
            if rt_active {
                self.record_rt_composite(camera, width, height, extent);
            }

            // Barrier: render pass output -> transfer read
            // When RT is active the composite pass ends at TRANSFER_SRC_OPTIMAL.
            // When raster-only the main render pass ends at TRANSFER_SRC_OPTIMAL.
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
    /// 4. Composite render pass (LOAD, additive blend, final_layout = TRANSFER_SRC_OPTIMAL)
    ///
    /// # Safety
    /// The command buffer must be in the recording state, outside any render pass.
    unsafe fn record_rt_composite(
        &self,
        camera: &Camera,
        width: u32,
        height: u32,
        extent: vk::Extent2D,
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

            // Step 4: Composite render pass (LOAD, additive blend fullscreen triangle).
            let composite_rp = self
                .composite_render_pass
                .expect("composite render pass present");
            let composite_fb = self
                .composite_framebuffer
                .expect("composite framebuffer present");
            let composite_pl = self.composite_pipeline.expect("composite pipeline present");
            let composite_layout = self
                .composite_pipeline_layout
                .expect("composite layout present");
            let composite_ds = self
                .composite_descriptor_set
                .expect("composite descriptor set present");

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(composite_rp)
                .framebuffer(composite_fb)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                });

            device.cmd_begin_render_pass(
                self.command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

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

            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                composite_pl,
            );
            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                composite_layout,
                0,
                &[composite_ds],
                &[],
            );

            // Fullscreen triangle: 3 vertices, no vertex buffer.
            device.cmd_draw(self.command_buffer, 3, 1, 0, 0);

            device.cmd_end_render_pass(self.command_buffer);
        } // unsafe
    }

    /// Create the offscreen composite pass for blending RT reflections.
    ///
    /// Returns a tuple of all Vulkan objects (render_pass, framebuffer, pipeline,
    /// pipeline_layout, descriptor_set_layout, descriptor_pool, descriptor_set,
    /// sampler, vert_module, frag_module).
    #[allow(clippy::type_complexity)]
    fn create_composite_pass(
        device: &ash::Device,
        format: vk::Format,
        color_view: vk::ImageView,
        rt_storage_view: vk::ImageView,
        width: u32,
        height: u32,
    ) -> Result<
        (
            vk::RenderPass,
            vk::Framebuffer,
            vk::Pipeline,
            vk::PipelineLayout,
            vk::DescriptorSetLayout,
            vk::DescriptorPool,
            vk::DescriptorSet,
            vk::Sampler,
            vk::ShaderModule,
            vk::ShaderModule,
        ),
        RenderError,
    > {
        // Render pass: color-only, LOAD existing content, final_layout = TRANSFER_SRC_OPTIMAL.
        let render_pass = GridPipeline::create_hud_render_pass(
            device,
            format,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;

        // Framebuffer wrapping the offscreen target's color image view.
        let fb_attachments = [color_view];
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&fb_attachments)
            .width(width)
            .height(height)
            .layers(1);
        // SAFETY: Creating framebuffer.
        let framebuffer = unsafe {
            device
                .create_framebuffer(&fb_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Sampler for the RT storage image (nearest — 1:1 pixel mapping).
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: Creating sampler.
        let sampler = unsafe {
            device
                .create_sampler(&sampler_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Descriptor set layout: binding 0 = combined image sampler.
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [binding];
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Creating descriptor set layout.
        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Pipeline layout (descriptor set, no push constants).
        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: Creating pipeline layout.
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Shaders.
        let vert_module = load_shader(device, include_bytes!("../shaders/composite.vert.spv"))?;
        let frag_module = load_shader(device, include_bytes!("../shaders/composite.frag.spv"))?;

        // Pipeline: fullscreen triangle, additive blending, no vertex input.
        let entry_point = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_point),
        ];

        // No vertex input (positions are hardcoded in the vertex shader).
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Additive blending: src + dst.
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD);
        let blend_attachments = [blend_attachment];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        // SAFETY: Creating graphics pipeline.
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .map_err(|(_pipelines, err)| RenderError::Vulkan(err))?[0]
        };

        // Descriptor pool + set.
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: Creating descriptor pool.
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(RenderError::Vulkan)?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: Allocating descriptor set.
        let descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(RenderError::Vulkan)?[0]
        };

        // Write the RT storage image into the descriptor set.
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(rt_storage_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(sampler);
        let image_infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);
        // SAFETY: Updating descriptor set.
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        Ok((
            render_pass,
            framebuffer,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            sampler,
            vert_module,
            frag_module,
        ))
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

/// Destroy the buffer in `slot`, if any, once the device is idle.
fn free_slot(gpu: &mut GpuContext, slot: &mut Option<HostBuffer>) -> Result<(), RenderError> {
    if let Some(old) = slot.take() {
        // SAFETY: The wait makes sure no pending work uses the old buffer.
        unsafe {
            if let Err(e) = gpu.device.device_wait_idle() {
                *slot = Some(old);
                return Err(RenderError::Vulkan(e));
            }
            old.destroy(&gpu.device, &mut gpu.allocator);
        }
    }
    Ok(())
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
            for slot in [
                &mut self.trail_line_buffer,
                &mut self.hud_line_buffer,
                &mut self.hud_fill_buffer,
                // Dynamic flight buffers (streaming mode).
                &mut self.flight_line_buffer,
                &mut self.flight_fill_buffer,
                &mut self.flight_glow_buffer,
            ] {
                if let Some(buffer) = slot.take() {
                    buffer.destroy(device, allocator);
                }
            }

            // Clean up RT composite resources.
            if let Some(fb) = self.composite_framebuffer {
                self.gpu.device.destroy_framebuffer(fb, None);
            }
            if let Some(pl) = self.composite_pipeline {
                self.gpu.device.destroy_pipeline(pl, None);
            }
            if let Some(layout) = self.composite_pipeline_layout {
                self.gpu.device.destroy_pipeline_layout(layout, None);
            }
            if let Some(pool) = self.composite_descriptor_pool {
                self.gpu.device.destroy_descriptor_pool(pool, None);
            }
            if let Some(dsl) = self.composite_descriptor_set_layout {
                self.gpu.device.destroy_descriptor_set_layout(dsl, None);
            }
            if let Some(s) = self.composite_sampler {
                self.gpu.device.destroy_sampler(s, None);
            }
            if let Some(rp) = self.composite_render_pass {
                self.gpu.device.destroy_render_pass(rp, None);
            }
            if let Some(m) = self.composite_vert_module {
                self.gpu.device.destroy_shader_module(m, None);
            }
            if let Some(m) = self.composite_frag_module {
                self.gpu.device.destroy_shader_module(m, None);
            }

            if let Some(rtp) = self.rt_pipeline.take() {
                rtp.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            if let Some(target) = self.target.take() {
                target.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            self.pipeline.destroy(&self.gpu.device);
        }
    }
}
