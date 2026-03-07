use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

use cf_scene::camera::Camera;
use cf_scene::color;
use cf_scene::grid::{GridConfig, GridVertex, generate_grid_vertices};
use cf_scene::tee::{
    TeeBox, generate_ball, generate_ball_at, generate_ball_glow, generate_ball_glow_at,
    generate_tee_border, generate_tee_fill,
};
use cf_scene::trail::{TrailPoint, generate_trail_glow, generate_trail_line};

use crate::context::{GpuConfig, GpuContext};
use crate::error::RenderError;
use crate::framebuffer::OffscreenTarget;
use crate::pipeline::{GridPipeline, GridPushConstants};
use crate::readback::FrameBuffers;
use crate::render::FlightRenderData;
use crate::rt_offscreen::build_scene_geometry;
use crate::rt_pipeline::{RtPipeline, RtPushConstants};

/// Offscreen renderer for headless PNG generation.
pub struct OffscreenRenderer {
    gpu: GpuContext,
    pipeline: GridPipeline,
    target: Option<OffscreenTarget>,
    vertex_buffer: vk::Buffer,
    vertex_allocation: Option<Allocation>,
    vertex_count: u32,
    /// Filled geometry: tee box fill + tee box border + ball.
    fill_buffer: vk::Buffer,
    fill_allocation: Option<Allocation>,
    tee_fill_count: u32,
    tee_border_count: u32,
    ball_count: u32,
    glow_buffer: vk::Buffer,
    glow_allocation: Option<Allocation>,
    glow_count: u32,
    /// Trail centerline (LINE_LIST, always visible regardless of viewing angle).
    trail_line_buffer: Option<vk::Buffer>,
    trail_line_allocation: Option<Allocation>,
    trail_line_count: u32,
    // Optional HUD overlay.
    hud_line_buffer: Option<vk::Buffer>,
    hud_line_allocation: Option<Allocation>,
    hud_line_count: u32,
    hud_fill_buffer: Option<vk::Buffer>,
    hud_fill_allocation: Option<Allocation>,
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
    grid_spacing_m: f32,
    grid_max_fade_dist: f32,
    /// Ball center position for RT push constants (xyz).
    rt_ball_center: glam::Vec3,
    /// Ball radius for RT push constants.
    rt_ball_radius: f32,
    /// RT trail fade distance (arc length of trimmed trail, meters).
    rt_trail_fade_dist: f32,
    /// Optional chase camera for split viewport (right 20%).
    chase_camera: Option<Camera>,

    // Dynamic per-frame flight geometry (pre-allocated for streaming use).
    flight_line_buffer: Option<vk::Buffer>,
    flight_line_allocation: Option<Allocation>,
    flight_line_count: u32,
    flight_line_capacity: vk::DeviceSize,
    flight_fill_buffer: Option<vk::Buffer>,
    flight_fill_allocation: Option<Allocation>,
    flight_fill_count: u32,
    flight_fill_capacity: vk::DeviceSize,
    flight_glow_buffer: Option<vk::Buffer>,
    flight_glow_allocation: Option<Allocation>,
    flight_glow_count: u32,
    flight_glow_capacity: vk::DeviceSize,
    /// When true, draw the static ball on tee. When false, draw from dynamic flight buffers.
    ball_on_tee_box: bool,
}

/// Offscreen image format (RGBA8 sRGB for correct gamma in PNG output).
const OFFSCREEN_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/// Clip bounds that effectively disable clipping (huge region).
const NO_CLIP: [f32; 4] = [-1e6, -1e6, 1e6, 1e6];


impl OffscreenRenderer {
    /// Create a headless renderer for the given grid configuration.
    /// Enables raytracing if hardware supports it.
    pub fn new(width: u32, height: u32, grid_config: &GridConfig) -> Result<Self, RenderError> {
        let config = GpuConfig {
            width,
            height,
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
            raster_final_layout,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            pipeline.render_pass,
        )?;

        let grid_verts = generate_grid_vertices(grid_config);
        let vertex_count = grid_verts.len() as u32;
        let (vertex_buffer, vertex_allocation) =
            Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &grid_verts)?;

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
        let (fill_buffer, fill_allocation) =
            Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &fill_verts)?;

        let glow_verts = generate_ball_glow(&tee, 16, 32);
        let glow_count = glow_verts.len() as u32;
        let (glow_buffer, glow_allocation) =
            Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &glow_verts)?;

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Grid params for RT push constants.
        let spacing_m = grid_config.unit.to_meters(f64::from(grid_config.spacing)) as f32;
        let dr = grid_config.unit.to_meters(f64::from(grid_config.downrange)) as f32;
        let lat = grid_config.unit.to_meters(f64::from(grid_config.lateral)) as f32;
        let max_fade_dist = (dr * dr + lat * lat).sqrt();

        // Build RT pipeline + composite pass if hardware supports it.
        let ball_center = glam::Vec3::new(0.0, tee.ball_radius, 0.0);
        let geometries = build_scene_geometry(grid_config, &tee, &[]);
        let (rt_pipeline, composite_render_pass, composite_framebuffer, composite_pipeline,
             composite_pipeline_layout, composite_descriptor_set_layout,
             composite_descriptor_pool, composite_descriptor_set, composite_sampler,
             composite_vert_module, composite_frag_module) =
            if rt_available {
                let rtp = RtPipeline::new(&mut gpu, width, height, &geometries)?;
                let composite = Self::create_composite_pass(
                    &gpu.device,
                    OFFSCREEN_FORMAT,
                    target.view,
                    rtp.storage_view,
                    width,
                    height,
                )?;
                (Some(rtp), Some(composite.0), Some(composite.1), Some(composite.2),
                 Some(composite.3), Some(composite.4), Some(composite.5),
                 Some(composite.6), Some(composite.7), Some(composite.8), Some(composite.9))
            } else {
                (None, None, None, None, None, None, None, None, None, None, None)
            };

        Ok(Self {
            gpu,
            pipeline,
            target: Some(target),
            vertex_buffer,
            vertex_allocation: Some(vertex_allocation),
            vertex_count,
            fill_buffer,
            fill_allocation: Some(fill_allocation),
            tee_fill_count,
            tee_border_count,
            ball_count,
            glow_buffer,
            glow_allocation: Some(glow_allocation),
            glow_count,
            trail_line_buffer: None,
            trail_line_allocation: None,
            trail_line_count: 0,
            hud_line_buffer: None,
            hud_line_allocation: None,
            hud_line_count: 0,
            hud_fill_buffer: None,
            hud_fill_allocation: None,
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
            grid_spacing_m: spacing_m,
            grid_max_fade_dist: max_fade_dist,
            rt_ball_center: ball_center,
            rt_ball_radius: tee.ball_radius,
            rt_trail_fade_dist: 0.0,
            chase_camera: None,

            flight_line_buffer: None,
            flight_line_allocation: None,
            flight_line_count: 0,
            flight_line_capacity: 0,
            flight_fill_buffer: None,
            flight_fill_allocation: None,
            flight_fill_count: 0,
            flight_fill_capacity: 0,
            flight_glow_buffer: None,
            flight_glow_allocation: None,
            flight_glow_count: 0,
            flight_glow_capacity: 0,
            ball_on_tee_box: true,
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
            width,
            height,
            validation: cfg!(debug_assertions),
            enable_raytracing: false,
        };

        let mut gpu = GpuContext::new_headless(config)?;

        let pipeline = GridPipeline::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            pipeline.render_pass,
        )?;

        // Grid (same as default)
        let grid_verts = generate_grid_vertices(grid_config);
        let vertex_count = grid_verts.len() as u32;
        let (vertex_buffer, vertex_allocation) =
            Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &grid_verts)?;

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
        let (fill_buffer, fill_allocation) =
            Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &fill_verts)?;

        // Ball glow sphere + trail glow ribbon (with endcap to bridge the junction).
        let mut glow_verts = generate_ball_glow_at(ball_center, tee.ball_radius, 16, 32);
        glow_verts.extend(generate_trail_glow(trail_points, current_time, max_lifetime, camera.position, tee.ball_radius));
        let glow_count = glow_verts.len() as u32;
        let (glow_buffer, glow_allocation) =
            Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &glow_verts)?;

        // Trail centerline (LINE_LIST — always visible regardless of viewing angle).
        let trail_line_verts = generate_trail_line(trail_points, current_time, max_lifetime);
        let (trail_line_buffer, trail_line_allocation, trail_line_count) = if trail_line_verts.is_empty() {
            (None, None, 0)
        } else {
            let count = trail_line_verts.len() as u32;
            let (buf, alloc) = Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &trail_line_verts)?;
            (Some(buf), Some(alloc), count)
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
            vertex_buffer,
            vertex_allocation: Some(vertex_allocation),
            vertex_count,
            fill_buffer,
            fill_allocation: Some(fill_allocation),
            tee_fill_count,
            tee_border_count,
            ball_count,
            glow_buffer,
            glow_allocation: Some(glow_allocation),
            glow_count,
            trail_line_buffer,
            trail_line_allocation,
            trail_line_count,
            hud_line_buffer: None,
            hud_line_allocation: None,
            hud_line_count: 0,
            hud_fill_buffer: None,
            hud_fill_allocation: None,
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
            grid_spacing_m: 0.0,
            grid_max_fade_dist: 0.0,
            rt_ball_center: ball_center,
            rt_ball_radius: tee.ball_radius,
            rt_trail_fade_dist: 0.0,
            chase_camera: None,

            flight_line_buffer: None,
            flight_line_allocation: None,
            flight_line_count: 0,
            flight_line_capacity: 0,
            flight_fill_buffer: None,
            flight_fill_allocation: None,
            flight_fill_count: 0,
            flight_fill_capacity: 0,
            flight_glow_buffer: None,
            flight_glow_allocation: None,
            flight_glow_count: 0,
            flight_glow_capacity: 0,
            ball_on_tee_box: true,
        })
    }

    /// Create a headless renderer with RT reflection compositing for the static range.
    ///
    /// Falls back to raster-only if RT hardware is not available.
    pub fn new_rt(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
    ) -> Result<Self, RenderError> {
        let tee = TeeBox::default();
        let ball_center = glam::Vec3::new(0.0, cf_scene::tee::TEE_ELEVATION + tee.ball_radius, 0.0);
        let geometries = build_scene_geometry(grid_config, &tee, &[]);
        Self::create_rt(width, height, grid_config, &tee, ball_center, &geometries, None, 0.0)
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

        // Build RT geometry from trail positions.
        let trail_positions: Vec<glam::Vec3> = trail_points.iter().map(|p| p.position).collect();
        let trimmed = crate::rt_offscreen::trim_trail_from_ball_pub(&trail_positions, 0.8);
        let trim_len: f32 = trimmed.windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum();
        let geometries = build_scene_geometry(grid_config, &tee, &trimmed);

        Self::create_rt(
            width, height, grid_config, &tee, ball_center, &geometries,
            Some((ball_pos, trail_points, current_time, max_lifetime, camera)),
            trim_len,
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
        flight: Option<(glam::Vec3, &[TrailPoint], f64, f64, &Camera)>,
        trail_fade_dist: f32,
    ) -> Result<Self, RenderError> {
        let config = GpuConfig {
            width,
            height,
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
            raster_final_layout,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            pipeline.render_pass,
        )?;

        // Grid vertices
        let grid_verts = generate_grid_vertices(grid_config);
        let vertex_count = grid_verts.len() as u32;
        let (vertex_buffer, vertex_allocation) =
            Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &grid_verts)?;

        // Tee box + ball geometry
        let (tee_fill_count, tee_border_count, ball_count, fill_buffer, fill_allocation,
             glow_count, glow_buffer, glow_allocation,
             trail_line_buffer, trail_line_allocation, trail_line_count) =
            if let Some((ball_pos, trail_points, current_time, max_lifetime, camera)) = flight {
                let tee_fill_verts = generate_tee_fill(tee);
                let tee_border_verts = generate_tee_border(tee);
                let ball_verts = generate_ball_at(ball_center, tee.ball_radius, 12, 24);
                let tfc = tee_fill_verts.len() as u32;
                let tbc = tee_border_verts.len() as u32;
                let bc = ball_verts.len() as u32;
                let mut fv = tee_fill_verts;
                fv.extend(tee_border_verts);
                fv.extend(ball_verts);
                let (fb, fa) = Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &fv)?;

                let mut gv = generate_ball_glow_at(ball_center, tee.ball_radius, 16, 32);
                gv.extend(generate_trail_glow(trail_points, current_time, max_lifetime, camera.position, tee.ball_radius));
                let gc = gv.len() as u32;
                let (gb, ga) = Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &gv)?;

                // Trail centerline (LINE_LIST).
                let tlv = generate_trail_line(trail_points, current_time, max_lifetime);
                let (tlb, tla, tlc) = if tlv.is_empty() {
                    (None, None, 0u32)
                } else {
                    let c = tlv.len() as u32;
                    let (b, a) = Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &tlv)?;
                    (Some(b), Some(a), c)
                };

                // Suppress unused variable warning for ball_pos (used via ball_center).
                let _ = ball_pos;

                (tfc, tbc, bc, fb, fa, gc, gb, ga, tlb, tla, tlc)
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
                let (fb, fa) = Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &fv)?;

                let gv = generate_ball_glow(tee, 16, 32);
                let gc = gv.len() as u32;
                let (gb, ga) = Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &gv)?;

                (tfc, tbc, bc, fb, fa, gc, gb, ga, None, None, 0u32)
            };

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Compute grid params for RT push constants.
        let spacing_m = grid_config.unit.to_meters(f64::from(grid_config.spacing)) as f32;
        let dr = grid_config.unit.to_meters(f64::from(grid_config.downrange)) as f32;
        let lat = grid_config.unit.to_meters(f64::from(grid_config.lateral)) as f32;
        let max_fade_dist = (dr * dr + lat * lat).sqrt();

        // Build RT pipeline + composite pass if hardware supports it.
        let (rt_pipeline, composite_render_pass, composite_framebuffer, composite_pipeline,
             composite_pipeline_layout, composite_descriptor_set_layout,
             composite_descriptor_pool, composite_descriptor_set, composite_sampler,
             composite_vert_module, composite_frag_module) =
            if rt_available {
                let rtp = RtPipeline::new(&mut gpu, width, height, geometries)?;

                let composite = Self::create_composite_pass(
                    &gpu.device,
                    OFFSCREEN_FORMAT,
                    target.view,
                    rtp.storage_view,
                    width,
                    height,
                )?;

                (Some(rtp), Some(composite.0), Some(composite.1), Some(composite.2),
                 Some(composite.3), Some(composite.4), Some(composite.5),
                 Some(composite.6), Some(composite.7), Some(composite.8), Some(composite.9))
            } else {
                (None, None, None, None, None, None, None, None, None, None, None)
            };

        Ok(Self {
            gpu,
            pipeline,
            target: Some(target),
            vertex_buffer,
            vertex_allocation: Some(vertex_allocation),
            vertex_count,
            fill_buffer,
            fill_allocation: Some(fill_allocation),
            tee_fill_count,
            tee_border_count,
            ball_count,
            glow_buffer,
            glow_allocation: Some(glow_allocation),
            glow_count,
            trail_line_buffer,
            trail_line_allocation,
            trail_line_count,
            hud_line_buffer: None,
            hud_line_allocation: None,
            hud_line_count: 0,
            hud_fill_buffer: None,
            hud_fill_allocation: None,
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
            grid_spacing_m: spacing_m,
            grid_max_fade_dist: max_fade_dist,
            rt_ball_center: ball_center,
            rt_ball_radius: tee.ball_radius,
            rt_trail_fade_dist: trail_fade_dist,
            chase_camera: None,

            flight_line_buffer: None,
            flight_line_allocation: None,
            flight_line_count: 0,
            flight_line_capacity: 0,
            flight_fill_buffer: None,
            flight_fill_allocation: None,
            flight_fill_count: 0,
            flight_fill_capacity: 0,
            flight_glow_buffer: None,
            flight_glow_allocation: None,
            flight_glow_count: 0,
            flight_glow_capacity: 0,
            ball_on_tee_box: true,
        })
    }

    /// Whether this renderer has RT reflection compositing enabled.
    #[must_use]
    fn compute_view_proj(camera: &Camera, aspect: f32) -> [[f32; 4]; 4] {
        let view = camera.view_matrix();
        let mut proj = camera.projection_matrix(aspect);
        // Vulkan NDC has +Y down; glam assumes OpenGL (+Y up). Flip Y.
        proj.y_axis.y *= -1.0;
        (proj * view).to_cols_array_2d()
    }

    /// Draw all scene geometry (grid, tee box, ball, glow) with the given view-projection.
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
            clip_bounds: NO_CLIP,
        };
        let pc_cyan_bytes: &[u8] = std::slice::from_raw_parts(
            std::ptr::from_ref(&pc_cyan).cast::<u8>(),
            std::mem::size_of::<GridPushConstants>(),
        );
        let pc_black = GridPushConstants {
            view_proj,
            color: [0.0, 0.0, 0.0, 1.0],
            clip_bounds: NO_CLIP,
        };
        let pc_black_bytes: &[u8] = std::slice::from_raw_parts(
            std::ptr::from_ref(&pc_black).cast::<u8>(),
            std::mem::size_of::<GridPushConstants>(),
        );
        let pc_magenta = GridPushConstants {
            view_proj,
            color: color::MAGENTA.into(),
            clip_bounds: NO_CLIP,
        };
        let pc_magenta_bytes: &[u8] = std::slice::from_raw_parts(
            std::ptr::from_ref(&pc_magenta).cast::<u8>(),
            std::mem::size_of::<GridPushConstants>(),
        );

        // Grid lines (LINE_LIST, cyan)
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
        device.cmd_set_line_width(cb, 1.0);
        device.cmd_push_constants(
            cb, self.pipeline.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_cyan_bytes,
        );
        device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer], &[0]);
        device.cmd_draw(cb, self.vertex_count, 1, 0, 0);

        // Tee box fill (black)
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.fill_pipeline);
        device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer], &[0]);
        device.cmd_push_constants(
            cb, self.pipeline.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_black_bytes,
        );
        device.cmd_draw(cb, self.tee_fill_count, 1, 0, 0);

        // Tee box border (cyan)
        device.cmd_push_constants(
            cb, self.pipeline.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_cyan_bytes,
        );
        device.cmd_draw(cb, self.tee_border_count, 1, self.tee_fill_count, 0);

        if self.ball_on_tee_box {
            // Static ball: trail + glow + ball from construction-time buffers.
            if let Some(trail_buf) = self.trail_line_buffer {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
                device.cmd_set_line_width(cb, 1.0);
                device.cmd_bind_vertex_buffers(cb, 0, &[trail_buf], &[0]);
                device.cmd_push_constants(
                    cb, self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_magenta_bytes,
                );
                device.cmd_draw(cb, self.trail_line_count, 1, 0, 0);
            }

            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.glow_pipeline);
            device.cmd_bind_vertex_buffers(cb, 0, &[self.glow_buffer], &[0]);
            device.cmd_push_constants(
                cb, self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_magenta_bytes,
            );
            device.cmd_draw(cb, self.glow_count, 1, 0, 0);

            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.fill_pipeline);
            device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer], &[0]);
            device.cmd_push_constants(
                cb, self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_black_bytes,
            );
            device.cmd_draw(cb, self.ball_count, 1, self.tee_fill_count + self.tee_border_count, 0);
        } else {
            // Dynamic flight geometry from pre-allocated streaming buffers.
            if let Some(flight_line_buf) = self.flight_line_buffer {
                if self.flight_line_count > 0 {
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
                    device.cmd_set_line_width(cb, 1.0);
                    device.cmd_bind_vertex_buffers(cb, 0, &[flight_line_buf], &[0]);
                    device.cmd_push_constants(
                        cb, self.pipeline.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_magenta_bytes,
                    );
                    device.cmd_draw(cb, self.flight_line_count, 1, 0, 0);
                }
            }
            if let Some(flight_glow_buf) = self.flight_glow_buffer {
                if self.flight_glow_count > 0 {
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.glow_pipeline);
                    device.cmd_bind_vertex_buffers(cb, 0, &[flight_glow_buf], &[0]);
                    device.cmd_push_constants(
                        cb, self.pipeline.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_magenta_bytes,
                    );
                    device.cmd_draw(cb, self.flight_glow_count, 1, 0, 0);
                }
            }
            if let Some(flight_fill_buf) = self.flight_fill_buffer {
                if self.flight_fill_count > 0 {
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.fill_pipeline);
                    device.cmd_bind_vertex_buffers(cb, 0, &[flight_fill_buf], &[0]);
                    device.cmd_push_constants(
                        cb, self.pipeline.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_black_bytes,
                    );
                    device.cmd_draw(cb, self.flight_fill_count, 1, 0, 0);
                }
            }
        }
        } // unsafe
    }

    pub fn has_rt(&self) -> bool {
        self.rt_pipeline.is_some()
    }

    /// Add HUD overlay geometry (call before `render()`).
    /// Set the chase camera for the right 20% split viewport.
    pub fn set_chase_camera(&mut self, camera: Camera) {
        self.chase_camera = Some(camera);
    }

    pub fn set_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) -> Result<(), RenderError> {
        if !lines.is_empty() {
            let (buf, alloc) =
                Self::create_vertex_buffer(&self.gpu.device, &mut self.gpu.allocator, lines)?;
            self.hud_line_buffer = Some(buf);
            self.hud_line_allocation = Some(alloc);
            self.hud_line_count = lines.len() as u32;
        }
        if !fills.is_empty() {
            let (buf, alloc) =
                Self::create_vertex_buffer(&self.gpu.device, &mut self.gpu.allocator, fills)?;
            self.hud_fill_buffer = Some(buf);
            self.hud_fill_allocation = Some(alloc);
            self.hud_fill_count = fills.len() as u32;
        }
        Ok(())
    }

    /// Allocate pre-sized dynamic buffers for streaming use.
    ///
    /// Call once after construction to enable `update_flight()` and `update_hud_streaming()`.
    /// Without this, the renderer only supports static geometry set at construction time.
    pub fn init_streaming(&mut self) -> Result<(), RenderError> {
        // Flight line: trail centerlines, ~2000 points × 2 verts × 16 bytes = ~64 KB
        let line_cap: vk::DeviceSize = 128 * 1024;
        let (lb, la) = Self::create_dynamic_buffer(
            &self.gpu.device, &mut self.gpu.allocator, line_cap, "stream flight line")?;
        self.flight_line_buffer = Some(lb);
        self.flight_line_allocation = Some(la);
        self.flight_line_capacity = line_cap;

        // Flight fill: up to 5 in-flight balls × 1728 verts = ~140 KB
        let fill_cap: vk::DeviceSize = 256 * 1024;
        let (fb, fa) = Self::create_dynamic_buffer(
            &self.gpu.device, &mut self.gpu.allocator, fill_cap, "stream flight fill")?;
        self.flight_fill_buffer = Some(fb);
        self.flight_fill_allocation = Some(fa);
        self.flight_fill_capacity = fill_cap;

        // Flight glow: up to 5 balls × (ball glow + trail glow) ≈ 6 MB
        let glow_cap: vk::DeviceSize = 8 * 1024 * 1024;
        let (gb, ga) = Self::create_dynamic_buffer(
            &self.gpu.device, &mut self.gpu.allocator, glow_cap, "stream flight glow")?;
        self.flight_glow_buffer = Some(gb);
        self.flight_glow_allocation = Some(ga);
        self.flight_glow_capacity = glow_cap;

        // Pre-allocate HUD buffers (replaces set_hud's per-call allocations).
        let hud_line_cap: vk::DeviceSize = 128 * 1024;
        let (hlb, hla) = Self::create_dynamic_buffer(
            &self.gpu.device, &mut self.gpu.allocator, hud_line_cap, "stream hud line")?;
        // Free old HUD buffers if set_hud was called before.
        if let Some(old) = self.hud_line_buffer {
            // SAFETY: Buffer is no longer in use after device_wait_idle in drop.
            unsafe { self.gpu.device.destroy_buffer(old, None); }
        }
        if let Some(alloc) = self.hud_line_allocation.take() {
            let _ = self.gpu.allocator.free(alloc);
        }
        self.hud_line_buffer = Some(hlb);
        self.hud_line_allocation = Some(hla);

        let hud_fill_cap: vk::DeviceSize = 4 * 1024;
        let (hfb, hfa) = Self::create_dynamic_buffer(
            &self.gpu.device, &mut self.gpu.allocator, hud_fill_cap, "stream hud fill")?;
        if let Some(old) = self.hud_fill_buffer {
            unsafe { self.gpu.device.destroy_buffer(old, None); }
        }
        if let Some(alloc) = self.hud_fill_allocation.take() {
            let _ = self.gpu.allocator.free(alloc);
        }
        self.hud_fill_buffer = Some(hfb);
        self.hud_fill_allocation = Some(hfa);

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
            glow_verts.extend(generate_ball_glow_at(flight.ball_pos, tee.ball_radius, 8, 16));

            if flight.trail_points.len() >= 2 {
                glow_verts.extend(generate_trail_glow(
                    &flight.trail_points, flight.current_time, DEFAULT_TRAIL_LIFETIME,
                    camera.position, tee.ball_radius,
                ));
                line_verts.extend(generate_trail_line(
                    &flight.trail_points, flight.current_time, DEFAULT_TRAIL_LIFETIME,
                ));
            }
        }

        if let Some(alloc) = &self.flight_line_allocation {
            self.flight_line_count = Self::upload_to_mapped(alloc, &line_verts, self.flight_line_capacity);
        }
        if let Some(alloc) = &self.flight_fill_allocation {
            self.flight_fill_count = Self::upload_to_mapped(alloc, &fill_verts, self.flight_fill_capacity);
        }
        if let Some(alloc) = &self.flight_glow_allocation {
            self.flight_glow_count = Self::upload_to_mapped(alloc, &glow_verts, self.flight_glow_capacity);
        }

        // Rebuild RT TLAS with updated ball/trail positions.
        if let Some(ref mut rt_pipeline) = self.rt_pipeline {
            // Collect trail points for RT geometry.
            let rt_trail_points: Vec<glam::Vec3> = flights
                .iter()
                .flat_map(|f| f.trail_points.iter().map(|tp| tp.position))
                .collect();

            if let Some(first_ball) = flights.first() {
                self.rt_ball_center = first_ball.ball_pos;
            }
            self.rt_trail_fade_dist = rt_trail_points
                .windows(2)
                .map(|w| (w[1] - w[0]).length())
                .sum();

            let grid_config = cf_scene::grid::GridConfig::default();
            let geometries = build_scene_geometry(&grid_config, &tee, &rt_trail_points);
            if let Err(e) = rt_pipeline.update_scene(&mut self.gpu, &geometries) {
                eprintln!("RT scene update failed: {e}");
            }
        }
    }

    /// Update HUD overlay geometry for this frame (streaming mode).
    ///
    /// Requires `init_streaming()` to have been called first.
    pub fn update_hud_streaming(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        if let Some(alloc) = &self.hud_line_allocation {
            self.hud_line_count = Self::upload_to_mapped(alloc, lines, 128 * 1024);
        }
        if let Some(alloc) = &self.hud_fill_allocation {
            self.hud_fill_count = Self::upload_to_mapped(alloc, fills, 4 * 1024);
        }
    }

    /// Render the grid and read back the result as RGBA8 pixels.
    ///
    /// When RT is enabled, renders the raster scene (skipping stencil reflections),
    /// traces RT reflection rays, and composites them onto the rasterized output.
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
            let view_proj = Self::compute_view_proj(camera, aspect);
            self.draw_scene_geometry(view_proj);

            // Chase camera: right 20% viewport
            if let Some(chase) = &self.chase_camera.clone() {
                let chase_frac: f32 = 0.20;
                let chase_x = (width as f32 * (1.0 - chase_frac)) as i32;
                let chase_w = (width as f32 * chase_frac) as u32;

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
                    rect: vk::Rect2D {
                        offset: vk::Offset2D { x: chase_x, y: 0 },
                        extent: vk::Extent2D { width: chase_w, height },
                    },
                    base_array_layer: 0,
                    layer_count: 1,
                };
                device.cmd_clear_attachments(self.command_buffer, &clear_attachments, &[clear_rect]);

                let chase_viewport = vk::Viewport {
                    x: chase_x as f32,
                    y: 0.0,
                    width: chase_w as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(self.command_buffer, 0, &[chase_viewport]);

                let chase_scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: chase_x, y: 0 },
                    extent: vk::Extent2D { width: chase_w, height },
                };
                device.cmd_set_scissor(self.command_buffer, 0, &[chase_scissor]);

                let chase_aspect = chase_w as f32 / height as f32;
                let chase_view_proj = Self::compute_view_proj(chase, chase_aspect);
                self.draw_scene_geometry(chase_view_proj);

                // Restore full viewport for HUD
                device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
                device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
            }

            // HUD overlay (2D, pixel coordinates)
            if self.hud_fill_count > 0 || self.hud_line_count > 0 {
                let w = width as f32;
                let h = height as f32;
                let hud_proj = glam::Mat4::orthographic_rh(0.0, w, 0.0, h, -1.0, 1.0);

                if let Some(hud_fill_buf) = self.hud_fill_buffer {
                    if self.hud_fill_count > 0 {
                        let pc_hud_bg = GridPushConstants {
                            view_proj: hud_proj.to_cols_array_2d(),
                            color: [0.0, 0.0, 0.0, 0.8],
                            clip_bounds: NO_CLIP,
                        };
                        let pc_hud_bg_bytes: &[u8] = std::slice::from_raw_parts(
                            std::ptr::from_ref(&pc_hud_bg).cast::<u8>(),
                            std::mem::size_of::<GridPushConstants>(),
                        );
                        device.cmd_bind_pipeline(
                            self.command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.fill_pipeline,
                        );
                        device.cmd_bind_vertex_buffers(
                            self.command_buffer,
                            0,
                            &[hud_fill_buf],
                            &[0],
                        );
                        device.cmd_push_constants(
                            self.command_buffer,
                            self.pipeline.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            pc_hud_bg_bytes,
                        );
                        device.cmd_draw(self.command_buffer, self.hud_fill_count, 1, 0, 0);
                    }
                }

                if let Some(hud_line_buf) = self.hud_line_buffer {
                    if self.hud_line_count > 0 {
                        let pc_hud_text = GridPushConstants {
                            view_proj: hud_proj.to_cols_array_2d(),
                            color: [1.0, 1.0, 1.0, 1.0],
                            clip_bounds: NO_CLIP,
                        };
                        let pc_hud_text_bytes: &[u8] = std::slice::from_raw_parts(
                            std::ptr::from_ref(&pc_hud_text).cast::<u8>(),
                            std::mem::size_of::<GridPushConstants>(),
                        );
                        device.cmd_bind_pipeline(
                            self.command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline.pipeline,
                        );
                        device.cmd_set_line_width(self.command_buffer, 1.0);
                        device.cmd_bind_vertex_buffers(
                            self.command_buffer,
                            0,
                            &[hud_line_buf],
                            &[0],
                        );
                        device.cmd_push_constants(
                            self.command_buffer,
                            self.pipeline.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            pc_hud_text_bytes,
                        );
                        device.cmd_draw(self.command_buffer, self.hud_line_count, 1, 0, 0);
                    }
                }
            }

            device.cmd_end_render_pass(self.command_buffer);

            // ── RT reflection compositing ──
            if rt_active {
                self.record_rt_composite(camera, width, height, extent, target);
            }

            // Barrier: render pass output -> transfer read
            // When RT is active the composite pass ends at TRANSFER_SRC_OPTIMAL.
            // When raster-only the main render pass ends at TRANSFER_SRC_OPTIMAL.
            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(target.image)
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
                target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target.staging_buffer,
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
        let mut pixels = vec![0u8; pixel_count];

        if let Some(mapped) = target.staging_allocation.mapped_ptr() {
            // SAFETY: Reading from mapped GPU memory within allocation bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped.as_ptr().cast::<u8>(),
                    pixels.as_mut_ptr(),
                    pixel_count,
                );
            }
        }

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
        _target: &OffscreenTarget,
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

        // Build RT push constants.
        let aspect = width as f32 / height as f32;
        let view = camera.view_matrix();
        let mut proj = camera.projection_matrix(aspect);
        proj.y_axis.y *= -1.0; // Vulkan Y-flip
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        let pc = RtPushConstants {
            camera_pos: [
                camera.position.x,
                camera.position.y,
                camera.position.z,
                0.0,
            ],
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            grid_params: [
                self.grid_spacing_m,
                0.15, // line half-width in meters
                self.grid_max_fade_dist,
                self.rt_trail_fade_dist,
            ],
            ball_pos: [
                self.rt_ball_center.x,
                self.rt_ball_center.y,
                self.rt_ball_center.z,
                self.rt_ball_radius,
            ],
        };

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
                .image(rt.storage_image)
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
                .image(rt.storage_image)
                .subresource_range(subresource_range)],
        );

        // Step 4: Composite render pass (LOAD, additive blend fullscreen triangle).
        let composite_rp = self.composite_render_pass.expect("composite render pass present");
        let composite_fb = self.composite_framebuffer.expect("composite framebuffer present");
        let composite_pl = self.composite_pipeline.expect("composite pipeline present");
        let composite_layout = self.composite_pipeline_layout.expect("composite layout present");
        let composite_ds = self.composite_descriptor_set.expect("composite descriptor set present");

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
    ) -> Result<(
        vk::RenderPass, vk::Framebuffer, vk::Pipeline, vk::PipelineLayout,
        vk::DescriptorSetLayout, vk::DescriptorPool, vk::DescriptorSet,
        vk::Sampler, vk::ShaderModule, vk::ShaderModule,
    ), RenderError> {
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
        let ds_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Creating descriptor set layout.
        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Pipeline layout (descriptor set, no push constants).
        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts);
        // SAFETY: Creating pipeline layout.
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Shaders.
        let vert_module = GridPipeline::create_shader_module_pub(
            device,
            include_bytes!("../shaders/composite.vert.spv"),
        )?;
        let frag_module = GridPipeline::create_shader_module_pub(
            device,
            include_bytes!("../shaders/composite.frag.spv"),
        )?;

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
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_attachments);

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
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[create_info],
                    None,
                )
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
            render_pass, framebuffer, pipeline, pipeline_layout,
            descriptor_set_layout, descriptor_pool, descriptor_set,
            sampler, vert_module, frag_module,
        ))
    }

    fn create_vertex_buffer(
        device: &ash::Device,
        allocator: &mut Allocator,
        vertices: &[GridVertex],
    ) -> Result<(vk::Buffer, Allocation), RenderError> {
        let size = std::mem::size_of_val(vertices) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Creating a buffer.
        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Getting memory requirements.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "grid vertices",
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RenderError::Allocator(e.to_string()))?;

        // SAFETY: Binding memory to buffer.
        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(RenderError::Vulkan)?;
        }

        if let Some(mapped) = allocation.mapped_ptr() {
            // SAFETY: Writing to mapped GPU memory within bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    vertices.as_ptr().cast::<u8>(),
                    mapped.as_ptr().cast::<u8>(),
                    size as usize,
                );
            }
        }

        Ok((buffer, allocation))
    }

    /// Create a pre-allocated dynamic buffer for per-frame geometry updates.
    fn create_dynamic_buffer(
        device: &ash::Device,
        allocator: &mut Allocator,
        capacity: vk::DeviceSize,
        name: &str,
    ) -> Result<(vk::Buffer, Allocation), RenderError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Creating a buffer.
        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Getting memory requirements for the buffer.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RenderError::Allocator(e.to_string()))?;

        // SAFETY: Binding memory to buffer.
        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(RenderError::Vulkan)?;
        }

        Ok((buffer, allocation))
    }

    /// Write vertices into a pre-allocated mapped buffer.
    ///
    /// Returns the number of vertices written. Silently truncates if over capacity.
    fn upload_to_mapped(allocation: &Allocation, vertices: &[GridVertex], capacity: vk::DeviceSize) -> u32 {
        if vertices.is_empty() {
            return 0;
        }
        let byte_size = std::mem::size_of_val(vertices);
        let write_size = byte_size.min(capacity as usize);
        let vert_count = write_size / std::mem::size_of::<GridVertex>();

        if let Some(mapped) = allocation.mapped_ptr() {
            // SAFETY: Writing to mapped GPU memory within allocation bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    vertices.as_ptr().cast::<u8>(),
                    mapped.as_ptr().cast::<u8>(),
                    write_size,
                );
            }
        }
        vert_count as u32
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

            self.gpu.device.destroy_fence(self.fence, None);
            self.gpu.device.destroy_buffer(self.vertex_buffer, None);
            self.gpu.device.destroy_buffer(self.fill_buffer, None);
            self.gpu.device.destroy_buffer(self.glow_buffer, None);
            if let Some(buf) = self.trail_line_buffer {
                self.gpu.device.destroy_buffer(buf, None);
            }
            if let Some(buf) = self.hud_line_buffer {
                self.gpu.device.destroy_buffer(buf, None);
            }
            if let Some(buf) = self.hud_fill_buffer {
                self.gpu.device.destroy_buffer(buf, None);
            }

            if let Some(alloc) = self.vertex_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.fill_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.glow_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.trail_line_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.hud_line_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.hud_fill_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }

            // Dynamic flight buffers (streaming mode).
            if let Some(buf) = self.flight_line_buffer {
                self.gpu.device.destroy_buffer(buf, None);
            }
            if let Some(buf) = self.flight_fill_buffer {
                self.gpu.device.destroy_buffer(buf, None);
            }
            if let Some(buf) = self.flight_glow_buffer {
                self.gpu.device.destroy_buffer(buf, None);
            }
            if let Some(alloc) = self.flight_line_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.flight_fill_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.flight_glow_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
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

            if let Some(mut rtp) = self.rt_pipeline.take() {
                rtp.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            if let Some(target) = self.target.take() {
                target.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            self.pipeline.destroy(&self.gpu.device);
        }
    }
}
