mod hud;
mod raster;
mod raytrace;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

use cf_scene::camera::Camera;
use cf_scene::color;
use cf_scene::grid::{GridConfig, GridVertex, generate_grid_vertices};
use cf_scene::tee::{TeeBox, generate_ball, generate_ball_glow, generate_tee_border, generate_tee_fill};

/// Per-frame flight render data: ball position + trail points (scene space).
pub struct FlightRenderData {
    pub ball_pos: glam::Vec3,
    pub trail_points: Vec<cf_scene::trail::TrailPoint>,
    pub current_time: f64,
}

use crate::context::GpuContext;
use crate::error::RenderError;
use crate::mode::RenderMode;
use crate::pipeline::{GridPipeline, DEPTH_STENCIL_FORMAT};
use crate::rt_offscreen::build_scene_geometry;
use crate::rt_pipeline::RtPipeline;
use crate::window::Swapchain;

/// Clip bounds that effectively disable clipping (huge region).
const NO_CLIP: [f32; 4] = [-1e6, -1e6, 1e6, 1e6];

/// Fraction of screen width used by the chase camera viewport (right side).
const CHASE_VIEWPORT_FRAC: f32 = 0.20;

/// RT reflection composite pass: fullscreen triangle that samples the RT storage
/// image and additively blends reflections onto the rasterized scene.
pub(super) struct CompositePass {
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
}

/// Render configuration.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Clear color (RGBA, linear).
    pub clear_color: [f32; 4],
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            clear_color: [
                color::BACKGROUND.x,
                color::BACKGROUND.y,
                color::BACKGROUND.z,
                color::BACKGROUND.w,
            ],
        }
    }
}

/// Complete renderer: owns GPU resources and draws frames.
pub struct Renderer {
    pub gpu: GpuContext,
    pub swapchain: Swapchain,
    pub pipeline: GridPipeline,
    pub(super) config: RenderConfig,
    pub(super) framebuffers: Vec<vk::Framebuffer>,
    pub(super) vertex_buffer: vk::Buffer,
    vertex_allocation: Option<Allocation>,
    pub(super) vertex_count: u32,
    pub(super) fill_buffer: vk::Buffer,
    fill_allocation: Option<Allocation>,
    pub(super) tee_fill_count: u32,
    pub(super) tee_border_count: u32,
    pub(super) ball_count: u32,
    pub(super) glow_buffer: vk::Buffer,
    glow_allocation: Option<Allocation>,
    pub(super) glow_count: u32,
    // Whether to draw the static ball on the tee box (hidden when a flight is active).
    pub(super) ball_on_tee_box: bool,
    // Dynamic per-frame flight geometry (in-flight balls + trail glow).
    pub(super) flight_line_buffer: vk::Buffer,
    flight_line_allocation: Option<Allocation>,
    pub(super) flight_line_count: u32,
    flight_line_capacity: vk::DeviceSize,
    pub(super) flight_fill_buffer: vk::Buffer,
    flight_fill_allocation: Option<Allocation>,
    pub(super) flight_fill_count: u32,
    flight_fill_capacity: vk::DeviceSize,
    pub(super) flight_glow_buffer: vk::Buffer,
    flight_glow_allocation: Option<Allocation>,
    pub(super) flight_glow_count: u32,
    flight_glow_capacity: vk::DeviceSize,
    // HUD overlay geometry (2D, pixel coordinates).
    pub(super) hud_line_buffer: vk::Buffer,
    hud_line_allocation: Option<Allocation>,
    pub(super) hud_line_count: u32,
    hud_line_capacity: vk::DeviceSize,
    pub(super) hud_fill_buffer: vk::Buffer,
    hud_fill_allocation: Option<Allocation>,
    pub(super) hud_fill_count: u32,
    hud_fill_capacity: vk::DeviceSize,
    // Depth/stencil image for depth testing.
    depth_stencil_image: vk::Image,
    depth_stencil_view: vk::ImageView,
    depth_stencil_allocation: Option<Allocation>,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    // HUD overlay pass (color-only, no depth/stencil).
    pub(super) hud_render_pass: vk::RenderPass,
    pub(super) hud_framebuffers: Vec<vk::Framebuffer>,
    pub(super) hud_fill_pipeline: vk::Pipeline,
    pub(super) hud_line_pipeline: vk::Pipeline,
    // Ray tracing (optional, only when hardware supports it).
    pub render_mode: RenderMode,
    pub(super) rt_pipeline: Option<RtPipeline>,
    // RT reflection composite pass (present only when RT pipeline exists).
    pub(super) composite: Option<CompositePass>,
    pub(super) grid_spacing_m: f32,
    pub(super) grid_max_fade_dist: f32,
    // Cached RT scene data for per-frame TLAS rebuilds.
    pub(super) rt_ball_center: glam::Vec3,
    rt_trail_points: Vec<glam::Vec3>,
    pub(super) rt_trail_fade_dist: f32,
    rt_needs_update: bool,
    /// Optional chase camera for the right 20% viewport (set per-frame).
    pub(super) chase_camera: Option<Camera>,
}

impl Renderer {
    /// Create a new renderer.
    pub fn new(
        mut gpu: GpuContext,
        swapchain: Swapchain,
        grid_config: &GridConfig,
    ) -> Result<Self, RenderError> {
        // Scene render pass ends at COLOR_ATTACHMENT_OPTIMAL — HUD pass handles final transition.
        let pipeline = GridPipeline::new(
            &gpu.device,
            swapchain.format,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;

        let (depth_stencil_image, depth_stencil_view, depth_stencil_allocation) =
            Self::create_depth_stencil(
                &gpu.device,
                &mut gpu.allocator,
                swapchain.extent.width,
                swapchain.extent.height,
            )?;

        let framebuffers = Self::create_framebuffers(
            &gpu.device,
            &swapchain,
            pipeline.render_pass,
            depth_stencil_view,
        )?;

        // HUD render pass: color-only overlay, preserves scene content, transitions to PRESENT_SRC.
        let hud_render_pass = GridPipeline::create_hud_render_pass(
            &gpu.device,
            swapchain.format,
            vk::ImageLayout::PRESENT_SRC_KHR,
        )?;
        let hud_framebuffers = Self::create_hud_framebuffers(
            &gpu.device,
            &swapchain,
            hud_render_pass,
        )?;
        let (hud_fill_pipeline, hud_line_pipeline) =
            pipeline.create_hud_pipelines(&gpu.device, hud_render_pass)?;

        let grid_verts = generate_grid_vertices(grid_config);
        let vertex_count = grid_verts.len() as u32;
        let (vertex_buffer, vertex_allocation) =
            Self::create_vertex_buffer(&gpu.device, &mut gpu.allocator, &grid_verts)?;

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

        // Pre-allocate dynamic flight buffers (empty initially).
        // Line: trail centerline (LINE_LIST), ~2000 points × 2 verts × 16 bytes = ~64 KB
        let flight_line_capacity: vk::DeviceSize = 128 * 1024;
        let (flight_line_buffer, flight_line_allocation) =
            Self::create_dynamic_buffer(&gpu.device, &mut gpu.allocator, flight_line_capacity, "flight line")?;
        // Fill: up to 5 in-flight balls × 1728 verts = ~140 KB
        let flight_fill_capacity: vk::DeviceSize = 256 * 1024;
        let (flight_fill_buffer, flight_fill_allocation) =
            Self::create_dynamic_buffer(&gpu.device, &mut gpu.allocator, flight_fill_capacity, "flight fill")?;
        // Glow: up to 5 in-flight balls × (ball glow + trail glow) ≈ 6 MB
        let flight_glow_capacity: vk::DeviceSize = 8 * 1024 * 1024;
        let (flight_glow_buffer, flight_glow_allocation) =
            Self::create_dynamic_buffer(&gpu.device, &mut gpu.allocator, flight_glow_capacity, "flight glow")?;

        // Pre-allocate HUD overlay buffers.
        // Lines: ~5000 text verts + decoration lines ≈ 128 KB
        let hud_line_capacity: vk::DeviceSize = 128 * 1024;
        let (hud_line_buffer, hud_line_allocation) =
            Self::create_dynamic_buffer(&gpu.device, &mut gpu.allocator, hud_line_capacity, "hud lines")?;
        // Fill: panel background = 6 verts × 16 bytes = 96 bytes, allocate 4 KB
        let hud_fill_capacity: vk::DeviceSize = 4 * 1024;
        let (hud_fill_buffer, hud_fill_allocation) =
            Self::create_dynamic_buffer(&gpu.device, &mut gpu.allocator, hud_fill_capacity, "hud fill")?;

        // Build RT pipeline if hardware supports it.
        let spacing_m = grid_config.unit.to_meters(f64::from(grid_config.spacing)) as f32;
        let dr = grid_config.unit.to_meters(f64::from(grid_config.downrange)) as f32;
        let lat = grid_config.unit.to_meters(f64::from(grid_config.lateral)) as f32;
        let grid_max_fade_dist = (dr * dr + lat * lat).sqrt();

        let (render_mode, rt_pipeline) = if gpu.rt_supported {
            let tee = TeeBox::default();
            let geometries = build_scene_geometry(grid_config, &tee, &[]);
            match RtPipeline::new(
                &mut gpu,
                swapchain.extent.width,
                swapchain.extent.height,
                &geometries,
            ) {
                Ok(rtp) => {
                    eprintln!("RT pipeline: initialized (windowed mode)");
                    (RenderMode::RayTraced, Some(rtp))
                }
                Err(e) => {
                    eprintln!("RT pipeline init failed, falling back to raster: {e}");
                    (RenderMode::Rasterized, None)
                }
            }
        } else {
            (RenderMode::Rasterized, None)
        };

        // Build composite pass if RT pipeline is available.
        let composite = if let Some(ref rtp) = rt_pipeline {
            Some(Self::create_composite_pass(
                &gpu.device,
                &swapchain,
                rtp.storage_view,
            )?)
        } else {
            None
        };

        let image_count = framebuffers.len();
        let command_buffers = Self::allocate_command_buffers(&gpu, image_count as u32)?;
        let (image_available, render_finished, in_flight_fences) =
            Self::create_sync_objects(&gpu.device, image_count)?;

        Ok(Self {
            gpu,
            swapchain,
            pipeline,
            config: RenderConfig::default(),
            framebuffers,
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
            flight_fill_buffer,
            ball_on_tee_box: true,
            flight_line_buffer,
            flight_line_allocation: Some(flight_line_allocation),
            flight_line_count: 0,
            flight_line_capacity,
            flight_fill_allocation: Some(flight_fill_allocation),
            flight_fill_count: 0,
            flight_fill_capacity,
            flight_glow_buffer,
            flight_glow_allocation: Some(flight_glow_allocation),
            flight_glow_count: 0,
            flight_glow_capacity,
            hud_line_buffer,
            hud_line_allocation: Some(hud_line_allocation),
            hud_line_count: 0,
            hud_line_capacity,
            hud_fill_buffer,
            hud_fill_allocation: Some(hud_fill_allocation),
            hud_fill_count: 0,
            hud_fill_capacity,
            depth_stencil_image,
            depth_stencil_view,
            depth_stencil_allocation: Some(depth_stencil_allocation),
            hud_render_pass,
            hud_framebuffers,
            hud_fill_pipeline,
            hud_line_pipeline,
            command_buffers,
            image_available,
            render_finished,
            in_flight_fences,
            current_frame: 0,
            render_mode,
            rt_pipeline,
            composite,
            grid_spacing_m: spacing_m,
            grid_max_fade_dist,
            rt_ball_center: glam::Vec3::new(0.0, cf_scene::tee::TEE_ELEVATION + tee.ball_radius, 0.0),
            rt_trail_points: Vec::new(),
            rt_trail_fade_dist: 0.0,
            rt_needs_update: false,
            chase_camera: None,
        })
    }

    /// Draw a frame with the current camera.
    ///
    /// Dispatches to the raster or RT scene recording, then overlays HUD,
    /// submits, and presents.
    pub fn draw_frame(&mut self, camera: &Camera) -> Result<(), RenderError> {
        let device = &self.gpu.device;
        let frame = self.current_frame;

        // Wait for this frame's previous use to complete
        // SAFETY: Waiting on a valid fence.
        unsafe {
            device
                .wait_for_fences(&[self.in_flight_fences[frame]], true, u64::MAX)
                .map_err(RenderError::Vulkan)?;
            device
                .reset_fences(&[self.in_flight_fences[frame]])
                .map_err(RenderError::Vulkan)?;
        }

        // Acquire next image
        // SAFETY: Acquiring swapchain image with valid semaphore.
        let (image_index, suboptimal) = unsafe {
            match self.swapchain.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.image_available[frame],
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.resize_to_surface()?;
                    return Ok(());
                }
                Err(e) => return Err(RenderError::Vulkan(e)),
            }
        };

        if suboptimal {
            self.resize_to_surface()?;
            return Ok(());
        }

        let cb = self.command_buffers[image_index as usize];

        // SAFETY: Recording commands into a valid command buffer.
        unsafe {
            device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(RenderError::Vulkan)?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cb, &begin_info)
                .map_err(RenderError::Vulkan)?;

            // Dispatch scene recording — leaves image at COLOR_ATTACHMENT_OPTIMAL.
            self.record_raster_scene(cb, camera, image_index);

            // If RT is active, rebuild TLAS with current geometry and composite reflections.
            let rt_active = self.render_mode == RenderMode::RayTraced
                && self.rt_pipeline.is_some()
                && self.composite.is_some();

            if rt_active {
                self.record_rt_reflections(cb, camera);
                self.record_composite_reflections(cb, image_index);
            }

            // HUD overlay (final_layout = PRESENT_SRC_KHR)
            self.record_hud_commands(cb, image_index, self.swapchain.extent);

            device
                .end_command_buffer(cb)
                .map_err(RenderError::Vulkan)?;
        }

        // Submit
        let wait_semaphores = [self.image_available[frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished[frame]];
        let command_buffers_arr = [cb];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers_arr)
            .signal_semaphores(&signal_semaphores);

        // SAFETY: Submitting to a valid queue with valid sync objects.
        unsafe {
            device
                .queue_submit(
                    self.gpu.graphics_queue,
                    &[submit_info],
                    self.in_flight_fences[frame],
                )
                .map_err(RenderError::Vulkan)?;
        }

        // Present
        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // SAFETY: Presenting with valid swapchain and semaphore.
        unsafe {
            match self
                .swapchain
                .swapchain_loader
                .queue_present(self.gpu.graphics_queue, &present_info)
            {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                    self.resize_to_surface()?;
                }
                Err(e) => return Err(RenderError::Vulkan(e)),
            }
        }

        self.current_frame = (self.current_frame + 1) % self.image_available.len();
        Ok(())
    }

    /// Query the current surface extent and resize the swapchain to match.
    fn resize_to_surface(&mut self) -> Result<(), RenderError> {
        // SAFETY: Querying surface capabilities for current extent.
        let caps = unsafe {
            self.swapchain
                .surface_loader
                .get_physical_device_surface_capabilities(
                    self.gpu.physical_device,
                    self.swapchain.surface,
                )
                .map_err(RenderError::Vulkan)?
        };
        self.resize(caps.current_extent.width, caps.current_extent.height)
    }

    /// Handle window resize.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        // Skip if dimensions haven't changed
        if width == self.swapchain.extent.width && height == self.swapchain.extent.height {
            return Ok(());
        }

        // SAFETY: Wait idle before rebuilding swapchain resources.
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .map_err(RenderError::Vulkan)?;
        }

        // Destroy old framebuffers (scene + HUD) and depth/stencil
        for fb in self.framebuffers.drain(..) {
            // SAFETY: Framebuffers are no longer in use (waited idle above).
            unsafe { self.gpu.device.destroy_framebuffer(fb, None) };
        }
        for fb in self.hud_framebuffers.drain(..) {
            // SAFETY: HUD framebuffers are no longer in use (waited idle above).
            unsafe { self.gpu.device.destroy_framebuffer(fb, None) };
        }
        // SAFETY: Depth/stencil no longer in use (waited idle above).
        unsafe {
            self.gpu
                .device
                .destroy_image_view(self.depth_stencil_view, None);
            self.gpu
                .device
                .destroy_image(self.depth_stencil_image, None);
        }
        if let Some(alloc) = self.depth_stencil_allocation.take() {
            let _ = self.gpu.allocator.free(alloc);
        }

        self.swapchain.recreate(
            &self.gpu.device,
            self.gpu.physical_device,
            self.gpu.queue_family_index,
            width,
            height,
        )?;

        // Use the swapchain's actual extent — it may differ from the requested
        // width/height after surface capabilities clamping.
        let actual_w = self.swapchain.extent.width;
        let actual_h = self.swapchain.extent.height;

        let (ds_image, ds_view, ds_alloc) = Self::create_depth_stencil(
            &self.gpu.device,
            &mut self.gpu.allocator,
            actual_w,
            actual_h,
        )?;
        self.depth_stencil_image = ds_image;
        self.depth_stencil_view = ds_view;
        self.depth_stencil_allocation = Some(ds_alloc);

        self.framebuffers = Self::create_framebuffers(
            &self.gpu.device,
            &self.swapchain,
            self.pipeline.render_pass,
            self.depth_stencil_view,
        )?;
        self.hud_framebuffers = Self::create_hud_framebuffers(
            &self.gpu.device,
            &self.swapchain,
            self.hud_render_pass,
        )?;
        // Resize RT storage image and update composite descriptor to match.
        if let Some(ref mut rtp) = self.rt_pipeline {
            rtp.resize_storage(
                &self.gpu.device,
                &mut self.gpu.allocator,
                actual_w,
                actual_h,
            )?;
        }
        if let Some(ref mut composite) = self.composite {
            for fb in composite.framebuffers.drain(..) {
                // SAFETY: Composite framebuffers no longer in use (waited idle above).
                unsafe { self.gpu.device.destroy_framebuffer(fb, None) };
            }
            composite.framebuffers = Self::create_hud_framebuffers(
                &self.gpu.device,
                &self.swapchain,
                composite.render_pass,
            )?;

            // Update composite descriptor set to point at the resized RT storage image.
            if let Some(ref rtp) = self.rt_pipeline {
                let image_info = vk::DescriptorImageInfo::default()
                    .image_view(rtp.storage_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .sampler(composite.sampler);
                let image_infos = [image_info];
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(composite.descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_infos);
                // SAFETY: Updating descriptor set with valid image view.
                unsafe { self.gpu.device.update_descriptor_sets(&[write], &[]) };
            }
        }

        // Reallocate command buffers if image count changed
        if self.command_buffers.len() != self.framebuffers.len() {
            // SAFETY: Old command buffers freed by pool reset.
            unsafe {
                self.gpu.device.free_command_buffers(
                    self.gpu.command_pool,
                    &self.command_buffers,
                );
            }
            self.command_buffers =
                Self::allocate_command_buffers(&self.gpu, self.framebuffers.len() as u32)?;
        }

        Ok(())
    }

    /// Switch between raster and ray-traced rendering.
    ///
    /// Falls back to `Rasterized` if RT is requested but not available.
    pub fn set_render_mode(&mut self, mode: RenderMode) {
        if mode == RenderMode::RayTraced && self.rt_pipeline.is_none() {
            self.render_mode = RenderMode::Rasterized;
        } else {
            self.render_mode = mode;
        }
    }

    /// Set or clear the chase camera for the right 20% viewport.
    ///
    /// Pass `Some(camera)` when a ball is in flight, `None` otherwise.
    pub fn set_chase_camera(&mut self, camera: Option<Camera>) {
        self.chase_camera = camera;
    }

    /// Update dynamic flight geometry for this frame.
    ///
    /// Generates ball + ball glow + trail glow vertices for each active flight
    /// and uploads them to the pre-allocated dynamic buffers. Also caches
    /// RT scene data for per-frame TLAS rebuild.
    pub fn update_flight_geometry(&mut self, flights: &[FlightRenderData], camera: &Camera) {
        use cf_scene::tee::{generate_ball_at, generate_ball_glow_at};
        use cf_scene::trail::{DEFAULT_TRAIL_LIFETIME, generate_trail_glow, generate_trail_line};
        use crate::rt_offscreen::trim_trail_from_ball_pub;

        self.ball_on_tee_box = flights.is_empty();

        let tee = TeeBox::default();
        let mut line_verts: Vec<GridVertex> = Vec::new();
        let mut fill_verts: Vec<GridVertex> = Vec::new();
        let mut glow_verts: Vec<GridVertex> = Vec::new();

        // Track RT scene state: use the first flight's ball + trail for RT reflections.
        let mut new_rt_ball = glam::Vec3::new(0.0, cf_scene::tee::TEE_ELEVATION + tee.ball_radius, 0.0);
        let mut new_rt_trail: Vec<glam::Vec3> = Vec::new();

        for (i, flight) in flights.iter().enumerate() {
            // Ball mesh at flight position
            fill_verts.extend(generate_ball_at(flight.ball_pos, tee.ball_radius, 8, 16));

            // Ball glow at flight position
            glow_verts.extend(generate_ball_glow_at(
                flight.ball_pos,
                tee.ball_radius,
                8,
                16,
            ));

            // Trail glow ribbon (with endcap) + wireframe centerline
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

            // Use the first flight for RT reflections.
            // Filter trail points by the same time-based fade as the raster trail
            // so the RT reflection disappears in sync with the visible tracer.
            if i == 0 {
                let ball_center = flight.ball_pos + glam::Vec3::new(0.0, tee.ball_radius, 0.0);
                let trail_positions: Vec<glam::Vec3> = flight.trail_points
                    .iter()
                    // COMMENT OUT THIS LINE TO IMPRINT TRACER REFLECTIONS
                    .filter(|p| (flight.current_time - p.time) < DEFAULT_TRAIL_LIFETIME)
                    .map(|p| p.position)
                    .collect();
                let trimmed = trim_trail_from_ball_pub(&trail_positions, 0.8);

                new_rt_ball = ball_center;
                new_rt_trail = trimmed;
            }
        }

        // Check if RT scene changed and needs TLAS rebuild.
        if self.rt_pipeline.is_some()
            && (new_rt_ball != self.rt_ball_center || new_rt_trail != self.rt_trail_points)
        {
            self.rt_ball_center = new_rt_ball;
            self.rt_trail_points = new_rt_trail.clone();
            // Compute trail fade distance (arc length of trimmed trail).
            self.rt_trail_fade_dist = new_rt_trail.windows(2)
                .map(|w| (w[1] - w[0]).length())
                .sum();
            self.rt_needs_update = true;
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

        // Rebuild RT TLAS if scene changed.
        if self.rt_needs_update {
            if let Some(ref mut rt_pipeline) = self.rt_pipeline {
                let grid_config = cf_scene::grid::GridConfig::default();
                let geometries = build_scene_geometry(
                    &grid_config,
                    &tee,
                    &self.rt_trail_points,
                );
                if let Err(e) = rt_pipeline.update_scene(&mut self.gpu, &geometries) {
                    eprintln!("RT scene update failed: {e}");
                }
            }
            self.rt_needs_update = false;
        }
    }

    /// Update HUD overlay geometry for this frame.
    pub fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        if let Some(alloc) = &self.hud_line_allocation {
            self.hud_line_count = Self::upload_to_mapped(alloc, lines, self.hud_line_capacity);
        }
        if let Some(alloc) = &self.hud_fill_allocation {
            self.hud_fill_count = Self::upload_to_mapped(alloc, fills, self.hud_fill_capacity);
        }
    }

    // ── Private helpers ──

    fn create_framebuffers(
        device: &ash::Device,
        swapchain: &Swapchain,
        render_pass: vk::RenderPass,
        depth_stencil_view: vk::ImageView,
    ) -> Result<Vec<vk::Framebuffer>, RenderError> {
        swapchain
            .image_views
            .iter()
            .map(|&view| {
                let attachments = [view, depth_stencil_view];
                let fb_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);

                // SAFETY: Creating framebuffer with valid attachments.
                unsafe { device.create_framebuffer(&fb_info, None).map_err(RenderError::Vulkan) }
            })
            .collect()
    }

    fn create_hud_framebuffers(
        device: &ash::Device,
        swapchain: &Swapchain,
        hud_render_pass: vk::RenderPass,
    ) -> Result<Vec<vk::Framebuffer>, RenderError> {
        swapchain
            .image_views
            .iter()
            .map(|&view| {
                let attachments = [view];
                let fb_info = vk::FramebufferCreateInfo::default()
                    .render_pass(hud_render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);

                // SAFETY: Creating framebuffer with valid attachment.
                unsafe { device.create_framebuffer(&fb_info, None).map_err(RenderError::Vulkan) }
            })
            .collect()
    }

    fn create_depth_stencil(
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, vk::ImageView, Allocation), RenderError> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(DEPTH_STENCIL_FORMAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: Creating depth/stencil image.
        let image = unsafe {
            device
                .create_image(&image_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Querying image memory requirements.
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "depth stencil",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RenderError::Allocator(e.to_string()))?;

        // SAFETY: Binding memory to image.
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(RenderError::Vulkan)?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(DEPTH_STENCIL_FORMAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        // SAFETY: Creating image view.
        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .map_err(RenderError::Vulkan)?
        };

        Ok((image, view, allocation))
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

        // SAFETY: Getting memory requirements for the buffer.
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

        // Copy vertex data
        if let Some(mapped) = allocation.mapped_ptr() {
            // SAFETY: Writing to mapped GPU memory within allocation bounds.
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

    /// Create the composite pass for blending RT reflections onto the rasterized scene.
    fn create_composite_pass(
        device: &ash::Device,
        swapchain: &Swapchain,
        rt_storage_view: vk::ImageView,
    ) -> Result<CompositePass, RenderError> {
        // Render pass: color-only, LOAD, stays at COLOR_ATTACHMENT_OPTIMAL.
        let render_pass = GridPipeline::create_hud_render_pass(
            device,
            swapchain.format,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;

        let framebuffers = Self::create_hud_framebuffers(device, swapchain, render_pass)?;

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

        // Shaders
        let vert_module = GridPipeline::create_shader_module_pub(
            device,
            include_bytes!("../../shaders/composite.vert.spv"),
        )?;
        let frag_module = GridPipeline::create_shader_module_pub(
            device,
            include_bytes!("../../shaders/composite.frag.spv"),
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

        Ok(CompositePass {
            render_pass,
            framebuffers,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            sampler,
            vert_module,
            frag_module,
        })
    }

    fn allocate_command_buffers(
        gpu: &GpuContext,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>, RenderError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(gpu.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        // SAFETY: Allocating command buffers from a valid pool.
        unsafe {
            gpu.device
                .allocate_command_buffers(&alloc_info)
                .map_err(RenderError::Vulkan)
        }
    }

    #[allow(clippy::type_complexity)]
    fn create_sync_objects(
        device: &ash::Device,
        count: usize,
    ) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>), RenderError> {
        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available = Vec::with_capacity(count);
        let mut render_finished = Vec::with_capacity(count);
        let mut fences = Vec::with_capacity(count);

        // SAFETY: Creating sync primitives.
        unsafe {
            for _ in 0..count {
                image_available.push(
                    device
                        .create_semaphore(&sem_info, None)
                        .map_err(RenderError::Vulkan)?,
                );
                render_finished.push(
                    device
                        .create_semaphore(&sem_info, None)
                        .map_err(RenderError::Vulkan)?,
                );
                fences.push(
                    device
                        .create_fence(&fence_info, None)
                        .map_err(RenderError::Vulkan)?,
                );
            }
        }

        Ok((image_available, render_finished, fences))
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.gpu.device.device_wait_idle().ok();

            for &sem in &self.image_available {
                self.gpu.device.destroy_semaphore(sem, None);
            }
            for &sem in &self.render_finished {
                self.gpu.device.destroy_semaphore(sem, None);
            }
            for &fence in &self.in_flight_fences {
                self.gpu.device.destroy_fence(fence, None);
            }

            for fb in &self.framebuffers {
                self.gpu.device.destroy_framebuffer(*fb, None);
            }
            for fb in &self.hud_framebuffers {
                self.gpu.device.destroy_framebuffer(*fb, None);
            }
            self.gpu.device.destroy_pipeline(self.hud_fill_pipeline, None);
            self.gpu.device.destroy_pipeline(self.hud_line_pipeline, None);
            self.gpu.device.destroy_render_pass(self.hud_render_pass, None);
            self.gpu.device.destroy_buffer(self.vertex_buffer, None);
            self.gpu.device.destroy_buffer(self.fill_buffer, None);
            self.gpu.device.destroy_buffer(self.glow_buffer, None);
            self.gpu.device.destroy_buffer(self.flight_line_buffer, None);
            self.gpu.device.destroy_buffer(self.flight_fill_buffer, None);
            self.gpu.device.destroy_buffer(self.flight_glow_buffer, None);
            self.gpu.device.destroy_buffer(self.hud_line_buffer, None);
            self.gpu.device.destroy_buffer(self.hud_fill_buffer, None);
            self.gpu
                .device
                .destroy_image_view(self.depth_stencil_view, None);
            self.gpu
                .device
                .destroy_image(self.depth_stencil_image, None);
            if let Some(alloc) = self.vertex_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.fill_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.glow_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
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
            if let Some(alloc) = self.hud_line_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.hud_fill_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }
            if let Some(alloc) = self.depth_stencil_allocation.take() {
                let _ = self.gpu.allocator.free(alloc);
            }

            if let Some(ref mut composite) = self.composite {
                for fb in &composite.framebuffers {
                    self.gpu.device.destroy_framebuffer(*fb, None);
                }
                self.gpu.device.destroy_pipeline(composite.pipeline, None);
                self.gpu.device.destroy_pipeline_layout(composite.pipeline_layout, None);
                self.gpu.device.destroy_descriptor_pool(composite.descriptor_pool, None);
                self.gpu.device.destroy_descriptor_set_layout(composite.descriptor_set_layout, None);
                self.gpu.device.destroy_sampler(composite.sampler, None);
                self.gpu.device.destroy_render_pass(composite.render_pass, None);
                self.gpu.device.destroy_shader_module(composite.vert_module, None);
                self.gpu.device.destroy_shader_module(composite.frag_module, None);
            }

            if let Some(ref mut rtp) = self.rt_pipeline {
                rtp.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            self.pipeline.destroy(&self.gpu.device);
            self.swapchain.destroy(&self.gpu.device);
        }
    }
}
