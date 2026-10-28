mod hud;
mod raster;
mod raytrace;

use std::mem::ManuallyDrop;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use cf_scene::camera::Camera;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex, generate_grid_vertices};
use cf_scene::tee::{
    TeeBox, generate_ball, generate_ball_glow, generate_tee_border, generate_tee_fill,
};

/// Per-frame flight render data: ball position + trail points (scene space).
pub struct FlightRenderData {
    pub ball_pos: glam::Vec3,
    pub trail_points: Vec<cf_scene::trail::TrailPoint>,
    pub current_time: f64,
}

use crate::context::GpuContext;
use crate::error::RenderError;
use crate::gpu::{HostBuffer, Image2d, Image2dDesc, load_shader};
use crate::green::GreenEdge;
use crate::mode::RenderMode;
use crate::pipeline::GridPipeline;
use crate::rt_offscreen::build_scene_geometry;
use crate::rt_pipeline::RtPipeline;
use crate::window::Swapchain;

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

/// Complete renderer: owns GPU resources and draws frames.
pub struct Renderer {
    pub gpu: GpuContext,
    pub swapchain: Swapchain,
    pub pipeline: GridPipeline,
    pub(super) framebuffers: Vec<vk::Framebuffer>,
    pub(super) vertex_buffer: ManuallyDrop<HostBuffer>,
    pub(super) vertex_count: u32,
    pub(super) fill_buffer: ManuallyDrop<HostBuffer>,
    pub(super) tee_fill_count: u32,
    pub(super) tee_border_count: u32,
    pub(super) ball_count: u32,
    pub(super) glow_buffer: ManuallyDrop<HostBuffer>,
    pub(super) glow_count: u32,
    // Whether to draw the static ball on the tee box (hidden when a flight is active).
    pub(super) ball_on_tee_box: bool,
    // Dynamic per-frame flight geometry (in-flight balls + trail glow).
    pub(super) flight_line_buffer: ManuallyDrop<HostBuffer>,
    pub(super) flight_line_count: u32,
    pub(super) flight_fill_buffer: ManuallyDrop<HostBuffer>,
    pub(super) flight_fill_count: u32,
    pub(super) flight_glow_buffer: ManuallyDrop<HostBuffer>,
    pub(super) flight_glow_count: u32,
    // HUD overlay geometry (2D, pixel coordinates).
    pub(super) hud_line_buffer: ManuallyDrop<HostBuffer>,
    pub(super) hud_line_count: u32,
    pub(super) hud_fill_buffer: ManuallyDrop<HostBuffer>,
    pub(super) hud_fill_count: u32,
    // Depth image for depth testing (`None` only after a failed rebuild).
    depth: Option<Image2d>,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    /// Last window size passed to `resize` (0×0 while minimised).
    window_extent: vk::Extent2D,
    /// Swapchain must be rebuilt before the next acquire (out of date or suboptimal).
    swapchain_stale: bool,
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
    // Cached RT scene data for per-frame TLAS rebuilds.
    pub(super) rt_ball_center: glam::Vec3,
    rt_trail_points: Vec<glam::Vec3>,
    pub(super) rt_trail_fade_dist: f32,
    pub(super) rt_trail_tail_t: f32,
    rt_needs_update: bool,
    /// Optional chase camera for the right 20% viewport (set per-frame).
    pub(super) chase_camera: Option<Camera>,
    /// Edge of the range's green, if any (see [`Renderer::set_green`]).
    pub(super) green_edge: Option<GreenEdge>,
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
            gpu.depth_format,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;

        let depth = Self::create_depth(
            &gpu.device,
            &mut gpu.allocator,
            gpu.depth_format,
            swapchain.extent,
        )?;

        let framebuffers =
            Self::create_framebuffers(&gpu.device, &swapchain, pipeline.render_pass, depth.view())?;

        // HUD render pass: color-only overlay, preserves scene content, transitions to PRESENT_SRC.
        let hud_render_pass = GridPipeline::create_hud_render_pass(
            &gpu.device,
            swapchain.format,
            vk::ImageLayout::PRESENT_SRC_KHR,
        )?;
        let hud_framebuffers =
            Self::create_hud_framebuffers(&gpu.device, &swapchain, hud_render_pass)?;
        let (hud_fill_pipeline, hud_line_pipeline) =
            pipeline.create_hud_pipelines(&gpu.device, hud_render_pass)?;

        let grid_verts = generate_grid_vertices(grid_config);
        let vertex_count = grid_verts.len() as u32;
        let vertex_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &grid_verts,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "grid",
        )?;

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

        // Pre-allocate dynamic flight buffers (empty initially).
        // Line: trail centerline (LINE_LIST), ~2000 points × 2 verts × 16 bytes = ~64 KB
        let flight_line_buffer = HostBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            128 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "flight line",
        )?;
        // Fill: up to 5 in-flight balls × 1728 verts = ~140 KB
        let flight_fill_buffer = HostBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            256 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "flight fill",
        )?;
        // Glow: up to 5 in-flight balls × (ball glow + trail glow) ≈ 6 MB
        let flight_glow_buffer = HostBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            8 * 1024 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "flight glow",
        )?;

        // Pre-allocate HUD overlay buffers.
        // Lines: ~5000 text verts + decoration lines ≈ 128 KB
        let hud_line_buffer = HostBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            128 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "hud lines",
        )?;
        // Fill: panel background = 6 verts × 16 bytes = 96 bytes, allocate 4 KB
        let hud_fill_buffer = HostBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            4 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "hud fill",
        )?;

        // Build RT pipeline if hardware supports it.
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
                rtp.storage.view(),
            )?)
        } else {
            None
        };

        let window_extent = swapchain.extent;
        let image_count = framebuffers.len();
        let command_buffers = Self::allocate_command_buffers(&gpu, image_count as u32)?;
        let (image_available, render_finished, in_flight_fences) =
            Self::create_sync_objects(&gpu.device, image_count)?;

        Ok(Self {
            gpu,
            swapchain,
            pipeline,
            framebuffers,
            vertex_buffer: ManuallyDrop::new(vertex_buffer),
            vertex_count,
            fill_buffer: ManuallyDrop::new(fill_buffer),
            tee_fill_count,
            tee_border_count,
            ball_count,
            glow_buffer: ManuallyDrop::new(glow_buffer),
            glow_count,
            flight_fill_buffer: ManuallyDrop::new(flight_fill_buffer),
            ball_on_tee_box: true,
            flight_line_buffer: ManuallyDrop::new(flight_line_buffer),
            flight_line_count: 0,
            flight_fill_count: 0,
            flight_glow_buffer: ManuallyDrop::new(flight_glow_buffer),
            flight_glow_count: 0,
            hud_line_buffer: ManuallyDrop::new(hud_line_buffer),
            hud_line_count: 0,
            hud_fill_buffer: ManuallyDrop::new(hud_fill_buffer),
            hud_fill_count: 0,
            depth: Some(depth),
            hud_render_pass,
            hud_framebuffers,
            hud_fill_pipeline,
            hud_line_pipeline,
            command_buffers,
            image_available,
            render_finished,
            in_flight_fences,
            current_frame: 0,
            window_extent,
            swapchain_stale: false,
            render_mode,
            rt_pipeline,
            composite,
            rt_ball_center: glam::Vec3::new(
                0.0,
                cf_scene::tee::TEE_ELEVATION + tee.ball_radius,
                0.0,
            ),
            rt_trail_points: Vec::new(),
            rt_trail_fade_dist: 0.0,
            rt_trail_tail_t: 0.0,
            rt_needs_update: false,
            chase_camera: None,
            green_edge: None,
        })
    }

    /// Draw a frame with the current camera.
    ///
    /// Dispatches to the raster or RT scene recording, then overlays HUD,
    /// submits, and presents. Does nothing while the window is minimised
    /// (zero extent). An out-of-date swapchain is rebuilt and the frame
    /// skipped; a suboptimal one is rendered and presented, then rebuilt.
    ///
    /// The in-flight fence is reset only immediately before the submit that
    /// signals it, so every early return leaves it signalled.
    pub fn draw_frame(&mut self, camera: &Camera) -> Result<(), RenderError> {
        if self.window_extent.width == 0 || self.window_extent.height == 0 {
            return Ok(());
        }
        if self.swapchain_stale && !self.recreate_swapchain()? {
            // Surface still has a zero extent (minimised).
            return Ok(());
        }

        let frame = self.current_frame;

        // Wait for this frame's previous use to complete
        // SAFETY: Waiting on a valid fence.
        unsafe {
            self.gpu
                .device
                .wait_for_fences(&[self.in_flight_fences[frame]], true, u64::MAX)
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
                    // No image acquired and the semaphore is not signalled.
                    // Rebuild now; the next frame renders.
                    self.swapchain_stale = true;
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(e) => return Err(RenderError::Vulkan(e)),
            }
        };

        // A suboptimal image is still presentable: render it, rebuild after present.
        if suboptimal {
            self.swapchain_stale = true;
        }

        let device = &self.gpu.device;
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

            device.end_command_buffer(cb).map_err(RenderError::Vulkan)?;
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

        // SAFETY: Resetting a fence that is signalled (waited above) and
        // submitting to a valid queue with valid sync objects.
        unsafe {
            device
                .reset_fences(&[self.in_flight_fences[frame]])
                .map_err(RenderError::Vulkan)?;
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

        // This frame's semaphores and fence are in use now, whatever the
        // present returns; the next frame takes the next set.
        self.current_frame = (self.current_frame + 1) % self.image_available.len();

        // SAFETY: Presenting with valid swapchain and semaphore.
        unsafe {
            match self
                .swapchain
                .swapchain_loader
                .queue_present(self.gpu.graphics_queue, &present_info)
            {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.swapchain_stale = true;
                }
                Err(e) => return Err(RenderError::Vulkan(e)),
            }
        }

        if self.swapchain_stale {
            self.recreate_swapchain()?;
        }
        Ok(())
    }

    /// Rebuild the swapchain at the current surface extent, even if the size
    /// is unchanged.
    ///
    /// Returns `Ok(false)` without rebuilding (leaving the swapchain marked
    /// stale) while the surface extent is zero, e.g. when minimised.
    fn recreate_swapchain(&mut self) -> Result<bool, RenderError> {
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
        // u32::MAX means the surface size is set by the swapchain: use the window size.
        let extent = if caps.current_extent.width == u32::MAX {
            self.window_extent
        } else {
            caps.current_extent
        };
        if extent.width == 0 || extent.height == 0 {
            return Ok(false);
        }
        self.rebuild_swapchain_resources(extent.width, extent.height)?;
        self.swapchain_stale = false;
        Ok(true)
    }

    /// Handle window resize.
    ///
    /// A zero width or height (minimised) pauses rendering until a non-zero
    /// size arrives.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        self.window_extent = vk::Extent2D { width, height };
        if width == 0 || height == 0 {
            return Ok(());
        }

        // Skip if dimensions haven't changed
        if width == self.swapchain.extent.width && height == self.swapchain.extent.height {
            return Ok(());
        }

        self.rebuild_swapchain_resources(width, height)?;
        self.swapchain_stale = false;
        Ok(())
    }

    /// Destroy and recreate the swapchain and every resource sized to it.
    fn rebuild_swapchain_resources(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        // SAFETY: Wait idle before rebuilding swapchain resources.
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .map_err(RenderError::Vulkan)?;
        }

        // Destroy old framebuffers (scene + HUD) and depth
        for fb in self.framebuffers.drain(..) {
            // SAFETY: Framebuffers are no longer in use (waited idle above).
            unsafe { self.gpu.device.destroy_framebuffer(fb, None) };
        }
        for fb in self.hud_framebuffers.drain(..) {
            // SAFETY: HUD framebuffers are no longer in use (waited idle above).
            unsafe { self.gpu.device.destroy_framebuffer(fb, None) };
        }
        if let Some(depth) = self.depth.take() {
            // SAFETY: Depth no longer in use (waited idle above).
            unsafe { depth.destroy(&self.gpu.device, &mut self.gpu.allocator) };
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

        let depth = Self::create_depth(
            &self.gpu.device,
            &mut self.gpu.allocator,
            self.gpu.depth_format,
            self.swapchain.extent,
        )?;
        let depth_view = depth.view();
        self.depth = Some(depth);

        self.framebuffers = Self::create_framebuffers(
            &self.gpu.device,
            &self.swapchain,
            self.pipeline.render_pass,
            depth_view,
        )?;
        self.hud_framebuffers =
            Self::create_hud_framebuffers(&self.gpu.device, &self.swapchain, self.hud_render_pass)?;
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
                    .image_view(rtp.storage.view())
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
                self.gpu
                    .device
                    .free_command_buffers(self.gpu.command_pool, &self.command_buffers);
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
        use crate::rt_offscreen::RtTrail;
        use cf_scene::tee::{generate_ball_at, generate_ball_glow_at};
        use cf_scene::trail::{DEFAULT_TRAIL_LIFETIME, generate_trail_glow, generate_trail_line};

        self.ball_on_tee_box = flights.is_empty();

        let tee = TeeBox::default();
        let mut line_verts: Vec<GridVertex> = Vec::new();
        let mut fill_verts: Vec<GridVertex> = Vec::new();
        let mut glow_verts: Vec<GridVertex> = Vec::new();

        // Track RT scene state: reflections show the newest flight's ball +
        // trail, the flight the HUD and chase camera follow.
        let mut new_rt_ball =
            glam::Vec3::new(0.0, cf_scene::tee::TEE_ELEVATION + tee.ball_radius, 0.0);
        let mut new_rt_trail = RtTrail::default();

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

            // Use the newest flight for RT reflections. The reflected trail is
            // the part of the trail the raster tracer still draws, so the
            // reflection disappears in sync with the visible tracer.
            if i + 1 == flights.len() {
                new_rt_ball = flight.ball_pos + glam::Vec3::new(0.0, tee.ball_radius, 0.0);
                new_rt_trail = RtTrail::new(
                    &flight.trail_points,
                    flight.current_time,
                    DEFAULT_TRAIL_LIFETIME,
                );
            }
        }

        // The tail fade value changes as the trail ages, even between TLAS
        // rebuilds.
        self.rt_trail_tail_t = new_rt_trail.tail_t;

        // Check if RT scene changed and needs TLAS rebuild.
        if self.rt_pipeline.is_some()
            && (new_rt_ball != self.rt_ball_center || new_rt_trail.points != self.rt_trail_points)
        {
            self.rt_ball_center = new_rt_ball;
            self.rt_trail_fade_dist = new_rt_trail.fade_dist;
            self.rt_trail_points = new_rt_trail.points;
            self.rt_needs_update = true;
        }

        self.flight_line_count = self.flight_line_buffer.write(&line_verts);
        self.flight_fill_count = self.flight_fill_buffer.write(&fill_verts);
        self.flight_glow_count = self.flight_glow_buffer.write(&glow_verts);

        // Rebuild RT TLAS if scene changed.
        if self.rt_needs_update {
            if let Some(ref mut rt_pipeline) = self.rt_pipeline {
                let grid_config = cf_scene::grid::GridConfig::default();
                let geometries = build_scene_geometry(&grid_config, &tee, &self.rt_trail_points);
                if let Err(e) = rt_pipeline.update_scene(&mut self.gpu, &geometries) {
                    eprintln!("RT scene update failed: {e}");
                }
            }
            self.rt_needs_update = false;
        }
    }

    /// Update HUD overlay geometry for this frame.
    pub fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.hud_line_count = self.hud_line_buffer.write(lines);
        self.hud_fill_count = self.hud_fill_buffer.write(fills);
    }

    // ── Private helpers ──

    fn create_framebuffers(
        device: &ash::Device,
        swapchain: &Swapchain,
        render_pass: vk::RenderPass,
        depth_view: vk::ImageView,
    ) -> Result<Vec<vk::Framebuffer>, RenderError> {
        swapchain
            .image_views
            .iter()
            .map(|&view| {
                let attachments = [view, depth_view];
                let fb_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);

                // SAFETY: Creating framebuffer with valid attachments.
                unsafe {
                    device
                        .create_framebuffer(&fb_info, None)
                        .map_err(RenderError::Vulkan)
                }
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
                unsafe {
                    device
                        .create_framebuffer(&fb_info, None)
                        .map_err(RenderError::Vulkan)
                }
            })
            .collect()
    }

    fn create_depth(
        device: &ash::Device,
        allocator: &mut Allocator,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Image2d, RenderError> {
        Image2d::new(
            device,
            allocator,
            &Image2dDesc {
                extent,
                format,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                aspect: vk::ImageAspectFlags::DEPTH,
                name: "depth",
            },
        )
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

        // Shaders
        let vert_module = load_shader(device, include_bytes!("../../shaders/composite.vert.spv"))?;
        let frag_module = load_shader(device, include_bytes!("../../shaders/composite.frag.spv"))?;

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
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

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

            if let Some(edge) = self.green_edge.take() {
                edge.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

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
            self.gpu
                .device
                .destroy_pipeline(self.hud_fill_pipeline, None);
            self.gpu
                .device
                .destroy_pipeline(self.hud_line_pipeline, None);
            self.gpu
                .device
                .destroy_render_pass(self.hud_render_pass, None);
            // SAFETY: Each `ManuallyDrop` buffer is taken once, here in `Drop`,
            // and never used after.
            for buffer in [
                &mut self.vertex_buffer,
                &mut self.fill_buffer,
                &mut self.glow_buffer,
                &mut self.flight_line_buffer,
                &mut self.flight_fill_buffer,
                &mut self.flight_glow_buffer,
                &mut self.hud_line_buffer,
                &mut self.hud_fill_buffer,
            ] {
                ManuallyDrop::take(buffer).destroy(&self.gpu.device, &mut self.gpu.allocator);
            }
            if let Some(depth) = self.depth.take() {
                depth.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            if let Some(ref mut composite) = self.composite {
                for fb in &composite.framebuffers {
                    self.gpu.device.destroy_framebuffer(*fb, None);
                }
                self.gpu.device.destroy_pipeline(composite.pipeline, None);
                self.gpu
                    .device
                    .destroy_pipeline_layout(composite.pipeline_layout, None);
                self.gpu
                    .device
                    .destroy_descriptor_pool(composite.descriptor_pool, None);
                self.gpu
                    .device
                    .destroy_descriptor_set_layout(composite.descriptor_set_layout, None);
                self.gpu.device.destroy_sampler(composite.sampler, None);
                self.gpu
                    .device
                    .destroy_render_pass(composite.render_pass, None);
                self.gpu
                    .device
                    .destroy_shader_module(composite.vert_module, None);
                self.gpu
                    .device
                    .destroy_shader_module(composite.frag_module, None);
            }

            if let Some(rtp) = self.rt_pipeline.take() {
                rtp.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            self.pipeline.destroy(&self.gpu.device);
            self.swapchain.destroy(&self.gpu.device);
        }
    }
}
