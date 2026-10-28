use std::mem::ManuallyDrop;

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use cf_scene::camera::Camera;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex};

use crate::context::GpuContext;
use crate::error::RenderError;
use crate::flight::FlightRenderData;
use crate::gpu::{Image2d, Image2dDesc};
use crate::mode::{RenderMode, RtMode};
use crate::recorder::{FrameTarget, SceneRecorder};
use crate::window::Swapchain;

/// Complete renderer: owns GPU resources and draws frames.
pub struct Renderer {
    /// Passes, scene, per-frame content, reflections, mode and chase camera.
    recorder: ManuallyDrop<SceneRecorder>,
    swapchain: Swapchain,
    /// Colour + depth framebuffers of the scene pass, one per swapchain image.
    framebuffers: Vec<vk::Framebuffer>,
    /// Colour-only framebuffers of the HUD and composite passes, one per
    /// swapchain image.
    overlay_framebuffers: Vec<vk::Framebuffer>,
    /// Depth image for depth testing (`None` after a failed rebuild, until
    /// the next one succeeds).
    depth: Option<Image2d>,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    /// Last window size passed to `resize` (0×0 while minimised).
    window_extent: vk::Extent2D,
    /// Swapchain must be rebuilt before the next acquire (out of date,
    /// suboptimal, or the last rebuild failed).
    swapchain_stale: bool,
    gpu: GpuContext,
}

impl Renderer {
    /// Create a renderer for `swapchain` on `gpu`, drawing the range
    /// `grid_config` with the ball on the tee. It ray-traces floor
    /// reflections when the GPU supports them and they set up (otherwise it
    /// mirrors them across the floor and says so on stderr).
    ///
    /// # Errors
    /// A part fails to create; what was created, and `swapchain`, are
    /// destroyed first.
    pub fn new(
        mut gpu: GpuContext,
        swapchain: Swapchain,
        grid_config: &GridConfig,
    ) -> Result<Self, RenderError> {
        // The HUD pass ends every frame and leaves the image ready to present.
        // Ray-traced reflections when the device supports them; mirrored if not.
        // One frame in flight per swapchain image (see `create_sync_objects`),
        // each with its own buffers and ray tracer structures.
        let recorder = match SceneRecorder::new(
            &mut gpu,
            swapchain.format,
            vk::ImageLayout::PRESENT_SRC_KHR,
            swapchain.extent,
            grid_config,
            RtMode::Auto,
            swapchain.image_views.len(),
        ) {
            Ok(recorder) => recorder,
            Err(e) => {
                // SAFETY: No GPU work has used the swapchain.
                unsafe { swapchain.destroy(&gpu.device) };
                return Err(e);
            }
        };
        if recorder.has_ray_tracing() {
            eprintln!("RT pipeline: initialized (windowed mode)");
        }

        // From here on an early return drops `renderer`, which destroys what
        // it holds so far.
        let window_extent = swapchain.extent;
        let mut renderer = Self {
            recorder: ManuallyDrop::new(recorder),
            swapchain,
            framebuffers: Vec::new(),
            overlay_framebuffers: Vec::new(),
            depth: None,
            command_buffers: Vec::new(),
            image_available: Vec::new(),
            render_finished: Vec::new(),
            in_flight_fences: Vec::new(),
            current_frame: 0,
            window_extent,
            swapchain_stale: false,
            gpu,
        };
        renderer.create_swapchain_targets()?;
        let image_count = renderer.framebuffers.len();
        renderer.command_buffers =
            Self::allocate_command_buffers(&renderer.gpu, image_count as u32)?;
        renderer.create_sync_objects(image_count)?;
        Ok(renderer)
    }

    /// Draw a frame with the current camera.
    ///
    /// Records the scene, the reflections (ray-traced in `RayTraced` mode,
    /// else mirrored) and the HUD, then submits and presents. Does nothing while the window is minimised
    /// (zero extent). An out-of-date swapchain is rebuilt and the frame
    /// skipped; a suboptimal one is rendered and presented, then rebuilt.
    /// After a failed rebuild every frame retries it first, and a frame
    /// without a complete set of swapchain resources is skipped.
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
        if !self.swapchain_resources_complete() {
            self.swapchain_stale = true;
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
        let target = FrameTarget {
            scene: self.framebuffers[image_index as usize],
            overlay: self.overlay_framebuffers[image_index as usize],
            extent: self.swapchain.extent,
        };

        // SAFETY: Recording commands into a valid command buffer. The
        // framebuffers of `image_index` were made with the recorder's passes
        // at the swapchain extent, the recorder was resized to it, and this
        // frame's fence was waited on above.
        unsafe {
            device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(RenderError::Vulkan)?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cb, &begin_info)
                .map_err(RenderError::Vulkan)?;

            // Scene, reflections and HUD; leaves the image in PRESENT_SRC_KHR.
            self.recorder
                .record_frame(device, cb, &target, camera, frame);

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

    /// Whether every swapchain image has its two framebuffers and a command
    /// buffer, as after a successful rebuild.
    fn swapchain_resources_complete(&self) -> bool {
        let images = self.swapchain.image_views.len();
        self.depth.is_some()
            && self.framebuffers.len() == images
            && self.overlay_framebuffers.len() == images
            && self.command_buffers.len() == images
    }

    /// Destroy and recreate the swapchain and every resource sized to it.
    /// On failure the swapchain is marked stale, so the next `draw_frame`
    /// retries before it records anything.
    fn rebuild_swapchain_resources(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        let rebuilt = self.try_rebuild_swapchain_resources(width, height);
        if rebuilt.is_err() {
            self.swapchain_stale = true;
        }
        rebuilt
    }

    fn try_rebuild_swapchain_resources(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        // SAFETY: Wait idle before rebuilding swapchain resources.
        unsafe {
            self.gpu
                .device
                .device_wait_idle()
                .map_err(RenderError::Vulkan)?;
        }

        // Destroy old framebuffers (scene + overlay) and depth
        for fb in self.framebuffers.drain(..) {
            // SAFETY: Framebuffers are not in use (waited idle above).
            unsafe { self.gpu.device.destroy_framebuffer(fb, None) };
        }
        for fb in self.overlay_framebuffers.drain(..) {
            // SAFETY: Overlay framebuffers are not in use (waited idle above).
            unsafe { self.gpu.device.destroy_framebuffer(fb, None) };
        }
        if let Some(depth) = self.depth.take() {
            // SAFETY: Depth is not in use (waited idle above).
            unsafe { depth.destroy(&self.gpu.device, &mut self.gpu.allocator) };
        }

        self.swapchain.recreate(
            &self.gpu.device,
            self.gpu.physical_device,
            self.gpu.queue_family_index,
            width,
            height,
        )?;

        self.create_swapchain_targets()?;
        // The swapchain's actual extent, which may differ from the requested
        // size after surface clamping.
        // SAFETY: The device is idle (waited above) and the old framebuffers
        // are destroyed.
        unsafe {
            self.recorder.resize(
                &self.gpu.device,
                &mut self.gpu.allocator,
                self.swapchain.extent,
            )?;
        }

        // Reallocate command buffers if image count changed
        if self.command_buffers.len() != self.framebuffers.len() {
            let old = std::mem::take(&mut self.command_buffers);
            if !old.is_empty() {
                // SAFETY: The device is idle (waited above); no pending
                // work uses the old command buffers.
                unsafe {
                    self.gpu
                        .device
                        .free_command_buffers(self.gpu.command_pool, &old);
                }
            }
            self.command_buffers =
                Self::allocate_command_buffers(&self.gpu, self.framebuffers.len() as u32)?;
        }

        Ok(())
    }

    /// The swapchain's current size in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    /// The current rendering mode.
    #[must_use]
    pub fn render_mode(&self) -> RenderMode {
        self.recorder.mode()
    }

    /// Whether the renderer can ray-trace its floor reflections: the GPU
    /// supports it and the ray tracer is set up and working.
    #[must_use]
    pub fn ray_tracing_available(&self) -> bool {
        self.recorder.has_ray_tracing()
    }

    /// Switch the floor reflections between ray-traced (`RayTraced`) and
    /// mirrored across the floor plane (`Rasterized`), from the next frame on.
    ///
    /// Falls back to `Rasterized` if RT is requested but not available.
    pub fn set_render_mode(&mut self, mode: RenderMode) {
        self.recorder.set_mode(mode);
    }

    /// Draw the edge of `green` as one ring on the range floor, faded like
    /// the renderer's grid; `None` draws nothing. Replaces any green set
    /// before.
    pub fn set_green(&mut self, green: Option<&GreenConfig>) -> Result<(), RenderError> {
        self.recorder.set_green(&mut self.gpu, green)
    }

    /// Set or clear the chase camera for the right 20% viewport.
    ///
    /// Pass `Some(camera)` when a ball is in flight, `None` otherwise.
    pub fn set_chase_camera(&mut self, camera: Option<Camera>) {
        self.recorder.set_chase_camera(camera);
    }

    /// Update dynamic flight geometry for the next frame.
    ///
    /// Builds ball + ball glow + tracer tube vertices for each active flight; the
    /// next `draw_frame` uploads them. Reflections show the newest flight's
    /// ball and its tracer, or the ball on the tee.
    pub fn update_flight(&mut self, flights: &[FlightRenderData]) {
        self.recorder.update_flight(flights);
    }

    /// Update HUD overlay geometry (pixel coordinates) for the next frame.
    pub fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.recorder.update_hud(lines, fills);
    }

    // ── Private helpers ──

    /// Create the depth image, the scene framebuffers and the overlay
    /// framebuffers at the swapchain extent, keeping each in `self` as soon as
    /// it exists. Requires none of them in `self`.
    fn create_swapchain_targets(&mut self) -> Result<(), RenderError> {
        let depth = Self::create_depth(
            &self.gpu.device,
            &mut self.gpu.allocator,
            self.gpu.depth_format,
            self.swapchain.extent,
        )?;
        let depth_view = depth.view();
        self.depth = Some(depth);

        Self::create_framebuffers(
            &self.gpu.device,
            &self.swapchain,
            self.recorder.scene_pass(),
            Some(depth_view),
            &mut self.framebuffers,
        )?;
        Self::create_framebuffers(
            &self.gpu.device,
            &self.swapchain,
            self.recorder.hud_pass(),
            None,
            &mut self.overlay_framebuffers,
        )
    }

    /// Push a framebuffer for `render_pass` over each swapchain image (plus
    /// `depth_view` if given) onto `framebuffers`, stopping at the first
    /// error.
    fn create_framebuffers(
        device: &ash::Device,
        swapchain: &Swapchain,
        render_pass: vk::RenderPass,
        depth_view: Option<vk::ImageView>,
        framebuffers: &mut Vec<vk::Framebuffer>,
    ) -> Result<(), RenderError> {
        for &view in &swapchain.image_views {
            let attachments: &[vk::ImageView] = match depth_view {
                Some(depth_view) => &[view, depth_view],
                None => &[view],
            };
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(attachments)
                .width(swapchain.extent.width)
                .height(swapchain.extent.height)
                .layers(1);

            // SAFETY: Creating framebuffer with valid attachments.
            let framebuffer = unsafe {
                device
                    .create_framebuffer(&fb_info, None)
                    .map_err(RenderError::Vulkan)?
            };
            framebuffers.push(framebuffer);
        }
        Ok(())
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

    /// Create `count` sets of frame semaphores and (signalled) fences,
    /// keeping each in `self` as soon as it exists.
    fn create_sync_objects(&mut self, count: usize) -> Result<(), RenderError> {
        let device = &self.gpu.device;
        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        // SAFETY: Creating sync primitives.
        unsafe {
            for _ in 0..count {
                self.image_available.push(
                    device
                        .create_semaphore(&sem_info, None)
                        .map_err(RenderError::Vulkan)?,
                );
                self.render_finished.push(
                    device
                        .create_semaphore(&sem_info, None)
                        .map_err(RenderError::Vulkan)?,
                );
                self.in_flight_fences.push(
                    device
                        .create_fence(&fence_info, None)
                        .map_err(RenderError::Vulkan)?,
                );
            }
        }

        Ok(())
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
            for fb in &self.overlay_framebuffers {
                self.gpu.device.destroy_framebuffer(*fb, None);
            }
            if let Some(depth) = self.depth.take() {
                depth.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }
            // SAFETY: Taken once, here in `Drop`, and never used after; the
            // device is idle.
            ManuallyDrop::take(&mut self.recorder)
                .destroy(&self.gpu.device, &mut self.gpu.allocator);
            self.swapchain.destroy(&self.gpu.device);
        }
    }
}
