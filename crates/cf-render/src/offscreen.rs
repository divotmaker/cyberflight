use std::mem::ManuallyDrop;

use ash::vk;

use cf_scene::camera::Camera;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex};
use cf_scene::tee::TeeBox;
use cf_scene::trail::TrailPoint;

use crate::context::{GpuConfig, GpuContext};
use crate::error::RenderError;
use crate::framebuffer::OffscreenTarget;
use crate::readback::FrameBuffers;
use crate::recorder::{FrameTarget, SceneRecorder};
use crate::reflections::RtSceneState;
use crate::render::FlightRenderData;
use crate::rt_offscreen::RtTrail;

/// Offscreen renderer for headless PNG generation.
pub struct OffscreenRenderer {
    /// Passes, scene, per-frame content, reflections and chase camera.
    recorder: ManuallyDrop<SceneRecorder>,
    target: ManuallyDrop<OffscreenTarget>,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    gpu: GpuContext,
}

/// Offscreen image format (RGBA8 sRGB for correct gamma in PNG output).
const OFFSCREEN_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

impl OffscreenRenderer {
    /// Create a headless renderer for the given grid configuration.
    /// Enables raytracing if hardware supports it and RT setup succeeds.
    pub fn new(width: u32, height: u32, grid_config: &GridConfig) -> Result<Self, RenderError> {
        let scene = RtSceneState::on_tee(&TeeBox::default());
        Self::create(width, height, grid_config, true, Some(scene))
    }

    /// Create a headless renderer with a ball in flight and a tracer trail,
    /// without ray tracing: [`OffscreenRenderer::update_flight`] with one
    /// flight on a raster renderer.
    ///
    /// `ball_pos` is dimple's reference point of the ball (see
    /// [`FlightRenderData`]). `trail_points` go from oldest (index 0) to
    /// newest (last = near ball_pos). The trail glow ribbon faces `camera`.
    /// The tracer fades over
    /// [`DEFAULT_TRAIL_LIFETIME`](cf_scene::trail::DEFAULT_TRAIL_LIFETIME);
    /// `_max_lifetime` is not used.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flight(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        ball_pos: glam::Vec3,
        trail_points: &[TrailPoint],
        current_time: f64,
        _max_lifetime: f64,
        camera: &Camera,
    ) -> Result<Self, RenderError> {
        let mut renderer = Self::create(width, height, grid_config, false, None)?;
        renderer.update_flight(
            &[FlightRenderData {
                ball_pos,
                trail_points: trail_points.to_vec(),
                current_time,
            }],
            camera,
        );
        Ok(renderer)
    }

    /// Create a headless renderer with RT reflection compositing for the static range.
    ///
    /// Falls back to raster-only if RT hardware is not available or RT setup fails.
    pub fn new_rt(width: u32, height: u32, grid_config: &GridConfig) -> Result<Self, RenderError> {
        let tee = TeeBox::default();
        let scene = RtSceneState::new(
            glam::Vec3::new(0.0, cf_scene::tee::TEE_ELEVATION + tee.ball_radius, 0.0),
            RtTrail::default(),
        );
        Self::create(width, height, grid_config, true, Some(scene))
    }

    /// Create a headless renderer with RT reflection compositing and a ball
    /// in flight: [`OffscreenRenderer::update_flight`] with one flight on a
    /// ray-traced renderer. The arguments are those of
    /// [`OffscreenRenderer::new_with_flight`].
    ///
    /// Falls back to raster-only if RT hardware is not available or RT setup fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flight_rt(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        ball_pos: glam::Vec3,
        trail_points: &[TrailPoint],
        current_time: f64,
        _max_lifetime: f64,
        camera: &Camera,
    ) -> Result<Self, RenderError> {
        let scene = RtSceneState::on_tee(&TeeBox::default());
        let mut renderer = Self::create(width, height, grid_config, true, Some(scene))?;
        renderer.update_flight(
            &[FlightRenderData {
                ball_pos,
                trail_points: trail_points.to_vec(),
                current_time,
            }],
            camera,
        );
        Ok(renderer)
    }

    /// The constructor behind every public one: a device with or without
    /// ray tracing; reflections showing `rt_scene` when it is given and the
    /// device supports them (raster when they fail to set up); the ball on
    /// the tee.
    fn create(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        enable_raytracing: bool,
        rt_scene: Option<RtSceneState>,
    ) -> Result<Self, RenderError> {
        let config = GpuConfig {
            validation: cfg!(debug_assertions),
            enable_raytracing,
        };

        let mut gpu = GpuContext::new_headless(config)?;

        // Reflections when `rt_scene` is given and the device supports them;
        // raster if they fail to set up.
        let recorder = SceneRecorder::new(
            &mut gpu,
            OFFSCREEN_FORMAT,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::Extent2D { width, height },
            grid_config,
            rt_scene,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            recorder.scene_pass(),
            recorder.hud_pass(),
        )?;

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        Ok(Self {
            recorder: ManuallyDrop::new(recorder),
            target: ManuallyDrop::new(target),
            command_buffer,
            fence,
            gpu,
        })
    }

    /// Whether this renderer has RT reflection compositing enabled.
    #[must_use]
    pub fn has_rt(&self) -> bool {
        self.recorder.has_reflections()
    }

    /// Draw the edge of `green` as one ring on the range floor, faded like
    /// the renderer's grid; `None` draws nothing. Replaces any green set
    /// before. `_grid` must be the grid the renderer was created with.
    pub fn set_green(
        &mut self,
        green: Option<&GreenConfig>,
        _grid: &GridConfig,
    ) -> Result<(), RenderError> {
        self.recorder.set_green(&mut self.gpu, green)
    }

    /// Set or clear the chase camera for the right 20% split viewport.
    ///
    /// Pass `Some(camera)` while a ball is being followed, `None` otherwise.
    pub fn set_chase_camera(&mut self, camera: Option<Camera>) {
        self.recorder.set_chase_camera(camera);
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

    /// Update dynamic flight geometry for the next `render`. Reflections
    /// show the newest flight's ball and the part of its trail the tracer
    /// still draws, or the ball on the tee when there is no flight.
    pub fn update_flight(&mut self, flights: &[FlightRenderData], camera: &Camera) {
        self.recorder.update_flight(&mut self.gpu, flights, camera);
    }

    /// Update HUD overlay geometry (pixel coordinates) for the next `render`.
    pub fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.recorder.update_hud(lines, fills);
    }

    /// Update HUD overlay geometry for the next `render`; the same as
    /// [`OffscreenRenderer::update_hud`].
    pub fn update_hud_streaming(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.update_hud(lines, fills);
    }

    /// Render the grid and read back the result as RGBA8 pixels.
    ///
    /// When RT is enabled, renders the raster scene, traces RT reflection
    /// rays, and composites them onto the rasterized output. The HUD is
    /// drawn last, over the scene and its reflections.
    pub fn render(&mut self, camera: &Camera) -> Result<FrameBuffers, RenderError> {
        let target = &self.target;
        let device = &self.gpu.device;
        let width = target.width;
        let height = target.height;
        let frame_target = FrameTarget {
            scene: target.scene,
            overlay: target.overlay,
            extent: vk::Extent2D { width, height },
        };

        // Record command buffer
        // SAFETY: Recording commands into a valid command buffer. The target's
        // framebuffers were made with the recorder's passes at the target
        // size, and the previous frame's fence was waited on.
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(RenderError::Vulkan)?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(RenderError::Vulkan)?;

            // Scene, reflections and HUD; leaves the image in TRANSFER_SRC_OPTIMAL.
            self.recorder
                .record_frame(device, self.command_buffer, &frame_target, camera);

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

            let device = &self.gpu.device;
            let allocator = &mut self.gpu.allocator;
            // SAFETY: Each `ManuallyDrop` field is taken once, here in `Drop`,
            // and never used after; the device is idle.
            ManuallyDrop::take(&mut self.target).destroy(device, allocator);
            ManuallyDrop::take(&mut self.recorder).destroy(device, allocator);
        }
    }
}
