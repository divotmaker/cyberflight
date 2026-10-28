use std::mem::ManuallyDrop;

use ash::vk;

use cf_scene::camera::Camera;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex};

use crate::context::{GpuConfig, GpuContext};
use crate::error::RenderError;
use crate::flight::FlightRenderData;
use crate::framebuffer::OffscreenTarget;
use crate::mode::{RenderMode, RtMode};
use crate::readback::Frame;
use crate::recorder::{FrameTarget, SceneRecorder};

/// Headless renderer: draws frames into an image and reads them back, for
/// screenshots and streaming.
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
    /// Create a `width` × `height` renderer of the range `grid_config`, with
    /// the ball on the tee and no HUD.
    ///
    /// `RtMode::Off` creates the device without ray tracing; `Auto` requests
    /// it and ray-traces floor reflections when the GPU supports them and
    /// they set up (otherwise it mirrors them across the floor and says so on
    /// stderr).
    pub fn new(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        rt: RtMode,
    ) -> Result<Self, RenderError> {
        let config = GpuConfig {
            validation: cfg!(debug_assertions),
            enable_raytracing: rt == RtMode::Auto,
        };

        let mut gpu = GpuContext::new_headless(config)?;

        let recorder = SceneRecorder::new(
            &mut gpu,
            OFFSCREEN_FORMAT,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::Extent2D { width, height },
            grid_config,
            rt,
        )?;

        let target = match OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            recorder.scene_pass(),
            recorder.hud_pass(),
        ) {
            Ok(target) => target,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used it.
                unsafe { recorder.destroy(&gpu.device, &mut gpu.allocator) };
                return Err(e);
            }
        };

        let sync = Self::allocate_command_buffer(&gpu).and_then(|command_buffer| {
            let fence_info = vk::FenceCreateInfo::default();
            // SAFETY: Creating a fence.
            let fence = unsafe {
                gpu.device
                    .create_fence(&fence_info, None)
                    .map_err(RenderError::Vulkan)?
            };
            Ok((command_buffer, fence))
        });
        let (command_buffer, fence) = match sync {
            Ok(sync) => sync,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them. A command
                // buffer is freed with the pool.
                unsafe {
                    target.destroy(&gpu.device, &mut gpu.allocator);
                    recorder.destroy(&gpu.device, &mut gpu.allocator);
                }
                return Err(e);
            }
        };

        Ok(Self {
            recorder: ManuallyDrop::new(recorder),
            target: ManuallyDrop::new(target),
            command_buffer,
            fence,
            gpu,
        })
    }

    /// `RayTraced` when the renderer ray-traces its floor reflections, else
    /// `Rasterized`: they are mirrored across the floor.
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
    /// mirrored (`Rasterized`), from the next `render` on. `RayTraced`
    /// without ray tracing gives `Rasterized`; check
    /// [`render_mode`](Self::render_mode) for the result.
    pub fn set_render_mode(&mut self, mode: RenderMode) {
        self.recorder.set_mode(mode);
    }

    /// Draw the edge of `green` as one ring on the range floor, faded like
    /// the renderer's grid; `None` draws nothing. Replaces any green set
    /// before.
    pub fn set_green(&mut self, green: Option<&GreenConfig>) -> Result<(), RenderError> {
        self.recorder.set_green(&mut self.gpu, green)
    }

    /// Set or clear the chase camera for the right 20% split viewport.
    ///
    /// Pass `Some(camera)` while a ball is being followed, `None` otherwise.
    pub fn set_chase_camera(&mut self, camera: Option<Camera>) {
        self.recorder.set_chase_camera(camera);
    }

    /// Update dynamic flight geometry for the next `render`. Reflections
    /// show the newest flight's ball and its tracer, or the ball on the tee
    /// when there is no flight.
    pub fn update_flight(&mut self, flights: &[FlightRenderData]) {
        self.recorder.update_flight(flights);
    }

    /// Update HUD overlay geometry (pixel coordinates) for the next `render`.
    pub fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.recorder.update_hud(lines, fills);
    }

    /// Render a frame seen from `camera` and read it back.
    ///
    /// Traces (`RayTraced` mode) or mirrors (`Rasterized`) the floor
    /// reflections and composites them onto the raster scene. The HUD is drawn last, over the scene and its
    /// reflections.
    pub fn render(&mut self, camera: &Camera) -> Result<Frame<'_>, RenderError> {
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

        // The staging buffer holds the frame until the next render.
        let pixel_count = (width * height * 4) as usize;
        let pixels = target.staging.bytes().get(..pixel_count).ok_or_else(|| {
            RenderError::Allocator("staging buffer smaller than the frame".into())
        })?;

        Ok(Frame {
            pixels,
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
