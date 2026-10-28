use ash::vk;

use cf_scene::camera::Camera;

use crate::recorder::ChaseViewport;
use crate::rt_pipeline::RtPushConstants;

use super::Renderer;

impl Renderer {
    /// Build RT push constants for a given camera and viewport offset.
    fn rt_push_constants(
        &self,
        camera: &Camera,
        aspect: f32,
        vp_offset_x: f32,
        vp_offset_y: f32,
    ) -> RtPushConstants {
        RtPushConstants::new(
            camera,
            aspect,
            [vp_offset_x, vp_offset_y],
            self.rt_ball_center,
            self.rt_trail_fade_dist,
            self.rt_trail_tail_t,
        )
    }

    /// Record ray tracing commands: trace reflection rays into the storage image,
    /// then transition it to `SHADER_READ_ONLY_OPTIMAL` for the composite pass.
    /// If a chase camera is set, traces reflections for both viewports.
    ///
    /// # Safety
    /// The command buffer must be in the recording state.
    pub(super) unsafe fn record_rt_reflections(&self, cb: vk::CommandBuffer, camera: &Camera) {
        let device = &self.gpu.device;
        let extent = self.swapchain.extent;
        let rt = self.rt_pipeline.as_ref().expect("RT pipeline present");

        // Main camera push constants (full screen, offset 0,0)
        let main_aspect = extent.width as f32 / extent.height as f32;
        let main_pc = self.rt_push_constants(camera, main_aspect, 0.0, 0.0);

        // SAFETY: Recording RT trace commands into a valid command buffer.
        unsafe {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            // Transition storage image: UNDEFINED → GENERAL
            device.cmd_pipeline_barrier(
                cb,
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

            // Trace main camera reflections (full screen)
            rt.record_trace(device, cb, &main_pc, extent.width, extent.height);

            // Trace chase camera reflections (right viewport)
            if let Some(chase) = &self.chase_camera {
                // Barrier: wait for main trace to finish writing before chase trace overwrites
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)],
                    &[],
                    &[],
                );

                let chase_vp = ChaseViewport::new(extent.width);
                let chase_pc = self.rt_push_constants(
                    chase,
                    chase_vp.aspect(extent.height),
                    chase_vp.x as f32,
                    0.0,
                );
                rt.record_trace(device, cb, &chase_pc, chase_vp.width, extent.height);
            }

            // Transition storage image: GENERAL → SHADER_READ_ONLY_OPTIMAL (for composite sampling)
            device.cmd_pipeline_barrier(
                cb,
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
        }
    }

    /// Composite RT reflections onto the rasterized scene with additive
    /// blending, in the overlay framebuffer of `image_index`. The image stays
    /// in `COLOR_ATTACHMENT_OPTIMAL` for the HUD pass.
    ///
    /// # Safety
    /// The command buffer must be in the recording state. The swapchain image
    /// must be in `COLOR_ATTACHMENT_OPTIMAL`. The RT storage image must be in
    /// `SHADER_READ_ONLY_OPTIMAL`.
    pub(super) unsafe fn record_composite_reflections(
        &self,
        cb: vk::CommandBuffer,
        image_index: u32,
    ) {
        if let Some(composite) = &self.composite {
            // SAFETY: The caller guarantees the command buffer and image states.
            unsafe {
                composite.record(
                    &self.gpu.device,
                    cb,
                    self.overlay_framebuffers[image_index as usize],
                    self.swapchain.extent,
                );
            }
        }
    }
}
