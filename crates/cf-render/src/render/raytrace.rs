use ash::vk;

use cf_scene::camera::Camera;

use crate::rt_pipeline::RtPushConstants;

use super::{Renderer, CHASE_VIEWPORT_FRAC};

impl Renderer {
    /// Build RT push constants for a given camera and viewport offset.
    fn rt_push_constants(&self, camera: &Camera, aspect: f32, vp_offset_x: f32, vp_offset_y: f32) -> RtPushConstants {
        let view = camera.view_matrix();
        let mut proj = camera.projection_matrix(aspect);
        proj.y_axis.y *= -1.0; // Vulkan Y-flip
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        RtPushConstants {
            camera_pos: [
                camera.position.x,
                camera.position.y,
                camera.position.z,
                vp_offset_x,
            ],
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            grid_params: [
                self.grid_spacing_m,
                0.15, // line half-width in meters
                self.grid_max_fade_dist,
                vp_offset_y,
            ],
            ball_pos: [
                self.rt_ball_center.x,
                self.rt_ball_center.y,
                self.rt_ball_center.z,
                self.rt_trail_fade_dist,
            ],
        }
    }

    /// Record ray tracing commands: trace reflection rays into the storage image,
    /// then transition it to `SHADER_READ_ONLY_OPTIMAL` for the composite pass.
    /// If a chase camera is set, traces reflections for both viewports.
    ///
    /// # Safety
    /// The command buffer must be in the recording state.
    pub(super) unsafe fn record_rt_reflections(
        &self,
        cb: vk::CommandBuffer,
        camera: &Camera,
    ) {
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
                    .image(rt.storage_image)
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

                let chase_x = (extent.width as f32 * (1.0 - CHASE_VIEWPORT_FRAC)) as u32;
                let chase_w = (extent.width as f32 * CHASE_VIEWPORT_FRAC) as u32;
                let chase_aspect = chase_w as f32 / extent.height as f32;
                let chase_pc = self.rt_push_constants(chase, chase_aspect, chase_x as f32, 0.0);
                rt.record_trace(device, cb, &chase_pc, chase_w, extent.height);
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
                    .image(rt.storage_image)
                    .subresource_range(subresource_range)],
            );
        }
    }

    /// Composite RT reflections onto the rasterized scene with additive blending.
    ///
    /// Draws a fullscreen triangle sampling the RT storage image.
    /// The render pass preserves scene content (load_op = LOAD) and leaves
    /// the image at `COLOR_ATTACHMENT_OPTIMAL` for the HUD pass.
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
        let device = &self.gpu.device;
        let composite = self.composite.as_ref().expect("composite pass present");
        let extent = self.swapchain.extent;

        // SAFETY: Recording composite render pass commands.
        unsafe {
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(composite.render_pass)
                .framebuffer(composite.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                });

            device.cmd_begin_render_pass(cb, &render_pass_info, vk::SubpassContents::INLINE);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cb, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            device.cmd_set_scissor(cb, 0, &[scissor]);

            device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                composite.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                composite.pipeline_layout,
                0,
                &[composite.descriptor_set],
                &[],
            );

            // Fullscreen triangle: 3 vertices, no vertex buffer
            device.cmd_draw(cb, 3, 1, 0, 0);

            device.cmd_end_render_pass(cb);
        }
    }
}
