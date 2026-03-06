use ash::vk;

use crate::pipeline::GridPushConstants;

use super::{NO_CLIP, Renderer};

impl Renderer {
    /// Record HUD overlay commands into the given command buffer.
    ///
    /// Begins the HUD render pass (load_op=LOAD, preserves scene), draws HUD geometry,
    /// and ends the render pass (final_layout=PRESENT_SRC_KHR).
    ///
    /// # Safety
    /// The command buffer must be in the recording state. The swapchain image at
    /// `image_index` must be in COLOR_ATTACHMENT_OPTIMAL layout.
    pub(super) unsafe fn record_hud_commands(
        &self,
        cb: vk::CommandBuffer,
        image_index: u32,
        extent: vk::Extent2D,
    ) {
        let device = &self.gpu.device;

        // SAFETY: Recording HUD render pass commands into a valid command buffer.
        // Image is in COLOR_ATTACHMENT_OPTIMAL (guaranteed by scene pass or RT barrier).
        unsafe {
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.hud_render_pass)
                .framebuffer(self.hud_framebuffers[image_index as usize])
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

            if self.hud_fill_count > 0 || self.hud_line_count > 0 {
                let w = extent.width as f32;
                let h = extent.height as f32;
                let hud_proj = glam::Mat4::orthographic_rh(0.0, w, 0.0, h, -1.0, 1.0);

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
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.hud_fill_pipeline,
                    );
                    device.cmd_bind_vertex_buffers(cb, 0, &[self.hud_fill_buffer], &[0]);
                    device.cmd_push_constants(
                        cb,
                        self.pipeline.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        pc_hud_bg_bytes,
                    );
                    device.cmd_draw(cb, self.hud_fill_count, 1, 0, 0);
                }

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
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.hud_line_pipeline,
                    );
                    device.cmd_set_line_width(cb, 1.0);
                    device.cmd_bind_vertex_buffers(cb, 0, &[self.hud_line_buffer], &[0]);
                    device.cmd_push_constants(
                        cb,
                        self.pipeline.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        pc_hud_text_bytes,
                    );
                    device.cmd_draw(cb, self.hud_line_count, 1, 0, 0);
                }
            }

            device.cmd_end_render_pass(cb);
        }
    }
}
