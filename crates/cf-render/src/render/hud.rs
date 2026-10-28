use ash::vk;

use super::Renderer;

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
                .render_pass(self.pipeline.hud_pass)
                .framebuffer(self.overlay_framebuffers[image_index as usize])
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

            self.frame.draw_hud(
                device,
                cb,
                self.pipeline.hud_fill,
                self.pipeline.hud_line,
                self.pipeline.layout,
                extent,
            );

            device.cmd_end_render_pass(cb);
        }
    }
}
