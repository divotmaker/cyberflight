use ash::vk;

use cf_scene::camera::Camera;
use cf_scene::color;

use crate::gpu::push_constants;
use crate::pipeline::GridPushConstants;
use crate::recorder::{ChaseViewport, view_proj};

use super::Renderer;

impl Renderer {
    /// Record raster scene commands into the command buffer.
    ///
    /// Begins the scene render pass, draws all geometry (grid, tee box, ball,
    /// glow, in-flight objects), and ends the render pass. If a chase camera
    /// is set, the right 20% of the screen is cleared and redrawn from the
    /// chase camera's perspective.
    ///
    /// Final layout is `COLOR_ATTACHMENT_OPTIMAL` — HUD pass handles the transition
    /// to `PRESENT_SRC_KHR`.
    ///
    /// # Safety
    /// The command buffer must be in the recording state.
    pub(super) unsafe fn record_raster_scene(
        &self,
        cb: vk::CommandBuffer,
        camera: &Camera,
        image_index: u32,
    ) {
        let device = &self.gpu.device;
        let extent = self.swapchain.extent;

        // SAFETY: Recording scene render pass commands into a valid command buffer.
        unsafe {
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: color::BACKGROUND.to_array(),
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
                .render_pass(self.pipeline.scene_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(cb, &render_pass_info, vk::SubpassContents::INLINE);

            // Main scene: full viewport
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

            let aspect = extent.width as f32 / extent.height as f32;
            self.draw_scene_geometry(cb, view_proj(camera, aspect));

            // Chase camera: right 20% viewport (only when a ball is in flight)
            if let Some(chase) = &self.chase_camera {
                let chase_vp = ChaseViewport::new(extent.width);

                let clear_attachments = [
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        color_attachment: 0,
                        clear_value: vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: color::BACKGROUND.to_array(),
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
                    rect: chase_vp.rect(extent.height),
                    base_array_layer: 0,
                    layer_count: 1,
                };
                device.cmd_clear_attachments(cb, &clear_attachments, &[clear_rect]);

                device.cmd_set_viewport(cb, 0, &[chase_vp.viewport(extent.height)]);
                device.cmd_set_scissor(cb, 0, &[chase_vp.rect(extent.height)]);

                let chase_view_proj = view_proj(chase, chase_vp.aspect(extent.height));
                self.draw_scene_geometry(cb, chase_view_proj);
            }

            // End scene render pass (final_layout = COLOR_ATTACHMENT_OPTIMAL)
            device.cmd_end_render_pass(cb);
        }
    }

    /// Draw all scene geometry with the given view-projection matrix.
    ///
    /// Assumes viewport and scissor are already set. Binds pipelines, sets
    /// push constants, and issues draw calls for grid, green edge, tee
    /// box, ball, and in-flight geometry.
    ///
    /// # Safety
    /// Must be called within an active render pass.
    unsafe fn draw_scene_geometry(&self, cb: vk::CommandBuffer, view_proj: [[f32; 4]; 4]) {
        let device = &self.gpu.device;

        // SAFETY: Drawing within an active render pass.
        unsafe {
            // Grid lines (LINE_LIST, cyan, 1px)
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.line);
            device.cmd_set_line_width(cb, 1.0);

            let pc_cyan = GridPushConstants {
                view_proj,
                color: color::CYAN.into(),
            };
            push_constants(
                device,
                cb,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc_cyan,
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer.handle()], &[0]);
            device.cmd_draw(cb, self.vertex_count, 1, 0, 0);

            // Green edge (LINE_LIST, neon green, 1px)
            if let Some(edge) = &self.green_edge {
                edge.record(device, cb, &self.pipeline, view_proj);
            }

            // Tee box fill (black quad — masks grid lines under the tee box)
            let pc_black = GridPushConstants {
                view_proj,
                color: [0.0, 0.0, 0.0, 1.0],
            };
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.fill);
            device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer.handle()], &[0]);
            push_constants(
                device,
                cb,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc_black,
            );
            device.cmd_draw(cb, self.tee_fill_count, 1, 0, 0);

            // Tee box border (cyan filled quads)
            push_constants(
                device,
                cb,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &pc_cyan,
            );
            device.cmd_draw(cb, self.tee_border_count, 1, self.tee_fill_count, 0);

            // Magenta push constants (shared by ball + flight geometry)
            let pc_magenta = GridPushConstants {
                view_proj,
                color: color::MAGENTA.into(),
            };

            // Static ball on tee box (hidden when a flight is active)
            if self.ball_on_tee_box {
                // Ball glow (additive blend, magenta — draw before ball so ball occludes center)
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.glow);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.glow_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc_magenta,
                );
                device.cmd_draw(cb, self.glow_count, 1, 0, 0);

                // Ball (black sphere — occludes glow center, revealing outline)
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.fill);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc_black,
                );
                device.cmd_draw(
                    cb,
                    self.ball_count,
                    1,
                    self.tee_fill_count + self.tee_border_count,
                    0,
                );
            }

            // In-flight trail wireframe core (LINE_LIST, magenta — 2px when supported)
            if self.flight_line_count > 0 {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.line);
                device.cmd_set_line_width(cb, if self.gpu.wide_lines { 2.0 } else { 1.0 });
                device.cmd_bind_vertex_buffers(cb, 0, &[self.flight_line_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc_magenta,
                );
                device.cmd_draw(cb, self.flight_line_count, 1, 0, 0);
            }

            // In-flight ball glow + trail glow (additive, magenta)
            if self.flight_glow_count > 0 {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.glow);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.flight_glow_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc_magenta,
                );
                device.cmd_draw(cb, self.flight_glow_count, 1, 0, 0);
            }

            // In-flight ball fill (black sphere — occludes glow center)
            if self.flight_fill_count > 0 {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline.fill);
                device.cmd_bind_vertex_buffers(cb, 0, &[self.flight_fill_buffer.handle()], &[0]);
                push_constants(
                    device,
                    cb,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc_black,
                );
                device.cmd_draw(cb, self.flight_fill_count, 1, 0, 0);
            }
        }
    }
}
