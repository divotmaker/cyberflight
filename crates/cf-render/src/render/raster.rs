use ash::vk;

use cf_scene::camera::Camera;
use cf_scene::color;

use crate::pipeline::GridPushConstants;

use super::{NO_CLIP, Renderer};

/// Fraction of screen width used by the chase camera viewport (right side).
const CHASE_VIEWPORT_FRAC: f32 = 0.20;

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
                        float32: self.config.clear_color,
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
                .render_pass(self.pipeline.render_pass)
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
            let view_proj = Self::compute_view_proj(camera, aspect);
            self.draw_scene_geometry(cb, view_proj);

            // Chase camera: right 20% viewport (only when a ball is in flight)
            if let Some(chase) = &self.chase_camera {
                let chase_x = (extent.width as f32 * (1.0 - CHASE_VIEWPORT_FRAC)) as i32;
                let chase_w = (extent.width as f32 * CHASE_VIEWPORT_FRAC) as u32;

                // Clear color + depth in the chase viewport region
                let clear_attachments = [
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        color_attachment: 0,
                        clear_value: vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: self.config.clear_color,
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
                    rect: vk::Rect2D {
                        offset: vk::Offset2D { x: chase_x, y: 0 },
                        extent: vk::Extent2D {
                            width: chase_w,
                            height: extent.height,
                        },
                    },
                    base_array_layer: 0,
                    layer_count: 1,
                };
                device.cmd_clear_attachments(cb, &clear_attachments, &[clear_rect]);

                let chase_viewport = vk::Viewport {
                    x: chase_x as f32,
                    y: 0.0,
                    width: chase_w as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cb, 0, &[chase_viewport]);

                let chase_scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: chase_x, y: 0 },
                    extent: vk::Extent2D {
                        width: chase_w,
                        height: extent.height,
                    },
                };
                device.cmd_set_scissor(cb, 0, &[chase_scissor]);

                let chase_aspect = chase_w as f32 / extent.height as f32;
                let chase_view_proj = Self::compute_view_proj(chase, chase_aspect);
                self.draw_scene_geometry(cb, chase_view_proj);
            }

            // End scene render pass (final_layout = COLOR_ATTACHMENT_OPTIMAL)
            device.cmd_end_render_pass(cb);
        }
    }

    /// Compute the Vulkan view-projection matrix for a camera at a given aspect ratio.
    fn compute_view_proj(camera: &Camera, aspect: f32) -> [[f32; 4]; 4] {
        let view = camera.view_matrix();
        let mut proj = camera.projection_matrix(aspect);
        // Vulkan NDC has +Y down; glam assumes OpenGL (+Y up). Flip Y.
        proj.y_axis.y *= -1.0;
        (proj * view).to_cols_array_2d()
    }

    /// Draw all scene geometry with the given view-projection matrix.
    ///
    /// Assumes viewport and scissor are already set. Binds pipelines, sets
    /// push constants, and issues draw calls for grid, tee box, ball, and
    /// in-flight geometry.
    ///
    /// # Safety
    /// Must be called within an active render pass.
    unsafe fn draw_scene_geometry(&self, cb: vk::CommandBuffer, view_proj: [[f32; 4]; 4]) {
        let device = &self.gpu.device;

        // SAFETY: Drawing within an active render pass.
        unsafe {
            // Grid lines (LINE_LIST, cyan, 1px)
            device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            device.cmd_set_line_width(cb, 1.0);

            let pc_cyan = GridPushConstants {
                view_proj,
                color: color::CYAN.into(),
                clip_bounds: NO_CLIP,
            };
            let pc_cyan_bytes: &[u8] = std::slice::from_raw_parts(
                std::ptr::from_ref(&pc_cyan).cast::<u8>(),
                std::mem::size_of::<GridPushConstants>(),
            );
            device.cmd_push_constants(
                cb,
                self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                pc_cyan_bytes,
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer], &[0]);
            device.cmd_draw(cb, self.vertex_count, 1, 0, 0);

            // Tee box fill (black quad — masks grid lines under the tee box)
            let pc_black = GridPushConstants {
                view_proj,
                color: [0.0, 0.0, 0.0, 1.0],
                clip_bounds: NO_CLIP,
            };
            let pc_black_bytes: &[u8] = std::slice::from_raw_parts(
                std::ptr::from_ref(&pc_black).cast::<u8>(),
                std::mem::size_of::<GridPushConstants>(),
            );
            device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.fill_pipeline,
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer], &[0]);
            device.cmd_push_constants(
                cb,
                self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                pc_black_bytes,
            );
            device.cmd_draw(cb, self.tee_fill_count, 1, 0, 0);

            // Tee box border (cyan filled quads)
            device.cmd_push_constants(
                cb,
                self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                pc_cyan_bytes,
            );
            device.cmd_draw(cb, self.tee_border_count, 1, self.tee_fill_count, 0);

            // Magenta push constants (shared by ball + flight geometry)
            let pc_magenta = GridPushConstants {
                view_proj,
                color: color::MAGENTA.into(),
                clip_bounds: NO_CLIP,
            };
            let pc_magenta_bytes: &[u8] = std::slice::from_raw_parts(
                std::ptr::from_ref(&pc_magenta).cast::<u8>(),
                std::mem::size_of::<GridPushConstants>(),
            );

            // Static ball on tee box (hidden when a flight is active)
            if self.ball_on_tee_box {
                // Ball glow (additive blend, magenta — draw before ball so ball occludes center)
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.glow_pipeline,
                );
                device.cmd_bind_vertex_buffers(cb, 0, &[self.glow_buffer], &[0]);
                device.cmd_push_constants(
                    cb,
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_magenta_bytes,
                );
                device.cmd_draw(cb, self.glow_count, 1, 0, 0);

                // Ball (black sphere — occludes glow center, revealing outline)
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.fill_pipeline,
                );
                device.cmd_bind_vertex_buffers(cb, 0, &[self.fill_buffer], &[0]);
                device.cmd_push_constants(
                    cb,
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_black_bytes,
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
                device.cmd_set_line_width(cb, if self.gpu.wide_lines { 2.0 } else { 1.0 });
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline, // LINE_LIST grid pipeline
                );
                device.cmd_bind_vertex_buffers(cb, 0, &[self.flight_line_buffer], &[0]);
                device.cmd_push_constants(
                    cb,
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_magenta_bytes,
                );
                device.cmd_draw(cb, self.flight_line_count, 1, 0, 0);
            }

            // In-flight ball glow + trail glow (additive, magenta)
            if self.flight_glow_count > 0 {
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.glow_pipeline,
                );
                device.cmd_bind_vertex_buffers(cb, 0, &[self.flight_glow_buffer], &[0]);
                device.cmd_push_constants(
                    cb,
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_magenta_bytes,
                );
                device.cmd_draw(cb, self.flight_glow_count, 1, 0, 0);
            }

            // In-flight ball fill (black sphere — occludes glow center)
            if self.flight_fill_count > 0 {
                device.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.fill_pipeline,
                );
                device.cmd_bind_vertex_buffers(cb, 0, &[self.flight_fill_buffer], &[0]);
                device.cmd_push_constants(
                    cb,
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_black_bytes,
                );
                device.cmd_draw(cb, self.flight_fill_count, 1, 0, 0);
            }
        }
    }
}
