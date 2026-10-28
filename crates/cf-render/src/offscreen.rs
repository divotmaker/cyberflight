use std::mem::ManuallyDrop;

use ash::vk;

use cf_scene::camera::Camera;
use cf_scene::color;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex};
use cf_scene::tee::TeeBox;
use cf_scene::trail::TrailPoint;

use crate::context::{GpuConfig, GpuContext};
use crate::error::RenderError;
use crate::flight::flight_vertices;
use crate::frame::{FrameContent, FrameResources};
use crate::framebuffer::OffscreenTarget;
use crate::pipeline::ScenePipelines;
use crate::readback::FrameBuffers;
use crate::recorder::{ChaseViewport, view_proj};
use crate::reflections::{Reflections, RtSceneState, TraceView};
use crate::render::FlightRenderData;
use crate::rt_offscreen::RtTrail;
use crate::scene::{SceneGeometry, Snapshot};

/// Offscreen renderer for headless PNG generation.
pub struct OffscreenRenderer {
    gpu: GpuContext,
    pipeline: ManuallyDrop<ScenePipelines>,
    target: ManuallyDrop<OffscreenTarget>,
    /// Grid, green edge, tee box and the ball on the tee (or a snapshot's
    /// ball in flight).
    geometry: ManuallyDrop<SceneGeometry>,
    /// Next frame's flights and HUD, uploaded into `frame` by `render`.
    content: FrameContent,
    /// Per-frame flight and HUD buffers.
    frame: ManuallyDrop<FrameResources>,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    /// Ray-traced floor reflections, when the device supports them.
    reflections: Option<Reflections>,
    /// Optional chase camera for split viewport (right 20%).
    chase_camera: Option<Camera>,
}

/// Offscreen image format (RGBA8 sRGB for correct gamma in PNG output).
const OFFSCREEN_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

impl OffscreenRenderer {
    /// Create a headless renderer for the given grid configuration.
    /// Enables raytracing if hardware supports it.
    pub fn new(width: u32, height: u32, grid_config: &GridConfig) -> Result<Self, RenderError> {
        let config = GpuConfig {
            validation: cfg!(debug_assertions),
            enable_raytracing: true,
        };

        let mut gpu = GpuContext::new_headless(config)?;
        let rt_available = gpu.rt_supported;

        let pipeline = ScenePipelines::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            pipeline.scene_pass,
            pipeline.hud_pass,
        )?;

        let tee = TeeBox::default();
        let geometry = SceneGeometry::new(
            &gpu.device,
            &mut gpu.allocator,
            grid_config,
            &tee,
            None,
            gpu.wide_lines,
        )?;

        let frame = FrameResources::new(&gpu.device, &mut gpu.allocator)?;

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Build RT reflections if hardware supports it.
        let reflections = if rt_available {
            let scene = RtSceneState::new(
                glam::Vec3::new(0.0, tee.ball_radius, 0.0),
                RtTrail::default(),
            );
            Some(Reflections::new(
                &mut gpu,
                OFFSCREEN_FORMAT,
                vk::Extent2D { width, height },
                grid_config,
                &tee,
                scene,
            )?)
        } else {
            None
        };

        Ok(Self {
            gpu,
            pipeline: ManuallyDrop::new(pipeline),
            target: ManuallyDrop::new(target),
            geometry: ManuallyDrop::new(geometry),
            content: FrameContent::default(),
            frame: ManuallyDrop::new(frame),
            command_buffer,
            fence,

            reflections,
            chase_camera: None,
        })
    }

    /// Create a headless renderer with a ball in flight and a tracer trail.
    ///
    /// `ball_pos` is the current ball center (bottom of sphere touches this Y - radius).
    /// `trail_points` go from oldest (index 0) to newest (last = near ball_pos).
    /// `camera` is used to orient the trail glow ribbon toward the viewer.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flight(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        ball_pos: glam::Vec3,
        trail_points: &[TrailPoint],
        current_time: f64,
        max_lifetime: f64,
        camera: &Camera,
    ) -> Result<Self, RenderError> {
        let config = GpuConfig {
            validation: cfg!(debug_assertions),
            enable_raytracing: false,
        };

        let mut gpu = GpuContext::new_headless(config)?;

        let pipeline = ScenePipelines::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            pipeline.scene_pass,
            pipeline.hud_pass,
        )?;

        // Tee box + ball at flight position, trail glow and core.
        let tee = TeeBox::default();
        let snapshot = Snapshot {
            ball_pos,
            trail_points,
            current_time,
            max_lifetime,
            camera_pos: camera.position,
        };
        let geometry = SceneGeometry::new(
            &gpu.device,
            &mut gpu.allocator,
            grid_config,
            &tee,
            Some(&snapshot),
            gpu.wide_lines,
        )?;

        let frame = FrameResources::new(&gpu.device, &mut gpu.allocator)?;

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        Ok(Self {
            gpu,
            pipeline: ManuallyDrop::new(pipeline),
            target: ManuallyDrop::new(target),
            geometry: ManuallyDrop::new(geometry),
            content: FrameContent::default(),
            frame: ManuallyDrop::new(frame),
            command_buffer,
            fence,

            reflections: None,
            chase_camera: None,
        })
    }

    /// Create a headless renderer with RT reflection compositing for the static range.
    ///
    /// Falls back to raster-only if RT hardware is not available.
    pub fn new_rt(width: u32, height: u32, grid_config: &GridConfig) -> Result<Self, RenderError> {
        let tee = TeeBox::default();
        let scene = RtSceneState::new(
            glam::Vec3::new(0.0, cf_scene::tee::TEE_ELEVATION + tee.ball_radius, 0.0),
            RtTrail::default(),
        );
        Self::create_rt(width, height, grid_config, &tee, scene, None)
    }

    /// Create a headless renderer with RT reflection compositing and a ball in flight.
    ///
    /// Falls back to raster-only if RT hardware is not available.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flight_rt(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        ball_pos: glam::Vec3,
        trail_points: &[TrailPoint],
        current_time: f64,
        max_lifetime: f64,
        camera: &Camera,
    ) -> Result<Self, RenderError> {
        let tee = TeeBox::default();
        let ball_center = ball_pos + glam::Vec3::new(0.0, tee.ball_radius, 0.0);

        // Reflect the part of the trail the raster tracer draws.
        let scene = RtSceneState::new(
            ball_center,
            RtTrail::new(trail_points, current_time, max_lifetime),
        );
        let snapshot = Snapshot {
            ball_pos,
            trail_points,
            current_time,
            max_lifetime,
            camera_pos: camera.position,
        };

        Self::create_rt(width, height, grid_config, &tee, scene, Some(&snapshot))
    }

    /// Internal RT constructor shared by `new_rt` and `new_with_flight_rt`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn create_rt(
        width: u32,
        height: u32,
        grid_config: &GridConfig,
        tee: &TeeBox,
        scene: RtSceneState,
        snapshot: Option<&Snapshot<'_>>,
    ) -> Result<Self, RenderError> {
        let config = GpuConfig {
            validation: cfg!(debug_assertions),
            enable_raytracing: true,
        };

        let mut gpu = GpuContext::new_headless(config)?;

        // Determine if RT is actually available.
        let rt_available = gpu.rt_supported;

        let pipeline = ScenePipelines::new(
            &gpu.device,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;

        let target = OffscreenTarget::new(
            &gpu.device,
            &mut gpu.allocator,
            width,
            height,
            OFFSCREEN_FORMAT,
            gpu.depth_format,
            pipeline.scene_pass,
            pipeline.hud_pass,
        )?;

        // Tee box + ball, on the tee or at the snapshot's flight position.
        let geometry = SceneGeometry::new(
            &gpu.device,
            &mut gpu.allocator,
            grid_config,
            tee,
            snapshot,
            gpu.wide_lines,
        )?;

        let frame = FrameResources::new(&gpu.device, &mut gpu.allocator)?;

        let command_buffer = Self::allocate_command_buffer(&gpu)?;

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating a fence.
        let fence = unsafe {
            gpu.device
                .create_fence(&fence_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Build RT reflections if hardware supports it.
        let reflections = if rt_available {
            Some(Reflections::new(
                &mut gpu,
                OFFSCREEN_FORMAT,
                vk::Extent2D { width, height },
                grid_config,
                tee,
                scene,
            )?)
        } else {
            None
        };

        Ok(Self {
            gpu,
            pipeline: ManuallyDrop::new(pipeline),
            target: ManuallyDrop::new(target),
            geometry: ManuallyDrop::new(geometry),
            content: FrameContent::default(),
            frame: ManuallyDrop::new(frame),
            command_buffer,
            fence,

            reflections,
            chase_camera: None,
        })
    }

    /// Draw all scene geometry (grid, green edge, tee box, ball, glow) with
    /// the given view-projection.
    ///
    /// Assumes viewport and scissor are already set, and we're inside a render pass.
    ///
    /// # Safety
    /// Must be called within an active render pass.
    unsafe fn draw_scene_geometry(&self, view_proj: [[f32; 4]; 4]) {
        // SAFETY: The caller guarantees an active scene pass.
        unsafe {
            self.geometry.draw(
                &self.gpu.device,
                self.command_buffer,
                &self.pipeline,
                &self.frame,
                view_proj,
            );
        }
    }

    /// Whether this renderer has RT reflection compositing enabled.
    #[must_use]
    pub fn has_rt(&self) -> bool {
        self.reflections.is_some()
    }

    /// Draw the edge of `green` as one ring on the range floor, faded like
    /// `grid`; `None` draws nothing. Replaces any green set before.
    pub fn set_green(
        &mut self,
        green: Option<&GreenConfig>,
        grid: &GridConfig,
    ) -> Result<(), RenderError> {
        self.geometry
            .set_green(&self.gpu.device, &mut self.gpu.allocator, green, grid)
    }

    /// Set or clear the chase camera for the right 20% split viewport.
    ///
    /// Pass `Some(camera)` while a ball is being followed, `None` otherwise.
    pub fn set_chase_camera(&mut self, camera: Option<Camera>) {
        self.chase_camera = camera;
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

    /// Update dynamic flight geometry for the next `render`, and rebuild the
    /// RT scene from the newest flight.
    pub fn update_flight(&mut self, flights: &[FlightRenderData], camera: &Camera) {
        use cf_scene::trail::DEFAULT_TRAIL_LIFETIME;

        let tee = TeeBox::default();
        flight_vertices(flights, camera.position, &tee, &mut self.content);

        // Reflections follow the newest flight, the one the HUD and chase
        // camera show, and reflect the part of its trail the raster tracer
        // still draws. With no flight the reflected ball stays where the
        // last flight left it. The acceleration structures are rebuilt only
        // when the reflected ball or trail moves.
        if let Some(reflections) = &mut self.reflections {
            let newest = flights.last();
            let trail = newest.map_or_else(RtTrail::default, |f| {
                RtTrail::new(&f.trail_points, f.current_time, DEFAULT_TRAIL_LIFETIME)
            });
            let ball_centre = newest.map_or(reflections.scene().ball_centre(), |f| f.ball_pos);
            reflections.set_scene(
                &mut self.gpu,
                RtSceneState::new(ball_centre, trail),
                &GridConfig::default(),
                &tee,
            );
        }
    }

    /// Update HUD overlay geometry (pixel coordinates) for the next `render`.
    pub fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.content.hud_lines.clear();
        self.content.hud_lines.extend_from_slice(lines);
        self.content.hud_fills.clear();
        self.content.hud_fills.extend_from_slice(fills);
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
        self.frame.upload(&self.content);
        let target = &self.target;
        let device = &self.gpu.device;
        let width = target.width;
        let height = target.height;
        let extent = vk::Extent2D { width, height };

        // Record command buffer
        // SAFETY: Recording commands into a valid command buffer.
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(RenderError::Vulkan)?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(RenderError::Vulkan)?;

            // Begin render pass
            let clear_color = [
                color::BACKGROUND.x,
                color::BACKGROUND.y,
                color::BACKGROUND.z,
                color::BACKGROUND.w,
            ];
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: clear_color,
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
                .framebuffer(target.scene)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .clear_values(&clear_values);

            device.cmd_begin_render_pass(
                self.command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.line,
            );
            device.cmd_set_line_width(self.command_buffer, 1.0);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);

            // Main scene
            let aspect = width as f32 / height as f32;
            self.draw_scene_geometry(view_proj(camera, aspect));

            // Chase camera: right 20% viewport
            if let Some(chase) = &self.chase_camera.clone() {
                let chase_vp = ChaseViewport::new(width);

                let clear_attachments = [
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        color_attachment: 0,
                        clear_value: vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [
                                    color::BACKGROUND.x,
                                    color::BACKGROUND.y,
                                    color::BACKGROUND.z,
                                    color::BACKGROUND.w,
                                ],
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
                    rect: chase_vp.rect(height),
                    base_array_layer: 0,
                    layer_count: 1,
                };
                device.cmd_clear_attachments(
                    self.command_buffer,
                    &clear_attachments,
                    &[clear_rect],
                );

                device.cmd_set_viewport(self.command_buffer, 0, &[chase_vp.viewport(height)]);
                device.cmd_set_scissor(self.command_buffer, 0, &[chase_vp.rect(height)]);

                self.draw_scene_geometry(view_proj(chase, chase_vp.aspect(height)));
            }

            device.cmd_end_render_pass(self.command_buffer);

            // Ray-traced reflections: main view, plus the chase inset when set.
            if let Some(reflections) = &self.reflections {
                let mut views = vec![TraceView {
                    camera: *camera,
                    offset: [0, 0],
                    extent,
                }];
                if let Some(chase) = self.chase_camera {
                    let chase_vp = ChaseViewport::new(width);
                    views.push(TraceView {
                        camera: chase,
                        offset: [chase_vp.x, 0],
                        extent: vk::Extent2D {
                            width: chase_vp.width,
                            height,
                        },
                    });
                }
                reflections.record(device, self.command_buffer, &views, target.overlay, extent);
            }

            // HUD overlay (2D, pixel coordinates) over the scene and its
            // reflections. The HUD pass leaves the image in TRANSFER_SRC_OPTIMAL.
            let hud_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.pipeline.hud_pass)
                .framebuffer(target.overlay)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                });
            device.cmd_begin_render_pass(
                self.command_buffer,
                &hud_pass_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
            self.frame.draw_hud(
                device,
                self.command_buffer,
                self.pipeline.hud_fill,
                self.pipeline.hud_line,
                self.pipeline.layout,
                extent,
            );
            device.cmd_end_render_pass(self.command_buffer);

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
            // SAFETY: Each `ManuallyDrop` buffer is taken once, here in `Drop`,
            // and never used after.
            ManuallyDrop::take(&mut self.geometry).destroy(device, allocator);
            ManuallyDrop::take(&mut self.frame).destroy(device, allocator);

            if let Some(reflections) = self.reflections.take() {
                reflections.destroy(&self.gpu.device, &mut self.gpu.allocator);
            }

            // SAFETY: Taken once, here in `Drop`, and never used after.
            ManuallyDrop::take(&mut self.target).destroy(&self.gpu.device, &mut self.gpu.allocator);

            // SAFETY: Taken once, here in `Drop`, and never used after.
            ManuallyDrop::take(&mut self.pipeline).destroy(&self.gpu.device);
        }
    }
}
