//! Frame recording shared by the windowed and the offscreen renderer.

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use cf_scene::camera::Camera;
use cf_scene::color;
use cf_scene::green::GreenConfig;
use cf_scene::grid::{GridConfig, GridVertex};
use cf_scene::tee::TeeBox;

use crate::context::GpuContext;
use crate::error::RenderError;
use crate::flight::{FlightRenderData, flight_vertices};
use crate::frame::{FrameContent, FrameResources, TracerBuffers};
use crate::mode::{RenderMode, RtMode};
use crate::pipeline::ScenePipelines;
use crate::reflections::{ReflectionFrame, Reflections, TraceView};
use crate::scene::SceneGeometry;

/// Fraction of the frame width the chase-camera inset takes (right side).
const CHASE_VIEWPORT_FRAC: f32 = 0.20;

/// Vulkan view-projection matrix for `camera` at `aspect` (Y flipped for
/// Vulkan NDC), as column arrays for push constants.
pub(crate) fn view_proj(camera: &Camera, aspect: f32) -> [[f32; 4]; 4] {
    let view = camera.view_matrix();
    let mut proj = camera.projection_matrix(aspect);
    // Vulkan NDC has +Y down; glam assumes OpenGL (+Y up). Flip Y.
    proj.y_axis.y *= -1.0;
    (proj * view).to_cols_array_2d()
}

/// The chase-camera inset: the right 20 % of the frame, full height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChaseViewport {
    /// Left edge in pixels.
    pub(crate) x: u32,
    /// Width in pixels.
    pub(crate) width: u32,
}

impl ChaseViewport {
    /// The inset of a frame `frame_width` pixels wide.
    pub(crate) fn new(frame_width: u32) -> Self {
        Self {
            x: (frame_width as f32 * (1.0 - CHASE_VIEWPORT_FRAC)) as u32,
            width: (frame_width as f32 * CHASE_VIEWPORT_FRAC) as u32,
        }
    }

    /// The inset of a frame `height` pixels high, seen from `camera`.
    pub(crate) fn view(self, camera: Camera, height: u32) -> TraceView {
        TraceView {
            camera,
            offset: [self.x, 0],
            extent: vk::Extent2D {
                width: self.width,
                height,
            },
        }
    }
}

/// The framebuffers one frame is recorded into, both over the same colour
/// image of `extent`.
pub(crate) struct FrameTarget {
    /// Colour + depth, created with [`SceneRecorder::scene_pass`].
    pub(crate) scene: vk::Framebuffer,
    /// Colour only, created with [`SceneRecorder::hud_pass`].
    pub(crate) overlay: vk::Framebuffer,
    pub(crate) extent: vk::Extent2D,
}

/// Everything both renderers draw a frame with, apart from the output image
/// and its synchronisation: the passes and pipelines, the static scene, the
/// next frame's flights and HUD, the floor reflections and the chase camera.
pub(crate) struct SceneRecorder {
    pipelines: ScenePipelines,
    geometry: SceneGeometry,
    /// Next frame's flights and HUD, uploaded into the buffers of the frame
    /// in flight by `record_frame`.
    content: FrameContent,
    /// The GPU buffers, one set per frame in flight.
    frames: Vec<FrameResources>,
    reflections: Option<Reflections>,
    mode: RenderMode,
    /// Chase camera for the right 20 % inset.
    chase: Option<Camera>,
    grid: GridConfig,
    tee: TeeBox,
}

impl SceneRecorder {
    /// Create the passes and pipelines for an `extent` image of
    /// `color_format` that the HUD pass leaves in `output_layout`, and the
    /// scene of `grid` with the default tee box and the ball on the tee.
    /// Every resource the CPU rewrites per frame (the vertex buffers and the
    /// ray tracer's structures built from them) exists `frames` times (at
    /// least once), one set for each frame in flight.
    ///
    /// It adds reflections of the ball on the tee, mirrored across the floor
    /// ([`RenderMode::Rasterized`]); with `RtMode::Auto` and a device that
    /// supports ray tracing, it also adds ray-traced ones and starts in
    /// `RayTraced` mode. If the ray tracer fails to set up, it says so on
    /// stderr and keeps the mirror. If the reflections fail to set up, it
    /// says so on stderr and renders without them.
    ///
    /// # Errors
    /// Any other part fails to create; what was created is destroyed first.
    pub(crate) fn new(
        gpu: &mut GpuContext,
        color_format: vk::Format,
        output_layout: vk::ImageLayout,
        extent: vk::Extent2D,
        grid: &GridConfig,
        rt: RtMode,
        frames: usize,
    ) -> Result<Self, RenderError> {
        let tee = TeeBox::default();
        let pipelines =
            ScenePipelines::new(&gpu.device, color_format, gpu.depth_format, output_layout)?;
        let geometry =
            match SceneGeometry::new(&gpu.device, &mut gpu.allocator, grid, &tee, gpu.wide_lines) {
                Ok(geometry) => geometry,
                Err(e) => {
                    // SAFETY: Just created; no GPU work has used it.
                    unsafe { pipelines.destroy(&gpu.device) };
                    return Err(e);
                }
            };
        let mut made: Vec<FrameResources> = Vec::with_capacity(frames.max(1));
        for _ in 0..frames.max(1) {
            match FrameResources::new(&gpu.device, &mut gpu.allocator, gpu.rt_supported) {
                Ok(frame) => made.push(frame),
                Err(e) => {
                    // SAFETY: Just created; no GPU work has used them.
                    unsafe {
                        for frame in made {
                            frame.destroy(&gpu.device, &mut gpu.allocator);
                        }
                        geometry.destroy(&gpu.device, &mut gpu.allocator);
                        pipelines.destroy(&gpu.device);
                    }
                    return Err(e);
                }
            }
        }
        let tracer_buffers: Option<Vec<TracerBuffers>> = made
            .iter()
            .map(FrameResources::tracer_buffers)
            .collect::<Option<_>>()
            .filter(|_| rt == RtMode::Auto);
        let reflections = match Reflections::new(
            gpu,
            color_format,
            extent,
            grid,
            &tee,
            tracer_buffers.as_deref(),
        ) {
            Ok(reflections) => Some(reflections),
            Err(e) => {
                eprintln!("Reflections setup failed, rendering without them: {e}");
                None
            }
        };
        let mode = if reflections
            .as_ref()
            .is_some_and(Reflections::has_ray_tracing)
        {
            RenderMode::RayTraced
        } else {
            RenderMode::Rasterized
        };
        Ok(Self {
            pipelines,
            geometry,
            content: FrameContent::default(),
            frames: made,
            reflections,
            mode,
            chase: None,
            grid: *grid,
            tee,
        })
    }

    /// Whether the recorder can ray-trace its reflections.
    pub(crate) fn has_ray_tracing(&self) -> bool {
        self.reflections
            .as_ref()
            .is_some_and(Reflections::has_ray_tracing)
    }

    /// Draw the edge of `green` as one ring on the range floor, faded like
    /// the recorder's grid; `None` draws nothing. Replaces any green set
    /// before, waiting for the device to go idle before it frees the old one.
    ///
    /// # Errors
    /// The wait or the upload fails.
    pub(crate) fn set_green(
        &mut self,
        gpu: &mut GpuContext,
        green: Option<&GreenConfig>,
    ) -> Result<(), RenderError> {
        self.geometry
            .set_green(&gpu.device, &mut gpu.allocator, green, &self.grid)
    }

    /// Set or clear the chase camera of the right 20 % inset.
    pub(crate) fn set_chase_camera(&mut self, camera: Option<Camera>) {
        self.chase = camera;
    }

    /// Switch the reflections between ray-traced and mirrored; `RayTraced`
    /// without a ray tracer gives `Rasterized`. The next frame shows the
    /// current scene with the path switched to.
    pub(crate) fn set_mode(&mut self, mode: RenderMode) {
        self.mode = if mode == RenderMode::RayTraced && !self.has_ray_tracing() {
            RenderMode::Rasterized
        } else {
            mode
        };
    }

    pub(crate) fn mode(&self) -> RenderMode {
        self.mode
    }

    /// Build the next frame's flight geometry from `flights` and point the
    /// reflections at the newest flight: its ball, and its tracer as drawn
    /// in the raster pass.
    pub(crate) fn update_flight(&mut self, flights: &[FlightRenderData]) {
        flight_vertices(flights, &self.tee, &mut self.content);
        if let Some(reflections) = &mut self.reflections {
            reflections.set_ball(self.content.ball_centre);
        }
    }

    /// Set the next frame's HUD geometry, in pixels.
    pub(crate) fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.content.hud_lines.clear();
        self.content.hud_lines.extend_from_slice(lines);
        self.content.hud_fills.clear();
        self.content.hud_fills.extend_from_slice(fills);
    }

    /// Size the reflection image to `extent`.
    ///
    /// # Safety
    /// The device is idle; framebuffers of the old extent are not in use.
    pub(crate) unsafe fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<(), RenderError> {
        if let Some(reflections) = &mut self.reflections {
            // SAFETY: The caller guarantees the device is idle.
            unsafe { reflections.resize(device, allocator, extent)? };
        }
        Ok(())
    }

    /// The render pass a [`FrameTarget::scene`] framebuffer is created with.
    pub(crate) fn scene_pass(&self) -> vk::RenderPass {
        self.pipelines.scene_pass
    }

    /// The render pass a [`FrameTarget::overlay`] framebuffer is created with.
    pub(crate) fn hud_pass(&self) -> vk::RenderPass {
        self.pipelines.hud_pass
    }

    /// Upload the next frame's content into the buffers of frame `frame` in
    /// flight and record the frame seen from `camera`: the scene pass (with
    /// the chase inset), then the reflections of the main view and the
    /// inset, ray-traced in `RayTraced` mode and mirrored otherwise, then
    /// the HUD pass, which leaves the colour image in the output layout.
    ///
    /// The upload is a host write into the buffers of `frame` alone, which no
    /// other frame in flight reads or builds the ray tracer's structures from
    /// ([`FrameResources::upload`]).
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass. The GPU has finished the
    /// frame that last used `frame` (its fence was waited on), and `frame`
    /// is less than the `frames` the recorder was created with. `target`'s
    /// framebuffers were created with [`Self::scene_pass`] and [`Self::hud_pass`] over one
    /// colour image of the recorder's colour format and `target.extent`
    /// (plus a depth image of the device's depth format for `scene`), and the
    /// recorder was [`resize`](Self::resize)d to `target.extent`.
    pub(crate) unsafe fn record_frame(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        target: &FrameTarget,
        camera: &Camera,
        frame: usize,
    ) {
        let resources = &mut self.frames[frame];
        resources.upload(&self.content);
        let resources = &self.frames[frame];
        let extent = target.extent;
        let main_view = TraceView {
            camera: *camera,
            offset: [0, 0],
            extent,
        };
        let full = main_view.rect();
        let viewport = main_view.viewport();
        let chase_view = self
            .chase
            .map(|chase| ChaseViewport::new(extent.width).view(chase, extent.height));

        // SAFETY: The caller guarantees the command buffer state and that the
        // framebuffers match the passes; every resource lives in `self`.
        unsafe {
            // Scene pass: main view, then the chase inset over it.
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
            let scene_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.pipelines.scene_pass)
                .framebuffer(target.scene)
                .render_area(full)
                .clear_values(&clear_values);
            device.cmd_begin_render_pass(cb, &scene_pass_info, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[full]);
            self.geometry.draw(
                device,
                cb,
                &self.pipelines,
                resources,
                view_proj(camera, main_view.aspect()),
            );

            if let Some(inset) = &chase_view {
                let clear_attachments = [
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        color_attachment: 0,
                        clear_value: clear_values[0],
                    },
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        color_attachment: 0,
                        clear_value: clear_values[1],
                    },
                ];
                let clear_rect = vk::ClearRect {
                    rect: inset.rect(),
                    base_array_layer: 0,
                    layer_count: 1,
                };
                device.cmd_clear_attachments(cb, &clear_attachments, &[clear_rect]);
                device.cmd_set_viewport(cb, 0, &[inset.viewport()]);
                device.cmd_set_scissor(cb, 0, &[inset.rect()]);
                self.geometry.draw(
                    device,
                    cb,
                    &self.pipelines,
                    resources,
                    view_proj(&inset.camera, inset.aspect()),
                );
            }
            device.cmd_end_render_pass(cb);

            // Floor reflections: main view, plus the chase inset when set.
            if let Some(reflections) = &self.reflections {
                let ray_traced = self.mode == RenderMode::RayTraced;
                let tracer = resources.newest_tracer();
                let with_chase;
                let main_only;
                let views: &[TraceView] = match chase_view {
                    Some(inset) => {
                        with_chase = [main_view, inset];
                        &with_chase
                    }
                    None => {
                        main_only = [main_view];
                        &main_only
                    }
                };
                let reflection_frame = ReflectionFrame {
                    views,
                    tracer: &tracer,
                    overlay: target.overlay,
                    extent,
                    frame,
                    ray_traced,
                };
                reflections.record(device, cb, &reflection_frame);
            }

            // HUD pass: the HUD over everything; ends in the output layout.
            let hud_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.pipelines.hud_pass)
                .framebuffer(target.overlay)
                .render_area(full);
            device.cmd_begin_render_pass(cb, &hud_pass_info, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[full]);
            resources.draw_hud(
                device,
                cb,
                self.pipelines.hud_fill,
                self.pipelines.hud_line,
                self.pipelines.layout,
                extent,
            );
            device.cmd_end_render_pass(cb);
        }
    }

    /// Destroy everything, in reverse creation order.
    ///
    /// # Safety
    /// The device is idle; `device` and `allocator` are the ones the recorder
    /// was created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees nothing is in use.
        unsafe {
            if let Some(reflections) = self.reflections {
                reflections.destroy(device, allocator);
            }
            for frame in self.frames {
                frame.destroy(device, allocator);
            }
            self.geometry.destroy(device, allocator);
            self.pipelines.destroy(device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chase_viewport_is_the_right_fifth() {
        for w in [1920u32, 3840, 1001] {
            let chase = ChaseViewport::new(w);
            assert_eq!(chase.x, (w as f32 * 0.8) as u32, "x at {w}");
            assert_eq!(chase.width, (w as f32 * 0.2) as u32, "width at {w}");
            assert!(chase.x + chase.width <= w, "inside the frame at {w}");
            let inset = chase.view(Camera::default(), 1080);
            let rect = inset.rect();
            assert_eq!(rect.offset.x as u32, chase.x, "i32 round trip at {w}");
            assert_eq!(rect.extent.width, chase.width);
            assert_eq!(rect.extent.height, 1080);
            assert_eq!(inset.aspect(), chase.width as f32 / 1080.0);
        }
    }

    #[test]
    fn chase_viewport_at_1080p() {
        assert_eq!(
            ChaseViewport::new(1920),
            ChaseViewport {
                x: 1536,
                width: 384
            }
        );
    }
}
