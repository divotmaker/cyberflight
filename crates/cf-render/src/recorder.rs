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
use crate::frame::{FrameContent, FrameResources};
use crate::mode::RenderMode;
use crate::pipeline::ScenePipelines;
use crate::reflections::{Reflections, RtSceneState, TraceView};
use crate::scene::{SceneGeometry, Snapshot};

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

    /// The inset's rectangle in a frame `height` pixels high (scissor and
    /// clear rect).
    pub(crate) fn rect(self, height: u32) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D {
                x: self.x as i32,
                y: 0,
            },
            extent: vk::Extent2D {
                width: self.width,
                height,
            },
        }
    }

    /// The inset's viewport in a frame `height` pixels high.
    pub(crate) fn viewport(self, height: u32) -> vk::Viewport {
        vk::Viewport {
            x: self.x as f32,
            y: 0.0,
            width: self.width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Width / height of the inset in a frame `height` pixels high.
    pub(crate) fn aspect(self, height: u32) -> f32 {
        self.width as f32 / height as f32
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
/// next frame's flights and HUD, the optional reflections and the chase
/// camera.
pub(crate) struct SceneRecorder {
    pipelines: ScenePipelines,
    geometry: SceneGeometry,
    /// Next frame's flights and HUD, uploaded into `frame` by `record_frame`.
    content: FrameContent,
    frame: FrameResources,
    reflections: Option<Reflections>,
    mode: RenderMode,
    /// Chase camera for the right 20 % inset.
    chase: Option<Camera>,
    color_format: vk::Format,
    grid: GridConfig,
    tee: TeeBox,
}

impl SceneRecorder {
    /// Create the passes and pipelines for a `color_format` image that the
    /// HUD pass leaves in `output_layout`, and the scene of `grid` with the
    /// default tee box: the ball on the tee, or `snapshot`'s ball and trail.
    /// No reflections; the mode is `Rasterized`.
    ///
    /// # Errors
    /// Any part fails to create; what was created is destroyed first.
    pub(crate) fn new(
        gpu: &mut GpuContext,
        color_format: vk::Format,
        output_layout: vk::ImageLayout,
        grid: &GridConfig,
        snapshot: Option<&Snapshot<'_>>,
    ) -> Result<Self, RenderError> {
        let tee = TeeBox::default();
        let pipelines =
            ScenePipelines::new(&gpu.device, color_format, gpu.depth_format, output_layout)?;
        let geometry = match SceneGeometry::new(
            &gpu.device,
            &mut gpu.allocator,
            grid,
            &tee,
            snapshot,
            gpu.wide_lines,
        ) {
            Ok(geometry) => geometry,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used it.
                unsafe { pipelines.destroy(&gpu.device) };
                return Err(e);
            }
        };
        let frame = match FrameResources::new(&gpu.device, &mut gpu.allocator) {
            Ok(frame) => frame,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe {
                    geometry.destroy(&gpu.device, &mut gpu.allocator);
                    pipelines.destroy(&gpu.device);
                }
                return Err(e);
            }
        };
        Ok(Self {
            pipelines,
            geometry,
            content: FrameContent::default(),
            frame,
            reflections: None,
            mode: RenderMode::Rasterized,
            chase: None,
            color_format,
            grid: *grid,
            tee,
        })
    }

    /// Add ray-traced reflections for an `extent` frame showing `scene`, and
    /// switch to `RayTraced`. The device must support ray tracing.
    ///
    /// # Errors
    /// The reflections fail to create; the recorder stays raster-only.
    pub(crate) fn enable_reflections(
        &mut self,
        gpu: &mut GpuContext,
        extent: vk::Extent2D,
        scene: RtSceneState,
    ) -> Result<(), RenderError> {
        let reflections =
            Reflections::new(gpu, self.color_format, extent, &self.grid, &self.tee, scene)?;
        self.reflections = Some(reflections);
        self.mode = RenderMode::RayTraced;
        Ok(())
    }

    /// Whether the recorder has reflections.
    pub(crate) fn has_reflections(&self) -> bool {
        self.reflections.is_some()
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

    /// Switch between raster and ray-traced frames; `RayTraced` without
    /// reflections gives `Rasterized`.
    pub(crate) fn set_mode(&mut self, mode: RenderMode) {
        self.mode = if mode == RenderMode::RayTraced && self.reflections.is_none() {
            RenderMode::Rasterized
        } else {
            mode
        };
    }

    pub(crate) fn mode(&self) -> RenderMode {
        self.mode
    }

    /// Build the next frame's flight geometry from `flights`, with trail
    /// glows facing `camera`. With reflections, `rt_scene` gets the state
    /// they show and returns the one to show next.
    pub(crate) fn update_flight(
        &mut self,
        gpu: &mut GpuContext,
        flights: &[FlightRenderData],
        camera: &Camera,
        rt_scene: impl FnOnce(&RtSceneState) -> RtSceneState,
    ) {
        flight_vertices(flights, camera.position, &self.tee, &mut self.content);
        if let Some(reflections) = &mut self.reflections {
            let scene = rt_scene(reflections.scene());
            reflections.set_scene(gpu, scene, &GridConfig::default(), &self.tee);
        }
    }

    /// Set the next frame's HUD geometry, in pixels.
    pub(crate) fn update_hud(&mut self, lines: &[GridVertex], fills: &[GridVertex]) {
        self.content.hud_lines.clear();
        self.content.hud_lines.extend_from_slice(lines);
        self.content.hud_fills.clear();
        self.content.hud_fills.extend_from_slice(fills);
    }

    /// Size the reflections' storage image to `extent`.
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

    /// Upload the next frame's content and record the frame seen from
    /// `camera`: the scene pass (with the chase inset), then, in
    /// `RayTraced` mode, the reflections of the main view and the inset,
    /// then the HUD pass, which leaves the colour image in the output layout.
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass. `target`'s framebuffers were
    /// created with [`Self::scene_pass`] and [`Self::hud_pass`] over one
    /// colour image of the recorder's colour format and `target.extent`
    /// (plus a depth image of the device's depth format for `scene`), and the
    /// recorder was [`resize`](Self::resize)d to `target.extent`. The GPU has
    /// finished the frames that read the per-frame buffers.
    pub(crate) unsafe fn record_frame(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        target: &FrameTarget,
        camera: &Camera,
    ) {
        self.frame.upload(&self.content);
        let extent = target.extent;
        let full = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let chase = self
            .chase
            .map(|chase| (chase, ChaseViewport::new(extent.width)));

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
            let aspect = extent.width as f32 / extent.height as f32;
            self.geometry.draw(
                device,
                cb,
                &self.pipelines,
                &self.frame,
                view_proj(camera, aspect),
            );

            if let Some((chase_camera, chase_vp)) = &chase {
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
                    rect: chase_vp.rect(extent.height),
                    base_array_layer: 0,
                    layer_count: 1,
                };
                device.cmd_clear_attachments(cb, &clear_attachments, &[clear_rect]);
                device.cmd_set_viewport(cb, 0, &[chase_vp.viewport(extent.height)]);
                device.cmd_set_scissor(cb, 0, &[chase_vp.rect(extent.height)]);
                self.geometry.draw(
                    device,
                    cb,
                    &self.pipelines,
                    &self.frame,
                    view_proj(chase_camera, chase_vp.aspect(extent.height)),
                );
            }
            device.cmd_end_render_pass(cb);

            // Ray-traced reflections: main view, plus the chase inset when set.
            if self.mode == RenderMode::RayTraced
                && let Some(reflections) = &self.reflections
            {
                let mut views = vec![TraceView {
                    camera: *camera,
                    offset: [0, 0],
                    extent,
                }];
                if let Some((chase_camera, chase_vp)) = chase {
                    views.push(TraceView {
                        camera: chase_camera,
                        offset: [chase_vp.x, 0],
                        extent: vk::Extent2D {
                            width: chase_vp.width,
                            height: extent.height,
                        },
                    });
                }
                reflections.record(device, cb, &views, target.overlay, extent);
            }

            // HUD pass: the HUD over everything; ends in the output layout.
            let hud_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.pipelines.hud_pass)
                .framebuffer(target.overlay)
                .render_area(full);
            device.cmd_begin_render_pass(cb, &hud_pass_info, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[full]);
            self.frame.draw_hud(
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
            self.frame.destroy(device, allocator);
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
            let rect = chase.rect(1080);
            assert_eq!(rect.offset.x as u32, chase.x, "i32 round trip at {w}");
            assert_eq!(rect.extent.width, chase.width);
            assert_eq!(rect.extent.height, 1080);
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
