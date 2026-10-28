//! Floor reflections: the reflection image, filled by the ray tracer or by
//! the raster mirror, and the composite pass that adds it onto the scene.

use ash::vk;
use glam::Vec3;
use gpu_allocator::vulkan::Allocator;

use cf_scene::camera::Camera;
use cf_scene::grid::GridConfig;
use cf_scene::tee::TeeBox;

use crate::composite::CompositePass;
use crate::context::GpuContext;
use crate::error::RenderError;
use crate::flight::tee_ball_centre;
use crate::frame::TracerBuffers;
use crate::gpu::{Image2d, Image2dDesc};
use crate::mirror::{Mirror, MirrorTracer};
use crate::rt_offscreen::build_scene_geometry;
use crate::rt_pipeline::{RtPipeline, RtPushConstants};

/// Format of the reflection image (`rgba16f` in `rt.rgen`).
///
/// It holds linear colour up to `MAX_RADIANCE` in `floor_shading.glsl`, which
/// the composite pass reads one texel per pixel and blends into an sRGB
/// target, so it needs more range than [0, 1] for emissive reflections and
/// more precision than 8 bits for dark tones. Vulkan requires storage,
/// sampled and blendable colour attachment support for this format.
pub(crate) const REFLECTION_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// One view of a frame: a camera and the rectangle of the frame (and of the
/// reflection image) it fills.
#[derive(Clone, Copy)]
pub(crate) struct TraceView {
    pub(crate) camera: Camera,
    /// Top-left corner in pixels.
    pub(crate) offset: [u32; 2],
    pub(crate) extent: vk::Extent2D,
}

impl TraceView {
    /// The view's rectangle (scissor and clear rect).
    pub(crate) fn rect(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D {
                x: self.offset[0] as i32,
                y: self.offset[1] as i32,
            },
            extent: self.extent,
        }
    }

    /// The view's viewport.
    pub(crate) fn viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: self.offset[0] as f32,
            y: self.offset[1] as f32,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Width / height of the view.
    pub(crate) fn aspect(&self) -> f32 {
        self.extent.width as f32 / self.extent.height as f32
    }
}

/// The ball the floor reflects analytically.
#[derive(Clone, Copy)]
pub(crate) struct Ball {
    pub(crate) centre: Vec3,
    pub(crate) radius: f32,
}

/// What one frame's reflections are drawn from and into.
pub(crate) struct ReflectionFrame<'a> {
    /// The views to fill, the main one first; later ones replace the
    /// rectangle they cover.
    pub(crate) views: &'a [TraceView],
    /// The newest flight's tracer as the raster pass uploaded it.
    pub(crate) tracer: &'a MirrorTracer,
    /// Framebuffer of the colour image the reflections are composited into.
    pub(crate) overlay: vk::Framebuffer,
    /// Extent of the colour image.
    pub(crate) extent: vk::Extent2D,
    /// The frame in flight: which of the ray tracer's per-frame structures
    /// this frame builds and traces.
    pub(crate) frame: usize,
    /// Trace the reflections (when a ray tracer is available) rather than
    /// mirror them.
    pub(crate) ray_traced: bool,
}

/// Floor reflections of the tracer and the ball: fill the reflection image
/// with the ray tracer (when the device has one) or the raster mirror, then
/// composite it onto the raster scene.
///
/// Both paths shade the floor with the code of `floor_shading.glsl` and
/// show the same tracer (the newest flight's, drawn from the vertex buffers
/// the raster pass draws from) and ball, so they show the same reflection.
pub(crate) struct Reflections {
    /// The image both paths write and the composite reads: the floor's
    /// colour, with the reflections in it, per pixel.
    target: Image2d,
    composite: CompositePass,
    mirror: Mirror,
    /// `None` without ray tracing.
    rt: Option<RtPipeline>,
    /// The reflected ball.
    ball: Ball,
}

/// The reflection image of `extent`.
fn create_target(
    device: &ash::Device,
    allocator: &mut Allocator,
    extent: vk::Extent2D,
) -> Result<Image2d, RenderError> {
    Image2d::new(
        device,
        allocator,
        &Image2dDesc {
            extent,
            format: REFLECTION_FORMAT,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "reflection image",
        },
    )
}

impl Reflections {
    /// Create the reflections for an `extent` frame of the floor of `grid`
    /// and the tee box of `tee`, showing the ball on the tee, composited onto
    /// a `color_format` target. The raster mirror is always there; given the
    /// `tracers` buffers (one set per frame in flight, not empty), so is the
    /// ray tracer, unless it fails to set up (it says so on stderr and the
    /// mirror remains).
    ///
    /// # Errors
    /// The image, the composite or the mirror fails to create; what was
    /// created is destroyed first.
    pub(crate) fn new(
        gpu: &mut GpuContext,
        color_format: vk::Format,
        extent: vk::Extent2D,
        grid: &GridConfig,
        tee: &TeeBox,
        tracers: Option<&[TracerBuffers]>,
    ) -> Result<Self, RenderError> {
        let target = create_target(&gpu.device, &mut gpu.allocator, extent)?;
        let composite = match CompositePass::new(&gpu.device, color_format, target.view()) {
            Ok(composite) => composite,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used it.
                unsafe { target.destroy(&gpu.device, &mut gpu.allocator) };
                return Err(e);
            }
        };
        let mirror = match Mirror::new(
            &gpu.device,
            &mut gpu.allocator,
            target.view(),
            extent,
            grid,
            tee,
            gpu.wide_lines,
        ) {
            Ok(mirror) => mirror,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe {
                    composite.destroy(&gpu.device);
                    target.destroy(&gpu.device, &mut gpu.allocator);
                }
                return Err(e);
            }
        };
        let rt = tracers.and_then(|tracers| {
            let geometries = build_scene_geometry(grid, tee);
            match RtPipeline::new(gpu, target.view(), &geometries, tracers) {
                Ok(rt) => Some(rt),
                Err(e) => {
                    eprintln!("RT setup failed, using mirror reflections: {e}");
                    None
                }
            }
        });
        Ok(Self {
            target,
            composite,
            mirror,
            rt,
            ball: Ball {
                centre: tee_ball_centre(tee),
                radius: tee.ball_radius,
            },
        })
    }

    /// Whether the ray tracer is available.
    pub(crate) fn has_ray_tracing(&self) -> bool {
        self.rt.is_some()
    }

    /// Reflect the ball with centre `ball_centre`.
    pub(crate) fn set_ball(&mut self, ball_centre: Vec3) {
        self.ball.centre = ball_centre;
    }

    /// Recreate the reflection image at `extent` (if it differs) and point
    /// the composite, the ray tracer and the mirror at it.
    ///
    /// # Safety
    /// The device is idle; framebuffers of the old extent are not in use.
    pub(crate) unsafe fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<(), RenderError> {
        if self.target.extent() == extent {
            return Ok(());
        }
        let target = create_target(device, allocator, extent)?;
        // SAFETY: The caller guarantees the device is idle.
        unsafe {
            if let Err(e) = self.mirror.resize(device, allocator, target.view(), extent) {
                target.destroy(device, allocator);
                return Err(e);
            }
            let old = std::mem::replace(&mut self.target, target);
            old.destroy(device, allocator);
            self.composite.bind_storage(device, self.target.view());
            if let Some(rt) = &self.rt {
                rt.bind_storage(device, self.target.view());
            }
        }
        Ok(())
    }

    /// Fill the reflection image for each view of `frame`, with the ray
    /// tracer when `frame.ray_traced` and it is available, else with the
    /// mirror, then composite it into `frame.overlay`.
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass; the buffers of
    /// `frame.tracer` are live and hold its vertices (and, with ray tracing,
    /// the line AABBs of its centre line); `frame.frame` is the frame in
    /// flight that owns those buffers and whose fence was waited on; the
    /// colour image is in `COLOR_ATTACHMENT_OPTIMAL`; the reflection image
    /// extent equals `frame.extent` and every view lies within it. Afterwards the
    /// reflection image is in `SHADER_READ_ONLY_OPTIMAL` and the colour
    /// image still in `COLOR_ATTACHMENT_OPTIMAL`.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame: &ReflectionFrame<'_>,
    ) {
        // SAFETY: The caller guarantees the command buffer and image states
        // and the tracer's buffers.
        unsafe {
            match &self.rt {
                Some(rt) if frame.ray_traced => {
                    self.record_trace(rt, device, cb, frame);
                }
                _ => self.mirror.record(
                    device,
                    cb,
                    frame.tracer,
                    frame.views,
                    frame.extent,
                    &self.ball,
                ),
            }
            self.composite
                .record(device, cb, frame.overlay, frame.extent);
        }
    }

    /// Push constants for tracing `view`.
    fn rt_push_constants(&self, view: &TraceView) -> RtPushConstants {
        RtPushConstants::new(
            &view.camera,
            [view.extent.width as f32, view.extent.height as f32],
            [view.offset[0] as f32, view.offset[1] as f32],
            self.ball.centre,
            self.ball.radius,
        )
    }

    /// Rebuild `rt`'s structures of `frame.frame` for `frame.tracer`, then
    /// trace each of `frame.views` into the reflection image.
    ///
    /// # Safety
    /// As [`Self::record`].
    unsafe fn record_trace(
        &self,
        rt: &RtPipeline,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame: &ReflectionFrame<'_>,
    ) {
        let storage = self.target.image();
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        // SAFETY: The caller guarantees the command buffer and image states
        // and the tracer's buffers.
        unsafe {
            rt.record_build(
                device,
                cb,
                frame.frame,
                frame.tracer.tube,
                frame.tracer.line,
            );

            // Storage image: UNDEFINED → GENERAL. The transition is a write,
            // so it waits for what last used the image: the composite
            // sampling it (fragment shader) and, after mirrored frames, the
            // mirror's floor pass writing it (colour attachment output).
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(storage)
                    .subresource_range(subresource_range)],
            );

            for (i, view) in frame.views.iter().enumerate() {
                if i > 0 {
                    // The previous trace finishes writing before the next one.
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
                }
                let pc = self.rt_push_constants(view);
                rt.record_trace(
                    device,
                    cb,
                    frame.frame,
                    &pc,
                    view.extent.width,
                    view.extent.height,
                );
            }

            // Storage image: GENERAL → SHADER_READ_ONLY_OPTIMAL for the composite.
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
                    .image(storage)
                    .subresource_range(subresource_range)],
            );
        }
    }

    /// Destroy the composite pass, the ray tracer, the mirror and the image.
    ///
    /// # Safety
    /// No pending GPU work uses them; `device` and `allocator` are the ones
    /// they were created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees nothing uses them.
        unsafe {
            if let Some(rt) = self.rt {
                rt.destroy(device, allocator);
            }
            self.mirror.destroy(device, allocator);
            self.composite.destroy(device);
            self.target.destroy(device, allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flight::{FlightRenderData, flight_vertices};
    use crate::frame::FrameContent;
    use cf_scene::tee::TEE_ELEVATION;
    use cf_scene::trail::{DEFAULT_TRAIL_LIFETIME, TrailPoint, generate_trail_tube};

    fn flight(z: f32, current_time: f64) -> FlightRenderData {
        FlightRenderData {
            ball_pos: Vec3::new(1.0, 5.0, z),
            trail_points: (0..40)
                .map(|i| TrailPoint {
                    position: Vec3::new(0.0, i as f32 * 0.1, i as f32 * z / 40.0),
                    time: f64::from(i) * 0.1,
                })
                .collect(),
            current_time,
        }
    }

    /// The frame content of `flights`.
    fn content(flights: &[FlightRenderData]) -> FrameContent {
        let mut content = FrameContent::default();
        flight_vertices(flights, &TeeBox::default(), &mut content);
        content
    }

    #[test]
    fn the_newest_flights_tracer_is_the_raster_tracer() {
        let tee = TeeBox::default();
        let newest = flight(60.0, 3.5);
        let content = content(&[flight(30.0, 3.9), flight(60.0, 3.5)]);
        let raster = generate_trail_tube(
            &newest.trail_points,
            newest.current_time,
            DEFAULT_TRAIL_LIFETIME,
            tee.ball_radius,
        );
        assert!(!raster.is_empty());
        assert_eq!(content.flight_glows[..content.newest_tube], raster[..]);
    }

    #[test]
    fn the_ball_is_the_newest_flights_one_radius_up_or_on_the_tee() {
        let tee = TeeBox::default();
        assert_eq!(
            content(&[]).ball_centre,
            Vec3::new(0.0, TEE_ELEVATION + tee.ball_radius, 0.0)
        );
        let newest = flight(60.0, 3.5);
        assert_eq!(
            content(&[flight(30.0, 3.9), flight(60.0, 3.5)]).ball_centre,
            newest.ball_pos + Vec3::new(0.0, tee.ball_radius, 0.0)
        );
    }

    #[test]
    fn push_constants_follow_the_view() {
        let ball = Vec3::new(1.0, 2.0, 3.0);
        let camera = Camera::driving_range();
        let view = TraceView {
            camera,
            offset: [1536, 0],
            extent: vk::Extent2D {
                width: 384,
                height: 1080,
            },
        };
        let pc = RtPushConstants::new(&camera, [384.0, 1080.0], [1536.0, 0.0], ball, 0.02);
        assert_eq!(pc.camera_pos[3], 1536.0);
        assert_eq!(pc.ball_pos[..3], [1.0, 2.0, 3.0]);
        assert_eq!(view.aspect(), 384.0 / 1080.0);
    }

    #[test]
    fn a_view_is_its_own_rectangle_and_viewport() {
        let inset = TraceView {
            camera: Camera::driving_range(),
            offset: [1536, 0],
            extent: vk::Extent2D {
                width: 384,
                height: 1080,
            },
        };
        let rect = inset.rect();
        assert_eq!((rect.offset.x, rect.offset.y), (1536, 0));
        assert_eq!((rect.extent.width, rect.extent.height), (384, 1080));
        let viewport = inset.viewport();
        assert_eq!((viewport.x, viewport.y), (1536.0, 0.0));
        assert_eq!((viewport.width, viewport.height), (384.0, 1080.0));
        assert_eq!((viewport.min_depth, viewport.max_depth), (0.0, 1.0));
    }
}
