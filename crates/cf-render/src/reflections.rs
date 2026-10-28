//! Ray-traced floor reflections: the RT pipeline, its composite pass and the
//! scene state they show, as one optional part of a renderer.

use ash::vk;
use glam::Vec3;
use gpu_allocator::vulkan::Allocator;

use cf_scene::camera::Camera;
use cf_scene::grid::{GridConfig, GridVertex};
use cf_scene::tee::{TEE_ELEVATION, TeeBox};

use crate::composite::CompositePass;
use crate::context::GpuContext;
use crate::error::RenderError;
use crate::flight::{FlightRenderData, flight_trail_line, flight_trail_tube};
use crate::rt_offscreen::build_scene_geometry;
use crate::rt_pipeline::{RtGeometry, RtPipeline, RtPushConstants};

/// What the reflections show: the analytic ball and the reflected tracer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RtSceneState {
    /// Centre of the reflected ball.
    ball_centre: Vec3,
    /// The tracer tube's triangles: those the raster pass draws.
    trail: Vec<GridVertex>,
    /// The tracer's centre line segments: those the raster pass draws.
    line: Vec<GridVertex>,
}

impl RtSceneState {
    pub(crate) fn new(ball_centre: Vec3, trail: &[GridVertex], line: &[GridVertex]) -> Self {
        Self {
            ball_centre,
            trail: trail.to_vec(),
            line: line.to_vec(),
        }
    }

    /// The ball resting on `tee`'s tee box, no trail.
    pub(crate) fn on_tee(tee: &TeeBox) -> Self {
        Self::new(
            Vec3::new(0.0, TEE_ELEVATION + tee.ball_radius, 0.0),
            &[],
            &[],
        )
    }

    /// The newest of `flights` (the one the HUD and chase camera follow):
    /// its ball, one radius above `ball_pos`, and its tracer. With no
    /// flight, the ball on the tee.
    pub(crate) fn from_flights(flights: &[FlightRenderData], tee: &TeeBox) -> Self {
        match flights.last() {
            Some(flight) => Self::new(
                flight.ball_pos + Vec3::new(0.0, tee.ball_radius, 0.0),
                &flight_trail_tube(flight, tee),
                &flight_trail_line(flight),
            ),
            None => Self::on_tee(tee),
        }
    }

    /// The acceleration-structure geometry: the floor of `grid`, the tee
    /// box of `tee` and the tracer.
    fn geometry(&self, grid: &GridConfig, tee: &TeeBox) -> Vec<RtGeometry> {
        build_scene_geometry(grid, tee, &self.trail, &self.line)
    }

    /// Whether both states build the same acceleration structures: the same
    /// tracer. The ball is a push constant, not geometry.
    fn same_geometry(&self, other: &Self) -> bool {
        self.trail == other.trail && self.line == other.line
    }

    /// Push constants for tracing `view`.
    fn push_constants(&self, view: &TraceView) -> RtPushConstants {
        RtPushConstants::new(
            &view.camera,
            [view.extent.width as f32, view.extent.height as f32],
            [view.offset[0] as f32, view.offset[1] as f32],
            self.ball_centre,
        )
    }
}

/// One trace dispatch: a camera and the storage-image rectangle it fills.
pub(crate) struct TraceView {
    pub(crate) camera: Camera,
    /// Top-left corner in storage-image pixels.
    pub(crate) offset: [u32; 2],
    pub(crate) extent: vk::Extent2D,
}

/// Ray-traced floor reflections: trace into the storage image, then
/// composite it onto the raster scene.
pub(crate) struct Reflections {
    pipeline: RtPipeline,
    composite: CompositePass,
    /// What the acceleration structures were built from, with the current
    /// trail fade values.
    scene: RtSceneState,
}

impl Reflections {
    /// Build the RT pipeline for an `extent` storage image, with the floor
    /// of `grid`, the tee box of `tee` and `scene`'s trail, and the composite
    /// pass for a `color_format` target.
    ///
    /// # Errors
    /// Either part fails to create; what was created is destroyed first.
    pub(crate) fn new(
        gpu: &mut GpuContext,
        color_format: vk::Format,
        extent: vk::Extent2D,
        grid: &GridConfig,
        tee: &TeeBox,
        scene: RtSceneState,
    ) -> Result<Self, RenderError> {
        let geometries = scene.geometry(grid, tee);
        let pipeline = RtPipeline::new(gpu, extent.width, extent.height, &geometries)?;
        let composite = match CompositePass::new(&gpu.device, color_format, pipeline.storage.view())
        {
            Ok(composite) => composite,
            Err(e) => {
                // SAFETY: The pipeline has not been used.
                unsafe { pipeline.destroy(&gpu.device, &mut gpu.allocator) };
                return Err(e);
            }
        };
        Ok(Self {
            pipeline,
            composite,
            scene,
        })
    }

    /// Show `scene`. Rebuilds the acceleration structures (with the floor of
    /// `grid` and the tee box of `tee`) only when the geometry changes;
    /// `update_scene` waits for the device to go idle first. The ball follows
    /// `scene` on every successful call.
    ///
    /// # Errors
    /// The rebuild fails. The old structures are destroyed by then, so the
    /// reflections must not record again; destroy them.
    pub(crate) fn set_scene(
        &mut self,
        gpu: &mut GpuContext,
        scene: RtSceneState,
        grid: &GridConfig,
        tee: &TeeBox,
    ) -> Result<(), RenderError> {
        if !scene.same_geometry(&self.scene) {
            let geometries = scene.geometry(grid, tee);
            self.pipeline.update_scene(gpu, &geometries)?;
        }
        self.scene = scene;
        Ok(())
    }

    /// Recreate the storage image at `extent` (if it differs) and point the
    /// composite at it.
    ///
    /// # Safety
    /// The device is idle; framebuffers of the old extent are not in use.
    pub(crate) unsafe fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<(), RenderError> {
        self.pipeline
            .resize_storage(device, allocator, extent.width, extent.height)?;
        // SAFETY: The caller guarantees the device is idle.
        unsafe {
            self.composite
                .bind_storage(device, self.pipeline.storage.view());
        }
        Ok(())
    }

    /// Trace each of `views` into the storage image, then composite it into
    /// `overlay_fb` (`extent`).
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass; the colour image is in
    /// `COLOR_ATTACHMENT_OPTIMAL`; the storage image extent equals `extent`
    /// and every view lies within it. Afterwards the storage image is in
    /// `SHADER_READ_ONLY_OPTIMAL` and the colour image still in
    /// `COLOR_ATTACHMENT_OPTIMAL`.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        views: &[TraceView],
        overlay_fb: vk::Framebuffer,
        extent: vk::Extent2D,
    ) {
        let storage = self.pipeline.storage.image();
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        // SAFETY: The caller guarantees the command buffer and image states.
        unsafe {
            // Storage image: UNDEFINED → GENERAL.
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
                    .image(storage)
                    .subresource_range(subresource_range)],
            );

            for (i, view) in views.iter().enumerate() {
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
                let pc = self.scene.push_constants(view);
                self.pipeline
                    .record_trace(device, cb, &pc, view.extent.width, view.extent.height);
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

            self.composite.record(device, cb, overlay_fb, extent);
        }
    }

    /// Destroy the composite pass and the RT pipeline.
    ///
    /// # Safety
    /// No pending GPU work uses them; `device` and `allocator` are the ones
    /// they were created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees nothing uses them.
        unsafe {
            self.composite.destroy(device);
            self.pipeline.destroy(device, allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cf_scene::trail::{DEFAULT_TRAIL_LIFETIME, TrailPoint, generate_trail_tube};

    fn tube(z: f32) -> Vec<GridVertex> {
        let v = |z: f32| GridVertex {
            position: [0.0, 1.0, z],
            fade: 0.5,
        };
        vec![v(z), v(z + 1.0), v(z + 2.0)]
    }

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

    #[test]
    fn no_flight_is_the_ball_on_the_tee() {
        let tee = TeeBox::default();
        let state = RtSceneState::from_flights(&[], &tee);
        assert_eq!(state, RtSceneState::on_tee(&tee));
        assert_eq!(
            state.ball_centre,
            Vec3::new(0.0, TEE_ELEVATION + tee.ball_radius, 0.0)
        );
        assert!(state.trail.is_empty());
        assert!(state.line.is_empty());
    }

    #[test]
    fn the_newest_flight_is_reflected_one_radius_up() {
        let tee = TeeBox::default();
        let older = flight(30.0, 3.9);
        let newest = flight(60.0, 3.5);
        let state = RtSceneState::from_flights(&[older, newest], &tee);
        let newest = flight(60.0, 3.5);
        assert_eq!(
            state.ball_centre,
            newest.ball_pos + Vec3::new(0.0, tee.ball_radius, 0.0)
        );
        assert!(!state.trail.is_empty());
        assert_eq!(state.trail, flight_trail_tube(&newest, &tee));
        assert_eq!(state.line, flight_trail_line(&newest));
    }

    #[test]
    fn the_reflected_tracer_is_the_raster_tracer() {
        let tee = TeeBox::default();
        let flight = flight(60.0, 3.5);
        let state = RtSceneState::from_flights(std::slice::from_ref(&flight), &tee);
        let raster = generate_trail_tube(
            &flight.trail_points,
            flight.current_time,
            DEFAULT_TRAIL_LIFETIME,
            tee.ball_radius,
        );
        assert_eq!(state.trail, raster);
    }

    #[test]
    fn geometry_uses_the_given_grid() {
        let tee = TeeBox::default();
        let state = RtSceneState::from_flights(&[flight(60.0, 3.5)], &tee);
        let grid = GridConfig {
            downrange: 200.0,
            ..GridConfig::default()
        };
        let built = state.geometry(&grid, &tee);
        let expected = build_scene_geometry(&grid, &tee, &state.trail, &state.line);
        let default_floor = &build_scene_geometry(&GridConfig::default(), &tee, &[], &[])[0];
        assert_eq!(built.len(), expected.len());
        for (b, e) in built.iter().zip(&expected) {
            assert_eq!(b.positions, e.positions);
            assert_eq!(b.geom_type, e.geom_type);
        }
        assert_ne!(built[0].positions, default_floor.positions);
    }

    #[test]
    fn only_the_tube_is_geometry() {
        let a = RtSceneState::new(Vec3::Y, &tube(1.0), &tube(1.0)[..2]);
        let moved = RtSceneState::new(Vec3::X, &tube(1.0), &tube(1.0)[..2]);
        let other_tube = RtSceneState::new(Vec3::Y, &tube(2.0), &tube(1.0)[..2]);
        let other_line = RtSceneState::new(Vec3::Y, &tube(1.0), &tube(2.0)[..2]);
        assert!(a.same_geometry(&moved));
        assert_ne!(a, moved);
        assert!(!a.same_geometry(&other_tube));
        assert!(!a.same_geometry(&other_line));
    }

    #[test]
    fn push_constants_follow_the_view() {
        let state = RtSceneState::new(Vec3::new(1.0, 2.0, 3.0), &tube(1.0), &[]);
        let camera = Camera::driving_range();
        let view = TraceView {
            camera,
            offset: [1536, 0],
            extent: vk::Extent2D {
                width: 384,
                height: 1080,
            },
        };
        let pc = state.push_constants(&view);
        let expected = RtPushConstants::new(
            &camera,
            [384.0, 1080.0],
            [1536.0, 0.0],
            Vec3::new(1.0, 2.0, 3.0),
        );
        assert_eq!(pc.camera_pos, expected.camera_pos);
        assert_eq!(pc.inv_view_proj, expected.inv_view_proj);
        assert_eq!(pc.ball_pos, expected.ball_pos);
    }
}
