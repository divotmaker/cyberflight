use ash::vk;
use gpu_allocator::vulkan::Allocator;

use cf_scene::camera::Camera;

use crate::context::GpuContext;
use crate::error::RenderError;
use crate::gpu::{DeviceBuffer, HostBuffer, Image2d, Image2dDesc, load_shader, push_constants};

/// Push constants for the ray tracing pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RtPushConstants {
    /// Camera position (xyz), w = viewport X offset in pixels.
    pub camera_pos: [f32; 4],
    /// Inverse view-projection matrix (screen → world).
    pub inv_view_proj: [[f32; 4]; 4],
    /// x = tracer fade value at the trail's oldest point, y = size of one
    /// pixel per metre of ray path (the viewport's pixel angle), z = unused
    /// (0), w = viewport Y offset in pixels.
    pub params: [f32; 4],
    /// Ball position (xyz), w = trail fade distance in meters.
    pub ball_pos: [f32; 4],
}

impl RtPushConstants {
    /// Pack push constants for one trace dispatch.
    ///
    /// `viewport_size` is the dispatch viewport's width and height and
    /// `viewport_offset` its top-left corner, both in storage-image pixels.
    /// `trail_fade_dist` is the trail reflection fade distance in meters (0
    /// selects the shader default) and `trail_tail_t` the tracer fade value at
    /// the trail's oldest point (see [`RtTrail`](crate::rt_offscreen::RtTrail)).
    #[must_use]
    pub fn new(
        camera: &Camera,
        viewport_size: [f32; 2],
        viewport_offset: [f32; 2],
        ball_center: glam::Vec3,
        trail_fade_dist: f32,
        trail_tail_t: f32,
    ) -> Self {
        let view = camera.view_matrix();
        let mut proj = camera.projection_matrix(viewport_size[0] / viewport_size[1]);
        // y_axis.y = 1 / tan(fov_y / 2): the viewport spans 2 / y_axis.y per
        // metre of distance over its height.
        let pixel_size = 2.0 / (proj.y_axis.y * viewport_size[1]);
        proj.y_axis.y *= -1.0; // Vulkan Y-flip
        let inv_view_proj = (proj * view).inverse();

        Self {
            camera_pos: [
                camera.position.x,
                camera.position.y,
                camera.position.z,
                viewport_offset[0],
            ],
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            params: [trail_tail_t, pixel_size, 0.0, viewport_offset[1]],
            ball_pos: [ball_center.x, ball_center.y, ball_center.z, trail_fade_dist],
        }
    }
}

/// Format of the RT reflection storage image (`rgba16f` in `rt.rgen`).
///
/// It holds linear colour up to `MAX_RADIANCE` in `rt.rgen`, which the
/// composite pass reads one texel per pixel and blends into an sRGB target,
/// so it needs more range than [0, 1] for emissive reflections and more
/// precision than 8 bits for dark tones. Vulkan requires storage and sampled
/// support for this format.
const STORAGE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Geometry type IDs used as `gl_InstanceCustomIndexEXT` in shaders.
pub const GEOM_FLOOR: u32 = 0;
// The ball has no RT geometry: the floor shader reflects it analytically.
pub const GEOM_TEE_BOX: u32 = 2;
pub const GEOM_TRAIL: u32 = 3;
pub const GEOM_TEE_BOX_BORDER: u32 = 4;

/// Largest radius of the trail tube in metres, and the padding of each
/// segment's AABB around its axis. Must equal `TUBE_MAX_RADIUS` in `rt.rint`.
pub const TRAIL_TUBE_MAX_RADIUS: f32 = 1.0;

/// Shader stages that read the push constants.
const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::RAYGEN_KHR.as_raw()
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR.as_raw()
        | vk::ShaderStageFlags::INTERSECTION_KHR.as_raw(),
);

/// Hit group of triangle geometry (`rt.rchit`), the SBT hit record offset of
/// its instances.
const HIT_GROUP_TRIANGLES: u32 = 0;
/// Hit group of the trail tube (`rt.rint` + `rt_trail.rchit`).
const HIT_GROUP_TUBE: u32 = 1;

/// A single acceleration structure (BLAS or TLAS) with its backing buffer.
struct AccelStruct {
    accel: vk::AccelerationStructureKHR,
    buffer: DeviceBuffer,
}

/// The pipeline's shader modules, layouts, pipeline and descriptor set.
/// Null handles stand for the objects not created yet; destroying a null
/// handle does nothing.
#[derive(Default)]
struct PipelineObjects {
    rgen_module: vk::ShaderModule,
    rmiss_module: vk::ShaderModule,
    rchit_module: vk::ShaderModule,
    rint_module: vk::ShaderModule,
    trail_rchit_module: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl PipelineObjects {
    /// Create the shader modules, the descriptor set layout (TLAS, storage
    /// image, trail points), the pipeline layout (push constants + set) and
    /// the pipeline, stopping at the first error.
    fn create_pipeline(
        &mut self,
        device: &ash::Device,
        rt_loader: &ash::khr::ray_tracing_pipeline::Device,
    ) -> Result<(), RenderError> {
        // ── Shaders ──
        self.rgen_module = load_shader(device, include_bytes!("../shaders/rt.rgen.spv"))?;
        self.rmiss_module = load_shader(device, include_bytes!("../shaders/rt.rmiss.spv"))?;
        self.rchit_module = load_shader(device, include_bytes!("../shaders/rt.rchit.spv"))?;
        self.rint_module = load_shader(device, include_bytes!("../shaders/rt.rint.spv"))?;
        self.trail_rchit_module =
            load_shader(device, include_bytes!("../shaders/rt_trail.rchit.spv"))?;

        // ── Descriptor set layout: TLAS, storage image, trail points ──
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::INTERSECTION_KHR),
        ];
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Creating descriptor set layout.
        self.descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // ── Pipeline layout (push constants + descriptor set) ──
        let push_range = vk::PushConstantRange::default()
            .stage_flags(PUSH_STAGES)
            .offset(0)
            .size(std::mem::size_of::<RtPushConstants>() as u32);
        let push_ranges = [push_range];
        let set_layouts = [self.descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: Creating pipeline layout.
        self.pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // ── RT pipeline ──
        let entry_point = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(self.rgen_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(self.rmiss_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(self.rchit_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
                .module(self.rint_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(self.trail_rchit_module)
                .name(entry_point),
        ];

        let groups = [
            // Group 0: raygen
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Group 1: miss
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Group 2: triangle hit (HIT_GROUP_TRIANGLES)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Group 3: trail tube hit (HIT_GROUP_TUBE)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(4)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(3),
        ];

        let rt_create_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(2) // primary + 1 reflection
            .layout(self.pipeline_layout);

        // SAFETY: Creating ray tracing pipeline.
        self.pipeline = unsafe {
            rt_loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[rt_create_info],
                    None,
                )
                .map_err(|(_pipelines, err)| RenderError::Vulkan(err))?[0]
        };
        Ok(())
    }

    /// Create the descriptor pool and allocate the one descriptor set.
    fn create_descriptor_set(&mut self, device: &ash::Device) -> Result<(), RenderError> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: Creating descriptor pool.
        self.descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(RenderError::Vulkan)?
        };

        let set_layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: Allocating descriptor set.
        self.descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(RenderError::Vulkan)?[0]
        };
        Ok(())
    }

    /// Destroy every object; the descriptor set goes with its pool.
    ///
    /// # Safety
    /// No pending GPU work uses them.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: The caller guarantees nothing is in use.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.rgen_module, None);
            device.destroy_shader_module(self.rmiss_module, None);
            device.destroy_shader_module(self.rchit_module, None);
            device.destroy_shader_module(self.rint_module, None);
            device.destroy_shader_module(self.trail_rchit_module, None);
        }
    }
}

/// Ray tracing pipeline: shaders, SBT, descriptors, and acceleration structures.
pub struct RtPipeline {
    accel_loader: ash::khr::acceleration_structure::Device,
    rt_loader: ash::khr::ray_tracing_pipeline::Device,

    // Shader modules, pipeline and descriptors
    objects: PipelineObjects,

    // Shader binding table
    sbt: HostBuffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,

    // Acceleration structures
    blas_list: Vec<AccelStruct>,
    tlas: Option<AccelStruct>,

    // Geometry buffers (kept alive for AS references)
    geometry_buffers: Vec<HostBuffer>,
    // Trail tube axis points read by `rt.rint` (one zero point without a
    // trail), bound at binding 2.
    trail_points: Option<HostBuffer>,
    // Instance buffer
    instance_buffer: Option<HostBuffer>,

    // Scratch buffers of the current BLAS/TLAS build, freed with the AS on the
    // next `update_scene` or on drop.
    scratch_buffers: Vec<DeviceBuffer>,

    // Storage image for RT output
    pub(crate) storage: Image2d,
}

/// What an [`RtGeometry`]'s positions describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPrimitive {
    /// A triangle list: 3 vertices per triangle.
    Triangles,
    /// The axis of the trail tube: a polyline, one vertex per point. Each
    /// segment is one AABB, padded by [`TRAIL_TUBE_MAX_RADIUS`], that
    /// `rt.rint` intersects.
    Tube,
}

/// Geometry for BLAS building.
pub struct RtGeometry {
    /// Packed f32x3 positions (3 floats per vertex).
    pub positions: Vec<f32>,
    /// How the positions form primitives.
    pub primitive: RtPrimitive,
    /// Geometry type ID (GEOM_FLOOR, GEOM_TEE_BOX, etc.).
    pub geom_type: u32,
    /// Instance transform (identity if not set); converted to the 3×4
    /// row-major `VkTransformMatrixKHR` when the TLAS is built.
    pub transform: glam::Mat4,
}

impl RtGeometry {
    /// Number of primitives: triangles, or tube segments.
    #[must_use]
    pub fn primitive_count(&self) -> u32 {
        let vertices = self.positions.len() / 3;
        match self.primitive {
            RtPrimitive::Triangles => (vertices / 3) as u32,
            RtPrimitive::Tube => vertices.saturating_sub(1) as u32,
        }
    }
}

/// The AABB of each segment of the polyline `positions` (packed f32x3), its
/// end points' bounds padded by `pad` on every side, as `VkAabbPositionsKHR`
/// (min xyz, max xyz).
fn tube_aabbs(positions: &[f32], pad: f32) -> Vec<f32> {
    let points: Vec<glam::Vec3> = positions
        .chunks_exact(3)
        .map(glam::Vec3::from_slice)
        .collect();
    let mut aabbs = Vec::with_capacity(points.len().saturating_sub(1) * 6);
    for w in points.windows(2) {
        let min = w[0].min(w[1]) - glam::Vec3::splat(pad);
        let max = w[0].max(w[1]) + glam::Vec3::splat(pad);
        aabbs.extend_from_slice(&[min.x, min.y, min.z, max.x, max.y, max.z]);
    }
    aabbs
}

/// The polyline `positions` (packed f32x3) as `vec4` points (w = 0), the
/// layout of `rt.rint`'s trail point buffer.
fn tube_points(positions: &[f32]) -> Vec<f32> {
    positions
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 0.0])
        .collect()
}

impl RtPipeline {
    /// Create the RT pipeline and build acceleration structures from the given geometry.
    ///
    /// # Errors
    /// A part fails to create or build; what was created is destroyed first.
    pub fn new(
        gpu: &mut GpuContext,
        width: u32,
        height: u32,
        geometries: &[RtGeometry],
    ) -> Result<Self, RenderError> {
        let accel_loader =
            ash::khr::acceleration_structure::Device::new(&gpu.instance, &gpu.device);
        let rt_loader = ash::khr::ray_tracing_pipeline::Device::new(&gpu.instance, &gpu.device);

        // Query RT pipeline properties for SBT alignment
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
        // SAFETY: Querying physical device properties.
        unsafe {
            gpu.instance
                .get_physical_device_properties2(gpu.physical_device, &mut props2);
        }

        let mut objects = PipelineObjects::default();
        let sbt = objects
            .create_pipeline(&gpu.device, &rt_loader)
            .and_then(|()| Self::create_sbt(gpu, &rt_loader, objects.pipeline, &rt_props));
        let (sbt, [raygen_region, miss_region, hit_region]) = match sbt {
            Ok(sbt) => sbt,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe { objects.destroy(&gpu.device) };
                return Err(e);
            }
        };

        let storage = objects.create_descriptor_set(&gpu.device).and_then(|()| {
            Self::create_storage_image(&gpu.device, &mut gpu.allocator, width, height)
        });
        let storage = match storage {
            Ok(storage) => storage,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe {
                    sbt.destroy(&gpu.device, &mut gpu.allocator);
                    objects.destroy(&gpu.device);
                }
                return Err(e);
            }
        };

        let mut pipeline = Self {
            accel_loader,
            rt_loader,
            objects,
            sbt,
            raygen_region,
            miss_region,
            hit_region,
            blas_list: Vec::new(),
            tlas: None,
            geometry_buffers: Vec::new(),
            trail_points: None,
            instance_buffer: None,
            scratch_buffers: Vec::new(),
            storage,
        };
        // SAFETY: The set was just allocated and no command buffer uses it.
        unsafe { pipeline.write_storage_descriptor(&gpu.device) };
        if let Err(e) = pipeline.build_accel_structures(gpu, geometries) {
            // SAFETY: No command buffer has traced with the pipeline. The
            // wait covers a failure inside a build submission.
            unsafe {
                gpu.device.device_wait_idle().ok();
                pipeline.destroy(&gpu.device, &mut gpu.allocator);
            }
            return Err(e);
        }
        Ok(pipeline)
    }

    /// The shader binding table of `pipeline`'s four groups (raygen, miss,
    /// triangle hit, tube hit), each handle at the start of a slot aligned as
    /// `rt_props` requires, and the raygen, miss and hit regions. The hit
    /// region holds both hit groups, indexed by the instances' SBT record
    /// offset ([`HIT_GROUP_TRIANGLES`], [`HIT_GROUP_TUBE`]).
    fn create_sbt(
        gpu: &mut GpuContext,
        rt_loader: &ash::khr::ray_tracing_pipeline::Device,
        pipeline: vk::Pipeline,
        rt_props: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'_>,
    ) -> Result<(HostBuffer, [vk::StridedDeviceAddressRegionKHR; 3]), RenderError> {
        let handle_size = rt_props.shader_group_handle_size;
        let handle_alignment = rt_props.shader_group_handle_alignment;
        let base_alignment = rt_props.shader_group_base_alignment;
        let aligned_handle_size = align_up(handle_size, handle_alignment);
        // Each SBT region's start address must be aligned to shaderGroupBaseAlignment.
        let region_stride = align_up(aligned_handle_size, base_alignment);
        let group_count = 4u32;
        let sbt_size = u64::from(region_stride) * u64::from(group_count);

        // SAFETY: Getting shader group handles.
        let handle_data = unsafe {
            rt_loader
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count,
                    (handle_size * group_count) as usize,
                )
                .map_err(RenderError::Vulkan)?
        };

        // Each handle at a region_stride offset for base alignment; the gaps are zero.
        let mut sbt_data = vec![0u8; sbt_size as usize];
        for i in 0..group_count as usize {
            let src = i * handle_size as usize;
            let dst = i * region_stride as usize;
            sbt_data[dst..dst + handle_size as usize]
                .copy_from_slice(&handle_data[src..src + handle_size as usize]);
        }
        let sbt = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &sbt_data,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "rt sbt",
        )?;

        let sbt_address = sbt.device_address(&gpu.device);
        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address,
            stride: u64::from(region_stride),
            size: u64::from(region_stride),
        };
        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address + u64::from(region_stride),
            stride: u64::from(region_stride),
            size: u64::from(region_stride),
        };
        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address + u64::from(region_stride) * 2,
            stride: u64::from(region_stride),
            size: u64::from(region_stride) * 2,
        };
        Ok((sbt, [raygen_region, miss_region, hit_region]))
    }

    /// Resize the RT storage image to new dimensions.
    ///
    /// Creates a new storage image/view and destroys the old one, then updates
    /// the RT descriptor set (binding 1) to point at the new image.
    /// The device must be idle before calling this.
    pub fn resize_storage(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        if self.storage.extent() == (vk::Extent2D { width, height }) {
            return Ok(());
        }

        let new = Self::create_storage_image(device, allocator, width, height)?;
        let old = std::mem::replace(&mut self.storage, new);
        // SAFETY: Device is idle (caller guarantees).
        unsafe {
            old.destroy(device, allocator);
            self.write_storage_descriptor(device);
        }

        Ok(())
    }

    /// Point the descriptor set's binding 1 at the storage image.
    ///
    /// # Safety
    /// No pending GPU work uses the descriptor set.
    unsafe fn write_storage_descriptor(&self, device: &ash::Device) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(self.storage.view())
            .image_layout(vk::ImageLayout::GENERAL);
        let image_infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.objects.descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos);
        // SAFETY: The image view is live; the caller guarantees the set is unused.
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    /// Record one ray-trace dispatch of `dispatch_w × dispatch_h` rays into
    /// the storage image (smaller than the image for a sub-viewport trace).
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass; the storage image is in
    /// `GENERAL` layout; the dispatch, offset by the push constants' viewport
    /// offset, lies within the storage image.
    pub(crate) unsafe fn record_trace(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        pc: &RtPushConstants,
        dispatch_w: u32,
        dispatch_h: u32,
    ) {
        let callable_region = vk::StridedDeviceAddressRegionKHR::default();

        // SAFETY: The caller guarantees the command buffer and image states.
        unsafe {
            device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.objects.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.objects.pipeline_layout,
                0,
                &[self.objects.descriptor_set],
                &[],
            );

            push_constants(device, cb, self.objects.pipeline_layout, PUSH_STAGES, pc);

            self.rt_loader.cmd_trace_rays(
                cb,
                &self.raygen_region,
                &self.miss_region,
                &self.hit_region,
                &callable_region,
                dispatch_w,
                dispatch_h,
                1,
            );
        }
    }

    // ── BLAS building ──

    /// Build one BLAS per geometry with primitives into `blas_list`, with its
    /// input buffers in `geometry_buffers` (and the tube's points in
    /// `trail_points`) and its scratch buffer in `scratch_buffers`.
    fn build_all_blas(
        &mut self,
        gpu: &mut GpuContext,
        geometries: &[RtGeometry],
    ) -> Result<(), RenderError> {
        let accel_loader = &self.accel_loader;
        for geom in geometries {
            let primitive_count = geom.primitive_count();
            if primitive_count == 0 {
                continue;
            }

            let as_geometry = match geom.primitive {
                RtPrimitive::Triangles => {
                    // Upload positions to a device-address-able buffer
                    let pos_buffer = HostBuffer::with_data(
                        &gpu.device,
                        &mut gpu.allocator,
                        &geom.positions,
                        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        "blas positions",
                    )?;
                    let pos_addr = pos_buffer.device_address(&gpu.device);
                    self.geometry_buffers.push(pos_buffer);

                    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: pos_addr,
                        })
                        .vertex_stride(12) // 3 × f32, tightly packed
                        .max_vertex(primitive_count * 3 - 1)
                        .index_type(vk::IndexType::NONE_KHR);
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                        .flags(vk::GeometryFlagsKHR::OPAQUE)
                }
                RtPrimitive::Tube => {
                    let aabb_buffer = HostBuffer::with_data(
                        &gpu.device,
                        &mut gpu.allocator,
                        &tube_aabbs(&geom.positions, TRAIL_TUBE_MAX_RADIUS),
                        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        "blas tube aabbs",
                    )?;
                    let aabb_addr = aabb_buffer.device_address(&gpu.device);
                    self.geometry_buffers.push(aabb_buffer);

                    let points = HostBuffer::with_data(
                        &gpu.device,
                        &mut gpu.allocator,
                        &tube_points(&geom.positions),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        "rt trail points",
                    )?;
                    if let Some(old) = self.trail_points.replace(points) {
                        // SAFETY: Created by this build; no GPU work uses it.
                        unsafe { old.destroy(&gpu.device, &mut gpu.allocator) };
                    }

                    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: aabb_addr,
                        })
                        .stride(24); // VkAabbPositionsKHR: 6 × f32
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::AABBS)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
                        .flags(vk::GeometryFlagsKHR::OPAQUE)
                }
            };

            let as_geometries = [as_geometry];
            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&as_geometries);

            let primitive_counts = [primitive_count];
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            // SAFETY: Querying build sizes.
            unsafe {
                accel_loader.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &primitive_counts,
                    &mut sizes,
                );
            }

            // Create AS buffer
            let as_buffer = DeviceBuffer::new(
                &gpu.device,
                &mut gpu.allocator,
                sizes.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                "blas",
            )?;

            let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(as_buffer.handle())
                .size(as_buffer.size())
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

            // SAFETY: Creating acceleration structure.
            let accel = match unsafe {
                accel_loader.create_acceleration_structure(&as_create_info, None)
            } {
                Ok(accel) => accel,
                Err(e) => {
                    // SAFETY: Just created; no GPU work has used it.
                    unsafe { as_buffer.destroy(&gpu.device, &mut gpu.allocator) };
                    return Err(RenderError::Vulkan(e));
                }
            };
            self.blas_list.push(AccelStruct {
                accel,
                buffer: as_buffer,
            });

            // Scratch buffer
            let scratch_buffer = DeviceBuffer::new(
                &gpu.device,
                &mut gpu.allocator,
                sizes.build_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                "blas scratch",
            )?;
            let scratch_addr = scratch_buffer.device_address(&gpu.device);
            self.scratch_buffers.push(scratch_buffer);

            // Build BLAS on device
            let build_info_final = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .dst_acceleration_structure(accel)
                .geometries(&as_geometries)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_addr,
                });

            let range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(primitive_count)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);
            let range_infos = [range_info];

            submit_immediate(gpu, |device, cb| {
                // SAFETY: Recording AS build command.
                unsafe {
                    accel_loader.cmd_build_acceleration_structures(
                        cb,
                        &[build_info_final],
                        &[&range_infos],
                    );
                    // Barrier: AS build → AS read
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                        .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
                    device.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
            })?;
        }

        Ok(())
    }

    // ── TLAS building ──

    /// Build the TLAS over `blas_list` into `tlas`, with its instance buffer
    /// in `instance_buffer` and its scratch buffer in `scratch_buffers`.
    /// Returns the TLAS handle.
    fn build_tlas(
        &mut self,
        gpu: &mut GpuContext,
        geometries: &[RtGeometry],
    ) -> Result<vk::AccelerationStructureKHR, RenderError> {
        let accel_loader = &self.accel_loader;
        let mut instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::new();

        let built = geometries.iter().filter(|g| g.primitive_count() > 0);
        for (geom, blas) in built.zip(self.blas_list.iter()) {
            let blas_addr = unsafe {
                accel_loader.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(blas.accel),
                )
            };

            // Convert glam::Mat4 to VkTransformMatrixKHR (3×4 row-major).
            // glam Mat4 is column-major: col[i] = x_axis, y_axis, z_axis, w_axis.
            // VkTransformMatrixKHR is 3 rows × 4 cols, row-major.
            let m = geom.transform;
            let transform = vk::TransformMatrixKHR {
                matrix: [
                    m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x, m.x_axis.y, m.y_axis.y,
                    m.z_axis.y, m.w_axis.y, m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z,
                ],
            };

            // Instance mask bits:
            //   Bit 0 (0x01): floor. Primary rays and trail continuation rays use
            //                 cull mask 0x01.
            //   Bit 1 (0x02): tee box border. No ray uses this bit.
            //   Bit 2 (0x04): trail. Floor reflection rays use cull mask 0x04.
            //   Bit 7 (0x80): tee box fill. No ray uses this bit.
            //
            // Tee box fill is excluded from all RT rays. At Y=0.001 it sits above
            // the floor and would occlude the ball in reflections and steal primary
            // hits from the glass floor shader. It only exists for raster depth.
            let mask = match geom.geom_type {
                GEOM_FLOOR => 0x01,
                GEOM_TEE_BOX => 0x80, // invisible to all RT rays
                GEOM_TRAIL => 0x04,
                _ => 0x02, // tee box border
            };
            let hit_group = match geom.primitive {
                RtPrimitive::Triangles => HIT_GROUP_TRIANGLES,
                RtPrimitive::Tube => HIT_GROUP_TUBE,
            };
            instances.push(vk::AccelerationStructureInstanceKHR {
                transform,
                instance_custom_index_and_mask: vk::Packed24_8::new(geom.geom_type, mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    hit_group,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas_addr,
                },
            });
        }

        let instance_count = instances.len() as u32;
        let instance_buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &instances,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "tlas instances",
        )?;
        let instance_addr = instance_buffer.device_address(&gpu.device);
        self.instance_buffer = Some(instance_buffer);

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_addr,
            });

        let tlas_geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instances_data,
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        let tlas_geometries = [tlas_geometry];
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&tlas_geometries);

        let counts = [instance_count];
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: Querying build sizes.
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &counts,
                &mut sizes,
            );
        }

        let tlas_buffer = DeviceBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "tlas",
        )?;

        let tlas_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(tlas_buffer.handle())
            .size(tlas_buffer.size())
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        // SAFETY: Creating TLAS.
        let tlas_accel =
            match unsafe { accel_loader.create_acceleration_structure(&tlas_create_info, None) } {
                Ok(accel) => accel,
                Err(e) => {
                    // SAFETY: Just created; no GPU work has used it.
                    unsafe { tlas_buffer.destroy(&gpu.device, &mut gpu.allocator) };
                    return Err(RenderError::Vulkan(e));
                }
            };
        self.tlas = Some(AccelStruct {
            accel: tlas_accel,
            buffer: tlas_buffer,
        });

        let scratch_buffer = DeviceBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "tlas scratch",
        )?;
        let scratch_addr = scratch_buffer.device_address(&gpu.device);
        self.scratch_buffers.push(scratch_buffer);

        let build_info_final = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(tlas_accel)
            .geometries(&tlas_geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            });

        let range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instance_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        let range_infos = [range_info];

        submit_immediate(gpu, |device, cb| {
            // SAFETY: Building TLAS.
            unsafe {
                accel_loader.cmd_build_acceleration_structures(
                    cb,
                    &[build_info_final],
                    &[&range_infos],
                );
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        })?;

        Ok(tlas_accel)
    }

    // ── Storage image ──

    fn create_storage_image(
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<Image2d, RenderError> {
        Image2d::new(
            device,
            allocator,
            &Image2dDesc {
                extent: vk::Extent2D { width, height },
                format: STORAGE_FORMAT,
                usage: vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "rt storage image",
            },
        )
    }

    /// Rebuild all acceleration structures with new geometry.
    ///
    /// Waits for device idle, destroys old AS resources, builds new ones,
    /// and updates the TLAS descriptor binding.
    ///
    /// # Errors
    /// The wait or a build fails. After a failed build the old structures
    /// are destroyed and the descriptor set still names the old TLAS, so the
    /// pipeline must not trace again; destroy it, which also frees what the
    /// failed build created.
    pub fn update_scene(
        &mut self,
        gpu: &mut GpuContext,
        geometries: &[RtGeometry],
    ) -> Result<(), RenderError> {
        // SAFETY: Waiting for device idle before destroying AS resources.
        unsafe {
            gpu.device.device_wait_idle().map_err(RenderError::Vulkan)?;
        }

        // SAFETY: Device is idle, AS resources are not in use.
        unsafe {
            self.destroy_accel_structures(&gpu.device, &mut gpu.allocator);
        }

        self.build_accel_structures(gpu, geometries)
    }

    /// Build a BLAS per geometry and the TLAS over them, then point the
    /// descriptor set's binding 0 at the TLAS and binding 2 at the trail
    /// points (one zero point without a trail). Each object is kept in `self`
    /// as soon as it exists, so after a failure
    /// [`destroy_accel_structures`](Self::destroy_accel_structures) frees what
    /// was built.
    ///
    /// Requires no acceleration structures in `self` and the descriptor set
    /// not in use by pending GPU work.
    fn build_accel_structures(
        &mut self,
        gpu: &mut GpuContext,
        geometries: &[RtGeometry],
    ) -> Result<(), RenderError> {
        self.build_all_blas(gpu, geometries)?;
        let tlas_handle = self.build_tlas(gpu, geometries)?;
        let trail_points = match &self.trail_points {
            Some(points) => points.handle(),
            None => {
                let points = HostBuffer::with_data(
                    &gpu.device,
                    &mut gpu.allocator,
                    &[0.0f32; 4],
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    "rt trail points",
                )?;
                self.trail_points.insert(points).handle()
            }
        };

        let mut as_write_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&tlas_handle));
        let buffer_infos = [vk::DescriptorBufferInfo::default()
            .buffer(trail_points)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.objects.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .push_next(&mut as_write_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.objects.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos),
        ];

        // SAFETY: Updating descriptor set with the valid TLAS and point buffer.
        unsafe {
            gpu.device.update_descriptor_sets(&writes, &[]);
        }

        Ok(())
    }

    /// Destroy acceleration structures and their backing resources.
    ///
    /// Leaves pipeline, SBT, descriptors, storage image intact.
    ///
    /// # Safety
    /// Device must be idle. All resources must not be in use.
    unsafe fn destroy_accel_structures(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: Caller guarantees device is idle.
        unsafe {
            if let Some(tlas) = self.tlas.take() {
                self.accel_loader
                    .destroy_acceleration_structure(tlas.accel, None);
                tlas.buffer.destroy(device, allocator);
            }

            if let Some(buffer) = self.instance_buffer.take() {
                buffer.destroy(device, allocator);
            }

            for blas in self.blas_list.drain(..) {
                self.accel_loader
                    .destroy_acceleration_structure(blas.accel, None);
                blas.buffer.destroy(device, allocator);
            }

            for buffer in self.geometry_buffers.drain(..) {
                buffer.destroy(device, allocator);
            }
            if let Some(buffer) = self.trail_points.take() {
                buffer.destroy(device, allocator);
            }
            for buffer in self.scratch_buffers.drain(..) {
                buffer.destroy(device, allocator);
            }
        }
    }

    /// Clean up all Vulkan resources.
    ///
    /// # Safety
    /// Must be called before destroying the device. Device must be idle.
    pub unsafe fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: No resource is in use (caller guarantees device idle).
        unsafe {
            self.destroy_accel_structures(device, allocator);

            // Storage image
            self.storage.destroy(device, allocator);

            // SBT
            self.sbt.destroy(device, allocator);

            // Pipeline, descriptors and shader modules
            self.objects.destroy(device);
        }
    }
}

// ── Helper functions ──

fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Submit a one-shot command buffer and wait for completion.
fn submit_immediate(
    gpu: &GpuContext,
    record: impl FnOnce(&ash::Device, vk::CommandBuffer),
) -> Result<(), RenderError> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(gpu.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    // SAFETY: Allocating + recording + submitting a one-shot command buffer.
    unsafe {
        let cb = gpu
            .device
            .allocate_command_buffers(&alloc_info)
            .map_err(RenderError::Vulkan)?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        gpu.device
            .begin_command_buffer(cb, &begin_info)
            .map_err(RenderError::Vulkan)?;

        record(&gpu.device, cb);

        gpu.device
            .end_command_buffer(cb)
            .map_err(RenderError::Vulkan)?;

        let cbs = [cb];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cbs);
        gpu.device
            .queue_submit(gpu.graphics_queue, &[submit_info], vk::Fence::null())
            .map_err(RenderError::Vulkan)?;
        gpu.device
            .queue_wait_idle(gpu.graphics_queue)
            .map_err(RenderError::Vulkan)?;

        gpu.device.free_command_buffers(gpu.command_pool, &cbs);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constants_size() {
        // The std430 push block in rt.rgen / rt.rchit, within the 128-byte
        // minimum guarantee.
        assert_eq!(std::mem::size_of::<RtPushConstants>(), 112);
    }

    #[test]
    fn push_constants_lanes() {
        let camera = Camera::default();
        let pc = RtPushConstants::new(
            &camera,
            [1920.0, 1080.0],
            [12.0, 34.0],
            glam::Vec3::new(1.0, 2.0, 3.0),
            56.0,
            0.25,
        );
        assert_eq!(
            pc.camera_pos,
            [
                camera.position.x,
                camera.position.y,
                camera.position.z,
                12.0
            ]
        );
        let mut proj = camera.projection_matrix(16.0 / 9.0);
        proj.y_axis.y *= -1.0;
        let inv_view_proj = (proj * camera.view_matrix()).inverse();
        assert_eq!(pc.inv_view_proj, inv_view_proj.to_cols_array_2d());
        assert_eq!(pc.params[0], 0.25);
        assert_eq!(pc.params[2..], [0.0, 34.0]);
        assert_eq!(pc.ball_pos, [1.0, 2.0, 3.0, 56.0]);
    }

    #[test]
    fn pixel_size_is_the_pixel_angle() {
        let camera = Camera::default();
        let pc = RtPushConstants::new(
            &camera,
            [384.0, 1080.0],
            [0.0; 2],
            glam::Vec3::ZERO,
            0.0,
            0.0,
        );
        let expected = 2.0 * (camera.fov_y / 2.0).tan() / 1080.0;
        assert!((pc.params[1] - expected).abs() < 1e-9, "{}", pc.params[1]);
    }

    fn geometry(positions: &[f32], primitive: RtPrimitive) -> RtGeometry {
        RtGeometry {
            positions: positions.to_vec(),
            primitive,
            geom_type: GEOM_TRAIL,
            transform: glam::Mat4::IDENTITY,
        }
    }

    #[test]
    fn primitive_counts() {
        let nine = [0.0; 9];
        assert_eq!(geometry(&nine, RtPrimitive::Triangles).primitive_count(), 1);
        assert_eq!(geometry(&nine, RtPrimitive::Tube).primitive_count(), 2);
        assert_eq!(geometry(&nine[..3], RtPrimitive::Tube).primitive_count(), 0);
        assert_eq!(geometry(&[], RtPrimitive::Tube).primitive_count(), 0);
    }

    #[test]
    fn tube_aabbs_pad_each_segment() {
        let aabbs = tube_aabbs(&[0.0, 1.0, 2.0, 3.0, -1.0, 2.0, 3.0, -1.0, 5.0], 0.5);
        assert_eq!(
            aabbs,
            vec![
                -0.5, -1.5, 1.5, 3.5, 1.5, 2.5, //
                2.5, -1.5, 1.5, 3.5, -0.5, 5.5,
            ]
        );
        assert!(tube_aabbs(&[1.0, 2.0, 3.0], 0.5).is_empty());
    }

    #[test]
    fn tube_points_are_vec4() {
        assert_eq!(
            tube_points(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]
        );
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }
}
