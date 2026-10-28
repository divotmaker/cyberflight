use std::cell::Cell;

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use cf_scene::camera::Camera;
use cf_scene::grid::GridVertex;

use crate::context::GpuContext;
use crate::error::RenderError;
use crate::frame::TracerBuffers;
use crate::gpu::{DeviceBuffer, HostBuffer, device_address, load_shader, push_constants};

/// Push constants for the ray tracing pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RtPushConstants {
    /// Camera position (xyz), w = viewport X offset in pixels.
    pub camera_pos: [f32; 4],
    /// Inverse view-projection matrix (screen → world).
    pub inv_view_proj: [[f32; 4]; 4],
    /// Ball position (xyz), w = viewport Y offset in pixels.
    pub ball_pos: [f32; 4],
    /// Size of one pixel per metre of ray path (the viewport's pixel angle).
    pub pixel_size: f32,
}

impl RtPushConstants {
    /// Pack push constants for one trace dispatch.
    ///
    /// `viewport_size` is the dispatch viewport's width and height and
    /// `viewport_offset` its top-left corner, both in storage-image pixels.
    #[must_use]
    pub fn new(
        camera: &Camera,
        viewport_size: [f32; 2],
        viewport_offset: [f32; 2],
        ball_center: glam::Vec3,
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
            ball_pos: [
                ball_center.x,
                ball_center.y,
                ball_center.z,
                viewport_offset[1],
            ],
            pixel_size,
        }
    }
}

/// Geometry type IDs used as `gl_InstanceCustomIndexEXT` in shaders.
pub const GEOM_FLOOR: u32 = 0;
// The ball has no RT geometry: the floor shader reflects it analytically.
pub const GEOM_TEE_BOX: u32 = 2;
pub const GEOM_TRAIL: u32 = 3;
pub const GEOM_TEE_BOX_BORDER: u32 = 4;
pub const GEOM_TRAIL_LINE: u32 = 5;

/// Largest radius of the tracer's centre line in metres, and the padding of
/// each segment's AABB around its axis. Must equal `LINE_MAX_RADIUS` in
/// `rt_line.rint`.
pub const TRAIL_LINE_MAX_RADIUS: f32 = 1.0;

/// Shader stages that read the push constants.
const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::RAYGEN_KHR.as_raw()
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR.as_raw()
        | vk::ShaderStageFlags::INTERSECTION_KHR.as_raw(),
);

/// Hit group of triangle geometry (`rt.rchit`), the SBT hit record offset of
/// its instances.
const HIT_GROUP_TRIANGLES: u32 = 0;
/// Hit group of the tracer tube's translucent triangles (`rt_trail.rahit`).
const HIT_GROUP_GLOW: u32 = 1;
/// Hit group of the tracer's centre line (`rt_line.rint` + `rt_line.rahit`).
const HIT_GROUP_LINE: u32 = 2;

/// The pipeline's shader modules, layouts, pipeline and descriptor set.
/// Null handles stand for the objects not created yet; destroying a null
/// handle does nothing.
#[derive(Default)]
struct PipelineObjects {
    rgen_module: vk::ShaderModule,
    rmiss_module: vk::ShaderModule,
    rchit_module: vk::ShaderModule,
    trail_rahit_module: vk::ShaderModule,
    line_rint_module: vk::ShaderModule,
    line_rahit_module: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl PipelineObjects {
    /// Create the shader modules, the descriptor set layout (TLAS, storage
    /// image, tracer vertices), the pipeline layout (push constants + set) and
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
        self.trail_rahit_module =
            load_shader(device, include_bytes!("../shaders/rt_trail.rahit.spv"))?;

        self.line_rint_module = load_shader(device, include_bytes!("../shaders/rt_line.rint.spv"))?;
        self.line_rahit_module =
            load_shader(device, include_bytes!("../shaders/rt_line.rahit.spv"))?;

        // ── Descriptor set layout: TLAS, storage image, tracer vertices ──
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
                .stage_flags(vk::ShaderStageFlags::ANY_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::INTERSECTION_KHR | vk::ShaderStageFlags::ANY_HIT_KHR,
                ),
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
                .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
                .module(self.trail_rahit_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
                .module(self.line_rint_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
                .module(self.line_rahit_module)
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
            // Group 3: tracer tube hit (HIT_GROUP_GLOW): any-hit only, it
            // adds the light of every surface the ray crosses.
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(3)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Group 4: tracer centre line hit (HIT_GROUP_LINE): a procedural
            // capsule per segment, any-hit only.
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(5)
                .intersection_shader(4),
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
                .descriptor_count(2),
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
            device.destroy_shader_module(self.trail_rahit_module, None);
            device.destroy_shader_module(self.line_rint_module, None);
            device.destroy_shader_module(self.line_rahit_module, None);
        }
    }
}

/// Bytes between two `GridVertex`es, the stride of the tracer tube's
/// triangle vertices.
const TRACER_VERTEX_STRIDE: u32 = size_of::<GridVertex>() as u32;

/// Bytes of one `VkAabbPositionsKHR` (min xyz, max xyz).
const AABB_STRIDE: u32 = 24;

/// Build flags of the floor and tee box structures, which are built once.
const STATIC_BUILD_FLAGS: vk::BuildAccelerationStructureFlagsKHR =
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;

/// Build flags of the structures rebuilt every frame.
const FRAME_BUILD_FLAGS: vk::BuildAccelerationStructureFlagsKHR =
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD;

/// A bottom-level or top-level structure with its storage buffer.
struct AccelStruct {
    accel: vk::AccelerationStructureKHR,
    buffer: DeviceBuffer,
}

impl AccelStruct {
    /// Create a structure of type `ty` with `size` bytes of storage.
    ///
    /// # Errors
    /// The buffer or the structure fails to create; nothing is left behind.
    fn new(
        loader: &ash::khr::acceleration_structure::Device,
        gpu: &mut GpuContext,
        ty: vk::AccelerationStructureTypeKHR,
        size: vk::DeviceSize,
        name: &'static str,
    ) -> Result<Self, RenderError> {
        let buffer = DeviceBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            name,
        )?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.handle())
            .size(buffer.size())
            .ty(ty);
        // SAFETY: `create_info` names the live buffer just created, whose
        // size it states.
        match unsafe { loader.create_acceleration_structure(&create_info, None) } {
            Ok(accel) => Ok(Self { accel, buffer }),
            Err(e) => {
                // SAFETY: Just created; no GPU work has used it.
                unsafe { buffer.destroy(&gpu.device, &mut gpu.allocator) };
                Err(RenderError::Vulkan(e))
            }
        }
    }

    /// The structure's device address, which instances refer to.
    fn address(&self, loader: &ash::khr::acceleration_structure::Device) -> vk::DeviceAddress {
        // SAFETY: `self.accel` is a live structure of `loader`'s device.
        unsafe {
            loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(self.accel),
            )
        }
    }

    /// # Safety
    /// No pending GPU work uses the structure; `loader`, `device` and
    /// `allocator` are the ones it was created with.
    unsafe fn destroy(
        self,
        loader: &ash::khr::acceleration_structure::Device,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) {
        // SAFETY: The caller guarantees nothing uses the structure.
        unsafe {
            loader.destroy_acceleration_structure(self.accel, None);
            self.buffer.destroy(device, allocator);
        }
    }
}

/// The sizes a build of `geometry` needs for up to `max_primitives`
/// primitives.
fn build_sizes(
    loader: &ash::khr::acceleration_structure::Device,
    ty: vk::AccelerationStructureTypeKHR,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    geometry: &vk::AccelerationStructureGeometryKHR<'_>,
    max_primitives: u32,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let geometries = [*geometry];
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(flags)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: Querying the build sizes of a well-formed build description.
    unsafe {
        loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[max_primitives],
            &mut sizes,
        );
    }
    sizes
}

/// A structure that is rebuilt from the same geometry description again and
/// again: the structure, its scratch buffer and how to build it.
struct Rebuilt {
    accel: AccelStruct,
    scratch: DeviceBuffer,
    scratch_address: vk::DeviceAddress,
    ty: vk::AccelerationStructureTypeKHR,
    geometry: vk::AccelerationStructureGeometryKHR<'static>,
    /// The most primitives a build may have.
    max_primitives: u32,
}

impl Rebuilt {
    /// Create a structure of type `ty` for builds of up to `max_primitives`
    /// primitives of `geometry`, and the scratch buffer they use.
    ///
    /// # Errors
    /// A buffer or the structure fails to create; nothing is left behind.
    fn new(
        loader: &ash::khr::acceleration_structure::Device,
        gpu: &mut GpuContext,
        ty: vk::AccelerationStructureTypeKHR,
        geometry: vk::AccelerationStructureGeometryKHR<'static>,
        max_primitives: u32,
        names: (&'static str, &'static str),
    ) -> Result<Self, RenderError> {
        let sizes = build_sizes(loader, ty, FRAME_BUILD_FLAGS, &geometry, max_primitives);
        let accel = AccelStruct::new(loader, gpu, ty, sizes.acceleration_structure_size, names.0)?;
        let scratch = match DeviceBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            names.1,
        ) {
            Ok(scratch) => scratch,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used it.
                unsafe { accel.destroy(loader, &gpu.device, &mut gpu.allocator) };
                return Err(e);
            }
        };
        let scratch_address = scratch.device_address(&gpu.device);
        Ok(Self {
            accel,
            scratch,
            scratch_address,
            ty,
            geometry,
            max_primitives,
        })
    }

    /// Record a build of `primitives` primitives (at most `max_primitives`).
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass; the buffers the geometry
    /// addresses are live and hold the primitives; a barrier orders every
    /// earlier use of the structure and its scratch buffer before this
    /// build.
    unsafe fn record_build(
        &self,
        loader: &ash::khr::acceleration_structure::Device,
        cb: vk::CommandBuffer,
        primitives: u32,
    ) {
        let geometries = [self.geometry];
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(self.ty)
            .flags(FRAME_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.accel.accel)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.scratch_address,
            });
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(primitives.min(self.max_primitives));
        // SAFETY: The caller guarantees the command buffer state, the
        // geometry's buffers and the ordering; the range stays within the
        // capacity the structure and scratch were sized for.
        unsafe { loader.cmd_build_acceleration_structures(cb, &[build_info], &[&[range]]) };
    }

    /// # Safety
    /// As [`AccelStruct::destroy`].
    unsafe fn destroy(
        self,
        loader: &ash::khr::acceleration_structure::Device,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) {
        // SAFETY: The caller guarantees nothing uses them.
        unsafe {
            self.accel.destroy(loader, device, allocator);
            self.scratch.destroy(device, allocator);
        }
    }
}

/// The tracer tube's geometry: the first vertices of the `GridVertex` buffer
/// at `vertices`, which holds up to `capacity` of them, as triangles whose
/// any-hit runs once per triangle.
fn tube_geometry(
    vertices: vk::DeviceAddress,
    capacity: u32,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertices,
        })
        .vertex_stride(u64::from(TRACER_VERTEX_STRIDE))
        .max_vertex(capacity.saturating_sub(1))
        .index_type(vk::IndexType::NONE_KHR);
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION)
}

/// The tracer's centre line geometry: one AABB per segment, from the
/// `VkAabbPositionsKHR` array at `aabbs`.
fn line_geometry(aabbs: vk::DeviceAddress) -> vk::AccelerationStructureGeometryKHR<'static> {
    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: aabbs,
        })
        .stride(u64::from(AABB_STRIDE));
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::AABBS)
        .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
        .flags(vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION)
}

/// `triangles` opaque triangles of tightly packed `f32` x3 positions at
/// `vertices`.
fn static_geometry(
    vertices: vk::DeviceAddress,
    triangles: u32,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    let data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertices,
        })
        .vertex_stride(12)
        .max_vertex((triangles * 3).saturating_sub(1))
        .index_type(vk::IndexType::NONE_KHR);
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: data })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

/// The instance array at `instances` as the top-level structure's geometry.
fn instances_geometry(
    instances: vk::DeviceAddress,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    let data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instances,
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

/// The tracer's bottom-level structures. They are sized for the frame
/// buffers they are built from (the largest tracer a buffer holds) and
/// rebuilt in each frame's command buffer.
struct TracerAccel {
    tube: Rebuilt,
    line: Rebuilt,
}

impl TracerAccel {
    /// Create the structures for builds from `buffers`.
    ///
    /// # Errors
    /// A buffer or a structure fails to create; nothing is left behind.
    fn new(
        loader: &ash::khr::acceleration_structure::Device,
        gpu: &mut GpuContext,
        buffers: &TracerBuffers,
    ) -> Result<Self, RenderError> {
        let tube_vertices = device_address(&gpu.device, buffers.tube);
        let aabbs = device_address(&gpu.device, buffers.line_aabbs);
        let tube = Rebuilt::new(
            loader,
            gpu,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            tube_geometry(tube_vertices, buffers.tube_capacity),
            buffers.tube_capacity / 3,
            ("tracer tube blas", "tracer tube scratch"),
        )?;
        let line = match Rebuilt::new(
            loader,
            gpu,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            line_geometry(aabbs),
            buffers.line_capacity / 2,
            ("tracer line blas", "tracer line scratch"),
        ) {
            Ok(line) => line,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used it.
                unsafe { tube.destroy(loader, &gpu.device, &mut gpu.allocator) };
                return Err(e);
            }
        };
        Ok(Self { tube, line })
    }

    /// # Safety
    /// As [`AccelStruct::destroy`].
    unsafe fn destroy(
        self,
        loader: &ash::khr::acceleration_structure::Device,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) {
        // SAFETY: The caller guarantees nothing uses them.
        unsafe {
            self.tube.destroy(loader, device, allocator);
            self.line.destroy(loader, device, allocator);
        }
    }
}

/// A bottom-level structure of opaque triangles, built once, with its
/// vertices.
struct StaticBlas {
    accel: AccelStruct,
    vertices: HostBuffer,
    /// Scratch of the build; freed once the build completed.
    scratch: Option<DeviceBuffer>,
    geometry: vk::AccelerationStructureGeometryKHR<'static>,
    triangles: u32,
    geom_type: u32,
    transform: glam::Mat4,
}

impl StaticBlas {
    /// Create the structure for `geom` and upload its vertices; the build is
    /// recorded separately.
    ///
    /// # Errors
    /// A buffer or the structure fails to create; nothing is left behind.
    fn new(
        loader: &ash::khr::acceleration_structure::Device,
        gpu: &mut GpuContext,
        geom: &RtGeometry,
    ) -> Result<Self, RenderError> {
        let triangles = geom.primitive_count();
        let vertices = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &geom.positions,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "blas positions",
        )?;
        let geometry = static_geometry(vertices.device_address(&gpu.device), triangles);
        let ty = vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL;
        let sizes = build_sizes(loader, ty, STATIC_BUILD_FLAGS, &geometry, triangles);
        let accel =
            match AccelStruct::new(loader, gpu, ty, sizes.acceleration_structure_size, "blas") {
                Ok(accel) => accel,
                Err(e) => {
                    // SAFETY: Just created; no GPU work has used it.
                    unsafe { vertices.destroy(&gpu.device, &mut gpu.allocator) };
                    return Err(e);
                }
            };
        let scratch = match DeviceBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "blas scratch",
        ) {
            Ok(scratch) => scratch,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe {
                    accel.destroy(loader, &gpu.device, &mut gpu.allocator);
                    vertices.destroy(&gpu.device, &mut gpu.allocator);
                }
                return Err(e);
            }
        };
        Ok(Self {
            accel,
            vertices,
            scratch: Some(scratch),
            geometry,
            triangles,
            geom_type: geom.geom_type,
            transform: geom.transform,
        })
    }

    /// Record the build, which needs the scratch buffer.
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass.
    unsafe fn record_build(
        &self,
        loader: &ash::khr::acceleration_structure::Device,
        device: &ash::Device,
        cb: vk::CommandBuffer,
    ) {
        let Some(scratch) = &self.scratch else {
            return;
        };
        let geometries = [self.geometry];
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(STATIC_BUILD_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.accel.accel)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch.device_address(device),
            });
        let range =
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(self.triangles);
        // SAFETY: The caller guarantees the command buffer state; the
        // vertices, structure and scratch buffer live in `self`.
        unsafe { loader.cmd_build_acceleration_structures(cb, &[build_info], &[&[range]]) };
    }

    /// # Safety
    /// As [`AccelStruct::destroy`].
    unsafe fn destroy(
        self,
        loader: &ash::khr::acceleration_structure::Device,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) {
        // SAFETY: The caller guarantees nothing uses them.
        unsafe {
            self.accel.destroy(loader, device, allocator);
            self.vertices.destroy(device, allocator);
            if let Some(scratch) = self.scratch {
                scratch.destroy(device, allocator);
            }
        }
    }
}

/// The scene the tracer is traced in: the static bottom-level structures,
/// the instances (those, then the tracer's tube and centre line) and the
/// top-level structure over them, which is rebuilt every frame.
///
/// Each part is stored as soon as it exists, so after a failure
/// [`destroy`](Self::destroy) frees what was built.
#[derive(Default)]
struct SceneAccel {
    statics: Vec<StaticBlas>,
    instances: Option<HostBuffer>,
    instance_count: u32,
    tlas: Option<Rebuilt>,
}

impl SceneAccel {
    /// Build the static structures of `geometries` (one submission, which
    /// completes before this returns), the instances of those and `tracer`,
    /// and create the top-level structure over them.
    ///
    /// # Errors
    /// A buffer or structure fails to create or the build submission fails;
    /// what exists stays in `self`.
    fn build(
        &mut self,
        loader: &ash::khr::acceleration_structure::Device,
        gpu: &mut GpuContext,
        geometries: &[RtGeometry],
        tracer: &TracerAccel,
    ) -> Result<(), RenderError> {
        for geom in geometries.iter().filter(|g| g.primitive_count() > 0) {
            let blas = StaticBlas::new(loader, gpu, geom)?;
            self.statics.push(blas);
        }
        let statics = &self.statics;
        submit_immediate(gpu, |device, cb| {
            for blas in statics {
                // SAFETY: `cb` is recording outside a render pass.
                unsafe { blas.record_build(loader, device, cb) };
            }
        })?;
        for blas in &mut self.statics {
            if let Some(scratch) = blas.scratch.take() {
                // SAFETY: The submission completed.
                unsafe { scratch.destroy(&gpu.device, &mut gpu.allocator) };
            }
        }

        let mut instances: Vec<vk::AccelerationStructureInstanceKHR> = self
            .statics
            .iter()
            .map(|blas| {
                instance(
                    blas.accel.address(loader),
                    blas.geom_type,
                    HIT_GROUP_TRIANGLES,
                    blas.transform,
                )
            })
            .collect();
        instances.push(instance(
            tracer.tube.accel.address(loader),
            GEOM_TRAIL,
            HIT_GROUP_GLOW,
            glam::Mat4::IDENTITY,
        ));
        instances.push(instance(
            tracer.line.accel.address(loader),
            GEOM_TRAIL_LINE,
            HIT_GROUP_LINE,
            glam::Mat4::IDENTITY,
        ));
        self.instance_count = instances.len() as u32;
        let buffer = HostBuffer::with_data(
            &gpu.device,
            &mut gpu.allocator,
            &instances,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "tlas instances",
        )?;
        let address = buffer.device_address(&gpu.device);
        self.instances = Some(buffer);
        self.tlas = Some(Rebuilt::new(
            loader,
            gpu,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            instances_geometry(address),
            self.instance_count,
            ("tlas", "tlas scratch"),
        )?);
        Ok(())
    }

    /// # Safety
    /// As [`AccelStruct::destroy`].
    unsafe fn destroy(
        self,
        loader: &ash::khr::acceleration_structure::Device,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) {
        // SAFETY: The caller guarantees nothing uses them.
        unsafe {
            if let Some(tlas) = self.tlas {
                tlas.destroy(loader, device, allocator);
            }
            if let Some(instances) = self.instances {
                instances.destroy(device, allocator);
            }
            for blas in self.statics {
                blas.destroy(loader, device, allocator);
            }
        }
    }
}

/// An instance of the structure at `blas` for geometry type `geom_type`,
/// hit with the SBT record offset `hit_group`.
fn instance(
    blas: vk::DeviceAddress,
    geom_type: u32,
    hit_group: u32,
    transform: glam::Mat4,
) -> vk::AccelerationStructureInstanceKHR {
    vk::AccelerationStructureInstanceKHR {
        transform: transform_matrix(transform),
        instance_custom_index_and_mask: vk::Packed24_8::new(geom_type, instance_mask(geom_type)),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            hit_group,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas,
        },
    }
}

/// `m` as a `VkTransformMatrixKHR`: 3 rows of 4, row-major, where glam
/// stores columns.
fn transform_matrix(m: glam::Mat4) -> vk::TransformMatrixKHR {
    vk::TransformMatrixKHR {
        matrix: [
            m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x, m.x_axis.y, m.y_axis.y, m.z_axis.y,
            m.w_axis.y, m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z,
        ],
    }
}

/// The instance mask of geometry type `geom_type`:
///
/// - Bit 0 (0x01): floor. Primary rays use cull mask 0x01.
/// - Bit 1 (0x02): tee box border. No ray uses this bit.
/// - Bit 2 (0x04): tracer tube and centre line. Floor reflection rays use
///   cull mask 0x04.
/// - Bit 7 (0x80): tee box fill. No ray uses this bit.
///
/// The tee box fill is excluded from all RT rays. At Y=0.001 it sits above
/// the floor and would occlude the ball in reflections and steal primary hits
/// from the glass floor shader. It only exists for raster depth.
fn instance_mask(geom_type: u32) -> u8 {
    match geom_type {
        GEOM_FLOOR => 0x01,
        GEOM_TEE_BOX => 0x80,
        GEOM_TRAIL | GEOM_TRAIL_LINE => 0x04,
        _ => 0x02,
    }
}

/// The AABB of each segment of `vertices` (pairs), its end points' bounds
/// padded by [`TRAIL_LINE_MAX_RADIUS`] on every side, as `VkAabbPositionsKHR`
/// (min xyz, max xyz). A trailing unpaired vertex has none.
pub(crate) fn line_aabbs(vertices: &[GridVertex]) -> Vec<f32> {
    let pad = glam::Vec3::splat(TRAIL_LINE_MAX_RADIUS);
    let mut aabbs = Vec::with_capacity(vertices.len() / 2 * 6);
    for seg in vertices.chunks_exact(2) {
        let a = glam::Vec3::from(seg[0].position);
        let b = glam::Vec3::from(seg[1].position);
        let (min, max) = (a.min(b) - pad, a.max(b) + pad);
        aabbs.extend_from_slice(&[min.x, min.y, min.z, max.x, max.y, max.z]);
    }
    aabbs
}

/// Ray tracing pipeline: shaders, SBT, descriptors, and acceleration
/// structures.
///
/// The floor and tee box are bottom-level structures built once. The
/// tracer's tube and centre line are rebuilt every frame in the frame's
/// command buffer ([`RtPipeline::record_build`]) from the frame's own vertex
/// buffers, with the top-level structure over everything.
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
    tracer: TracerAccel,
    /// `None` until [`Self::set_static_geometry`] built it.
    scene: Option<SceneAccel>,
    /// The structures were last built with no tracer in them.
    tracer_empty: Cell<bool>,
}

/// Geometry of the static scene: opaque triangles of one instance.
pub struct RtGeometry {
    /// Packed vertices of 3 floats each, 3 vertices per triangle.
    pub positions: Vec<f32>,
    /// Geometry type ID (GEOM_FLOOR, GEOM_TEE_BOX, etc.).
    pub geom_type: u32,
    /// Instance transform (identity if not set); converted to the 3×4
    /// row-major `VkTransformMatrixKHR` for the instance.
    pub transform: glam::Mat4,
}

impl RtGeometry {
    /// Number of triangles.
    #[must_use]
    pub fn primitive_count(&self) -> u32 {
        (self.positions.len() / 9) as u32
    }
}

impl RtPipeline {
    /// Create the RT pipeline, writing `storage_view` (the reflection image,
    /// `rgba16f`, `STORAGE` usage), tracing `geometries` and the tracer in
    /// `tracer`'s buffers.
    ///
    /// # Errors
    /// A part fails to create or build; what was created is destroyed first.
    pub(crate) fn new(
        gpu: &mut GpuContext,
        storage_view: vk::ImageView,
        geometries: &[RtGeometry],
        tracer: &TracerBuffers,
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

        let created = objects
            .create_descriptor_set(&gpu.device)
            .and_then(|()| TracerAccel::new(&accel_loader, gpu, tracer));
        let tracer_accel = match created {
            Ok(tracer_accel) => tracer_accel,
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
            tracer: tracer_accel,
            scene: None,
            tracer_empty: Cell::new(false),
        };
        // SAFETY: The set was just allocated and no command buffer uses it.
        unsafe {
            pipeline.bind_storage(&gpu.device, storage_view);
            pipeline.bind_tracer_buffers(&gpu.device, tracer);
        }
        if let Err(e) = pipeline.set_static_geometry(gpu, geometries) {
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
    /// The shader binding table of `pipeline`'s five groups (raygen, miss,
    /// triangle hit, tracer tube hit, tracer line hit), each handle at the
    /// start of a slot aligned as `rt_props` requires, and the raygen, miss
    /// and hit regions. The hit region holds the three hit groups, indexed by
    /// the instances' SBT record offset ([`HIT_GROUP_TRIANGLES`],
    /// [`HIT_GROUP_GLOW`], [`HIT_GROUP_LINE`]).
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
        let group_count = 5u32;
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
            size: u64::from(region_stride) * 3,
        };
        Ok((sbt, [raygen_region, miss_region, hit_region]))
    }

    /// Point the descriptor set's binding 1 at `storage_view`, e.g. after
    /// the reflection image was recreated at a new size.
    ///
    /// # Safety
    /// No pending GPU work uses the descriptor set; `storage_view` is a live
    /// view of an `rgba16f` image with `STORAGE` usage.
    pub(crate) unsafe fn bind_storage(&self, device: &ash::Device, storage_view: vk::ImageView) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(storage_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let image_infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.objects.descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos);
        // SAFETY: The caller guarantees the view is live and the set unused.
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    /// Record one ray-trace dispatch of `dispatch_w × dispatch_h` rays into
    /// the bound storage image (smaller than the image for a sub-viewport
    /// trace).
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

    // ── Acceleration structures ──

    /// Build the structures of `geometries`, once, and the instances and
    /// top-level structure over them and the tracer, replacing the ones
    /// built before. Waits for the device to go idle first.
    ///
    /// # Errors
    /// The wait or a build fails. The old structures are destroyed by then
    /// and the descriptor set may name a destroyed top-level structure, so
    /// the pipeline must not trace again; destroy it, which also frees what
    /// the failed build created.
    pub(crate) fn set_static_geometry(
        &mut self,
        gpu: &mut GpuContext,
        geometries: &[RtGeometry],
    ) -> Result<(), RenderError> {
        // SAFETY: Waiting for device idle before destroying AS resources.
        unsafe {
            gpu.device.device_wait_idle().map_err(RenderError::Vulkan)?;
        }
        if let Some(old) = self.scene.take() {
            // SAFETY: The device is idle, so nothing uses the old scene.
            unsafe { old.destroy(&self.accel_loader, &gpu.device, &mut gpu.allocator) };
        }
        let scene = self.scene.insert(SceneAccel::default());
        scene.build(&self.accel_loader, gpu, geometries, &self.tracer)?;
        let Some(tlas) = &scene.tlas else {
            return Ok(());
        };
        let mut as_write_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&tlas.accel.accel));
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.objects.descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .push_next(&mut as_write_info);
        // SAFETY: The device is idle, so no command buffer uses the set; the
        // structure is live.
        unsafe { gpu.device.update_descriptor_sets(&[write], &[]) };
        // The new top-level structure is not built yet.
        self.tracer_empty.set(false);
        Ok(())
    }

    /// Point the descriptor set's bindings 2 and 3 at the frame buffers the
    /// tracer's tube and centre line are read from.
    ///
    /// # Safety
    /// No pending GPU work uses the descriptor set; the buffers are live and
    /// outlive the pipeline's use of the set.
    unsafe fn bind_tracer_buffers(&self, device: &ash::Device, buffers: &TracerBuffers) {
        let tube = [vk::DescriptorBufferInfo::default()
            .buffer(buffers.tube)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let line = [vk::DescriptorBufferInfo::default()
            .buffer(buffers.line)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.objects.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&tube),
            vk::WriteDescriptorSet::default()
                .dst_set(self.objects.descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&line),
        ];
        // SAFETY: The caller guarantees the set is unused and the buffers live.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Record the rebuild of the tracer's structures from the first
    /// `tube_vertices` vertices of the frame's glow buffer and the first
    /// `line_vertices` of its line buffer, and of the top-level structure,
    /// so the next trace sees them. Does nothing while the structures
    /// already hold no tracer and there is still none.
    ///
    /// The builds are ordered after every earlier trace and build, so frames
    /// in flight share the structures.
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass. The frame buffers are live
    /// and hold those vertices, and the line AABB buffer the segments' bounds
    /// ([`line_aabbs`]). The command buffer is submitted before the next
    /// call that skips the rebuild.
    pub(crate) unsafe fn record_build(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        tube_vertices: u32,
        line_vertices: u32,
    ) {
        let empty = tube_vertices == 0 && line_vertices == 0;
        let Some(scene) = &self.scene else {
            return;
        };
        let Some(tlas) = &scene.tlas else {
            return;
        };
        if empty && self.tracer_empty.get() {
            return;
        }
        let loader = &self.accel_loader;
        let build = vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR;
        let read = vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR;
        let write = vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR;

        // SAFETY: The caller guarantees the command buffer state and the
        // buffers the builds read; the structures live in `self`, and the
        // barriers order each build after the work that used what it writes.
        unsafe {
            // Earlier frames' traces read the structures and their builds
            // wrote them and the scratch buffers.
            memory_barrier(
                device,
                cb,
                build | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                read | write,
                build,
                read | write,
            );
            self.tracer.tube.record_build(loader, cb, tube_vertices / 3);
            self.tracer.line.record_build(loader, cb, line_vertices / 2);
            // The top-level build reads the bottom-level structures.
            memory_barrier(device, cb, build, write, build, read);
            tlas.record_build(loader, cb, scene.instance_count);
            // The traces read the top-level structure.
            memory_barrier(
                device,
                cb,
                build,
                write,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                read,
            );
        }
        self.tracer_empty.set(empty);
    }

    /// Destroy acceleration structures and their backing resources.
    ///
    /// Leaves pipeline, SBT and descriptors intact.
    ///
    /// # Safety
    /// Device must be idle. All resources must not be in use.
    unsafe fn destroy_accel_structures(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: Caller guarantees device is idle.
        unsafe {
            if let Some(scene) = self.scene.take() {
                scene.destroy(&self.accel_loader, device, allocator);
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
            self.tracer.destroy(&self.accel_loader, device, allocator);

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

/// Record a global memory barrier.
///
/// # Safety
/// `cb` is recording, outside a render pass.
unsafe fn memory_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    // SAFETY: The caller guarantees the command buffer state.
    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
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
        assert_eq!(std::mem::size_of::<RtPushConstants>(), 100);
    }

    #[test]
    fn push_constants_lanes() {
        let camera = Camera::default();
        let pc = RtPushConstants::new(
            &camera,
            [1920.0, 1080.0],
            [12.0, 34.0],
            glam::Vec3::new(1.0, 2.0, 3.0),
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
        assert_eq!(pc.ball_pos, [1.0, 2.0, 3.0, 34.0]);
    }

    #[test]
    fn pixel_size_is_the_pixel_angle() {
        let camera = Camera::default();
        let pc = RtPushConstants::new(&camera, [384.0, 1080.0], [0.0; 2], glam::Vec3::ZERO);
        let expected = 2.0 * (camera.fov_y / 2.0).tan() / 1080.0;
        assert!((pc.pixel_size - expected).abs() < 1e-9, "{}", pc.pixel_size);
    }

    fn geometry(positions: &[f32]) -> RtGeometry {
        RtGeometry {
            positions: positions.to_vec(),
            geom_type: GEOM_FLOOR,
            transform: glam::Mat4::IDENTITY,
        }
    }

    #[test]
    fn primitive_counts() {
        assert_eq!(geometry(&[0.0; 9]).primitive_count(), 1);
        assert_eq!(geometry(&[0.0; 18]).primitive_count(), 2);
        assert_eq!(geometry(&[]).primitive_count(), 0);
    }

    fn vertex(x: f32, y: f32, z: f32) -> GridVertex {
        GridVertex {
            position: [x, y, z],
            fade: 0.5,
        }
    }

    #[test]
    fn line_aabbs_pad_each_segment() {
        let pad = TRAIL_LINE_MAX_RADIUS;
        let aabbs = line_aabbs(&[
            vertex(0.0, 1.0, 2.0),
            vertex(3.0, -1.0, 2.0),
            vertex(3.0, -1.0, 2.0),
            vertex(3.0, -1.0, 5.0),
        ]);
        assert_eq!(
            aabbs,
            vec![
                -pad,
                -1.0 - pad,
                2.0 - pad,
                3.0 + pad,
                1.0 + pad,
                2.0 + pad,
                3.0 - pad,
                -1.0 - pad,
                2.0 - pad,
                3.0 + pad,
                -1.0 + pad,
                5.0 + pad,
            ]
        );
        assert!(line_aabbs(&[vertex(1.0, 2.0, 3.0)]).is_empty());
    }

    #[test]
    fn masks_separate_what_each_ray_sees() {
        assert_eq!(instance_mask(GEOM_FLOOR), 0x01);
        assert_eq!(instance_mask(GEOM_TRAIL), 0x04);
        assert_eq!(instance_mask(GEOM_TRAIL_LINE), 0x04);
        assert_eq!(instance_mask(GEOM_TEE_BOX), 0x80);
        assert_eq!(instance_mask(GEOM_TEE_BOX_BORDER), 0x02);
    }

    #[test]
    fn transforms_are_row_major_3x4() {
        let m = glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let t = transform_matrix(m).matrix;
        assert_eq!(t, [1., 0., 0., 1., 0., 1., 0., 2., 0., 0., 1., 3.]);
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }
}
