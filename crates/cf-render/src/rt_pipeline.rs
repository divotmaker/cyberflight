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
    /// x = tracer fade value at the trail's oldest point, y/z = unused (0),
    /// w = viewport Y offset in pixels.
    pub params: [f32; 4],
    /// Ball position (xyz), w = trail fade distance in meters.
    pub ball_pos: [f32; 4],
}

impl RtPushConstants {
    /// Pack push constants for one trace dispatch.
    ///
    /// `aspect` is the dispatch viewport's width / height and `viewport_offset`
    /// its top-left corner in storage-image pixels. `trail_fade_dist` is the
    /// trail reflection fade distance in meters (0 selects the shader default)
    /// and `trail_tail_t` the tracer fade value at the trail's oldest point
    /// (see [`RtTrail`](crate::rt_offscreen::RtTrail)).
    #[must_use]
    pub fn new(
        camera: &Camera,
        aspect: f32,
        viewport_offset: [f32; 2],
        ball_center: glam::Vec3,
        trail_fade_dist: f32,
        trail_tail_t: f32,
    ) -> Self {
        let view = camera.view_matrix();
        let mut proj = camera.projection_matrix(aspect);
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
            params: [trail_tail_t, 0.0, 0.0, viewport_offset[1]],
            ball_pos: [ball_center.x, ball_center.y, ball_center.z, trail_fade_dist],
        }
    }
}

/// Format of the RT reflection storage image (`rgba16f` in `rt.rgen`).
///
/// It holds linear colour in [0, 1] that the composite pass samples (nearest)
/// and blends into an sRGB target, so it needs more precision than 8 bits for
/// dark tones. Vulkan requires storage and sampled support for this format.
const STORAGE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Geometry type IDs used as `gl_InstanceCustomIndexEXT` in shaders.
pub const GEOM_FLOOR: u32 = 0;
// The ball has no RT geometry: the floor shader reflects it analytically.
pub const GEOM_TEE_BOX: u32 = 2;
pub const GEOM_TRAIL: u32 = 3;
pub const GEOM_TEE_BOX_BORDER: u32 = 4;

/// A single acceleration structure (BLAS or TLAS) with its backing buffer.
struct AccelStruct {
    accel: vk::AccelerationStructureKHR,
    buffer: DeviceBuffer,
}

/// Ray tracing pipeline: shaders, SBT, descriptors, and acceleration structures.
pub struct RtPipeline {
    accel_loader: ash::khr::acceleration_structure::Device,
    rt_loader: ash::khr::ray_tracing_pipeline::Device,

    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,

    // Shader binding table
    sbt: HostBuffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,

    // Shader modules
    rgen_module: vk::ShaderModule,
    rmiss_module: vk::ShaderModule,
    rchit_module: vk::ShaderModule,

    // Acceleration structures
    blas_list: Vec<AccelStruct>,
    tlas: Option<AccelStruct>,

    // Geometry buffers (kept alive for AS references)
    geometry_buffers: Vec<HostBuffer>,
    // Instance buffer
    instance_buffer: Option<HostBuffer>,

    // Scratch buffers of the current BLAS/TLAS build, freed with the AS on the
    // next `update_scene` or on drop.
    scratch_buffers: Vec<DeviceBuffer>,

    // Storage image for RT output
    pub(crate) storage: Image2d,
}

/// Triangle geometry for BLAS building.
pub struct RtGeometry {
    /// Packed f32x3 positions (3 floats per vertex, 3 vertices per triangle).
    pub positions: Vec<f32>,
    /// Geometry type ID (GEOM_FLOOR, GEOM_TEE_BOX, etc.).
    pub geom_type: u32,
    /// Instance transform (identity if not set); converted to the 3×4
    /// row-major `VkTransformMatrixKHR` when the TLAS is built.
    pub transform: glam::Mat4,
}

impl RtPipeline {
    /// Create the RT pipeline and build acceleration structures from the given geometry.
    #[allow(clippy::too_many_lines)]
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
        let handle_size = rt_props.shader_group_handle_size;
        let handle_alignment = rt_props.shader_group_handle_alignment;
        let base_alignment = rt_props.shader_group_base_alignment;

        // ── Shaders ──
        let rgen_module = load_shader(&gpu.device, include_bytes!("../shaders/rt.rgen.spv"))?;
        let rmiss_module = load_shader(&gpu.device, include_bytes!("../shaders/rt.rmiss.spv"))?;
        let rchit_module = load_shader(&gpu.device, include_bytes!("../shaders/rt.rchit.spv"))?;

        // ── Descriptor set layout: TLAS + storage image ──
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
        ];
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Creating descriptor set layout.
        let descriptor_set_layout = unsafe {
            gpu.device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // ── Pipeline layout (push constants + descriptor set) ──
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .offset(0)
            .size(std::mem::size_of::<RtPushConstants>() as u32);
        let push_ranges = [push_range];
        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: Creating pipeline layout.
        let pipeline_layout = unsafe {
            gpu.device
                .create_pipeline_layout(&layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // ── RT pipeline ──
        let entry_point = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(rgen_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(rmiss_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(rchit_module)
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
            // Group 2: closest hit
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        let rt_create_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(2) // primary + 1 reflection
            .layout(pipeline_layout);

        // SAFETY: Creating ray tracing pipeline.
        let pipeline = unsafe {
            rt_loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[rt_create_info],
                    None,
                )
                .map_err(|(_pipelines, err)| RenderError::Vulkan(err))?[0]
        };

        // ── Shader binding table ──
        let aligned_handle_size = align_up(handle_size, handle_alignment);
        // Each SBT region's start address must be aligned to shaderGroupBaseAlignment.
        let region_stride = align_up(aligned_handle_size, base_alignment);
        let group_count = 3u32;
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
            size: u64::from(region_stride),
        };

        // ── Descriptor pool + set ──
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: Creating descriptor pool.
        let descriptor_pool = unsafe {
            gpu.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(RenderError::Vulkan)?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        // SAFETY: Allocating descriptor set.
        let descriptor_set = unsafe {
            gpu.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(RenderError::Vulkan)?[0]
        };

        // ── Storage image ──
        let storage = Self::create_storage_image(&gpu.device, &mut gpu.allocator, width, height)?;

        // ── Build acceleration structures ──
        let (blas_list, geometry_buffers, scratch_buffers) =
            Self::build_all_blas(gpu, &accel_loader, geometries)?;

        let (tlas, instance_buffer, tlas_scratch) =
            Self::build_tlas(gpu, &accel_loader, geometries, &blas_list)?;

        let mut all_scratch = scratch_buffers;
        all_scratch.push(tlas_scratch);

        // ── Update descriptor set ──
        let tlas_handle = tlas.accel;
        let mut as_write_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&tlas_handle));

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(storage.view())
            .image_layout(vk::ImageLayout::GENERAL);
        let image_infos = [image_info];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .push_next(&mut as_write_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&image_infos),
        ];

        // SAFETY: Updating descriptor sets.
        unsafe {
            gpu.device.update_descriptor_sets(&writes, &[]);
        }

        Ok(Self {
            accel_loader,
            rt_loader,
            pipeline,
            pipeline_layout,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            sbt,
            raygen_region,
            miss_region,
            hit_region,
            rgen_module,
            rmiss_module,
            rchit_module,
            blas_list,
            tlas: Some(tlas),
            geometry_buffers,
            instance_buffer: Some(instance_buffer),
            scratch_buffers: all_scratch,
            storage,
        })
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
        unsafe { old.destroy(device, allocator) };
        let view = self.storage.view();

        // Update RT descriptor set binding 1 (storage image).
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL);
        let image_infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos);
        // SAFETY: Updating descriptor set with valid image view.
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        Ok(())
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
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipeline);

            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            push_constants(
                device,
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                pc,
            );

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

    #[allow(clippy::type_complexity)]
    fn build_all_blas(
        gpu: &mut GpuContext,
        accel_loader: &ash::khr::acceleration_structure::Device,
        geometries: &[RtGeometry],
    ) -> Result<(Vec<AccelStruct>, Vec<HostBuffer>, Vec<DeviceBuffer>), RenderError> {
        let mut blas_list = Vec::new();
        let mut geo_buffers = Vec::new();
        let mut scratch_buffers = Vec::new();

        for geom in geometries {
            let tri_count = geom.positions.len() as u32 / 9; // 3 verts × 3 floats
            if tri_count == 0 {
                continue;
            }

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

            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: pos_addr,
                })
                .vertex_stride(12) // 3 × f32, tightly packed
                .max_vertex(tri_count * 3 - 1)
                .index_type(vk::IndexType::NONE_KHR);

            let as_geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                .flags(vk::GeometryFlagsKHR::OPAQUE);

            let as_geometries = [as_geometry];
            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&as_geometries);

            let tri_counts = [tri_count];
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            // SAFETY: Querying build sizes.
            unsafe {
                accel_loader.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &tri_counts,
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
            let accel = unsafe {
                accel_loader
                    .create_acceleration_structure(&as_create_info, None)
                    .map_err(RenderError::Vulkan)?
            };

            // Scratch buffer
            let scratch_buffer = DeviceBuffer::new(
                &gpu.device,
                &mut gpu.allocator,
                sizes.build_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                "blas scratch",
            )?;

            let scratch_addr = scratch_buffer.device_address(&gpu.device);

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
                .primitive_count(tri_count)
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

            blas_list.push(AccelStruct {
                accel,
                buffer: as_buffer,
            });
            geo_buffers.push(pos_buffer);
            scratch_buffers.push(scratch_buffer);
        }

        Ok((blas_list, geo_buffers, scratch_buffers))
    }

    // ── TLAS building ──

    #[allow(clippy::type_complexity)]
    fn build_tlas(
        gpu: &mut GpuContext,
        accel_loader: &ash::khr::acceleration_structure::Device,
        geometries: &[RtGeometry],
        blas_list: &[AccelStruct],
    ) -> Result<(AccelStruct, HostBuffer, DeviceBuffer), RenderError> {
        let mut instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::new();

        for (geom, blas) in geometries.iter().zip(blas_list.iter()) {
            if geom.positions.is_empty() {
                continue;
            }

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
            instances.push(vk::AccelerationStructureInstanceKHR {
                transform,
                instance_custom_index_and_mask: vk::Packed24_8::new(geom.geom_type, mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
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
        let tlas_accel = unsafe {
            accel_loader
                .create_acceleration_structure(&tlas_create_info, None)
                .map_err(RenderError::Vulkan)?
        };

        let scratch_buffer = DeviceBuffer::new(
            &gpu.device,
            &mut gpu.allocator,
            sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "tlas scratch",
        )?;
        let scratch_addr = scratch_buffer.device_address(&gpu.device);

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

        let tlas = AccelStruct {
            accel: tlas_accel,
            buffer: tlas_buffer,
        };

        Ok((tlas, instance_buffer, scratch_buffer))
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

        let (blas_list, geometry_buffers, scratch_buffers) =
            Self::build_all_blas(gpu, &self.accel_loader, geometries)?;

        let (tlas, instance_buffer, tlas_scratch) =
            Self::build_tlas(gpu, &self.accel_loader, geometries, &blas_list)?;

        let mut all_scratch = scratch_buffers;
        all_scratch.push(tlas_scratch);

        self.blas_list = blas_list;
        self.geometry_buffers = geometry_buffers;
        self.scratch_buffers = all_scratch;
        // Update TLAS descriptor binding (handle changed on rebuild).
        let tlas_handle = tlas.accel;
        self.tlas = Some(tlas);
        self.instance_buffer = Some(instance_buffer);

        let mut as_write_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&tlas_handle));

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .push_next(&mut as_write_info);

        // SAFETY: Updating descriptor set with valid TLAS.
        unsafe {
            gpu.device.update_descriptor_sets(&[write], &[]);
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

            // Pipeline
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Shader modules
            device.destroy_shader_module(self.rgen_module, None);
            device.destroy_shader_module(self.rmiss_module, None);
            device.destroy_shader_module(self.rchit_module, None);
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
            16.0 / 9.0,
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
        assert_eq!(pc.params, [0.25, 0.0, 0.0, 34.0]);
        assert_eq!(pc.ball_pos, [1.0, 2.0, 3.0, 56.0]);
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }
}
