use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

use crate::context::GpuContext;
use crate::error::RenderError;

/// Push constants for the ray tracing pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RtPushConstants {
    /// Camera position (xyz), w = viewport X offset in pixels.
    pub camera_pos: [f32; 4],
    /// Inverse view-projection matrix (screen → world).
    pub inv_view_proj: [[f32; 4]; 4],
    /// Grid parameters: [spacing_m, line_half_width, max_fade_dist, viewport_y_offset].
    pub grid_params: [f32; 4],
    /// Ball position (xyz), w = trail fade distance in meters.
    pub ball_pos: [f32; 4],
}

/// Geometry type IDs used as `gl_InstanceCustomIndexEXT` in shaders.
pub const GEOM_FLOOR: u32 = 0;
// GEOM_BALL (1) removed — ball reflections are analytical in the floor shader.
pub const GEOM_TEE_BOX: u32 = 2;
pub const GEOM_TRAIL: u32 = 3;
pub const GEOM_TEE_BOX_BORDER: u32 = 4;

/// A single acceleration structure (BLAS or TLAS) with its backing buffer.
struct AccelStruct {
    accel: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
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
    sbt_buffer: vk::Buffer,
    sbt_allocation: Option<Allocation>,
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
    geometry_buffers: Vec<(vk::Buffer, Option<Allocation>)>,
    // Instance buffer
    instance_buffer: Option<vk::Buffer>,
    instance_allocation: Option<Allocation>,

    // Scratch buffers (kept alive until build completes, then could be freed)
    scratch_buffers: Vec<(vk::Buffer, Option<Allocation>)>,

    // Storage image for RT output
    pub storage_image: vk::Image,
    pub storage_view: vk::ImageView,
    storage_allocation: Option<Allocation>,
    pub width: u32,
    pub height: u32,
}

/// Triangle geometry for BLAS building.
pub struct RtGeometry {
    /// Packed f32x3 positions (3 floats per vertex, 3 vertices per triangle).
    pub positions: Vec<f32>,
    /// Geometry type ID (GEOM_FLOOR, GEOM_TEE_BOX, etc.).
    pub geom_type: u32,
    /// 3×4 row-major transform (identity if not set).
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
        let rt_loader =
            ash::khr::ray_tracing_pipeline::Device::new(&gpu.instance, &gpu.device);

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
        let rgen_module = Self::create_shader_module(
            &gpu.device,
            include_bytes!("../shaders/rt.rgen.spv"),
        )?;
        let rmiss_module = Self::create_shader_module(
            &gpu.device,
            include_bytes!("../shaders/rt.rmiss.spv"),
        )?;
        let rchit_module = Self::create_shader_module(
            &gpu.device,
            include_bytes!("../shaders/rt.rchit.spv"),
        )?;

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
        let ds_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Creating descriptor set layout.
        let descriptor_set_layout = unsafe {
            gpu.device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // ── Pipeline layout (push constants + descriptor set) ──
        let push_range = vk::PushConstantRange::default()
            .stage_flags(
                vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )
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

        let (sbt_buffer, sbt_allocation) = create_buffer(
            &gpu.device,
            &mut gpu.allocator,
            sbt_size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            "rt sbt",
        )?;

        // Write handles into SBT buffer (each at region_stride offset for base alignment)
        if let Some(mapped) = sbt_allocation.mapped_ptr() {
            let dst = mapped.as_ptr().cast::<u8>();
            for i in 0..group_count as usize {
                let src_offset = i * handle_size as usize;
                let dst_offset = i * region_stride as usize;
                // SAFETY: Writing within mapped allocation bounds.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        handle_data.as_ptr().add(src_offset),
                        dst.add(dst_offset),
                        handle_size as usize,
                    );
                }
            }
        }

        let sbt_address = buffer_device_address(&gpu.device, sbt_buffer);
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
        let (storage_image, storage_view, storage_allocation) =
            Self::create_storage_image(&gpu.device, &mut gpu.allocator, width, height)?;

        // ── Build acceleration structures ──
        let (blas_list, geometry_buffers, scratch_buffers) =
            Self::build_all_blas(gpu, &accel_loader, geometries)?;

        let (tlas, instance_buffer, instance_allocation, tlas_scratch) =
            Self::build_tlas(gpu, &accel_loader, geometries, &blas_list)?;

        let mut all_scratch = scratch_buffers;
        all_scratch.push(tlas_scratch);

        // ── Update descriptor set ──
        let tlas_handle = tlas.accel;
        let mut as_write_info =
            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(std::slice::from_ref(&tlas_handle));

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(storage_view)
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
            sbt_buffer,
            sbt_allocation: Some(sbt_allocation),
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
            instance_allocation: Some(instance_allocation),
            scratch_buffers: all_scratch,
            storage_image,
            storage_view,
            storage_allocation: Some(storage_allocation),
            width,
            height,
        })
    }

    /// Resize the RT storage image to new dimensions.
    ///
    /// Destroys the old storage image/view and creates a new one, then updates
    /// the RT descriptor set (binding 1) to point at the new image.
    /// The device must be idle before calling this.
    pub fn resize_storage(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        if width == self.width && height == self.height {
            return Ok(());
        }

        // SAFETY: Device is idle (caller guarantees).
        unsafe {
            device.destroy_image_view(self.storage_view, None);
            device.destroy_image(self.storage_image, None);
        }
        if let Some(alloc) = self.storage_allocation.take() {
            let _ = allocator.free(alloc);
        }

        let (image, view, allocation) =
            Self::create_storage_image(device, allocator, width, height)?;
        self.storage_image = image;
        self.storage_view = view;
        self.storage_allocation = Some(allocation);
        self.width = width;
        self.height = height;

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

    /// Record RT dispatch commands into the given command buffer.
    ///
    /// The storage image must be in `GENERAL` layout before calling this.
    /// `dispatch_w` and `dispatch_h` control the trace dimensions (may be
    /// smaller than the storage image for sub-viewport traces).
    pub fn record_trace(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        push_constants: &RtPushConstants,
        dispatch_w: u32,
        dispatch_h: u32,
    ) {
        let callable_region = vk::StridedDeviceAddressRegionKHR::default();

        // SAFETY: Recording RT commands into a valid command buffer.
        unsafe {
            device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            let pc_bytes: &[u8] = std::slice::from_raw_parts(
                std::ptr::from_ref(push_constants).cast::<u8>(),
                std::mem::size_of::<RtPushConstants>(),
            );
            device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                0,
                pc_bytes,
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
    ) -> Result<
        (
            Vec<AccelStruct>,
            Vec<(vk::Buffer, Option<Allocation>)>,
            Vec<(vk::Buffer, Option<Allocation>)>,
        ),
        RenderError,
    > {
        let mut blas_list = Vec::new();
        let mut geo_buffers = Vec::new();
        let mut scratch_buffers = Vec::new();

        for geom in geometries {
            let tri_count = geom.positions.len() as u32 / 9; // 3 verts × 3 floats
            if tri_count == 0 {
                continue;
            }

            // Upload positions to a device-address-able buffer
            let pos_bytes = geom.positions.len() * std::mem::size_of::<f32>();
            let (pos_buffer, pos_allocation) = create_buffer(
                &gpu.device,
                &mut gpu.allocator,
                pos_bytes as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
                "blas positions",
            )?;
            if let Some(mapped) = pos_allocation.mapped_ptr() {
                // SAFETY: Writing geometry data to mapped buffer.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        geom.positions.as_ptr().cast::<u8>(),
                        mapped.as_ptr().cast::<u8>(),
                        pos_bytes,
                    );
                }
            }
            let pos_addr = buffer_device_address(&gpu.device, pos_buffer);

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
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles,
                })
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
            let (as_buffer, as_allocation) = create_buffer(
                &gpu.device,
                &mut gpu.allocator,
                sizes.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                "blas",
            )?;

            let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(as_buffer)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

            // SAFETY: Creating acceleration structure.
            let accel = unsafe {
                accel_loader
                    .create_acceleration_structure(&as_create_info, None)
                    .map_err(RenderError::Vulkan)?
            };

            // Scratch buffer
            let (scratch_buffer, scratch_allocation) = create_buffer(
                &gpu.device,
                &mut gpu.allocator,
                sizes.build_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
                "blas scratch",
            )?;

            let scratch_addr = buffer_device_address(&gpu.device, scratch_buffer);

            // Build BLAS on device
            let build_info_final =
                vk::AccelerationStructureBuildGeometryInfoKHR::default()
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
                        .src_access_mask(
                            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                        );
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
                allocation: Some(as_allocation),
            });
            geo_buffers.push((pos_buffer, Some(pos_allocation)));
            scratch_buffers.push((scratch_buffer, Some(scratch_allocation)));
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
    ) -> Result<
        (
            AccelStruct,
            vk::Buffer,
            Allocation,
            (vk::Buffer, Option<Allocation>),
        ),
        RenderError,
    > {
        let mut instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::new();

        for (i, (geom, blas)) in geometries.iter().zip(blas_list.iter()).enumerate() {
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
                    m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x,
                    m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y,
                    m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z,
                ],
            };

            // Instance mask bits:
            //   Bit 0 (0x01): visible to primary rays (floor only)
            //   Bit 1 (0x02): visible to reflection rays (ball, tee box border)
            //   Bit 2 (0x04): visible to reflections, skipped by trail continuation
            //   Bit 7 (0x80): RT-invisible (tee box fill — raster-only occlusion)
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
            instances.push(
                vk::AccelerationStructureInstanceKHR {
                    transform,
                    instance_custom_index_and_mask: vk::Packed24_8::new(geom.geom_type, mask),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0,
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw()
                            as u8,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: blas_addr,
                    },
                },
            );
            let _ = i; // suppress unused warning
        }

        let instance_count = instances.len() as u32;
        let instance_bytes =
            instances.len() * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>();

        let (instance_buffer, instance_allocation) = create_buffer(
            &gpu.device,
            &mut gpu.allocator,
            instance_bytes as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            "tlas instances",
        )?;

        if let Some(mapped) = instance_allocation.mapped_ptr() {
            // SAFETY: Writing instance data to mapped buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    instances.as_ptr().cast::<u8>(),
                    mapped.as_ptr().cast::<u8>(),
                    instance_bytes,
                );
            }
        }

        let instance_addr = buffer_device_address(&gpu.device, instance_buffer);

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

        let (tlas_buffer, tlas_allocation) = create_buffer(
            &gpu.device,
            &mut gpu.allocator,
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
            "tlas",
        )?;

        let tlas_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(tlas_buffer)
            .size(sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        // SAFETY: Creating TLAS.
        let tlas_accel = unsafe {
            accel_loader
                .create_acceleration_structure(&tlas_create_info, None)
                .map_err(RenderError::Vulkan)?
        };

        let (scratch_buffer, scratch_allocation) = create_buffer(
            &gpu.device,
            &mut gpu.allocator,
            sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
            "tlas scratch",
        )?;
        let scratch_addr = buffer_device_address(&gpu.device, scratch_buffer);

        let build_info_final =
            vk::AccelerationStructureBuildGeometryInfoKHR::default()
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
            allocation: Some(tlas_allocation),
        };

        Ok((
            tlas,
            instance_buffer,
            instance_allocation,
            (scratch_buffer, Some(scratch_allocation)),
        ))
    }

    // ── Storage image ──

    fn create_storage_image(
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, vk::ImageView, Allocation), RenderError> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: Creating storage image.
        let image = unsafe {
            device
                .create_image(&image_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Querying image memory requirements.
        let reqs = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "rt storage image",
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RenderError::Allocator(e.to_string()))?;

        // SAFETY: Binding memory.
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(RenderError::Vulkan)?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        // SAFETY: Creating image view.
        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .map_err(RenderError::Vulkan)?
        };

        Ok((image, view, allocation))
    }

    fn create_shader_module(
        device: &ash::Device,
        spirv: &[u8],
    ) -> Result<vk::ShaderModule, RenderError> {
        assert!(spirv.len() % 4 == 0, "SPIR-V size must be 4-byte aligned");
        let code: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        // SAFETY: Creating shader module from valid SPIR-V.
        unsafe {
            device
                .create_shader_module(&create_info, None)
                .map_err(RenderError::Vulkan)
        }
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
            gpu.device
                .device_wait_idle()
                .map_err(RenderError::Vulkan)?;
        }

        // SAFETY: Device is idle, AS resources are not in use.
        unsafe {
            self.destroy_accel_structures(&gpu.device, &mut gpu.allocator);
        }

        let (blas_list, geometry_buffers, scratch_buffers) =
            Self::build_all_blas(gpu, &self.accel_loader, geometries)?;

        let (tlas, instance_buffer, instance_allocation, tlas_scratch) =
            Self::build_tlas(gpu, &self.accel_loader, geometries, &blas_list)?;

        let mut all_scratch = scratch_buffers;
        all_scratch.push(tlas_scratch);

        self.blas_list = blas_list;
        self.geometry_buffers = geometry_buffers;
        self.scratch_buffers = all_scratch;
        self.tlas = Some(tlas);
        self.instance_buffer = Some(instance_buffer);
        self.instance_allocation = Some(instance_allocation);

        // Update TLAS descriptor binding (handle changed on rebuild).
        let tlas_handle = self.tlas.as_ref().expect("just built").accel;
        let mut as_write_info =
            vk::WriteDescriptorSetAccelerationStructureKHR::default()
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
                device.destroy_buffer(tlas.buffer, None);
                if let Some(alloc) = tlas.allocation {
                    let _ = allocator.free(alloc);
                }
            }

            if let Some(buf) = self.instance_buffer.take() {
                device.destroy_buffer(buf, None);
            }
            if let Some(alloc) = self.instance_allocation.take() {
                let _ = allocator.free(alloc);
            }

            for blas in self.blas_list.drain(..) {
                self.accel_loader
                    .destroy_acceleration_structure(blas.accel, None);
                device.destroy_buffer(blas.buffer, None);
                if let Some(alloc) = blas.allocation {
                    let _ = allocator.free(alloc);
                }
            }

            for (buf, alloc) in self.geometry_buffers.drain(..) {
                device.destroy_buffer(buf, None);
                if let Some(alloc) = alloc {
                    let _ = allocator.free(alloc);
                }
            }
            for (buf, alloc) in self.scratch_buffers.drain(..) {
                device.destroy_buffer(buf, None);
                if let Some(alloc) = alloc {
                    let _ = allocator.free(alloc);
                }
            }
        }
    }

    /// Clean up all Vulkan resources.
    ///
    /// # Safety
    /// Must be called before destroying the device. Device must be idle.
    pub unsafe fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: All resources are no longer in use (caller guarantees device idle).
        unsafe {
            self.destroy_accel_structures(device, allocator);

            // Storage image
            device.destroy_image_view(self.storage_view, None);
            device.destroy_image(self.storage_image, None);
            if let Some(alloc) = self.storage_allocation.take() {
                let _ = allocator.free(alloc);
            }

            // SBT
            device.destroy_buffer(self.sbt_buffer, None);
            if let Some(alloc) = self.sbt_allocation.take() {
                let _ = allocator.free(alloc);
            }

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

fn buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    // SAFETY: Buffer was created with SHADER_DEVICE_ADDRESS usage.
    unsafe {
        device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(buffer),
        )
    }
}

fn create_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    name: &'static str,
) -> Result<(vk::Buffer, Allocation), RenderError> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // SAFETY: Creating a buffer.
    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(RenderError::Vulkan)?
    };

    // SAFETY: Getting memory requirements.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| RenderError::Allocator(e.to_string()))?;

    // SAFETY: Binding memory to buffer.
    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(RenderError::Vulkan)?;
    }

    Ok((buffer, allocation))
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

        gpu.device
            .free_command_buffers(gpu.command_pool, &cbs);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constants_size() {
        // Must fit in 128-byte minimum guarantee
        assert!(std::mem::size_of::<RtPushConstants>() <= 128);
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }
}
