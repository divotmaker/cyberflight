//! Raster floor reflections: the tracer mirrored across the floor plane.
//!
//! The floor is the plane `y = 0`, so its reflection of the tracer is the
//! tracer itself mirrored across it, seen through the floor. This pass draws
//! that on any Vulkan device, into the same reflection image the ray tracer
//! writes, for the same composite to add onto the scene. Two render passes
//! per frame:
//!
//! 1. The *tracer pass* draws the newest flight's tracer tube and centre
//!    line, mirrored (`mirror.vert`), additively into a scratch image. It
//!    draws them from the vertex buffers the raster pass draws them from.
//!    Each pixel ends up with what the floor shader gets along a reflected
//!    ray: the summed light of the tracer surfaces behind the floor in that
//!    pixel, and the light-weighted sum of their heights (`mirror.frag`).
//! 2. The *floor pass* draws the floor's triangles (the same ones the ray
//!    tracer's floor is made of, tee box hole included) and shades each
//!    pixel with `shade_floor` of `floor_shading.glsl`, the code `rt.rchit`
//!    runs, from that scratch light (`mirror_floor.frag`).
//!
//! The ball is not mirrored geometry: `shade_floor` reflects it
//! analytically in both paths.
//!
//! Winding does not matter: the mirror flips it, and, as in the direct
//! draw, both faces of the tube's translucent shells are drawn and summed,
//! so nothing is culled. Surfaces below the floor (negative height before
//! the mirror) are discarded, as no ray leaving the floor reaches them.
//! Nothing is depth tested: the sum is additive, and a floor pixel shows
//! the mirrored tracer behind it whatever the raster scene has in front.
//!
//! The mirror plane is the floor's plane `y = 0`, flat everywhere, as the
//! ray tracer's floor geometry (`generate_floor_quad`) and the range
//! terrain are. The camera is above the floor.

use ash::vk;
use glam::Vec3;
use gpu_allocator::vulkan::Allocator;

use cf_scene::grid::{GridConfig, GridVertex, generate_floor_quad};
use cf_scene::tee::TeeBox;

use crate::error::RenderError;
use crate::gpu::{HostBuffer, Image2d, Image2dDesc, load_shader, push_constants};
use crate::recorder::view_proj;
use crate::reflections::{REFLECTION_FORMAT, TraceView};

/// The newest flight's tracer as the raster pass uploaded it: the mirror
/// draws these very vertices, mirrored.
pub(crate) struct MirrorTracer {
    /// Vertex buffer holding the tube's triangles.
    pub(crate) tube_buffer: vk::Buffer,
    /// The tube's vertices: the first ones of `tube_buffer`.
    pub(crate) tube: u32,
    /// Vertex buffer holding the centre line's segments.
    pub(crate) line_buffer: vk::Buffer,
    /// The centre line's vertices: the first ones of `line_buffer`.
    pub(crate) line: u32,
}

/// Push constants of the tracer pass (`mirror.vert`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct TracerPushConstants {
    /// View-projection matrix of the view.
    pub(crate) view_proj: [[f32; 4]; 4],
}

/// Push constants of the floor pass (`mirror_floor.vert`, `mirror_floor.frag`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct FloorPushConstants {
    /// View-projection matrix of the view.
    pub(crate) view_proj: [[f32; 4]; 4],
    /// xyz = camera position.
    pub(crate) camera_pos: [f32; 4],
    /// xyz = ball centre.
    pub(crate) ball_pos: [f32; 4],
}

/// The floor's vertices.
struct FloorBuffer {
    buffer: HostBuffer,
    count: u32,
}

impl FloorBuffer {
    /// Upload the floor of `grid` with the tee box hole of `tee`.
    ///
    /// # Errors
    /// The allocation fails.
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        grid: &GridConfig,
        tee: &TeeBox,
    ) -> Result<Self, RenderError> {
        let vertices = generate_floor_quad(grid, tee.half_size);
        let buffer = HostBuffer::with_data(
            device,
            allocator,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "mirror floor",
        )?;
        Ok(Self {
            buffer,
            count: vertices.len() as u32,
        })
    }

    /// # Safety
    /// No pending GPU work uses the buffer.
    unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees the buffer is unused.
        unsafe { self.buffer.destroy(device, allocator) };
    }
}

/// The scratch image of the tracer pass and the framebuffers of both
/// passes; recreated when the frame is resized.
struct MirrorTarget {
    /// Summed tracer light per pixel, `rgba16f`.
    tracer: Image2d,
    tracer_framebuffer: vk::Framebuffer,
    floor_framebuffer: vk::Framebuffer,
}

impl MirrorTarget {
    /// Create the scratch image of `extent` and the framebuffers of the two
    /// passes, the floor pass over `reflection_view`.
    ///
    /// # Errors
    /// A part fails to create; what was created is destroyed first.
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        passes: &MirrorPasses,
        reflection_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<Self, RenderError> {
        let tracer = Image2d::new(
            device,
            allocator,
            &Image2dDesc {
                extent,
                format: REFLECTION_FORMAT,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "mirror tracer image",
            },
        )?;
        let framebuffer = |pass: vk::RenderPass, view: vk::ImageView| {
            let views = [view];
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(pass)
                .attachments(&views)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            // SAFETY: `view` is a live view of an image of `extent` that
            // matches `pass`.
            unsafe {
                device
                    .create_framebuffer(&info, None)
                    .map_err(RenderError::Vulkan)
            }
        };
        let tracer_framebuffer = match framebuffer(passes.tracer, tracer.view()) {
            Ok(fb) => fb,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used it.
                unsafe { tracer.destroy(device, allocator) };
                return Err(e);
            }
        };
        let floor_framebuffer = match framebuffer(passes.floor, reflection_view) {
            Ok(fb) => fb,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe {
                    device.destroy_framebuffer(tracer_framebuffer, None);
                    tracer.destroy(device, allocator);
                }
                return Err(e);
            }
        };
        Ok(Self {
            tracer,
            tracer_framebuffer,
            floor_framebuffer,
        })
    }

    /// # Safety
    /// No pending GPU work uses the target.
    unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees the target is unused.
        unsafe {
            device.destroy_framebuffer(self.floor_framebuffer, None);
            device.destroy_framebuffer(self.tracer_framebuffer, None);
            self.tracer.destroy(device, allocator);
        }
    }
}

/// Render passes, pipelines and the descriptor set of the two passes.
struct MirrorPasses {
    /// Clears the scratch image, leaves it `SHADER_READ_ONLY_OPTIMAL`.
    tracer: vk::RenderPass,
    /// Clears the reflection image, leaves it `SHADER_READ_ONLY_OPTIMAL`.
    floor: vk::RenderPass,
    tracer_layout: vk::PipelineLayout,
    floor_layout: vk::PipelineLayout,
    /// TRIANGLE_LIST, additive.
    tube: vk::Pipeline,
    /// LINE_LIST, additive.
    line: vk::Pipeline,
    /// TRIANGLE_LIST, opaque.
    floor_pipeline: vk::Pipeline,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
}

impl MirrorPasses {
    /// Create everything for passes over `REFLECTION_FORMAT` images, or
    /// nothing.
    ///
    /// # Errors
    /// Any Vulkan object fails to create; what was created is destroyed first.
    fn new(device: &ash::Device) -> Result<Self, RenderError> {
        let modules = ShaderModules::new(device)?;
        let mut passes = Self {
            tracer: vk::RenderPass::null(),
            floor: vk::RenderPass::null(),
            tracer_layout: vk::PipelineLayout::null(),
            floor_layout: vk::PipelineLayout::null(),
            tube: vk::Pipeline::null(),
            line: vk::Pipeline::null(),
            floor_pipeline: vk::Pipeline::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            sampler: vk::Sampler::null(),
        };
        let created = passes.create(device, &modules);
        // SAFETY: The pipelines keep what they need; the modules are unused.
        unsafe { modules.destroy(device) };
        match created {
            Ok(()) => Ok(passes),
            Err(e) => {
                // SAFETY: Nothing created here has been used; null handles
                // are ignored by the destroy calls.
                unsafe { passes.destroy(device) };
                Err(e)
            }
        }
    }

    /// Fill in every handle, stopping at the first error.
    fn create(&mut self, device: &ash::Device, modules: &ShaderModules) -> Result<(), RenderError> {
        self.tracer = reflection_pass(device)?;
        self.floor = reflection_pass(device)?;

        // Nearest: the floor shader reads whole texels with `texelFetch`.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: Creating a sampler.
        self.sampler = unsafe {
            device
                .create_sampler(&sampler_info, None)
                .map_err(RenderError::Vulkan)?
        };

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Creating a descriptor set layout.
        self.set_layout = unsafe {
            device
                .create_descriptor_set_layout(&set_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        let tracer_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<TracerPushConstants>() as u32)];
        let tracer_layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&tracer_range);
        // SAFETY: Creating a pipeline layout.
        self.tracer_layout = unsafe {
            device
                .create_pipeline_layout(&tracer_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };
        let set_layouts = [self.set_layout];
        let floor_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<FloorPushConstants>() as u32)];
        let floor_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&floor_range);
        // SAFETY: Creating a pipeline layout.
        self.floor_layout = unsafe {
            device
                .create_pipeline_layout(&floor_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // `GridVertex`: position at 0, fade at 12; the floor reads only the position.
        let position = vk::VertexInputAttributeDescription::default()
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0);
        let fade = vk::VertexInputAttributeDescription::default()
            .location(1)
            .format(vk::Format::R32_SFLOAT)
            .offset(12);
        self.tube = create_pipeline(
            device,
            &PipelineDesc {
                pass: self.tracer,
                layout: self.tracer_layout,
                vert: modules.tracer_vert,
                frag: modules.tracer_frag,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                attributes: &[position, fade],
                additive: true,
            },
        )?;
        self.line = create_pipeline(
            device,
            &PipelineDesc {
                pass: self.tracer,
                layout: self.tracer_layout,
                vert: modules.tracer_vert,
                frag: modules.tracer_frag,
                topology: vk::PrimitiveTopology::LINE_LIST,
                attributes: &[position, fade],
                additive: true,
            },
        )?;
        self.floor_pipeline = create_pipeline(
            device,
            &PipelineDesc {
                pass: self.floor,
                layout: self.floor_layout,
                vert: modules.floor_vert,
                frag: modules.floor_frag,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                attributes: &[position],
                additive: false,
            },
        )?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: Creating a descriptor pool.
        self.pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(RenderError::Vulkan)?
        };
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&set_layouts);
        // SAFETY: Allocating a descriptor set (freed with the pool).
        self.set = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(RenderError::Vulkan)?[0]
        };
        Ok(())
    }

    /// Point the floor pass's sampler at `tracer_view`.
    ///
    /// # Safety
    /// No pending command buffer uses the descriptor set; `tracer_view` is
    /// a live view.
    unsafe fn bind_tracer(&self, device: &ash::Device, tracer_view: vk::ImageView) {
        let image_infos = [vk::DescriptorImageInfo::default()
            .image_view(tracer_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(self.sampler)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);
        // SAFETY: The caller guarantees the view is live and the set unused.
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    /// Destroy every object (the set goes with its pool).
    ///
    /// # Safety
    /// No pending GPU work uses them; `device` is the one they were created on.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: The caller guarantees nothing uses them.
        unsafe {
            device.destroy_pipeline(self.floor_pipeline, None);
            device.destroy_pipeline(self.line, None);
            device.destroy_pipeline(self.tube, None);
            device.destroy_pipeline_layout(self.floor_layout, None);
            device.destroy_pipeline_layout(self.tracer_layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_render_pass(self.floor, None);
            device.destroy_render_pass(self.tracer, None);
        }
    }
}

/// The four shader modules of the two passes.
struct ShaderModules {
    tracer_vert: vk::ShaderModule,
    tracer_frag: vk::ShaderModule,
    floor_vert: vk::ShaderModule,
    floor_frag: vk::ShaderModule,
}

impl ShaderModules {
    fn new(device: &ash::Device) -> Result<Self, RenderError> {
        let mut made: Vec<vk::ShaderModule> = Vec::with_capacity(4);
        let sources: [&[u8]; 4] = [
            include_bytes!("../shaders/mirror.vert.spv"),
            include_bytes!("../shaders/mirror.frag.spv"),
            include_bytes!("../shaders/mirror_floor.vert.spv"),
            include_bytes!("../shaders/mirror_floor.frag.spv"),
        ];
        for spv in sources {
            match load_shader(device, spv) {
                Ok(module) => made.push(module),
                Err(e) => {
                    for module in made {
                        // SAFETY: Just created and unused.
                        unsafe { device.destroy_shader_module(module, None) };
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self {
            tracer_vert: made[0],
            tracer_frag: made[1],
            floor_vert: made[2],
            floor_frag: made[3],
        })
    }

    /// # Safety
    /// No pipeline creation uses the modules any more.
    unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: The caller guarantees the modules are unused.
        unsafe {
            device.destroy_shader_module(self.tracer_vert, None);
            device.destroy_shader_module(self.tracer_frag, None);
            device.destroy_shader_module(self.floor_vert, None);
            device.destroy_shader_module(self.floor_frag, None);
        }
    }
}

/// A colour-only pass over a `REFLECTION_FORMAT` image: clears it, and
/// leaves it readable by fragment shaders.
///
/// The first dependency waits for the previous frame's fragment shaders to
/// finish reading the image; the second makes the pass's writes visible to
/// the fragment shaders that read it next (the floor pass, the composite).
fn reflection_pass(device: &ash::Device) -> Result<vk::RenderPass, RenderError> {
    let attachments = [vk::AttachmentDescription::default()
        .format(REFLECTION_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let color_refs = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)];
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    // SAFETY: Creating a render pass with a valid configuration.
    unsafe {
        device
            .create_render_pass(&info, None)
            .map_err(RenderError::Vulkan)
    }
}

/// What distinguishes the mirror pass's pipelines.
struct PipelineDesc<'a> {
    pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    topology: vk::PrimitiveTopology,
    /// Attributes read from binding 0, a buffer of `GridVertex`es.
    attributes: &'a [vk::VertexInputAttributeDescription],
    /// Add the fragment output to the image (`ONE`, `ONE`), else replace it.
    additive: bool,
}

/// A pipeline without depth test or culling, with dynamic viewport and
/// scissor (and line width for lines).
fn create_pipeline(
    device: &ash::Device,
    desc: &PipelineDesc<'_>,
) -> Result<vk::Pipeline, RenderError> {
    let entry_point = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(desc.vert)
            .name(entry_point),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(desc.frag)
            .name(entry_point),
    ];
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<GridVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(desc.attributes);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(desc.topology)
        .primitive_restart_enable(false);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // Culling is off: the mirror flips the winding, and the tube's shells
    // are drawn on both faces anyway.
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(desc.additive)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
        .alpha_blend_op(vk::BlendOp::ADD);
    let blend_attachments = [blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    if desc.topology == vk::PrimitiveTopology::LINE_LIST {
        dynamic_states.push(vk::DynamicState::LINE_WIDTH);
    }
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(desc.layout)
        .render_pass(desc.pass)
        .subpass(0);
    // SAFETY: Creating a graphics pipeline from live modules, layout and pass.
    let pipelines = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .map_err(|(_pipelines, err)| RenderError::Vulkan(err))?
    };
    Ok(pipelines[0])
}

/// Clear the rectangle of `view` of the pass's colour attachment to zero.
///
/// # Safety
/// `cb` is recording inside a render pass with one colour attachment.
unsafe fn clear_view(device: &ash::Device, cb: vk::CommandBuffer, view: &TraceView) {
    let attachments = [vk::ClearAttachment {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        color_attachment: 0,
        clear_value: vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        },
    }];
    let rect = vk::ClearRect {
        rect: view.rect(),
        base_array_layer: 0,
        layer_count: 1,
    };
    // SAFETY: The caller guarantees the command buffer state.
    unsafe { device.cmd_clear_attachments(cb, &attachments, &[rect]) };
}

/// The planar-mirror reflections: a tracer pass and a floor pass that fill
/// the reflection image.
pub(crate) struct Mirror {
    floor: FloorBuffer,
    passes: MirrorPasses,
    target: MirrorTarget,
    /// Width of the mirrored centre line in pixels: as the raster pass
    /// draws it.
    line_width: f32,
}

impl Mirror {
    /// Create the mirror for the floor of `grid` with the tee box of `tee`,
    /// drawing into `reflection_view` (an `extent` image of
    /// `REFLECTION_FORMAT` with `COLOR_ATTACHMENT` usage). The centre line is
    /// 2 px wide where `wide_lines` is enabled, else 1 px.
    ///
    /// # Errors
    /// A part fails to create; what was created is destroyed first.
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        reflection_view: vk::ImageView,
        extent: vk::Extent2D,
        grid: &GridConfig,
        tee: &TeeBox,
        wide_lines: bool,
    ) -> Result<Self, RenderError> {
        let floor = FloorBuffer::new(device, allocator, grid, tee)?;
        let passes = match MirrorPasses::new(device) {
            Ok(passes) => passes,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe { floor.destroy(device, allocator) };
                return Err(e);
            }
        };
        let target = match MirrorTarget::new(device, allocator, &passes, reflection_view, extent) {
            Ok(target) => target,
            Err(e) => {
                // SAFETY: Just created; no GPU work has used them.
                unsafe {
                    passes.destroy(device);
                    floor.destroy(device, allocator);
                }
                return Err(e);
            }
        };
        // SAFETY: The set was just allocated and no command buffer uses it.
        unsafe { passes.bind_tracer(device, target.tracer.view()) };
        Ok(Self {
            floor,
            passes,
            target,
            line_width: if wide_lines { 2.0 } else { 1.0 },
        })
    }

    /// Recreate the scratch image and the framebuffers for an `extent`
    /// reflection image whose view is `reflection_view`.
    ///
    /// # Safety
    /// The device is idle; the old reflection image may be destroyed after.
    pub(crate) unsafe fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        reflection_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<(), RenderError> {
        let target = MirrorTarget::new(device, allocator, &self.passes, reflection_view, extent)?;
        let old = std::mem::replace(&mut self.target, target);
        // SAFETY: The caller guarantees the device is idle.
        unsafe {
            old.destroy(device, allocator);
            self.passes.bind_tracer(device, self.target.tracer.view());
        }
        Ok(())
    }

    /// Fill the reflection image for `views` with the mirrored `tracer` and
    /// the ball centre `ball_centre`: the tracer pass, then the floor pass.
    /// Afterwards the reflection image is in `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// Later views replace the rectangle they cover (the chase inset over
    /// the main view).
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass; no pending GPU work for a
    /// previous frame writes the images; the buffers of `tracer` are live and
    /// hold its vertices; the reflection image has the extent `extent` that
    /// the mirror was created or resized with, and every view lies within it.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        tracer: &MirrorTracer,
        views: &[TraceView],
        extent: vk::Extent2D,
        ball_centre: Vec3,
    ) {
        let full = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let zero = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        }];
        let passes = &self.passes;

        // SAFETY: The caller guarantees the command buffer state, the images
        // and the tracer's buffers; the floor buffer, pipelines and
        // framebuffers live in `self`, and the push constants match the
        // layouts.
        unsafe {
            // Tracer pass: the mirrored tracer's light, summed per pixel.
            let tracer_info = vk::RenderPassBeginInfo::default()
                .render_pass(passes.tracer)
                .framebuffer(self.target.tracer_framebuffer)
                .render_area(full)
                .clear_values(&zero);
            device.cmd_begin_render_pass(cb, &tracer_info, vk::SubpassContents::INLINE);
            for (i, view) in views.iter().enumerate() {
                if i > 0 {
                    clear_view(device, cb, view);
                }
                device.cmd_set_viewport(cb, 0, &[view.viewport()]);
                device.cmd_set_scissor(cb, 0, &[view.rect()]);
                let pc = TracerPushConstants {
                    view_proj: view_proj(&view.camera, view.aspect()),
                };
                if tracer.tube > 0 {
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, passes.tube);
                    device.cmd_bind_vertex_buffers(cb, 0, &[tracer.tube_buffer], &[0]);
                    push_constants(
                        device,
                        cb,
                        passes.tracer_layout,
                        vk::ShaderStageFlags::VERTEX,
                        &pc,
                    );
                    device.cmd_draw(cb, tracer.tube, 1, 0, 0);
                }
                if tracer.line > 0 {
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, passes.line);
                    device.cmd_set_line_width(cb, self.line_width);
                    device.cmd_bind_vertex_buffers(cb, 0, &[tracer.line_buffer], &[0]);
                    push_constants(
                        device,
                        cb,
                        passes.tracer_layout,
                        vk::ShaderStageFlags::VERTEX,
                        &pc,
                    );
                    device.cmd_draw(cb, tracer.line, 1, 0, 0);
                }
            }
            device.cmd_end_render_pass(cb);

            // Floor pass: shade the floor pixels from that light.
            let floor_info = vk::RenderPassBeginInfo::default()
                .render_pass(passes.floor)
                .framebuffer(self.target.floor_framebuffer)
                .render_area(full)
                .clear_values(&zero);
            device.cmd_begin_render_pass(cb, &floor_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, passes.floor_pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                passes.floor_layout,
                0,
                &[passes.set],
                &[],
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[self.floor.buffer.handle()], &[0]);
            for (i, view) in views.iter().enumerate() {
                if i > 0 {
                    clear_view(device, cb, view);
                }
                device.cmd_set_viewport(cb, 0, &[view.viewport()]);
                device.cmd_set_scissor(cb, 0, &[view.rect()]);
                let camera = view.camera.position;
                let pc = FloorPushConstants {
                    view_proj: view_proj(&view.camera, view.aspect()),
                    camera_pos: [camera.x, camera.y, camera.z, 0.0],
                    ball_pos: [ball_centre.x, ball_centre.y, ball_centre.z, 0.0],
                };
                push_constants(
                    device,
                    cb,
                    passes.floor_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &pc,
                );
                device.cmd_draw(cb, self.floor.count, 1, 0, 0);
            }
            device.cmd_end_render_pass(cb);
        }
    }

    /// Destroy everything, in reverse creation order.
    ///
    /// # Safety
    /// No pending GPU work uses the mirror; `device` and `allocator` are the
    /// ones it was created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees nothing uses the mirror.
        unsafe {
            self.target.destroy(device, allocator);
            self.passes.destroy(device);
            self.floor.destroy(device, allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constants_fit_the_guaranteed_size() {
        assert_eq!(std::mem::size_of::<TracerPushConstants>(), 64);
        assert_eq!(std::mem::size_of::<FloorPushConstants>(), 96);
    }
}
