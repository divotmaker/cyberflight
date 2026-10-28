//! The reflection composite pass: a fullscreen triangle that samples the
//! ray-traced storage image and adds the reflections onto the raster scene.

use ash::vk;

use crate::error::RenderError;
use crate::gpu::load_shader;
use crate::pipeline::overlay_render_pass;

/// Render pass, pipeline and descriptor set of the reflection composite.
///
/// The pass loads the colour image in `COLOR_ATTACHMENT_OPTIMAL` and leaves
/// it there. It is compatible with the HUD pass (they differ only in final
/// layout), so it draws into the HUD pass's overlay framebuffer.
pub(crate) struct CompositePass {
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
}

impl CompositePass {
    /// Create the pass for a `color_format` target, sampling `storage_view`.
    ///
    /// # Errors
    /// Any Vulkan object fails to create; what was created is destroyed first.
    pub(crate) fn new(
        device: &ash::Device,
        color_format: vk::Format,
        storage_view: vk::ImageView,
    ) -> Result<Self, RenderError> {
        let vert = load_shader(device, include_bytes!("../shaders/composite.vert.spv"))?;
        let frag = match load_shader(device, include_bytes!("../shaders/composite.frag.spv")) {
            Ok(frag) => frag,
            Err(e) => {
                // SAFETY: `vert` was just created and is unused.
                unsafe { device.destroy_shader_module(vert, None) };
                return Err(e);
            }
        };

        let mut pass = Self {
            render_pass: vk::RenderPass::null(),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            sampler: vk::Sampler::null(),
        };
        let created = pass.create(device, color_format, vert, frag);

        // SAFETY: The pipeline keeps what it needs; the modules are unused.
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }

        match created {
            Ok(()) => {
                // SAFETY: The set was just allocated and no command buffer uses it.
                unsafe { pass.bind_storage(device, storage_view) };
                Ok(pass)
            }
            Err(e) => {
                // SAFETY: Nothing created here has been used; null handles
                // are ignored by the destroy calls.
                unsafe { pass.destroy(device) };
                Err(e)
            }
        }
    }

    /// Fill in every handle, stopping at the first error.
    fn create(
        &mut self,
        device: &ash::Device,
        color_format: vk::Format,
        vert: vk::ShaderModule,
        frag: vk::ShaderModule,
    ) -> Result<(), RenderError> {
        self.render_pass = overlay_render_pass(
            device,
            color_format,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        )?;

        // Sampler for the RT storage image (nearest — 1:1 pixel mapping).
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: Creating sampler.
        self.sampler = unsafe {
            device
                .create_sampler(&sampler_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Descriptor set layout: binding 0 = combined image sampler.
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [binding];
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Creating descriptor set layout.
        self.set_layout = unsafe {
            device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Pipeline layout (descriptor set, no push constants).
        let set_layouts = [self.set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: Creating pipeline layout.
        self.layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Pipeline: fullscreen triangle, additive blending, no vertex input.
        let entry_point = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(entry_point),
        ];

        // No vertex input (positions are hardcoded in the vertex shader).
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Additive blending: src + dst.
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD);
        let blend_attachments = [blend_attachment];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
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
            .layout(self.layout)
            .render_pass(self.render_pass)
            .subpass(0);

        // SAFETY: Creating graphics pipeline.
        self.pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .map_err(|(_pipelines, err)| RenderError::Vulkan(err))?[0]
        };

        // Descriptor pool + set.
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);
        let pool_sizes = [pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: Creating descriptor pool.
        self.pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(RenderError::Vulkan)?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&set_layouts);
        // SAFETY: Allocating descriptor set (freed with the pool).
        self.set = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(RenderError::Vulkan)?[0]
        };
        Ok(())
    }

    /// Point the descriptor set at `storage_view`, e.g. after the storage
    /// image was recreated at a new size.
    ///
    /// # Safety
    /// No pending command buffer uses the descriptor set (device idle).
    pub(crate) unsafe fn bind_storage(&self, device: &ash::Device, storage_view: vk::ImageView) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(storage_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(self.sampler);
        let image_infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);
        // SAFETY: The caller guarantees the set is not in use.
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    /// Record the composite into `overlay_fb` (colour only, `extent`).
    ///
    /// # Safety
    /// `cb` is recording, outside a render pass. The colour image is in
    /// `COLOR_ATTACHMENT_OPTIMAL` and the bound storage image is in
    /// `SHADER_READ_ONLY_OPTIMAL`, with extent equal to `extent`.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        overlay_fb: vk::Framebuffer,
        extent: vk::Extent2D,
    ) {
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(overlay_fb)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            });
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        // SAFETY: The caller guarantees the command buffer and image states.
        unsafe {
            device.cmd_begin_render_pass(cb, &render_pass_info, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[scissor]);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            // Fullscreen triangle: 3 vertices, no vertex buffer.
            device.cmd_draw(cb, 3, 1, 0, 0);
            device.cmd_end_render_pass(cb);
        }
    }

    /// Destroy every object of the pass (the set goes with its pool).
    ///
    /// # Safety
    /// No pending GPU work uses the pass; `device` is the one it was created on.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: The caller guarantees nothing uses the pass.
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_render_pass(self.render_pass, None);
        }
    }
}
