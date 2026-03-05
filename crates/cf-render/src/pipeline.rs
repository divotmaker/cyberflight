use ash::vk;

use crate::error::RenderError;

/// Push constants for the grid pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GridPushConstants {
    /// View-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Neon grid color (RGBA, HDR).
    pub color: [f32; 4],
    /// World-space clip bounds [min_x, min_z, max_x, max_z].
    /// Fragments outside these XZ bounds are discarded.
    /// Set to large values (e.g. ±1e6) to disable clipping.
    pub clip_bounds: [f32; 4],
}

/// Rasterization pipelines for the neon scene.
///
/// Contains both a LINE_LIST pipeline (grid) and a TRIANGLE_LIST pipeline
/// (filled geometry like tee box, ball). Both share the same render pass,
/// layout, shaders, and vertex format.
/// Depth/stencil format for floor reflection masking.
pub const DEPTH_STENCIL_FORMAT: vk::Format = vk::Format::D24_UNORM_S8_UINT;

pub struct GridPipeline {
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    /// LINE_LIST pipeline for grid lines.
    pub pipeline: vk::Pipeline,
    /// TRIANGLE_LIST pipeline for filled geometry (alpha blend).
    pub fill_pipeline: vk::Pipeline,
    /// TRIANGLE_LIST pipeline for glow effects (additive blend).
    pub glow_pipeline: vk::Pipeline,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
}

impl GridPipeline {
    /// Create the grid rendering pipeline.
    ///
    /// `final_layout` controls the image layout after the render pass:
    /// - `PRESENT_SRC_KHR` for windowed rendering
    /// - `TRANSFER_SRC_OPTIMAL` for offscreen readback
    pub fn new(
        device: &ash::Device,
        color_format: vk::Format,
        final_layout: vk::ImageLayout,
    ) -> Result<Self, RenderError> {
        let render_pass = Self::create_render_pass(device, color_format, final_layout)?;
        let vert_module = Self::create_shader_module(
            device,
            include_bytes!("../shaders/grid.vert.spv"),
        )?;
        let frag_module = Self::create_shader_module(
            device,
            include_bytes!("../shaders/grid.frag.spv"),
        )?;
        let pipeline_layout = Self::create_pipeline_layout(device)?;
        let pipeline = Self::create_pipeline(
            device,
            render_pass,
            pipeline_layout,
            vert_module,
            frag_module,
            vk::PrimitiveTopology::LINE_LIST,
            false,
            false,
            false,
        )?;
        let fill_pipeline = Self::create_pipeline(
            device,
            render_pass,
            pipeline_layout,
            vert_module,
            frag_module,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            false,
            true,
            true,
        )?;
        let glow_pipeline = Self::create_pipeline(
            device,
            render_pass,
            pipeline_layout,
            vert_module,
            frag_module,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            true,
            true,
            false,
        )?;

        Ok(Self {
            render_pass,
            pipeline_layout,
            pipeline,
            fill_pipeline,
            glow_pipeline,
            vert_module,
            frag_module,
        })
    }

    fn create_render_pass(
        device: &ash::Device,
        color_format: vk::Format,
        final_layout: vk::ImageLayout,
    ) -> Result<vk::RenderPass, RenderError> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(final_layout);

        let depth_stencil_attachment = vk::AttachmentDescription::default()
            .format(DEPTH_STENCIL_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_stencil_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_refs = [color_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)
            .depth_stencil_attachment(&depth_stencil_ref);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let attachments = [color_attachment, depth_stencil_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency];

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        // SAFETY: Creating render pass with valid configuration.
        unsafe {
            device
                .create_render_pass(&create_info, None)
                .map_err(RenderError::Vulkan)
        }
    }

    fn create_shader_module(
        device: &ash::Device,
        spirv: &[u8],
    ) -> Result<vk::ShaderModule, RenderError> {
        // Align the SPIR-V bytes to u32
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

    fn create_pipeline_layout(device: &ash::Device) -> Result<vk::PipelineLayout, RenderError> {
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<GridPushConstants>() as u32);

        let ranges = [push_constant_range];
        let create_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges);

        // SAFETY: Creating pipeline layout.
        unsafe {
            device
                .create_pipeline_layout(&create_info, None)
                .map_err(RenderError::Vulkan)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
        vert_module: vk::ShaderModule,
        frag_module: vk::ShaderModule,
        topology: vk::PrimitiveTopology,
        additive_blend: bool,
        depth_test: bool,
        depth_write: bool,
    ) -> Result<vk::Pipeline, RenderError> {
        let entry_point = c"main";

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_point),
        ];

        // GridVertex: position [f32; 3] at offset 0, fade f32 at offset 12
        let binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(16) // sizeof(GridVertex)
            .input_rate(vk::VertexInputRate::VERTEX);

        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32_SFLOAT)
                .offset(12),
        ];

        let bindings = [binding];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Additive blend for glow, standard alpha blend otherwise.
        let blend_attachment = if additive_blend {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                .alpha_blend_op(vk::BlendOp::ADD)
        } else {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
        };

        let blend_attachments = [blend_attachment];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&blend_attachments);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_test)
            .depth_write_enable(depth_write)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .stencil_test_enable(false);

        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::LINE_WIDTH,
        ];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let create_infos = [create_info];

        // SAFETY: Creating graphics pipeline.
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
                .map_err(|(_pipelines, err)| RenderError::Vulkan(err))?
        };

        Ok(pipelines[0])
    }

    /// Create fill and line pipelines compatible with a given render pass.
    ///
    /// Returns `(fill_pipeline, line_pipeline)`. Used for HUD pipelines bound to
    /// the HUD render pass (which has no depth/stencil).
    pub fn create_hud_pipelines(
        &self,
        device: &ash::Device,
        render_pass: vk::RenderPass,
    ) -> Result<(vk::Pipeline, vk::Pipeline), RenderError> {
        let fill = Self::create_pipeline(
            device,
            render_pass,
            self.pipeline_layout,
            self.vert_module,
            self.frag_module,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            false,
            false,
            false,
        )?;
        let line = Self::create_pipeline(
            device,
            render_pass,
            self.pipeline_layout,
            self.vert_module,
            self.frag_module,
            vk::PrimitiveTopology::LINE_LIST,
            false,
            false,
            false,
        )?;
        Ok((fill, line))
    }

    /// Create a color-only render pass for HUD overlay.
    ///
    /// Uses `load_op = LOAD` to preserve the scene, `final_layout` for the post-HUD transition.
    pub fn create_hud_render_pass(
        device: &ash::Device,
        color_format: vk::Format,
        final_layout: vk::ImageLayout,
    ) -> Result<vk::RenderPass, RenderError> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(final_layout);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_refs = [color_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let attachments = [color_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency];

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        // SAFETY: Creating render pass with valid configuration.
        unsafe {
            device
                .create_render_pass(&create_info, None)
                .map_err(RenderError::Vulkan)
        }
    }

    /// Create a shader module from SPIR-V bytes (public for use by composite pass).
    pub fn create_shader_module_pub(
        device: &ash::Device,
        spirv: &[u8],
    ) -> Result<vk::ShaderModule, RenderError> {
        Self::create_shader_module(device, spirv)
    }

    /// Clean up Vulkan resources.
    ///
    /// # Safety
    /// Must be called before destroying the device.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: Pipeline resources are no longer in use (caller guarantees device idle).
        unsafe {
            device.destroy_pipeline(self.glow_pipeline, None);
            device.destroy_pipeline(self.fill_pipeline, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_shader_module(self.vert_module, None);
            device.destroy_shader_module(self.frag_module, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constants_size() {
        // Vulkan push constants must fit in 128 bytes minimum guarantee
        assert!(std::mem::size_of::<GridPushConstants>() <= 128);
    }
}
