use ash::vk;

use crate::error::RenderError;
use crate::gpu::load_shader;

/// Push constants of the scene and HUD pipelines.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GridPushConstants {
    /// View-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Neon grid color (RGBA, HDR).
    pub color: [f32; 4],
}

/// The raster render passes and pipelines of the neon scene.
///
/// The scene pass clears colour and depth and ends in
/// `COLOR_ATTACHMENT_OPTIMAL`. The HUD pass loads the colour image, has no
/// depth, and ends in the renderer's output layout. The scene pipelines
/// (lines, alpha-blended fills, additive glow) belong to the scene pass;
/// the depth-less HUD fill and line pipelines to the HUD pass. All share one
/// layout, shaders and vertex format.
pub struct ScenePipelines {
    /// Colour CLEAR, `UNDEFINED` → `COLOR_ATTACHMENT_OPTIMAL`; depth CLEAR.
    pub(crate) scene_pass: vk::RenderPass,
    /// Colour LOAD, `COLOR_ATTACHMENT_OPTIMAL` → output layout; no depth.
    pub(crate) hud_pass: vk::RenderPass,
    /// [`GridPushConstants`] for the vertex and fragment stages.
    pub(crate) layout: vk::PipelineLayout,
    /// LINE_LIST, depth-tested (scene pass).
    pub(crate) line: vk::Pipeline,
    /// TRIANGLE_LIST, alpha blend, depth test + write (scene pass).
    pub(crate) fill: vk::Pipeline,
    /// TRIANGLE_LIST, additive blend, depth test (scene pass).
    pub(crate) glow: vk::Pipeline,
    /// TRIANGLE_LIST, alpha blend, no depth (HUD pass).
    pub(crate) hud_fill: vk::Pipeline,
    /// LINE_LIST, no depth (HUD pass).
    pub(crate) hud_line: vk::Pipeline,
}

impl ScenePipelines {
    /// Create both passes and all five pipelines. `output_layout` is the
    /// layout the HUD pass leaves the colour image in: `PRESENT_SRC_KHR`
    /// for a swapchain, `TRANSFER_SRC_OPTIMAL` for readback.
    ///
    /// # Errors
    /// Any Vulkan object fails to create; what was created is destroyed first.
    pub fn new(
        device: &ash::Device,
        color_format: vk::Format,
        depth_format: vk::Format,
        output_layout: vk::ImageLayout,
    ) -> Result<Self, RenderError> {
        let vert = load_shader(device, include_bytes!("../shaders/grid.vert.spv"))?;
        let frag = match load_shader(device, include_bytes!("../shaders/grid.frag.spv")) {
            Ok(frag) => frag,
            Err(e) => {
                // SAFETY: `vert` was just created and is unused.
                unsafe { device.destroy_shader_module(vert, None) };
                return Err(e);
            }
        };

        let mut pipes = Self {
            scene_pass: vk::RenderPass::null(),
            hud_pass: vk::RenderPass::null(),
            layout: vk::PipelineLayout::null(),
            line: vk::Pipeline::null(),
            fill: vk::Pipeline::null(),
            glow: vk::Pipeline::null(),
            hud_fill: vk::Pipeline::null(),
            hud_line: vk::Pipeline::null(),
        };
        let created = pipes.create(
            device,
            color_format,
            depth_format,
            output_layout,
            vert,
            frag,
        );

        // SAFETY: The pipelines keep what they need; the modules are unused.
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }

        match created {
            Ok(()) => Ok(pipes),
            Err(e) => {
                // SAFETY: Nothing created here has been used; null handles
                // are ignored by the destroy calls.
                unsafe { pipes.destroy(device) };
                Err(e)
            }
        }
    }

    /// Fill in every handle, stopping at the first error.
    fn create(
        &mut self,
        device: &ash::Device,
        color_format: vk::Format,
        depth_format: vk::Format,
        output_layout: vk::ImageLayout,
        vert: vk::ShaderModule,
        frag: vk::ShaderModule,
    ) -> Result<(), RenderError> {
        self.scene_pass = Self::create_scene_pass(device, color_format, depth_format)?;
        self.hud_pass = overlay_render_pass(device, color_format, output_layout)?;
        self.layout = Self::create_pipeline_layout(device)?;
        let scene = |topology, additive_blend, depth_test, depth_write| {
            Self::create_pipeline(
                device,
                self.scene_pass,
                self.layout,
                vert,
                frag,
                topology,
                additive_blend,
                depth_test,
                depth_write,
            )
        };
        self.line = scene(vk::PrimitiveTopology::LINE_LIST, false, false, false)?;
        self.fill = scene(vk::PrimitiveTopology::TRIANGLE_LIST, false, true, true)?;
        self.glow = scene(vk::PrimitiveTopology::TRIANGLE_LIST, true, true, false)?;
        let hud = |topology| {
            Self::create_pipeline(
                device,
                self.hud_pass,
                self.layout,
                vert,
                frag,
                topology,
                false,
                false,
                false,
            )
        };
        self.hud_fill = hud(vk::PrimitiveTopology::TRIANGLE_LIST)?;
        self.hud_line = hud(vk::PrimitiveTopology::LINE_LIST)?;
        Ok(())
    }

    fn create_scene_pass(
        device: &ash::Device,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<vk::RenderPass, RenderError> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_refs = [color_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)
            .depth_stencil_attachment(&depth_ref);

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

        let attachments = [color_attachment, depth_attachment];
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

        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        if topology == vk::PrimitiveTopology::LINE_LIST {
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

    /// Destroy both passes, the layout and the pipelines.
    ///
    /// # Safety
    /// No pending GPU work uses them; `device` is the one they were created on.
    pub unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: The caller guarantees nothing uses them.
        unsafe {
            device.destroy_pipeline(self.hud_line, None);
            device.destroy_pipeline(self.hud_fill, None);
            device.destroy_pipeline(self.glow, None);
            device.destroy_pipeline(self.fill, None);
            device.destroy_pipeline(self.line, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_render_pass(self.hud_pass, None);
            device.destroy_render_pass(self.scene_pass, None);
        }
    }
}

/// A colour-only render pass that loads the image in
/// `COLOR_ATTACHMENT_OPTIMAL`, stores it, and leaves it in `final_layout`:
/// the HUD pass and the reflection composite pass.
pub(crate) fn overlay_render_pass(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constants_size() {
        // Vulkan push constants must fit in 128 bytes minimum guarantee
        assert!(std::mem::size_of::<GridPushConstants>() <= 128);
    }
}
