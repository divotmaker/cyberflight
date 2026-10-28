use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::error::RenderError;
use crate::gpu::{HostBuffer, Image2d, Image2dDesc};
use crate::pipeline::DEPTH_STENCIL_FORMAT;

/// Offscreen render target for headless rendering.
pub struct OffscreenTarget {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub(crate) color: Image2d,
    pub(crate) depth_stencil: Image2d,
    pub framebuffer: vk::Framebuffer,
    /// Readback buffer: `width * height * 4` bytes of RGBA8.
    pub(crate) staging: HostBuffer,
}

impl OffscreenTarget {
    /// Create an offscreen render target with staging buffer for readback.
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
        render_pass: vk::RenderPass,
    ) -> Result<Self, RenderError> {
        let extent = vk::Extent2D { width, height };
        let color = Image2d::new(
            device,
            allocator,
            &Image2dDesc {
                extent,
                format,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "offscreen color",
            },
        )?;

        // Create depth/stencil image for stencil masking
        let depth_stencil = Image2d::new(
            device,
            allocator,
            &Image2dDesc {
                extent,
                format: DEPTH_STENCIL_FORMAT,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                aspect: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                name: "offscreen depth stencil",
            },
        )?;

        // Create framebuffer (color + depth/stencil)
        let attachments = [color.view(), depth_stencil.view()];
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);

        // SAFETY: Creating framebuffer.
        let framebuffer = unsafe {
            device
                .create_framebuffer(&fb_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Staging buffer for readback (4 bytes per pixel)
        let staging = HostBuffer::new(
            device,
            allocator,
            u64::from(width) * u64::from(height) * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "offscreen staging",
        )?;

        Ok(Self {
            width,
            height,
            format,
            color,
            depth_stencil,
            framebuffer,
            staging,
        })
    }

    /// Clean up Vulkan resources.
    ///
    /// # Safety
    /// Must be called before destroying the device, with the device idle and the
    /// allocator the target was created with.
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: Resources are no longer in use (caller guarantees device idle).
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            self.color.destroy(device, allocator);
            self.depth_stencil.destroy(device, allocator);
            self.staging.destroy(device, allocator);
        }
    }
}
