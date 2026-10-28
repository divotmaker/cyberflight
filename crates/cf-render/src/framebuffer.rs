use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::error::RenderError;
use crate::gpu::HostBuffer;
use crate::pipeline::DEPTH_STENCIL_FORMAT;

/// Offscreen render target for headless rendering.
pub struct OffscreenTarget {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub image_allocation: Allocation,
    pub depth_stencil_image: vk::Image,
    pub depth_stencil_view: vk::ImageView,
    pub depth_stencil_allocation: Allocation,
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
        // Create the color image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: Creating a Vulkan image.
        let image = unsafe {
            device
                .create_image(&image_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Querying image memory requirements.
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let image_allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "offscreen color",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RenderError::Allocator(e.to_string()))?;

        // SAFETY: Binding memory to image.
        unsafe {
            device
                .bind_image_memory(image, image_allocation.memory(), image_allocation.offset())
                .map_err(RenderError::Vulkan)?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
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

        // Create depth/stencil image for stencil masking
        let ds_image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(DEPTH_STENCIL_FORMAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: Creating depth/stencil image.
        let depth_stencil_image = unsafe {
            device
                .create_image(&ds_image_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Querying depth/stencil memory requirements.
        let ds_reqs = unsafe { device.get_image_memory_requirements(depth_stencil_image) };

        let depth_stencil_allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "offscreen depth stencil",
                requirements: ds_reqs,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RenderError::Allocator(e.to_string()))?;

        // SAFETY: Binding memory to depth/stencil image.
        unsafe {
            device
                .bind_image_memory(
                    depth_stencil_image,
                    depth_stencil_allocation.memory(),
                    depth_stencil_allocation.offset(),
                )
                .map_err(RenderError::Vulkan)?;
        }

        let ds_view_info = vk::ImageViewCreateInfo::default()
            .image(depth_stencil_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(DEPTH_STENCIL_FORMAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        // SAFETY: Creating depth/stencil image view.
        let depth_stencil_view = unsafe {
            device
                .create_image_view(&ds_view_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Create framebuffer (color + depth/stencil)
        let attachments = [view, depth_stencil_view];
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
            image,
            view,
            image_allocation,
            depth_stencil_image,
            depth_stencil_view,
            depth_stencil_allocation,
            framebuffer,
            staging,
        })
    }

    /// Clean up Vulkan resources.
    ///
    /// # Safety
    /// Must be called before destroying the device. Allocations must be freed by caller.
    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: Resources are no longer in use (caller guarantees device idle).
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.destroy_image_view(self.depth_stencil_view, None);
            device.destroy_image(self.depth_stencil_image, None);
            self.staging.destroy(device, allocator);
        }
        let _ = allocator.free(self.image_allocation);
        let _ = allocator.free(self.depth_stencil_allocation);
    }
}
