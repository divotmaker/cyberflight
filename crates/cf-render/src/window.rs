use ash::vk;

use crate::error::RenderError;

/// Swapchain and surface management.
pub struct Swapchain {
    pub surface_loader: ash::khr::surface::Instance,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

impl Swapchain {
    /// Create a swapchain for the given surface.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: ash::khr::surface::Instance,
        queue_family_index: u32,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);

        // SAFETY: Querying surface capabilities.
        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Querying surface formats.
        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Querying present modes.
        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .map_err(RenderError::Vulkan)?
        };

        // Prefer B8G8R8A8_SRGB, fall back to first available
        let surface_format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| {
                formats.iter().find(|f| {
                    f.format == vk::Format::B8G8R8A8_UNORM
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
            })
            .unwrap_or(&formats[0]);

        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO // always available
        };

        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };

        let image_count = (capabilities.min_image_count + 1).min(
            if capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                capabilities.max_image_count
            },
        );

        let queue_family_indices = [queue_family_index];
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        // SAFETY: Creating swapchain with valid parameters.
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // SAFETY: Getting swapchain images.
        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(RenderError::Vulkan)?
        };

        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                // SAFETY: Creating image views for swapchain images.
                unsafe { device.create_image_view(&view_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(RenderError::Vulkan)?;

        Ok(Self {
            surface_loader,
            swapchain_loader,
            surface,
            swapchain,
            images,
            image_views,
            extent,
            format: surface_format.format,
        })
    }

    /// Destroy and recreate the swapchain for a new size.
    #[allow(clippy::too_many_arguments)]
    pub fn recreate(
        &mut self,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        // SAFETY: Querying surface capabilities.
        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(physical_device, self.surface)
                .map_err(RenderError::Vulkan)?
        };

        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };

        let image_count = (capabilities.min_image_count + 1).min(
            if capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                capabilities.max_image_count
            },
        );

        let queue_family_indices = [queue_family_index];
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(self.format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(self.swapchain);

        // SAFETY: Creating new swapchain, retiring the old one.
        let new_swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&create_info, None)
                .map_err(RenderError::Vulkan)?
        };

        // Destroy old resources
        // SAFETY: Old swapchain was retired via old_swapchain field.
        unsafe {
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        self.swapchain = new_swapchain;
        self.extent = extent;

        // SAFETY: Getting new swapchain images.
        self.images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(self.swapchain)
                .map_err(RenderError::Vulkan)?
        };

        self.image_views = self
            .images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                // SAFETY: Creating image views for new swapchain images.
                unsafe { device.create_image_view(&view_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(RenderError::Vulkan)?;

        Ok(())
    }

    /// Clean up Vulkan resources.
    ///
    /// # Safety
    /// Must be called before destroying the device.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: Swapchain resources are no longer in use (caller guarantees device idle).
        unsafe {
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
