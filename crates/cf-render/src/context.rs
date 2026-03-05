use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::error::RenderError;

/// GPU configuration.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Render width in pixels.
    pub width: u32,
    /// Render height in pixels.
    pub height: u32,
    /// Enable Vulkan validation layers.
    pub validation: bool,
    /// Request ray tracing support (enabled if hardware supports it).
    pub enable_raytracing: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            validation: cfg!(debug_assertions),
            enable_raytracing: false,
        }
    }
}

/// Extensions required for hardware ray tracing.
const RT_DEVICE_EXTENSIONS: &[&CStr] = &[
    ash::khr::acceleration_structure::NAME,
    ash::khr::ray_tracing_pipeline::NAME,
    ash::khr::deferred_host_operations::NAME,
];

/// Vulkan GPU context.
///
/// Owns the Vulkan instance, device, queues, and allocator.
pub struct GpuContext {
    pub config: GpuConfig,
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub allocator: ManuallyDrop<Allocator>,
    /// Whether hardware ray tracing is available on this device.
    pub rt_supported: bool,
    /// Whether the device supports line widths > 1.0.
    pub wide_lines: bool,
}

impl GpuContext {
    /// Create a new GPU context with window surface support.
    ///
    /// # Errors
    /// Returns `RenderError` if Vulkan initialization fails.
    pub fn new(
        config: GpuConfig,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<(Self, vk::SurfaceKHR, ash::khr::surface::Instance), RenderError> {
        // SAFETY: Loading the Vulkan library from the system.
        let entry = unsafe { ash::Entry::load()? };

        let surface_extensions =
            ash_window::enumerate_required_extensions(display_handle)
                .map_err(|e| RenderError::Window(format!("enumerate extensions: {e}")))?
                .to_vec();

        let instance = Self::create_instance(&entry, &config, &surface_extensions)?;
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // SAFETY: Creating a Vulkan surface from valid window handles.
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .map_err(|e| RenderError::Window(format!("surface creation failed: {e}")))?
        };

        let (physical_device, queue_family_index) =
            Self::pick_physical_device(&instance, Some((&surface_loader, surface)))?;

        let want_rt = config.enable_raytracing;
        let has_rt = want_rt && Self::device_supports_rt(&instance, physical_device);
        if want_rt && !has_rt {
            eprintln!("Ray tracing requested but not supported by this device");
        }

        let device = Self::create_device(
            &instance,
            physical_device,
            queue_family_index,
            true, // enable swapchain
            has_rt,
        )?;

        let ctx = Self::finish(config, entry, instance, physical_device, device, queue_family_index, has_rt)?;
        Ok((ctx, surface, surface_loader))
    }

    /// Create a headless GPU context (no window or surface).
    ///
    /// # Errors
    /// Returns `RenderError` if Vulkan initialization fails.
    pub fn new_headless(config: GpuConfig) -> Result<Self, RenderError> {
        // SAFETY: Loading the Vulkan library from the system.
        let entry = unsafe { ash::Entry::load()? };

        let instance = Self::create_instance(&entry, &config, &[])?;

        let (physical_device, queue_family_index) =
            Self::pick_physical_device(&instance, None)?;

        let want_rt = config.enable_raytracing;
        let has_rt = want_rt && Self::device_supports_rt(&instance, physical_device);
        if want_rt && !has_rt {
            eprintln!("Ray tracing requested but not supported by this device");
        }

        let device = Self::create_device(
            &instance,
            physical_device,
            queue_family_index,
            false, // no swapchain
            has_rt,
        )?;

        Self::finish(config, entry, instance, physical_device, device, queue_family_index, has_rt)
    }

    fn finish(
        config: GpuConfig,
        entry: ash::Entry,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        queue_family_index: u32,
        rt_supported: bool,
    ) -> Result<Self, RenderError> {
        // SAFETY: Queue was created with the device.
        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let command_pool = Self::create_command_pool(&device, queue_family_index)?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: rt_supported,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })
        .map_err(|e| RenderError::Allocator(e.to_string()))?;

        if rt_supported {
            eprintln!("Ray tracing: enabled");
        }

        // SAFETY: Querying physical device features.
        let supported =
            unsafe { instance.get_physical_device_features(physical_device) };
        let wide_lines = supported.wide_lines == vk::TRUE;

        Ok(Self {
            config,
            entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            queue_family_index,
            command_pool,
            allocator: ManuallyDrop::new(allocator),
            rt_supported,
            wide_lines,
        })
    }

    fn create_instance(
        entry: &ash::Entry,
        config: &GpuConfig,
        surface_extensions: &[*const i8],
    ) -> Result<ash::Instance, RenderError> {
        let app_name = CString::new("cyberflight").expect("valid cstr");
        let engine_name = CString::new("cyberflight").expect("valid cstr");

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let mut extensions = surface_extensions.to_vec();

        let mut layers: Vec<*const i8> = Vec::new();
        let validation_layer =
            CString::new("VK_LAYER_KHRONOS_validation").expect("valid cstr");

        if config.validation {
            layers.push(validation_layer.as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);

        // SAFETY: Vulkan instance creation with valid create info.
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        Ok(instance)
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface: Option<(&ash::khr::surface::Instance, vk::SurfaceKHR)>,
    ) -> Result<(vk::PhysicalDevice, u32), RenderError> {
        // SAFETY: Enumerating Vulkan physical devices.
        let devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(RenderError::Vulkan)?
        };

        if devices.is_empty() {
            return Err(RenderError::NoSuitableDevice);
        }

        let mut best: Option<(vk::PhysicalDevice, u32, bool)> = None;

        for &pd in &devices {
            // SAFETY: Querying physical device properties.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;

            // SAFETY: Querying queue family properties.
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };

            for (i, qf) in queue_families.iter().enumerate() {
                let i = i as u32;
                if !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    continue;
                }

                // Check surface support if a surface was provided
                if let Some((surface_loader, surface_khr)) = &surface {
                    // SAFETY: Checking surface support for this queue family.
                    let surface_support = unsafe {
                        surface_loader
                            .get_physical_device_surface_support(pd, i, *surface_khr)
                            .unwrap_or(false)
                    };
                    if !surface_support {
                        continue;
                    }
                }

                match best {
                    None => best = Some((pd, i, is_discrete)),
                    Some((_, _, was_discrete)) if !was_discrete && is_discrete => {
                        best = Some((pd, i, is_discrete));
                    }
                    _ => {}
                }
            }
        }

        let (pd, qi, _) = best.ok_or(RenderError::NoSuitableQueue)?;

        // SAFETY: Reading device name for logging.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        eprintln!("GPU: {}", name.to_string_lossy());

        Ok((pd, qi))
    }

    /// Check if a physical device supports all required RT extensions.
    fn device_supports_rt(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> bool {
        // SAFETY: Enumerating device extension properties.
        let extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };

        RT_DEVICE_EXTENSIONS.iter().all(|required| {
            extensions
                .iter()
                .any(|ext| {
                    // SAFETY: Extension name is a valid C string from Vulkan.
                    let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                    name == *required
                })
        })
    }

    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        enable_swapchain: bool,
        enable_rt: bool,
    ) -> Result<ash::Device, RenderError> {
        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let queue_create_infos = [queue_create_info];

        let mut extensions: Vec<*const i8> = Vec::new();
        if enable_swapchain {
            extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }
        if enable_rt {
            for ext in RT_DEVICE_EXTENSIONS {
                extensions.push(ext.as_ptr());
            }
        }

        // SAFETY: Querying physical device features.
        let supported_features =
            unsafe { instance.get_physical_device_features(physical_device) };
        let wide_lines = supported_features.wide_lines == vk::TRUE;
        let features = vk::PhysicalDeviceFeatures::default().wide_lines(wide_lines);

        // RT requires buffer_device_address + acceleration_structure + rt_pipeline features.
        // Chain these via pNext when RT is enabled.
        let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
            .buffer_device_address(true);
        let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let mut rt_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
            .ray_tracing_pipeline(true);

        let mut create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        if enable_rt {
            create_info = create_info
                .push_next(&mut bda_features)
                .push_next(&mut as_features)
                .push_next(&mut rt_features);
        }

        // SAFETY: Creating a Vulkan logical device.
        let device = unsafe {
            instance
                .create_device(physical_device, &create_info, None)
                .map_err(RenderError::Vulkan)?
        };

        Ok(device)
    }

    fn create_command_pool(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<vk::CommandPool, RenderError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        // SAFETY: Creating a command pool.
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .map_err(RenderError::Vulkan)?
        };

        Ok(pool)
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            // SAFETY: Dropping allocator before device.
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
