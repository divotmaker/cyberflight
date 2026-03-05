use thiserror::Error;

/// Errors from the Vulkan rendering system.
#[derive(Debug, Error)]
pub enum RenderError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),

    #[error("Vulkan loading error: {0}")]
    Loading(#[from] ash::LoadingError),

    #[error("no suitable GPU found")]
    NoSuitableDevice,

    #[error("no suitable queue family")]
    NoSuitableQueue,

    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("GPU allocator error: {0}")]
    Allocator(String),

    #[error("window creation failed: {0}")]
    Window(String),

    #[error("swapchain error: {0}")]
    Swapchain(String),
}
