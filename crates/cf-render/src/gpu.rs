//! GPU helpers shared by both renderers: the byte view of GPU-facing data,
//! push constants, shader modules and buffers.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use cf_scene::grid::GridVertex;

use crate::error::RenderError;
use crate::pipeline::GridPushConstants;
use crate::rt_pipeline::RtPushConstants;

/// Plain data the GPU reads byte-for-byte.
///
/// # Safety
/// Implementors are `#[repr(C)]` (or primitive), `Copy`, contain no padding,
/// pointers or references, and every byte of every value is initialised.
pub(crate) unsafe trait GpuData: Copy + 'static {}

// SAFETY: A primitive integer: one byte, always initialised.
unsafe impl GpuData for u8 {}
const _: () = assert!(size_of::<u8>() == 1);

// SAFETY: A primitive float: four bytes, no padding.
unsafe impl GpuData for f32 {}
const _: () = assert!(size_of::<f32>() == 4);

// SAFETY: `#[repr(C)]` with `position: [f32; 3]` and `fade: f32`: 16 bytes of
// `f32`, no padding.
unsafe impl GpuData for GridVertex {}
const _: () = assert!(size_of::<GridVertex>() == 16);

// SAFETY: `#[repr(C)]` with `view_proj: [[f32; 4]; 4]` and `color: [f32; 4]`:
// 80 bytes of `f32`, no padding.
unsafe impl GpuData for GridPushConstants {}
const _: () = assert!(size_of::<GridPushConstants>() == 80);

// SAFETY: `#[repr(C)]` with `camera_pos: [f32; 4]`, `inv_view_proj:
// [[f32; 4]; 4]`, `params: [f32; 4]` and `ball_pos: [f32; 4]`: 112 bytes of
// `f32`, no padding.
unsafe impl GpuData for RtPushConstants {}
const _: () = assert!(size_of::<RtPushConstants>() == 112);

// SAFETY: `#[repr(C)]` Vulkan struct: a 3×4 `f32` transform (48 bytes), two
// `Packed24_8` (`u32` each) and an 8-byte acceleration-structure reference
// union whose two variants are both 8 bytes wide, so writing either one
// initialises it whole. 64 bytes, no padding.
unsafe impl GpuData for vk::AccelerationStructureInstanceKHR {}
const _: () = assert!(size_of::<vk::AccelerationStructureInstanceKHR>() == 64);

/// The bytes of `data`, as the GPU reads them.
#[must_use]
pub(crate) fn as_bytes<T: GpuData>(data: &[T]) -> &[u8] {
    // SAFETY: `T: GpuData` has no padding and every byte of every value is
    // initialised, so `data` is `size_of_val(data)` initialised bytes; `u8`
    // has alignment 1 and the borrow keeps `data` alive.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
}

/// Record `value` as push constants at offset 0 for `stages`.
///
/// # Safety
/// `cb` is in the recording state, and `layout` has a push-constant range
/// covering `[0, size_of::<T>())` for `stages`.
pub(crate) unsafe fn push_constants<T: GpuData>(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    stages: vk::ShaderStageFlags,
    value: &T,
) {
    // SAFETY: The caller guarantees the command buffer state and the range.
    unsafe {
        device.cmd_push_constants(cb, layout, stages, 0, as_bytes(std::slice::from_ref(value)));
    }
}

/// The 32-bit words of the SPIR-V binary `spv`.
///
/// # Errors
/// [`RenderError::ShaderCompilation`] when `spv` is not a whole number of
/// words or does not start with the SPIR-V magic number.
pub(crate) fn spirv_words(spv: &[u8]) -> Result<Vec<u32>, RenderError> {
    ash::util::read_spv(&mut std::io::Cursor::new(spv))
        .map_err(|e| RenderError::ShaderCompilation(e.to_string()))
}

/// Create a shader module from the SPIR-V binary `spv`.
///
/// # Errors
/// [`RenderError::ShaderCompilation`] for a malformed binary (see
/// [`spirv_words`]), [`RenderError::Vulkan`] if module creation fails.
pub(crate) fn load_shader(
    device: &ash::Device,
    spv: &[u8],
) -> Result<vk::ShaderModule, RenderError> {
    let code = spirv_words(spv)?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    // SAFETY: `code` is a whole number of SPIR-V words starting with the magic
    // number, and `create_info` borrows it for the duration of the call.
    unsafe {
        device
            .create_shader_module(&create_info, None)
            .map_err(RenderError::Vulkan)
    }
}

/// Create a buffer of `size` bytes and bind fresh memory at `location` to it.
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
    // SAFETY: Valid create info; the buffer is destroyed below on error or by
    // its owner's `destroy`.
    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(RenderError::Vulkan)?
    };
    // SAFETY: `buffer` was just created on `device`.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let allocation = match allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    }) {
        Ok(allocation) => allocation,
        Err(e) => {
            // SAFETY: `buffer` is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(RenderError::Allocator(e.to_string()));
        }
    };
    // SAFETY: `allocation` satisfies `buffer`'s requirements and is unbound.
    if let Err(e) =
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
    {
        // SAFETY: `buffer` is unused.
        unsafe { device.destroy_buffer(buffer, None) };
        let _ = allocator.free(allocation);
        return Err(RenderError::Vulkan(e));
    }
    Ok((buffer, allocation))
}

/// Device address of `buffer`, which was created with `SHADER_DEVICE_ADDRESS`.
fn device_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    // SAFETY: Callers only pass buffers created with `SHADER_DEVICE_ADDRESS`
    // usage on a device with `bufferDeviceAddress` enabled.
    unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    }
}

/// How many of `len` elements of `elem_size` bytes fit in `capacity_bytes`.
fn fit(capacity_bytes: usize, elem_size: usize, len: usize) -> usize {
    if elem_size == 0 {
        return len;
    }
    len.min(capacity_bytes / elem_size)
}

/// Host-visible, persistently mapped buffer that knows its size. Writes are
/// bounded by that size; it never writes past the allocation.
pub(crate) struct HostBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    size: vk::DeviceSize,
    name: &'static str,
    warned: bool,
}

impl HostBuffer {
    /// A mapped buffer of `size` bytes at `location` (`CpuToGpu` or
    /// `GpuToCpu`, both host-coherent, so writes and reads need no flush).
    ///
    /// # Errors
    /// Buffer creation or allocation fails, or the memory is not mapped.
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &'static str,
    ) -> Result<Self, RenderError> {
        let (buffer, allocation) = create_buffer(device, allocator, size, usage, location, name)?;
        if allocation.mapped_ptr().is_none() {
            // SAFETY: `buffer` is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            let _ = allocator.free(allocation);
            return Err(RenderError::Allocator(format!(
                "{name}: memory is not host-mapped"
            )));
        }
        Ok(Self {
            buffer,
            allocation,
            size,
            name,
            warned: false,
        })
    }

    /// A `CpuToGpu` buffer holding exactly `data`, at least one element in
    /// size so the buffer is never empty.
    ///
    /// # Errors
    /// As [`HostBuffer::new`].
    pub(crate) fn with_data<T: GpuData>(
        device: &ash::Device,
        allocator: &mut Allocator,
        data: &[T],
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> Result<Self, RenderError> {
        let size = size_of_val(data).max(size_of::<T>()) as vk::DeviceSize;
        let mut buffer = Self::new(
            device,
            allocator,
            size,
            usage,
            MemoryLocation::CpuToGpu,
            name,
        )?;
        buffer.write(data);
        Ok(buffer)
    }

    /// Write `data` from offset 0 and return the number of elements written:
    /// all of them, or the leading ones that fit. The first truncation of
    /// this buffer prints one warning.
    ///
    /// The copy is bounded by the buffer size, so it is memory-safe on the
    /// host. If pending GPU work still reads the buffer, the GPU may see a mix
    /// of old and new contents: wait for that work first to avoid it.
    pub(crate) fn write<T: GpuData>(&mut self, data: &[T]) -> u32 {
        let count = fit(self.size as usize, size_of::<T>(), data.len());
        if count < data.len() && !self.warned {
            eprintln!(
                "WARN: {}: {} elements do not fit in {} bytes; using the first {count}",
                self.name,
                data.len(),
                self.size
            );
            self.warned = true;
        }
        let bytes = as_bytes(&data[..count]);
        let Some(mapped) = self.allocation.mapped_slice_mut() else {
            return 0;
        };
        mapped[..bytes.len()].copy_from_slice(bytes);
        count as u32
    }

    /// The buffer's contents (`size` bytes).
    pub(crate) fn bytes(&self) -> &[u8] {
        self.allocation
            .mapped_slice()
            .map_or(&[], |mapped| &mapped[..self.size as usize])
    }

    pub(crate) fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Device address; the buffer must have `SHADER_DEVICE_ADDRESS` usage.
    pub(crate) fn device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        device_address(device, self.buffer)
    }

    /// Destroy the buffer and free its memory.
    ///
    /// # Safety
    /// No pending GPU work uses the buffer; `device` and `allocator` are the
    /// ones it was created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees the buffer is unused.
        unsafe { device.destroy_buffer(self.buffer, None) };
        let _ = allocator.free(self.allocation);
    }
}

/// Device-local buffer (acceleration-structure storage, scratch). No host access.
pub(crate) struct DeviceBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    size: vk::DeviceSize,
}

impl DeviceBuffer {
    /// A `GpuOnly` buffer of `size` bytes.
    ///
    /// # Errors
    /// Buffer creation or allocation fails.
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> Result<Self, RenderError> {
        let (buffer, allocation) = create_buffer(
            device,
            allocator,
            size,
            usage,
            MemoryLocation::GpuOnly,
            name,
        )?;
        Ok(Self {
            buffer,
            allocation,
            size,
        })
    }

    pub(crate) fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    pub(crate) fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Device address; the buffer must have `SHADER_DEVICE_ADDRESS` usage.
    pub(crate) fn device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        device_address(device, self.buffer)
    }

    /// Destroy the buffer and free its memory.
    ///
    /// # Safety
    /// No pending GPU work uses the buffer; `device` and `allocator` are the
    /// ones it was created with.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        // SAFETY: The caller guarantees the buffer is unused.
        unsafe { device.destroy_buffer(self.buffer, None) };
        let _ = allocator.free(self.allocation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_bytes_covers_every_element() {
        let verts = [GridVertex {
            position: [1.0, 2.0, 3.0],
            fade: 0.5,
        }; 2];
        let bytes = as_bytes(&verts);
        assert_eq!(bytes.len(), 32);
        assert_eq!(&bytes[..4], &1.0_f32.to_ne_bytes());
        assert_eq!(&bytes[28..], &0.5_f32.to_ne_bytes());
    }

    #[test]
    fn spirv_words_match_a_little_endian_decode() {
        let spv = include_bytes!("../shaders/grid.vert.spv");
        let expected: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(spirv_words(spv).unwrap(), expected);
    }

    #[test]
    fn spirv_words_reject_a_partial_word() {
        let spv = include_bytes!("../shaders/grid.vert.spv");
        assert!(matches!(
            spirv_words(&spv[..spv.len() - 1]),
            Err(RenderError::ShaderCompilation(_))
        ));
    }

    #[test]
    fn spirv_words_reject_a_bad_magic_number() {
        assert!(matches!(
            spirv_words(&[0u8; 8]),
            Err(RenderError::ShaderCompilation(_))
        ));
    }

    #[test]
    fn fit_keeps_everything_that_fits() {
        assert_eq!(fit(64, 16, 4), 4);
        assert_eq!(fit(64, 16, 3), 3);
    }

    #[test]
    fn fit_keeps_the_leading_whole_elements_on_overflow() {
        assert_eq!(fit(64, 16, 5), 4);
        assert_eq!(fit(70, 16, 100), 4);
    }

    #[test]
    fn fit_of_nothing_is_nothing() {
        assert_eq!(fit(64, 16, 0), 0);
        assert_eq!(fit(0, 16, 0), 0);
    }

    #[test]
    fn fit_is_zero_below_one_element() {
        assert_eq!(fit(15, 16, 3), 0);
        assert_eq!(fit(0, 16, 3), 0);
    }
}
