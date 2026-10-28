//! GPU helpers shared by both renderers: the byte view of GPU-facing data and
//! push constants.

use ash::vk;

use cf_scene::grid::GridVertex;

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
}
