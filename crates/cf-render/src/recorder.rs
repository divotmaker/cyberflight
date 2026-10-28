//! Frame recording shared by the windowed and the offscreen renderer.

use ash::vk;

use cf_scene::camera::Camera;

/// Fraction of the frame width the chase-camera inset takes (right side).
const CHASE_VIEWPORT_FRAC: f32 = 0.20;

/// Vulkan view-projection matrix for `camera` at `aspect` (Y flipped for
/// Vulkan NDC), as column arrays for push constants.
pub(crate) fn view_proj(camera: &Camera, aspect: f32) -> [[f32; 4]; 4] {
    let view = camera.view_matrix();
    let mut proj = camera.projection_matrix(aspect);
    // Vulkan NDC has +Y down; glam assumes OpenGL (+Y up). Flip Y.
    proj.y_axis.y *= -1.0;
    (proj * view).to_cols_array_2d()
}

/// The chase-camera inset: the right 20 % of the frame, full height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChaseViewport {
    /// Left edge in pixels.
    pub(crate) x: u32,
    /// Width in pixels.
    pub(crate) width: u32,
}

impl ChaseViewport {
    /// The inset of a frame `frame_width` pixels wide.
    pub(crate) fn new(frame_width: u32) -> Self {
        Self {
            x: (frame_width as f32 * (1.0 - CHASE_VIEWPORT_FRAC)) as u32,
            width: (frame_width as f32 * CHASE_VIEWPORT_FRAC) as u32,
        }
    }

    /// The inset's rectangle in a frame `height` pixels high (scissor and
    /// clear rect).
    pub(crate) fn rect(self, height: u32) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D {
                x: self.x as i32,
                y: 0,
            },
            extent: vk::Extent2D {
                width: self.width,
                height,
            },
        }
    }

    /// The inset's viewport in a frame `height` pixels high.
    pub(crate) fn viewport(self, height: u32) -> vk::Viewport {
        vk::Viewport {
            x: self.x as f32,
            y: 0.0,
            width: self.width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Width / height of the inset in a frame `height` pixels high.
    pub(crate) fn aspect(self, height: u32) -> f32 {
        self.width as f32 / height as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chase_viewport_is_the_right_fifth() {
        for w in [1920u32, 3840, 1001] {
            let chase = ChaseViewport::new(w);
            assert_eq!(chase.x, (w as f32 * 0.8) as u32, "x at {w}");
            assert_eq!(chase.width, (w as f32 * 0.2) as u32, "width at {w}");
            assert!(chase.x + chase.width <= w, "inside the frame at {w}");
            let rect = chase.rect(1080);
            assert_eq!(rect.offset.x as u32, chase.x, "i32 round trip at {w}");
            assert_eq!(rect.extent.width, chase.width);
            assert_eq!(rect.extent.height, 1080);
        }
    }

    #[test]
    fn chase_viewport_at_1080p() {
        assert_eq!(
            ChaseViewport::new(1920),
            ChaseViewport {
                x: 1536,
                width: 384
            }
        );
    }
}
