use glam::{Mat4, Vec3};

/// Camera for viewing the driving range.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Camera position in world space.
    pub position: Vec3,
    /// Point the camera looks at.
    pub target: Vec3,
    /// Up vector.
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl Camera {
    /// Default driving range camera: behind and above the tee box, looking downrange.
    ///
    /// Positioned at the south end of the range looking north (+Z).
    #[must_use]
    pub fn driving_range() -> Self {
        Self {
            position: Vec3::new(0.0, 1.7, -4.5),      // behind tee box, standing eye height
            target: Vec3::new(0.0, 1.7, 100.0),     // looking at horizon, downrange
            up: Vec3::Y,
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            near: 0.1,
            far: 600.0, // 450 yd ≈ 411m, need margin for diagonal
        }
    }

    /// Compute the view matrix (world -> camera).
    #[must_use]
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Compute the projection matrix for a given aspect ratio.
    #[must_use]
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect_ratio, self.near, self.far)
    }

    /// Chase camera: follows a ball in flight from behind.
    ///
    /// Positioned 10m behind the ball (horizontal), elevated above it,
    /// looking toward the landing zone. The ball appears in the lower
    /// half of the viewport.
    #[must_use]
    pub fn chase(ball_pos: Vec3, velocity: Vec3, landing_pos: Vec3) -> Self {
        let horizontal_dir = Vec3::new(velocity.x, 0.0, velocity.z);
        let dir = if horizontal_dir.length_squared() > 0.01 {
            horizontal_dir.normalize()
        } else {
            Vec3::Z // +Z = downrange fallback
        };

        let cam_pos = ball_pos - dir * 10.0 + Vec3::Y * 5.0;

        Self {
            position: cam_pos,
            target: landing_pos,
            up: Vec3::Y,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 600.0,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::driving_range()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_matrix_not_identity() {
        let cam = Camera::driving_range();
        let view = cam.view_matrix();
        assert_ne!(view, Mat4::IDENTITY);
    }

    #[test]
    fn projection_determinant_nonzero() {
        let cam = Camera::driving_range();
        let proj = cam.projection_matrix(16.0 / 9.0);
        assert!(proj.determinant().abs() > 1e-6);
    }

    #[test]
    fn far_plane_covers_range() {
        let cam = Camera::driving_range();
        // 450 yards ≈ 411m, far plane must exceed this
        assert!(cam.far > 411.0);
    }
}
