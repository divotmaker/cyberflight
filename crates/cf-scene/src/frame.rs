//! The boundary between dimple's physics frame and the scene frame.
//!
//! Both frames are right-handed with +Y up and the tee at the origin:
//!
//! | Frame | Downrange | Golfer's right | Golfer's left |
//! |---|---|---|---|
//! | dimple (physics) | +X | +Z | −Z |
//! | scene | +Z | −X | +X |
//!
//! The range camera looks down +Z with a right-handed view
//! ([`Camera::view_matrix`](crate::camera::Camera::view_matrix)), so scene +X
//! is on the left of the screen and a ball right of the target line has a
//! negative scene X.
//!
//! The map is a rotation about +Y, so it applies unchanged to positions,
//! velocities and angular velocities. Every value crossing between dimple
//! and the scene goes through [`physics_to_scene`] or [`scene_to_physics`].

use glam::DVec3;

/// Map a dimple vector `(downrange, up, right)` to the scene frame:
/// scene `(−right, up, downrange)`.
#[must_use]
pub fn physics_to_scene(p: DVec3) -> DVec3 {
    DVec3::new(-p.z, p.y, p.x)
}

/// Map a scene vector to dimple's frame; the inverse of
/// [`physics_to_scene`].
#[must_use]
pub fn scene_to_physics(p: DVec3) -> DVec3 {
    DVec3::new(p.z, p.y, -p.x)
}

/// Horizontal NDC position of a scene point under the range camera, as the
/// renderer projects it: negative is the left of the screen, positive the
/// right.
#[cfg(test)]
pub(crate) fn screen_x(scene: DVec3) -> f32 {
    let camera = crate::camera::Camera::driving_range();
    let mut proj = camera.projection_matrix(16.0 / 9.0);
    proj.y_axis.y *= -1.0; // the renderer's Vulkan Y-flip
    let clip = proj * camera.view_matrix() * scene.as_vec3().extend(1.0);
    clip.x / clip.w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downrange_maps_to_scene_z() {
        let p = physics_to_scene(DVec3::new(100.0, 5.0, 0.0));
        assert_eq!(p, DVec3::new(0.0, 5.0, 100.0));
    }

    #[test]
    fn golfer_right_maps_to_negative_scene_x() {
        let p = physics_to_scene(DVec3::new(100.0, 5.0, 10.0));
        assert_eq!(p, DVec3::new(-10.0, 5.0, 100.0));
    }

    #[test]
    fn round_trips() {
        let p = DVec3::new(123.0, 4.5, -6.7);
        assert_eq!(scene_to_physics(physics_to_scene(p)), p);
        assert_eq!(physics_to_scene(scene_to_physics(p)), p);
    }

    #[test]
    fn is_a_rotation() {
        // Right-handed to right-handed: X × Y = Z survives the map, so spin
        // vectors keep their sense.
        let x = physics_to_scene(DVec3::X);
        let y = physics_to_scene(DVec3::Y);
        let z = physics_to_scene(DVec3::Z);
        assert_eq!(x.cross(y), z);
    }

    #[test]
    fn golfer_right_renders_on_screen_right() {
        let right = physics_to_scene(DVec3::new(100.0, 0.0, 10.0));
        let left = physics_to_scene(DVec3::new(100.0, 0.0, -10.0));
        assert!(screen_x(right) > 0.0, "right of line: {}", screen_x(right));
        assert!(screen_x(left) < 0.0, "left of line: {}", screen_x(left));
        // The target line itself stays centred.
        let centre = physics_to_scene(DVec3::new(100.0, 0.0, 0.0));
        assert!(screen_x(centre).abs() < 1e-6);
    }
}
