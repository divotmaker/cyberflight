//! Shot data model — bridges flightrelay (FRP) schemas to the dimple physics engine.
//!
//! Converts `flightrelay::CompletedShot` into `dimple::trajectory::ShotInput` for flight simulation,
//! and extracts club/ball data for HUD display.

use dimple::trajectory::ShotInput;

/// Club delivery data extracted from an FRP shot.
///
/// All angles in degrees. Fields are optional because not all launch monitors
/// provide them (e.g. Mevo+ doesn't report face angle or dynamic loft).
#[derive(Debug, Clone, Default)]
pub struct ClubDelivery {
    pub club_speed_mph: Option<f64>,
    pub path_deg: Option<f64>,
    pub attack_angle_deg: Option<f64>,
    pub face_angle_deg: Option<f64>,
    pub dynamic_loft_deg: Option<f64>,
    pub smash_factor: Option<f64>,
}

/// A complete shot received from flightrelay, ready for simulation and display.
#[derive(Debug, Clone)]
pub struct ReceivedShot {
    pub device: String,
    pub shot_number: u32,
    pub input: ShotInput,
    pub club: ClubDelivery,
    /// Carry distance reported by the launch monitor (yards), if available.
    pub lm_carry_yards: Option<f64>,
}

impl ReceivedShot {
    /// Convert an FRP `CompletedShot` into a `ReceivedShot`.
    ///
    /// Returns `None` if the shot has no ball flight data (nothing to simulate).
    /// Backspin and sidespin come from the FRP components, or from
    /// `total_spin_rpm` + `spin_axis` when the device sends only that pair.
    #[must_use]
    pub fn from_frp(shot: &flightrelay::CompletedShot) -> Option<Self> {
        let mut ball = shot.ball.clone()?;
        ball.complete_spin();

        let input = ShotInput {
            ball_speed_mph: ball.launch_speed?.as_mph(),
            launch_angle_deg: ball.launch_elevation.unwrap_or(0.0),
            launch_azimuth_deg: ball.launch_azimuth.unwrap_or(0.0),
            backspin_rpm: ball.backspin_rpm.map_or(2500.0, |r| r as f64),
            sidespin_rpm: ball.sidespin_rpm.map_or(0.0, |r| r as f64),
        };

        let club = shot
            .club
            .as_ref()
            .map_or_else(ClubDelivery::default, |c| ClubDelivery {
                club_speed_mph: c.club_speed.map(|v| v.as_mph()),
                path_deg: c.path,
                attack_angle_deg: c.attack_angle,
                face_angle_deg: c.face_angle,
                dynamic_loft_deg: c.dynamic_loft,
                smash_factor: c.smash_factor,
            });

        let lm_carry_yards = ball.carry_distance.map(|d| d.as_yards());

        Some(Self {
            device: shot.device.clone(),
            shot_number: shot.key.shot_number,
            input,
            club,
            lm_carry_yards,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flightrelay::{BallFlight, ClubData, CompletedShot, ShotKey, Velocity};

    fn sample_completed_shot() -> CompletedShot {
        CompletedShot {
            device: "mevo.0".to_owned(),
            key: ShotKey {
                shot_id: "test-uuid".to_owned(),
                shot_number: 1,
            },
            ball: Some(BallFlight {
                launch_speed: Some(Velocity::MilesPerHour(167.0)),
                launch_elevation: Some(10.5),
                launch_azimuth: Some(-1.2),
                backspin_rpm: Some(2700),
                sidespin_rpm: Some(-300),
                ..BallFlight::default()
            }),
            club: Some(ClubData {
                club_speed: Some(Velocity::MilesPerHour(112.0)),
                path: Some(-2.1),
                attack_angle: Some(-1.5),
                face_angle: Some(0.8),
                dynamic_loft: Some(14.2),
                smash_factor: Some(1.49),
                ..ClubData::default()
            }),
            impact: None,
        }
    }

    #[test]
    fn converts_ball_flight_to_shot_input() {
        let shot = ReceivedShot::from_frp(&sample_completed_shot()).unwrap();
        assert!((shot.input.ball_speed_mph - 167.0).abs() < 0.01);
        assert!((shot.input.launch_angle_deg - 10.5).abs() < 0.01);
        assert!((shot.input.launch_azimuth_deg - -1.2).abs() < 0.01);
        assert!((shot.input.backspin_rpm - 2700.0).abs() < 0.01);
        assert!((shot.input.sidespin_rpm - -300.0).abs() < 0.01);
    }

    #[test]
    fn extracts_club_delivery() {
        let shot = ReceivedShot::from_frp(&sample_completed_shot()).unwrap();
        assert!((shot.club.club_speed_mph.unwrap() - 112.0).abs() < 0.01);
        assert!((shot.club.path_deg.unwrap() - -2.1).abs() < 0.01);
        assert!((shot.club.attack_angle_deg.unwrap() - -1.5).abs() < 0.01);
        assert!((shot.club.face_angle_deg.unwrap() - 0.8).abs() < 0.01);
        assert!((shot.club.dynamic_loft_deg.unwrap() - 14.2).abs() < 0.01);
        assert!((shot.club.smash_factor.unwrap() - 1.49).abs() < 0.01);
    }

    #[test]
    fn handles_missing_club_data() {
        let mut data = sample_completed_shot();
        data.club = None;
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert!(shot.club.club_speed_mph.is_none());
        assert!(shot.club.path_deg.is_none());
    }

    #[test]
    fn handles_missing_spin() {
        let mut data = sample_completed_shot();
        data.ball.as_mut().unwrap().backspin_rpm = None;
        data.ball.as_mut().unwrap().sidespin_rpm = None;
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert!((shot.input.backspin_rpm - 2500.0).abs() < 0.01);
        assert!((shot.input.sidespin_rpm - 0.0).abs() < 0.01);
    }

    #[test]
    fn uses_total_spin_and_axis_without_components() {
        let mut data = sample_completed_shot();
        let ball = data.ball.as_mut().unwrap();
        ball.backspin_rpm = None;
        ball.sidespin_rpm = None;
        ball.total_spin_rpm = Some(3000);
        ball.spin_axis = Some(-10.0);
        let shot = ReceivedShot::from_frp(&data).unwrap();
        // 3000 × cos(-10°) = 2954.4, 3000 × sin(-10°) = -520.9
        assert!((shot.input.backspin_rpm - 2954.0).abs() < 0.01);
        assert!((shot.input.sidespin_rpm - -521.0).abs() < 0.01);
    }

    #[test]
    fn returns_none_without_ball_data() {
        let mut data = sample_completed_shot();
        data.ball = None;
        assert!(ReceivedShot::from_frp(&data).is_none());
    }

    #[test]
    fn preserves_metadata() {
        let shot = ReceivedShot::from_frp(&sample_completed_shot()).unwrap();
        assert_eq!(shot.device, "mevo.0");
        assert_eq!(shot.shot_number, 1);
    }
}
