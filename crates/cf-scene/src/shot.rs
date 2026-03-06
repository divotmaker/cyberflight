//! Shot data model — bridges flighthook schemas to cyberflight's physics engine.
//!
//! Converts `flighthook::ShotData` into `cf_math::ShotInput` for flight simulation,
//! and extracts club/ball data for HUD display.

use cf_math::trajectory::ShotInput;

/// Club delivery data extracted from a flighthook shot.
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

/// A complete shot received from flighthook, ready for simulation and display.
#[derive(Debug, Clone)]
pub struct ReceivedShot {
    pub source: String,
    pub shot_number: i32,
    pub input: ShotInput,
    pub club: ClubDelivery,
    pub estimated: bool,
    /// Carry distance reported by the launch monitor (yards), if available.
    pub lm_carry_yards: Option<f64>,
}

impl ReceivedShot {
    /// Convert a flighthook `ShotData` into a `ReceivedShot`.
    #[must_use]
    pub fn from_flighthook(shot: &flighthook::ShotData) -> Self {
        let ball = &shot.ball;

        let input = ShotInput {
            ball_speed_mph: ball.launch_speed.as_mph(),
            launch_angle_deg: ball.launch_elevation,
            launch_azimuth_deg: ball.launch_azimuth,
            backspin_rpm: ball.backspin_rpm.map_or(2500.0, |r| r as f64),
            sidespin_rpm: ball.sidespin_rpm.map_or(0.0, |r| r as f64),
        };

        let club = shot.club.as_ref().map_or_else(ClubDelivery::default, |c| {
            ClubDelivery {
                club_speed_mph: Some(c.club_speed.as_mph()),
                path_deg: c.path,
                attack_angle_deg: c.attack_angle,
                face_angle_deg: c.face_angle,
                dynamic_loft_deg: c.dynamic_loft,
                smash_factor: c.smash_factor,
            }
        });

        let lm_carry_yards = ball.carry_distance.map(|d| d.as_yards());

        Self {
            source: shot.source.clone(),
            shot_number: shot.shot_number,
            input,
            club,
            estimated: shot.estimated,
            lm_carry_yards,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flighthook::{BallFlight, ClubData, ShotData, Velocity};

    fn sample_shot_data() -> ShotData {
        ShotData {
            source: "mevo.0".to_owned(),
            shot_number: 1,
            ball: BallFlight {
                launch_speed: Velocity::MilesPerHour(167.0),
                launch_elevation: 10.5,
                launch_azimuth: -1.2,
                carry_distance: None,
                total_distance: None,
                max_height: None,
                flight_time: None,
                roll_distance: None,
                backspin_rpm: Some(2700),
                sidespin_rpm: Some(-300),
            },
            club: Some(ClubData {
                club_speed: Velocity::MilesPerHour(112.0),
                path: Some(-2.1),
                attack_angle: Some(-1.5),
                face_angle: Some(0.8),
                dynamic_loft: Some(14.2),
                smash_factor: Some(1.49),
                club_speed_post: None,
                swing_plane_horizontal: None,
                swing_plane_vertical: None,
                club_offset: None,
                club_height: None,
            }),
            estimated: false,
        }
    }

    #[test]
    fn converts_ball_flight_to_shot_input() {
        let shot = ReceivedShot::from_flighthook(&sample_shot_data());
        assert!((shot.input.ball_speed_mph - 167.0).abs() < 0.01);
        assert!((shot.input.launch_angle_deg - 10.5).abs() < 0.01);
        assert!((shot.input.launch_azimuth_deg - -1.2).abs() < 0.01);
        assert!((shot.input.backspin_rpm - 2700.0).abs() < 0.01);
        assert!((shot.input.sidespin_rpm - -300.0).abs() < 0.01);
    }

    #[test]
    fn extracts_club_delivery() {
        let shot = ReceivedShot::from_flighthook(&sample_shot_data());
        assert!((shot.club.club_speed_mph.unwrap() - 112.0).abs() < 0.01);
        assert!((shot.club.path_deg.unwrap() - -2.1).abs() < 0.01);
        assert!((shot.club.attack_angle_deg.unwrap() - -1.5).abs() < 0.01);
        assert!((shot.club.face_angle_deg.unwrap() - 0.8).abs() < 0.01);
        assert!((shot.club.dynamic_loft_deg.unwrap() - 14.2).abs() < 0.01);
        assert!((shot.club.smash_factor.unwrap() - 1.49).abs() < 0.01);
    }

    #[test]
    fn handles_missing_club_data() {
        let mut data = sample_shot_data();
        data.club = None;
        let shot = ReceivedShot::from_flighthook(&data);
        assert!(shot.club.club_speed_mph.is_none());
        assert!(shot.club.path_deg.is_none());
    }

    #[test]
    fn handles_missing_spin() {
        let mut data = sample_shot_data();
        data.ball.backspin_rpm = None;
        data.ball.sidespin_rpm = None;
        let shot = ReceivedShot::from_flighthook(&data);
        assert!((shot.input.backspin_rpm - 2500.0).abs() < 0.01);
        assert!((shot.input.sidespin_rpm - 0.0).abs() < 0.01);
    }

    #[test]
    fn preserves_metadata() {
        let shot = ReceivedShot::from_flighthook(&sample_shot_data());
        assert_eq!(shot.source, "mevo.0");
        assert_eq!(shot.shot_number, 1);
        assert!(!shot.estimated);
    }
}
