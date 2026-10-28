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

/// Backspin the simulation assumes when the device measured no spin (rpm),
/// a mid-iron value. The HUD shows such spin as not measured.
pub const ASSUMED_BACKSPIN_RPM: f64 = 2500.0;

// Physical envelope for launch values; anything outside is clamped to it.
// Each limit sits well beyond what a golf shot produces, so only garbage
// (a misread, a unit error) is ever clamped.

/// Fastest ball speed (mph). Long-drive ball speeds top out near 230 mph.
pub const MAX_BALL_SPEED_MPH: f64 = 250.0;
/// Lowest launch elevation (degrees). Topped shots launch a few degrees
/// down at most.
pub const MIN_LAUNCH_ELEVATION_DEG: f64 = -20.0;
/// Highest launch elevation (degrees). Lob and flop shots launch below 60°.
pub const MAX_LAUNCH_ELEVATION_DEG: f64 = 70.0;
/// Widest launch azimuth either side of the target line (degrees). Shanks
/// start up to about 45° offline.
pub const MAX_LAUNCH_AZIMUTH_DEG: f64 = 60.0;
/// Largest backspin or sidespin magnitude (rpm). Wedges peak near 12 000 rpm.
pub const MAX_SPIN_RPM: f64 = 20_000.0;

/// Why an FRP shot cannot be simulated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RejectReason {
    /// The shot carries no ball flight data.
    #[error("no ball flight data")]
    NoBallFlight,
    /// The ball flight has no launch speed.
    #[error("no launch speed")]
    NoLaunchSpeed,
    /// The ball flight has no launch elevation.
    #[error("no launch elevation")]
    NoLaunchElevation,
    /// A launch value is NaN or infinite.
    #[error("{0} is not a finite number")]
    NotFinite(&'static str),
}

/// A launch value outside the physical envelope, and the limit it was
/// clamped to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clamped {
    /// FRP field name.
    pub field: &'static str,
    /// Value as received, in the unit of `limit`.
    pub received: f64,
    /// Value used for the simulation.
    pub limit: f64,
}

impl std::fmt::Display for Clamped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {:.1} clamped to {:.1}",
            self.field, self.received, self.limit
        )
    }
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
    /// Whether the device measured the backspin in `input` (directly or as
    /// total spin). When false it is [`ASSUMED_BACKSPIN_RPM`].
    pub backspin_measured: bool,
    /// Whether the device measured the sidespin in `input`. When false it
    /// is 0.
    pub sidespin_measured: bool,
    /// Launch values that were clamped to the physical envelope.
    pub clamped: Vec<Clamped>,
}

/// `value` clamped to `min..=max`, recording `field` in `clamped` when it
/// was outside.
fn clamp_logged(
    field: &'static str,
    value: f64,
    min: f64,
    max: f64,
    clamped: &mut Vec<Clamped>,
) -> f64 {
    let limited = value.clamp(min, max);
    if limited != value {
        clamped.push(Clamped {
            field,
            received: value,
            limit: limited,
        });
    }
    limited
}

/// `value`, or [`RejectReason::NotFinite`] for `field`.
fn finite(field: &'static str, value: f64) -> Result<f64, RejectReason> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RejectReason::NotFinite(field))
    }
}

impl ReceivedShot {
    /// Convert an FRP `CompletedShot` into a `ReceivedShot`.
    ///
    /// The shot needs a ball flight with a launch speed and elevation; a
    /// missing azimuth is a straight shot (0°). NaN or infinite launch
    /// values reject the shot, and values outside the physical envelope
    /// (the `MAX_*`/`MIN_*` constants) are clamped and listed in
    /// [`clamped`](Self::clamped).
    ///
    /// Spin comes from the FRP components, or from `total_spin_rpm` +
    /// `spin_axis` when the device sends only that pair. A total spin
    /// without an axis is taken as backspin (less any measured sidespin).
    /// Without any spin the simulation uses [`ASSUMED_BACKSPIN_RPM`] and no
    /// sidespin, flagged in [`backspin_measured`](Self::backspin_measured)
    /// and [`sidespin_measured`](Self::sidespin_measured).
    ///
    /// Display-only values (club data, the device's carry) that are not
    /// finite are dropped.
    ///
    /// # Errors
    ///
    /// Returns why the shot cannot be simulated.
    pub fn from_frp(shot: &flightrelay::CompletedShot) -> Result<Self, RejectReason> {
        let mut ball = shot.ball.clone().ok_or(RejectReason::NoBallFlight)?;
        if let Some(axis) = ball.spin_axis {
            finite("spin_axis", axis)?;
        }
        ball.complete_spin();

        let speed = ball.launch_speed.ok_or(RejectReason::NoLaunchSpeed)?;
        let speed = finite("launch_speed", speed.as_mph())?;
        let elevation = ball
            .launch_elevation
            .ok_or(RejectReason::NoLaunchElevation)?;
        let elevation = finite("launch_elevation", elevation)?;
        let azimuth = finite("launch_azimuth", ball.launch_azimuth.unwrap_or(0.0))?;

        let sidespin = ball.sidespin_rpm.map(f64::from);
        let backspin = ball.backspin_rpm.map(f64::from).or_else(|| {
            ball.total_spin_rpm.map(|total| {
                let (total, side) = (f64::from(total), sidespin.unwrap_or(0.0));
                (total * total - side * side).max(0.0).sqrt()
            })
        });

        let mut clamped = Vec::new();
        let input = ShotInput {
            ball_speed_mph: clamp_logged(
                "launch_speed",
                speed,
                0.0,
                MAX_BALL_SPEED_MPH,
                &mut clamped,
            ),
            launch_angle_deg: clamp_logged(
                "launch_elevation",
                elevation,
                MIN_LAUNCH_ELEVATION_DEG,
                MAX_LAUNCH_ELEVATION_DEG,
                &mut clamped,
            ),
            launch_azimuth_deg: clamp_logged(
                "launch_azimuth",
                azimuth,
                -MAX_LAUNCH_AZIMUTH_DEG,
                MAX_LAUNCH_AZIMUTH_DEG,
                &mut clamped,
            ),
            backspin_rpm: clamp_logged(
                "backspin_rpm",
                backspin.unwrap_or(ASSUMED_BACKSPIN_RPM),
                -MAX_SPIN_RPM,
                MAX_SPIN_RPM,
                &mut clamped,
            ),
            sidespin_rpm: clamp_logged(
                "sidespin_rpm",
                sidespin.unwrap_or(0.0),
                -MAX_SPIN_RPM,
                MAX_SPIN_RPM,
                &mut clamped,
            ),
        };

        let display = |v: Option<f64>| v.filter(|v| v.is_finite());
        let club = shot
            .club
            .as_ref()
            .map_or_else(ClubDelivery::default, |c| ClubDelivery {
                club_speed_mph: display(c.club_speed.map(|v| v.as_mph())),
                path_deg: display(c.path),
                attack_angle_deg: display(c.attack_angle),
                face_angle_deg: display(c.face_angle),
                dynamic_loft_deg: display(c.dynamic_loft),
                smash_factor: display(c.smash_factor),
            });

        let lm_carry_yards = display(ball.carry_distance.map(|d| d.as_yards()));

        Ok(Self {
            device: shot.device.clone(),
            shot_number: shot.key.shot_number,
            input,
            club,
            lm_carry_yards,
            backspin_measured: backspin.is_some(),
            sidespin_measured: sidespin.is_some(),
            clamped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dimple::aero::BallModel;
    use dimple::environment::Environment;
    use dimple::trajectory::simulate_flight;
    use flightrelay::{BallFlight, ClubData, CompletedShot, Distance, ShotKey, Velocity};

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

    fn ball(data: &mut CompletedShot) -> &mut BallFlight {
        data.ball.as_mut().unwrap()
    }

    #[test]
    fn measured_spin_is_flagged() {
        let shot = ReceivedShot::from_frp(&sample_completed_shot()).unwrap();
        assert!(shot.backspin_measured);
        assert!(shot.sidespin_measured);
        assert!(shot.clamped.is_empty());
    }

    #[test]
    fn missing_spin_is_assumed_and_flagged() {
        let mut data = sample_completed_shot();
        ball(&mut data).backspin_rpm = None;
        ball(&mut data).sidespin_rpm = None;
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert!((shot.input.backspin_rpm - ASSUMED_BACKSPIN_RPM).abs() < 0.01);
        assert!((shot.input.sidespin_rpm - 0.0).abs() < 0.01);
        assert!(!shot.backspin_measured);
        assert!(!shot.sidespin_measured);
    }

    #[test]
    fn uses_total_spin_without_axis_as_backspin() {
        let mut data = sample_completed_shot();
        let b = ball(&mut data);
        b.backspin_rpm = None;
        b.sidespin_rpm = None;
        b.total_spin_rpm = Some(9000);
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert!((shot.input.backspin_rpm - 9000.0).abs() < 0.01);
        assert!((shot.input.sidespin_rpm - 0.0).abs() < 0.01);
        assert!(shot.backspin_measured);
        assert!(!shot.sidespin_measured);
    }

    #[test]
    fn total_spin_and_sidespin_give_backspin() {
        let mut data = sample_completed_shot();
        let b = ball(&mut data);
        b.backspin_rpm = None;
        b.sidespin_rpm = Some(-600);
        b.total_spin_rpm = Some(1000);
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert!((shot.input.backspin_rpm - 800.0).abs() < 0.01);
        assert!((shot.input.sidespin_rpm - -600.0).abs() < 0.01);
        assert!(shot.backspin_measured && shot.sidespin_measured);
    }

    #[test]
    fn negative_backspin_passes_through() {
        let mut data = sample_completed_shot();
        ball(&mut data).backspin_rpm = Some(-1500);
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert!((shot.input.backspin_rpm - -1500.0).abs() < 0.01);
        assert!(shot.clamped.is_empty());
        let flight = simulate_flight(&shot.input, &BallModel::TOUR, &Environment::SEA_LEVEL);
        assert!(flight.carry_yards.is_finite() && flight.carry_yards > 0.0);
    }

    #[test]
    fn rejects_missing_launch_elevation() {
        let mut data = sample_completed_shot();
        ball(&mut data).launch_elevation = None;
        assert_eq!(
            ReceivedShot::from_frp(&data).unwrap_err(),
            RejectReason::NoLaunchElevation
        );
    }

    #[test]
    fn rejects_missing_launch_speed() {
        let mut data = sample_completed_shot();
        ball(&mut data).launch_speed = None;
        assert_eq!(
            ReceivedShot::from_frp(&data).unwrap_err(),
            RejectReason::NoLaunchSpeed
        );
    }

    #[test]
    fn missing_azimuth_is_straight() {
        let mut data = sample_completed_shot();
        ball(&mut data).launch_azimuth = None;
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert_eq!(shot.input.launch_azimuth_deg, 0.0);
    }

    #[test]
    fn rejects_non_finite_launch_values() {
        type Corrupt = fn(&mut BallFlight);
        let cases: [(&str, Corrupt); 6] = [
            ("launch_speed", |b| {
                b.launch_speed = Some(Velocity::MilesPerHour(f64::NAN));
            }),
            ("launch_speed", |b| {
                b.launch_speed = Some(Velocity::MetersPerSecond(f64::INFINITY));
            }),
            ("launch_elevation", |b| b.launch_elevation = Some(f64::NAN)),
            ("launch_elevation", |b| {
                b.launch_elevation = Some(f64::NEG_INFINITY);
            }),
            ("launch_azimuth", |b| b.launch_azimuth = Some(f64::NAN)),
            ("spin_axis", |b| {
                b.backspin_rpm = None;
                b.sidespin_rpm = None;
                b.total_spin_rpm = Some(3000);
                b.spin_axis = Some(f64::NAN);
            }),
        ];
        for (field, corrupt) in cases {
            let mut data = sample_completed_shot();
            corrupt(ball(&mut data));
            assert_eq!(
                ReceivedShot::from_frp(&data).unwrap_err(),
                RejectReason::NotFinite(field)
            );
        }
    }

    #[test]
    fn clamps_to_the_physical_envelope() {
        let mut data = sample_completed_shot();
        let b = ball(&mut data);
        b.launch_speed = Some(Velocity::MilesPerHour(400.0));
        b.launch_elevation = Some(85.0);
        b.launch_azimuth = Some(-90.0);
        b.backspin_rpm = Some(40_000);
        b.sidespin_rpm = Some(-25_000);
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert_eq!(shot.input.ball_speed_mph, MAX_BALL_SPEED_MPH);
        assert_eq!(shot.input.launch_angle_deg, MAX_LAUNCH_ELEVATION_DEG);
        assert_eq!(shot.input.launch_azimuth_deg, -MAX_LAUNCH_AZIMUTH_DEG);
        assert_eq!(shot.input.backspin_rpm, MAX_SPIN_RPM);
        assert_eq!(shot.input.sidespin_rpm, -MAX_SPIN_RPM);
        let fields: Vec<_> = shot.clamped.iter().map(|c| c.field).collect();
        assert_eq!(
            fields,
            [
                "launch_speed",
                "launch_elevation",
                "launch_azimuth",
                "backspin_rpm",
                "sidespin_rpm"
            ]
        );
        assert_eq!(shot.clamped[0].received, 400.0);
        assert_eq!(
            shot.clamped[0].to_string(),
            "launch_speed 400.0 clamped to 250.0"
        );
    }

    #[test]
    fn negative_speed_is_a_motionless_ball() {
        let mut data = sample_completed_shot();
        ball(&mut data).launch_speed = Some(Velocity::MilesPerHour(-20.0));
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert_eq!(shot.input.ball_speed_mph, 0.0);
        assert_eq!(shot.clamped.len(), 1);
    }

    #[test]
    fn converts_metric_units() {
        let mut data = sample_completed_shot();
        let b = ball(&mut data);
        b.launch_speed = Some(Velocity::MetersPerSecond(70.0));
        b.carry_distance = Some(Distance::Meters(150.0));
        data.club.as_mut().unwrap().club_speed = Some(Velocity::KilometersPerHour(160.0));
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert!((shot.input.ball_speed_mph - 156.586).abs() < 0.01);
        assert!((shot.lm_carry_yards.unwrap() - 164.042).abs() < 0.01);
        assert!((shot.club.club_speed_mph.unwrap() - 99.419).abs() < 0.01);
    }

    #[test]
    fn drops_non_finite_display_values() {
        let mut data = sample_completed_shot();
        ball(&mut data).carry_distance = Some(Distance::Yards(f64::NAN));
        let club = data.club.as_mut().unwrap();
        club.club_speed = Some(Velocity::MilesPerHour(f64::INFINITY));
        club.path = Some(f64::NAN);
        let shot = ReceivedShot::from_frp(&data).unwrap();
        assert!(shot.lm_carry_yards.is_none());
        assert!(shot.club.club_speed_mph.is_none());
        assert!(shot.club.path_deg.is_none());
        assert!(shot.club.attack_angle_deg.is_some());
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
    fn rejects_shot_without_ball_data() {
        let mut data = sample_completed_shot();
        data.ball = None;
        assert_eq!(
            ReceivedShot::from_frp(&data).unwrap_err(),
            RejectReason::NoBallFlight
        );
    }

    #[test]
    fn preserves_metadata() {
        let shot = ReceivedShot::from_frp(&sample_completed_shot()).unwrap();
        assert_eq!(shot.device, "mevo.0");
        assert_eq!(shot.shot_number, 1);
    }
}
