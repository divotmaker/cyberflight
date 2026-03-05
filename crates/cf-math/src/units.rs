/// Standard gravity (m/s^2).
pub const G: f64 = 9.80665;

// --- Unit conversions ---

/// Yards to meters.
#[must_use]
pub fn yards_to_meters(yards: f64) -> f64 {
    yards * 0.9144
}

/// Meters to yards.
#[must_use]
pub fn meters_to_yards(meters: f64) -> f64 {
    meters / 0.9144
}

/// MPH to m/s.
#[must_use]
pub fn mph_to_mps(mph: f64) -> f64 {
    mph * 0.44704
}

/// m/s to MPH.
#[must_use]
pub fn mps_to_mph(mps: f64) -> f64 {
    mps / 0.44704
}

/// Degrees to radians.
#[must_use]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

/// Radians to degrees.
#[must_use]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / std::f64::consts::PI
}

/// RPM to rad/s.
#[must_use]
pub fn rpm_to_rads(rpm: f64) -> f64 {
    rpm * std::f64::consts::PI / 30.0
}

/// Meters to feet.
#[must_use]
pub fn meters_to_feet(meters: f64) -> f64 {
    meters / 0.3048
}

/// Feet to meters.
#[must_use]
pub fn feet_to_meters(feet: f64) -> f64 {
    feet * 0.3048
}

/// Display unit for distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Yards,
    Meters,
}

impl Unit {
    /// Convert a value in this unit to meters.
    #[must_use]
    pub fn to_meters(self, value: f64) -> f64 {
        match self {
            Self::Yards => yards_to_meters(value),
            Self::Meters => value,
        }
    }

    /// Convert meters to this unit.
    #[must_use]
    pub fn from_meters(self, meters: f64) -> f64 {
        match self {
            Self::Yards => meters_to_yards(meters),
            Self::Meters => meters,
        }
    }

    /// Short label for display.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Yards => "yd",
            Self::Meters => "m",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_yards_meters() {
        let yards = 150.0;
        let result = meters_to_yards(yards_to_meters(yards));
        assert!((result - yards).abs() < 1e-10);
    }

    #[test]
    fn roundtrip_mph_mps() {
        let mph = 100.0;
        let result = mps_to_mph(mph_to_mps(mph));
        assert!((result - mph).abs() < 1e-10);
    }

    #[test]
    fn roundtrip_deg_rad() {
        let deg = 45.0;
        let result = rad_to_deg(deg_to_rad(deg));
        assert!((result - deg).abs() < 1e-10);
    }

    #[test]
    fn known_conversion_100mph() {
        let mps = mph_to_mps(100.0);
        assert!((mps - 44.704).abs() < 1e-3);
    }

    #[test]
    fn known_conversion_100yards() {
        let meters = yards_to_meters(100.0);
        assert!((meters - 91.44).abs() < 1e-2);
    }
}
