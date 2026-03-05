/// Atmospheric environment for ball flight simulation.
#[derive(Debug, Clone, Copy)]
pub struct Environment {
    /// Altitude above sea level (meters).
    pub altitude_m: f64,
    /// Ambient temperature (Celsius).
    pub temperature_c: f64,
}

impl Environment {
    /// Sea level, 15°C (ISA standard).
    pub const SEA_LEVEL: Self = Self {
        altitude_m: 0.0,
        temperature_c: 15.0,
    };

    /// Denver, CO — 5,280 ft (1,609 m), ~17°C summer average.
    pub const DENVER: Self = Self {
        altitude_m: 1609.0,
        temperature_c: 17.0,
    };

    /// Mexico City — 7,350 ft (2,240 m).
    pub const MEXICO_CITY: Self = Self {
        altitude_m: 2240.0,
        temperature_c: 16.0,
    };

    /// Air density (kg/m³) using the barometric formula for the troposphere.
    ///
    /// ρ = ρ₀ × (1 - L·h / T₀)^(g·M / (R·L))
    ///
    /// where:
    /// - ρ₀ = 1.225 kg/m³ (ISA sea level density at 15°C)
    /// - L = 0.0065 K/m (temperature lapse rate)
    /// - T₀ = 288.15 K (ISA sea level temperature)
    /// - g = 9.80665 m/s²
    /// - M = 0.0289644 kg/mol (molar mass of dry air)
    /// - R = 8.31447 J/(mol·K) (universal gas constant)
    ///
    /// Temperature correction: scales by T_standard / T_actual at altitude.
    #[must_use]
    pub fn air_density(&self) -> f64 {
        const RHO_0: f64 = 1.225;
        const L: f64 = 0.0065; // K/m
        const T0: f64 = 288.15; // K
        const EXPONENT: f64 = 5.2559; // g·M / (R·L)

        let h = self.altitude_m.max(0.0);

        // Standard temperature at this altitude
        let t_standard = T0 - L * h;
        // Actual temperature
        let t_actual = self.temperature_c + 273.15;

        // Barometric formula (pressure-based density)
        let rho_standard = RHO_0 * (1.0 - L * h / T0).powf(EXPONENT);

        // Temperature correction: ideal gas law, ρ ∝ 1/T at constant pressure
        rho_standard * (t_standard / t_actual)
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::SEA_LEVEL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sea_level_density() {
        let env = Environment::SEA_LEVEL;
        let rho = env.air_density();
        assert!(
            (rho - 1.225).abs() < 0.001,
            "sea level density should be ~1.225, got {rho:.4}"
        );
    }

    #[test]
    fn denver_less_dense() {
        let sea = Environment::SEA_LEVEL.air_density();
        let denver = Environment::DENVER.air_density();
        let reduction = 1.0 - denver / sea;
        assert!(
            reduction > 0.12 && reduction < 0.25,
            "Denver should be 12-25% less dense, got {:.1}%",
            reduction * 100.0
        );
    }

    #[test]
    fn mexico_city_less_than_denver() {
        let denver = Environment::DENVER.air_density();
        let mexico = Environment::MEXICO_CITY.air_density();
        assert!(
            mexico < denver,
            "Mexico City ({mexico:.4}) should be less dense than Denver ({denver:.4})"
        );
    }

    #[test]
    fn hot_day_less_dense() {
        let cool = Environment {
            altitude_m: 0.0,
            temperature_c: 10.0,
        };
        let hot = Environment {
            altitude_m: 0.0,
            temperature_c: 35.0,
        };
        assert!(
            hot.air_density() < cool.air_density(),
            "hot air should be less dense"
        );
    }
}
