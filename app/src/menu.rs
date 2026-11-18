//! Which range cyberflight plays: the `--range` argument.

use cf_scene::range::Range;

/// A range the player can pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeChoice {
    /// The flat driving range.
    DrivingRange,
    /// The hole: [`Range::hole`].
    Hole,
}

impl RangeChoice {
    /// The range this choice plays.
    #[must_use]
    pub fn range(self) -> Range {
        match self {
            RangeChoice::DrivingRange => Range::new(),
            RangeChoice::Hole => Range::hole(),
        }
    }
}

/// The range named by `--range driving|hole` in `args`, or `None` when the
/// argument is absent.
///
/// # Errors
/// `--range` without a value, or with a value other than `driving` or
/// `hole`.
pub fn parse_range(args: &[String]) -> anyhow::Result<Option<RangeChoice>> {
    let Some(i) = args.iter().position(|a| a == "--range") else {
        return Ok(None);
    };
    match args.get(i + 1).map(String::as_str) {
        Some("driving") => Ok(Some(RangeChoice::DrivingRange)),
        Some("hole") => Ok(Some(RangeChoice::Hole)),
        Some(other) => anyhow::bail!("--range {other}: expected driving or hole"),
        None => anyhow::bail!("--range needs a value: driving or hole"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        std::iter::once("cyberflight")
            .chain(list.iter().copied())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn parse_range_reads_the_argument() {
        assert_eq!(parse_range(&args(&[])).unwrap(), None);
        assert_eq!(
            parse_range(&args(&["--range", "driving"])).unwrap(),
            Some(RangeChoice::DrivingRange)
        );
        assert_eq!(
            parse_range(&args(&["--windowed", "--range", "hole"])).unwrap(),
            Some(RangeChoice::Hole)
        );
        assert!(parse_range(&args(&["--range", "green"])).is_err());
        assert!(parse_range(&args(&["--range"])).is_err());
    }
}
