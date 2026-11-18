//! Which range cyberflight plays: the `--range` argument, or the startup
//! menu when it is absent.

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

/// The menu's items, in order, as they are drawn.
pub const MENU_ITEMS: [&str; 2] = ["DRIVING RANGE", "THE HOLE   PAR 4   300 YDS"];
/// The range each of [`MENU_ITEMS`] picks, in the same order.
const MENU_CHOICES: [RangeChoice; MENU_ITEMS.len()] =
    [RangeChoice::DrivingRange, RangeChoice::Hole];

/// A key the startup menu reacts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuKey {
    Up,
    Down,
    Enter,
}

/// The startup menu: one of [`MENU_ITEMS`] is selected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Menu {
    /// Index into [`MENU_ITEMS`].
    pub selected: usize,
}

impl Menu {
    /// Apply `key`: Up and Down move the selection and stop at the ends;
    /// Enter returns the selected range.
    pub fn key(&mut self, key: MenuKey) -> Option<RangeChoice> {
        match key {
            MenuKey::Up => self.selected = self.selected.saturating_sub(1),
            MenuKey::Down => self.selected = (self.selected + 1).min(MENU_ITEMS.len() - 1),
            MenuKey::Enter => return MENU_CHOICES.get(self.selected).copied(),
        }
        None
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
    fn the_menu_picks_a_range() {
        let mut menu = Menu::default();
        assert_eq!(menu.key(MenuKey::Enter), Some(RangeChoice::DrivingRange));
        assert_eq!(menu.key(MenuKey::Up), None);
        assert_eq!(menu.selected, 0, "Up at the top stays");
        assert_eq!(menu.key(MenuKey::Down), None);
        assert_eq!(menu.key(MenuKey::Down), None);
        assert_eq!(menu.selected, 1, "Down at the bottom stays");
        assert_eq!(menu.key(MenuKey::Enter), Some(RangeChoice::Hole));
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
