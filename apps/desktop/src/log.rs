//! Optional diagnostics. Explorer starts the app as a COM server and the Finder through
//! Services, so there is no console to print to; with `log = 1` in the settings file each
//! process appends what it did to `archiver.log` next to it.

use std::io::Write;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Turns logging on or off for this process (the first call wins).
pub fn init(enabled: bool) {
    let _ = ENABLED.set(enabled);
}

pub fn enabled() -> bool {
    ENABLED.get().copied().unwrap_or(false)
}

/// Appends one line to the log if logging is on. Never fails.
pub fn log(msg: impl AsRef<str>) {
    if !enabled() {
        return;
    }
    let Some(dir) = crate::settings::config_dir() else {
        return;
    };
    let _ = std::fs::create_dir_all(&dir);
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("archiver.log"))
    {
        let _ = writeln!(f, "[{secs:.3} pid {}] {}", std::process::id(), msg.as_ref());
    }
}
