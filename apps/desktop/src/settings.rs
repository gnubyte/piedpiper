//! User settings, kept in a small `key = value` file:
//!
//! * Windows: `%APPDATA%\PiedPiper\archiver.ini`
//! * macOS: `~/Library/Application Support/PiedPiper/archiver.ini`
//! * elsewhere: `$XDG_CONFIG_HOME/piedpiper/archiver.ini` (or `~/.config/...`)
//!
//! Keys: `level` (1-9, compression level for new archives), `gpu` (run the neural model on the
//! graphics card where that is faster: AMD Radeon and other Vulkan GPUs), `log` (write a debug
//! log next to the settings file) and `ui = none` (no windows at all; used by automated tests of
//! the Explorer and Finder integration, where nobody could click a dialog away).

use std::fs;
use std::io;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// Compression level for new archives (1 = fastest ... 9 = strongest).
    pub level: u32,
    /// Use the graphics card (Vulkan; AMD Radeon tested) for large files. Never changes the
    /// archives, which extract the same on any computer.
    pub gpu: bool,
    /// Append diagnostics to `archiver.log` in the settings folder.
    pub log: bool,
    /// Never show a window; report problems only in the log.
    pub headless: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            level: piedpiper::DEFAULT_LEVEL,
            gpu: false,
            log: false,
            headless: false,
        }
    }
}

/// The folder holding the settings file and the log.
pub fn config_dir() -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if cfg!(windows) {
        env("APPDATA").map(|d| d.join("PiedPiper"))
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Application Support/PiedPiper"))
    } else {
        env("XDG_CONFIG_HOME")
            .or_else(|| env("HOME").map(|h| h.join(".config")))
            .map(|d| d.join("piedpiper"))
    }
}

/// Path of the settings file.
pub fn settings_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("archiver.ini"))
}

fn parse_bool(v: &str) -> Option<bool> {
    match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

impl Settings {
    /// Reads the settings file; anything missing or unreadable keeps its default.
    pub fn load() -> Settings {
        settings_path()
            .and_then(|p| fs::read_to_string(p).ok())
            .map(|text| Settings::parse(&text))
            .unwrap_or_default()
    }

    pub fn parse(text: &str) -> Settings {
        let mut s = Settings::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let value = value.trim();
            match key.trim().to_ascii_lowercase().as_str() {
                "level" => {
                    if let Ok(l) = value.parse::<u32>() {
                        if (piedpiper::MIN_LEVEL..=piedpiper::MAX_LEVEL).contains(&l) {
                            s.level = l;
                        }
                    }
                }
                "gpu" => s.gpu = parse_bool(value).unwrap_or(s.gpu),
                "log" => s.log = parse_bool(value).unwrap_or(s.log),
                "ui" => s.headless = value.eq_ignore_ascii_case("none"),
                _ => {}
            }
        }
        s
    }

    pub fn to_text(&self) -> String {
        let mut t = String::from("# PiedPiper Archiver settings\n");
        t += &format!(
            "# Compression level for new archives: 1 (fastest) to 9 (strongest).\nlevel = {}\n",
            self.level
        );
        t += &format!(
            "# Use the graphics card (AMD Radeon or another Vulkan GPU) for large files: on or off.\ngpu = {}\n",
            if self.gpu { "on" } else { "off" }
        );
        if self.log {
            t += "log = 1\n";
        }
        if self.headless {
            t += "ui = none\n";
        }
        t
    }

    /// Writes the settings file, creating its folder if needed.
    pub fn save(&self) -> io::Result<()> {
        let path = settings_path().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no settings folder"))?;
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, self.to_text())
    }
}

/// What the settings windows say about GPU acceleration.
pub const GPU_ABOUT: &str = "PiedPiper can run its neural network on the graphics card instead of the processor. \
It works with AMD Radeon graphics (tested on a Radeon RX 6800 XT) and other GPUs whose Vulkan driver keeps \
exact arithmetic. The graphics card computes exactly the same numbers as the processor, so the archives are \
identical either way and extract on any computer, with or without a GPU.\n\n\
With the option on, PiedPiper uses the graphics card where it is faster: compressing files of a few hundred KB \
and more (about 2x to 3x faster on such files, and many times faster on large ones and at the higher levels), \
and extracting large archives. Small files stay on the processor.";

/// The checkbox under the levels.
pub const GPU_OPTION: &str = "Use the graphics card (AMD Radeon or another Vulkan GPU) to compress large files faster";

/// One line on whether this computer has a usable GPU, for the settings window.
pub fn gpu_line() -> String {
    if !piedpiper::GPU_SUPPORT {
        return "GPU acceleration is available in the Windows app and the piedpiper command-line tool.".into();
    }
    match piedpiper::gpu_status() {
        piedpiper::GpuStatus::Available { device, .. } => format!("This computer: {device} is ready."),
        piedpiper::GpuStatus::Unavailable(why) => format!("This computer: no usable GPU ({why})."),
    }
}

/// Short description of each level for the settings window.
pub fn level_label(level: u32) -> String {
    let what = match level {
        1 => "fastest",
        2 => "balanced, the default",
        3 => "best for files up to a few MB",
        4 => "bigger network, for inputs of tens of MB",
        5 => "bigger network, slower",
        6 => "large network, slow",
        7 => "large network, slower",
        8 => "very large network, very slow",
        _ => "largest network, slowest",
    };
    format!("Level {level}: {what}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_print() {
        let s = Settings::parse("# c\nlevel = 3\ngpu = on\nlog=yes\nui = none\nbogus = 1\n");
        assert_eq!(
            s,
            Settings {
                level: 3,
                gpu: true,
                log: true,
                headless: true
            }
        );
        assert!(!Settings::parse("gpu = off").gpu);
        assert!(!Settings::parse("").gpu);
        assert_eq!(Settings::parse(&s.to_text()), s);
        assert_eq!(Settings::parse("level = 42").level, piedpiper::DEFAULT_LEVEL);
        assert_eq!(Settings::parse("").level, piedpiper::DEFAULT_LEVEL);
    }
}
