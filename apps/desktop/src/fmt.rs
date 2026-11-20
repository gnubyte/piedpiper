//! Human-readable sizes, durations and counts for progress windows and messages.

/// `532 bytes`, `12.3 KB`, `4.56 MB`, `1.23 GB` (decimal units, as the Finder and the CLI use).
pub fn size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.2} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1} KB", b / 1e3)
    } else if bytes == 1 {
        "1 byte".to_string()
    } else {
        format!("{bytes} bytes")
    }
}

/// A rough remaining time: `less than a minute`, `about 4 minutes`, `about 2 hours 10 minutes`.
pub fn remaining(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s < 60 {
        "less than a minute".to_string()
    } else if s < 3600 {
        let m = (s + 30) / 60;
        format!("about {}", count(m as usize, "minute", "minutes"))
    } else {
        let h = s / 3600;
        let m = (s % 3600 + 30) / 60;
        if m == 0 || h >= 10 {
            format!("about {}", count(h as usize, "hour", "hours"))
        } else {
            format!(
                "about {} {}",
                count(h as usize, "hour", "hours"),
                count(m as usize, "minute", "minutes")
            )
        }
    }
}

/// `1 item`, `3 items`.
pub fn count(n: usize, singular: &str, plural: &str) -> String {
    if n == 1 {
        format!("1 {singular}")
    } else {
        format!("{n} {plural}")
    }
}

/// Wraps a file name in curly quotes for headlines: “Photos.tar.pp”.
pub fn quoted(name: &str) -> String {
    format!("\u{201c}{name}\u{201d}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(size(0), "0 bytes");
        assert_eq!(size(1), "1 byte");
        assert_eq!(size(999), "999 bytes");
        assert_eq!(size(12_345), "12.3 KB");
        assert_eq!(size(4_560_000), "4.56 MB");
        assert_eq!(size(1_230_000_000), "1.23 GB");
    }

    #[test]
    fn durations() {
        assert_eq!(remaining(5.0), "less than a minute");
        assert_eq!(remaining(61.0), "about 1 minute");
        assert_eq!(remaining(250.0), "about 4 minutes");
        assert_eq!(remaining(3600.0), "about 1 hour");
        assert_eq!(remaining(7800.0), "about 2 hours 10 minutes");
        assert_eq!(remaining(40000.0), "about 11 hours");
    }
}
