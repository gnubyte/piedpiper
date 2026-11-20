//! A job is one right-click command on a selection. It runs on a worker thread and reports
//! through a shared [`Status`] that the progress window polls; the window cancels it the
//! same way.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::archive::{self, CompressRequest, Error, ExtractMode, Outcome};
use crate::fmt;
use crate::naming;

/// The commands offered in Explorer and the Finder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// All selected items into one archive.
    Compress,
    /// Each selected archive next to itself, in a new folder unless it holds a single item
    /// (what a double-click does, like the Finder's Archive Utility).
    Extract,
    /// Each selected archive's contents straight into the folder it is in.
    ExtractHere,
    /// Each selected archive into a new folder named after it.
    ExtractToFolder,
}

impl Action {
    pub const ALL: [Action; 4] = [
        Action::Compress,
        Action::Extract,
        Action::ExtractHere,
        Action::ExtractToFolder,
    ];

    /// Command-line spelling.
    pub fn arg(self) -> &'static str {
        match self {
            Action::Compress => "compress",
            Action::Extract => "extract",
            Action::ExtractHere => "extract-here",
            Action::ExtractToFolder => "extract-to",
        }
    }

    pub fn from_arg(s: &str) -> Option<Action> {
        Action::ALL.into_iter().find(|a| a.arg() == s)
    }

    pub fn is_extract(self) -> bool {
        self != Action::Compress
    }
}

/// One command on a selection.
#[derive(Clone, Debug)]
pub struct Job {
    pub action: Action,
    pub paths: Vec<PathBuf>,
    /// Compression level (ignored when extracting).
    pub level: u32,
    /// Let the codec use the graphics card where it is faster (output is unchanged).
    pub gpu: bool,
}

impl Job {
    pub fn new(action: Action, paths: Vec<PathBuf>, level: u32) -> Job {
        Job { action, paths, level, gpu: false }
    }

    /// The same job, with GPU acceleration on or off.
    pub fn with_gpu(mut self, gpu: bool) -> Job {
        self.gpu = gpu;
        self
    }

    /// First line of the progress window: `Compressing “report.docx”`, `Extracting 3 archives`.
    pub fn headline(&self) -> String {
        let n = self.paths.len();
        let one = || fmt::quoted(&naming::display_name(&self.paths[0]));
        match (self.action, n) {
            (Action::Compress, 1) => format!("Compressing {}", one()),
            (Action::Compress, _) => format!("Compressing {}", fmt::count(n, "item", "items")),
            (_, 1) => format!("Extracting {}", one()),
            (_, _) => format!("Extracting {}", fmt::count(n, "archive", "archives")),
        }
    }

    /// Runs the job to completion on the calling thread.
    pub fn run(&self, st: &Status) -> Result<Outcome, Error> {
        crate::log::log(format!(
            "{} {} item(s), level {}{}",
            self.action.arg(),
            self.paths.len(),
            self.level,
            if self.gpu { ", GPU allowed" } else { "" }
        ));
        let r = match self.action {
            Action::Compress => archive::compress(
                &CompressRequest {
                    inputs: self.paths.clone(),
                    level: self.level,
                    dest_dir: None,
                    gpu: self.gpu,
                },
                st,
            ),
            Action::Extract => archive::extract_with(&self.paths, ExtractMode::Smart, self.gpu, st),
            Action::ExtractHere => archive::extract_with(&self.paths, ExtractMode::Here, self.gpu, st),
            Action::ExtractToFolder => archive::extract_with(&self.paths, ExtractMode::ToFolder, self.gpu, st),
        };
        match &r {
            Ok(o) => crate::log::log(format!(
                "done: created {:?}, {} warning(s), {} error(s)",
                o.created,
                o.warnings.len(),
                o.errors.len()
            )),
            Err(e) => crate::log::log(format!("failed: {e}")),
        }
        r
    }
}

/// What the progress window shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub headline: String,
    pub detail: String,
    /// Overall progress, 0.0 ..= 1.0.
    pub fraction: f64,
}

/// Progress and cancellation shared between a job and its window.
#[derive(Debug, Default)]
pub struct Status {
    state: Mutex<Snapshot>,
    cancelled: AtomicBool,
}

impl Status {
    pub fn new(headline: impl Into<String>) -> Status {
        Status {
            state: Mutex::new(Snapshot {
                headline: headline.into(),
                ..Default::default()
            }),
            cancelled: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Snapshot> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn snapshot(&self) -> Snapshot {
        self.lock().clone()
    }

    pub fn set_detail(&self, detail: impl Into<String>) {
        self.lock().detail = detail.into();
    }

    pub fn set_fraction(&self, fraction: f64) {
        let mut s = self.lock();
        // Never move backwards: rounding between phases would make the bar flicker.
        s.fraction = s.fraction.max(fraction.clamp(0.0, 1.0));
    }

    pub fn update(&self, fraction: f64, detail: impl Into<String>) {
        let mut s = self.lock();
        s.fraction = s.fraction.max(fraction.clamp(0.0, 1.0));
        s.detail = detail.into();
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// A slice of the overall progress bar given to one phase of a job.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub start: f64,
    pub len: f64,
}

impl Span {
    pub const FULL: Span = Span { start: 0.0, len: 1.0 };

    /// The part from `a` to `b` (fractions of this span).
    pub fn sub(self, a: f64, b: f64) -> Span {
        Span {
            start: self.start + self.len * a,
            len: self.len * (b - a),
        }
    }

    /// Overall fraction at `x` (0..1) through this span.
    pub fn at(self, x: f64) -> f64 {
        self.start + self.len * x.clamp(0.0, 1.0)
    }
}

/// Estimates the remaining time from recent progress (the last 30 seconds).
#[derive(Debug)]
pub struct Eta {
    started: Instant,
    samples: VecDeque<(Instant, f64)>,
}

impl Default for Eta {
    fn default() -> Self {
        Eta::new()
    }
}

impl Eta {
    pub fn new() -> Eta {
        Eta {
            started: Instant::now(),
            samples: VecDeque::new(),
        }
    }

    /// Feeds the current fraction; returns the estimated seconds left once there is enough
    /// history to say anything.
    pub fn update(&mut self, fraction: f64) -> Option<f64> {
        let now = Instant::now();
        self.samples.push_back((now, fraction));
        while self.samples.len() > 2 && now.duration_since(self.samples[0].0) > Duration::from_secs(30) {
            self.samples.pop_front();
        }
        let (t0, f0) = *self.samples.front()?;
        let dt = now.duration_since(t0).as_secs_f64();
        let df = fraction - f0;
        if now.duration_since(self.started) < Duration::from_secs(3) || dt < 1.0 || df <= 0.0 || fraction >= 1.0 {
            return None;
        }
        Some((1.0 - fraction) * dt / df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_round_trip() {
        for a in Action::ALL {
            assert_eq!(Action::from_arg(a.arg()), Some(a));
        }
        assert_eq!(Action::from_arg("nope"), None);
    }

    #[test]
    fn headlines() {
        let j = Job::new(Action::Compress, vec![PathBuf::from("a/report.docx")], 2);
        assert_eq!(j.headline(), "Compressing \u{201c}report.docx\u{201d}");
        let j = Job::new(
            Action::ExtractHere,
            vec![PathBuf::from("x.pp"), PathBuf::from("y.pp")],
            2,
        );
        assert_eq!(j.headline(), "Extracting 2 archives");
    }

    #[test]
    fn spans_and_status() {
        let s = Span::FULL.sub(0.5, 1.0).sub(0.0, 0.5);
        assert_eq!(s, Span { start: 0.5, len: 0.25 });
        assert_eq!(s.at(1.0), 0.75);
        let st = Status::new("x");
        st.set_fraction(0.5);
        st.set_fraction(0.25);
        assert_eq!(st.snapshot().fraction, 0.5);
        assert!(!st.is_cancelled());
        st.cancel();
        assert!(st.is_cancelled());
    }
}
