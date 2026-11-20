//! The engine behind PiedPiper's right-click commands in Windows File Explorer and the macOS
//! Finder: **Compress**, **Extract** (what a double-click does), **Extract here** and
//! **Extract to folder**, each on any number of selected items.
//!
//! Archive format: a single selected file becomes `name.ext.pp`, a plain PiedPiper stream
//! (the same thing `piedpiper name.ext` writes). Several items, or a folder, become
//! `Name.tar.pp`: a tarball in one PiedPiper stream, so the model learns across files and
//! the core hands photos, documents and archives inside the tarball to their own recoders.
//! Both are ordinary PiedPiper streams that the command-line tool reads too.
//!
//! The platform front ends (progress windows, Explorer's COM verbs, Finder services) live in
//! the binary; everything here is plain Rust and runs the same on every OS.

pub mod archive;
pub mod explorer;
pub mod fmt;
pub mod job;
pub mod log;
pub mod naming;
pub mod settings;

pub use archive::{compress, extract, CompressRequest, Error, ExtractMode, Outcome};
pub use job::{Action, Eta, Job, Snapshot, Span, Status};
pub use settings::Settings;
