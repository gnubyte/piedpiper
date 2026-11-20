//! Compressing selections into archives and extracting archives.
//!
//! PiedPiper's codec works on whole buffers, so both directions hold one archive in memory:
//! compression builds the tarball in memory, then compresses it; extraction decompresses an
//! archive, then writes its files. Nothing is written until the slow part has finished, so a
//! cancelled or failed job leaves no partial files behind.

use std::collections::{HashMap, HashSet};
use std::fmt as stdfmt;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

use piedpiper::{CompressOptions, DecompressOptions};

use crate::fmt;
use crate::job::{Span, Status};
use crate::naming;

/// Why a job stopped.
#[derive(Debug)]
pub enum Error {
    /// The user cancelled.
    Cancelled,
    Io {
        path: PathBuf,
        source: io::Error,
    },
    /// The file does not start with PiedPiper's magic bytes.
    NotArchive(PathBuf),
    /// The codec refused the data (damaged, newer format, custom prior, ...).
    Codec {
        path: PathBuf,
        source: piedpiper::Error,
    },
    /// Nothing to do, with a message for the user.
    Nothing(String),
}

impl stdfmt::Display for Error {
    fn fmt(&self, f: &mut stdfmt::Formatter<'_>) -> stdfmt::Result {
        match self {
            Error::Cancelled => write!(f, "Cancelled."),
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::NotArchive(p) => write!(
                f,
                "{} is not a PiedPiper archive.",
                fmt::quoted(&naming::display_name(p))
            ),
            Error::Codec { path, source } => {
                let n = fmt::quoted(&naming::display_name(path));
                match source {
                    piedpiper::Error::Checksum { .. } => {
                        write!(f, "{n} is damaged: its contents do not match their checksum.")
                    }
                    piedpiper::Error::Prior(_) => write!(
                        f,
                        "{n} was compressed with a custom prior (piedpiper --prior). Extract it with the piedpiper command-line tool and the same prior."
                    ),
                    piedpiper::Error::Version(v) => write!(
                        f,
                        "{n} was made by a newer version of PiedPiper (format version {v}). Update PiedPiper to extract it."
                    ),
                    piedpiper::Error::Format(_) => write!(f, "{n} is damaged or not a PiedPiper archive ({source})."),
                    other => write!(f, "{n}: {other}"),
                }
            }
            Error::Nothing(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

fn io_err(path: &Path, source: io::Error) -> Error {
    Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn codec_err(path: &Path, source: piedpiper::Error) -> Error {
    match source {
        piedpiper::Error::Cancelled => Error::Cancelled,
        source => Error::Codec {
            path: path.to_path_buf(),
            source,
        },
    }
}

fn check_cancel(st: &Status) -> Result<(), Error> {
    if st.is_cancelled() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

/// What a finished job produced.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// Top-level paths created: the new archive, or what each archive was extracted to.
    pub created: Vec<PathBuf>,
    /// Things skipped along the way (special files, unsafe names, links Windows refused).
    pub warnings: Vec<String>,
    /// Archives that could not be extracted when several were selected.
    pub errors: Vec<String>,
    /// Uncompressed and compressed byte counts.
    pub bytes_in: u64,
    pub bytes_out: u64,
}

impl Outcome {
    /// What went wrong, as text for a message box: archives that failed, then things skipped.
    /// `None` when everything worked. Long lists are cut after `max_lines` items each.
    pub fn problems(&self, max_lines: usize) -> Option<String> {
        fn bullets(items: &[String], max: usize, out: &mut Vec<String>) {
            out.extend(items.iter().take(max).map(|i| format!("\u{2022} {i}")));
            if items.len() > max {
                out.push(format!("\u{2022} and {} more", items.len() - max));
            }
        }
        let mut lines = Vec::new();
        if !self.errors.is_empty() {
            lines.push(format!(
                "{} could not be extracted:",
                fmt::count(self.errors.len(), "archive", "archives")
            ));
            bullets(&self.errors, max_lines, &mut lines);
        }
        if !self.warnings.is_empty() {
            if !lines.is_empty() {
                lines.push(String::new());
            }
            bullets(&self.warnings, max_lines, &mut lines);
        }
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    fn merge(&mut self, o: Outcome) {
        self.created.extend(o.created);
        self.warnings.extend(o.warnings);
        self.errors.extend(o.errors);
        self.bytes_in += o.bytes_in;
        self.bytes_out += o.bytes_out;
    }
}

/// Absolute paths of the selected items, in order, without duplicates. Fails on the first
/// item that does not exist.
fn absolute_inputs(paths: &[PathBuf]) -> Result<Vec<PathBuf>, Error> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for p in paths {
        let abs = std::path::absolute(p).map_err(|e| io_err(p, e))?;
        fs::symlink_metadata(&abs).map_err(|e| io_err(&abs, e))?;
        if seen.insert(abs.clone()) {
            out.push(abs);
        }
    }
    Ok(out)
}

/// Total size in bytes of the files in a selection (folders walked, links not followed).
pub fn selection_size(paths: &[PathBuf]) -> u64 {
    fn walk(p: &Path) -> u64 {
        match fs::symlink_metadata(p) {
            Ok(m) if m.is_dir() => fs::read_dir(p)
                .map(|rd| rd.filter_map(|e| e.ok()).map(|e| walk(&e.path())).sum())
                .unwrap_or(0),
            Ok(m) if m.is_file() => m.len(),
            _ => 0,
        }
    }
    paths.iter().map(|p| walk(p)).sum()
}

/// Writes `data` to a new file in `dir`, named `name` or a numbered variant if that is taken.
/// Never replaces an existing file; removes its own partial file on failure.
fn write_new_file(dir: &Path, name: &str, data: &[u8]) -> Result<PathBuf, Error> {
    loop {
        let path = naming::unique_path(dir, name, false);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut f) => {
                let r = f.write_all(data).and_then(|_| f.sync_all());
                drop(f);
                return match r {
                    Ok(()) => Ok(path),
                    Err(e) => {
                        let _ = fs::remove_file(&path);
                        Err(io_err(&path, e))
                    }
                };
            }
            // Someone created it between our check and our open: try the next name.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(io_err(&path, e)),
        }
    }
}

/// Creates a new folder in `dir` named `name`, or a numbered variant if that is taken.
fn create_unique_dir(dir: &Path, name: &str) -> Result<PathBuf, Error> {
    loop {
        let path = naming::unique_path(dir, name, true);
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(io_err(&path, e)),
        }
    }
}

fn remove_created(p: &Path) {
    match fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => {
            let _ = fs::remove_dir_all(p);
        }
        Ok(_) => {
            let _ = fs::remove_file(p);
        }
        Err(_) => {}
    }
}

// ---------------------------------------------------------------------------------------------
// Compression

/// A Compress command.
#[derive(Clone, Debug)]
pub struct CompressRequest {
    /// Selected files and folders.
    pub inputs: Vec<PathBuf>,
    /// Compression level, 1..=9.
    pub level: u32,
    /// Folder for the new archive; by default the folder of the first selected item.
    pub dest_dir: Option<PathBuf>,
    /// Let the codec use the graphics card where it is faster (the archive is unchanged).
    pub gpu: bool,
}

fn gpu_mode(gpu: bool) -> piedpiper::GpuMode {
    if gpu {
        piedpiper::GpuMode::Auto
    } else {
        piedpiper::GpuMode::Off
    }
}

/// Compresses the selection into one new archive (see the crate docs for the naming rules).
pub fn compress(req: &CompressRequest, st: &Status) -> Result<Outcome, Error> {
    let inputs = absolute_inputs(&req.inputs)?;
    let Some(first) = inputs.first() else {
        return Err(Error::Nothing("There is nothing to compress.".into()));
    };
    let dest_dir = match &req.dest_dir {
        Some(d) => d.clone(),
        None => first
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| Error::Nothing(format!("Cannot compress {}: it is not in a folder.", first.display())))?,
    };

    let single_file = inputs.len() == 1 && fs::metadata(first).map(|m| m.is_file()).unwrap_or(false);
    let (name, data, warnings) = if single_file {
        st.update(0.0, format!("Reading {}", fmt::quoted(&naming::display_name(first))));
        let data = fs::read(first).map_err(|e| io_err(first, e))?;
        (
            naming::single_archive_name(&naming::display_name(first)),
            data,
            Vec::new(),
        )
    } else {
        let base = if inputs.len() == 1 {
            naming::display_name(first)
        } else {
            naming::MULTI_NAME.to_string()
        };
        let (tar, warnings) = build_tar(&inputs, st, Span::FULL.sub(0.0, 0.02))?;
        (naming::tar_archive_name(&base), tar, warnings)
    };
    check_cancel(st)?;

    let span = Span::FULL.sub(0.02, 0.99);
    // The codec sets up its model before it reports progress; say what is happening meanwhile.
    st.update(span.at(0.0), format!("Compressing {}", fmt::size(data.len() as u64)));
    let mut opts = CompressOptions::level(req.level.clamp(piedpiper::MIN_LEVEL, piedpiper::MAX_LEVEL));
    opts.gpu = gpu_mode(req.gpu);
    let packed = piedpiper::compress_with_progress(&data, &opts, &mut |done, total| {
        let total = total.max(1);
        st.update(
            span.at(done as f64 / total as f64),
            format!("Compressing: {} of {}", fmt::size(done), fmt::size(total)),
        );
        !st.is_cancelled()
    })
    .map_err(|e| codec_err(&dest_dir.join(&name), e))?;
    check_cancel(st)?;

    st.update(0.99, format!("Saving {}", fmt::quoted(&name)));
    let out = write_new_file(&dest_dir, &name, &packed)?;
    st.set_fraction(1.0);
    Ok(Outcome {
        created: vec![out],
        warnings,
        errors: Vec::new(),
        bytes_in: data.len() as u64,
        bytes_out: packed.len() as u64,
    })
}

enum Kind {
    Dir,
    File,
    Link(PathBuf),
}

struct Item {
    src: PathBuf,
    /// Path inside the archive.
    path: PathBuf,
    kind: Kind,
    meta: Metadata,
}

/// Walks one selected item. Links are stored as links, never followed.
fn collect(
    src: &Path,
    path: PathBuf,
    meta: Metadata,
    items: &mut Vec<Item>,
    warnings: &mut Vec<String>,
    st: &Status,
) -> Result<(), Error> {
    check_cancel(st)?;
    let ft = meta.file_type();
    if ft.is_symlink() {
        match fs::read_link(src) {
            Ok(target) => items.push(Item {
                src: src.to_path_buf(),
                path,
                kind: Kind::Link(target),
                meta,
            }),
            Err(e) => warnings.push(format!("Skipped the link {}: {e}", src.display())),
        }
    } else if ft.is_dir() {
        let mut children = fs::read_dir(src)
            .map_err(|e| io_err(src, e))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| io_err(src, e))?;
        children.sort_by_key(|c| c.file_name());
        items.push(Item {
            src: src.to_path_buf(),
            path: path.clone(),
            kind: Kind::Dir,
            meta,
        });
        for child in children {
            let name = child.file_name();
            if name == ".DS_Store" {
                continue; // Finder view settings: never wanted in an archive
            }
            let child_path = child.path();
            let m = fs::symlink_metadata(&child_path).map_err(|e| io_err(&child_path, e))?;
            collect(&child_path, path.join(&name), m, items, warnings, st)?;
        }
    } else if ft.is_file() {
        items.push(Item {
            src: src.to_path_buf(),
            path,
            kind: Kind::File,
            meta,
        });
    } else {
        warnings.push(format!("Skipped {}: not a file or folder", src.display()));
    }
    Ok(())
}

fn tar_header(meta: &Metadata, kind: tar::EntryType) -> tar::Header {
    let mut h = tar::Header::new_gnu();
    h.set_metadata_in_mode(meta, tar::HeaderMode::Complete);
    h.set_entry_type(kind);
    h.set_size(0);
    h
}

/// Builds the tarball for a multi-item selection. Folders come first (parents before
/// children), then files grouped by extension, so that similar content sits together in the
/// solid stream the model learns from, then links.
fn build_tar(inputs: &[PathBuf], st: &Status, span: Span) -> Result<(Vec<u8>, Vec<String>), Error> {
    st.update(span.at(0.0), "Looking at the selected items".to_string());
    let mut items = Vec::new();
    let mut warnings = Vec::new();
    let mut taken = HashSet::new();
    for input in inputs {
        let meta = fs::symlink_metadata(input).map_err(|e| io_err(input, e))?;
        let top = naming::unique_name(&mut taken, &naming::display_name(input), meta.is_dir());
        collect(input, PathBuf::from(top), meta, &mut items, &mut warnings, st)?;
    }

    let (mut dirs, mut files, mut links) = (Vec::new(), Vec::new(), Vec::new());
    for it in items {
        match it.kind {
            Kind::Dir => dirs.push(it),
            Kind::File => files.push(it),
            Kind::Link(_) => links.push(it),
        }
    }
    let ext_key = |it: &Item| {
        let name = naming::display_name(&it.path).to_lowercase();
        let ext = naming::split_ext(&name).1.to_string();
        (ext, name, it.path.clone())
    };
    files.sort_by_cached_key(ext_key);

    let total: u64 = files.iter().map(|f| f.meta.len()).sum();
    let headers = 1024 * (dirs.len() + files.len() + links.len() + 2);
    let mut b = tar::Builder::new(Vec::with_capacity(total as usize + headers));
    for d in &dirs {
        let mut h = tar_header(&d.meta, tar::EntryType::Directory);
        b.append_data(&mut h, &d.path, io::empty())
            .map_err(|e| io_err(&d.src, e))?;
    }
    let mut done = 0u64;
    for (i, f) in files.iter().enumerate() {
        check_cancel(st)?;
        st.update(
            span.at(done as f64 / total.max(1) as f64),
            format!(
                "Reading {} ({} of {})",
                fmt::quoted(&naming::display_name(&f.src)),
                i + 1,
                files.len()
            ),
        );
        let data = fs::read(&f.src).map_err(|e| io_err(&f.src, e))?;
        done += data.len() as u64;
        let mut h = tar_header(&f.meta, tar::EntryType::Regular);
        h.set_size(data.len() as u64);
        b.append_data(&mut h, &f.path, &data[..])
            .map_err(|e| io_err(&f.src, e))?;
    }
    for l in &links {
        let Kind::Link(target) = &l.kind else { unreachable!() };
        // Archives use forward slashes, also for links made on Windows.
        let target = if cfg!(windows) {
            PathBuf::from(target.to_string_lossy().replace('\\', "/"))
        } else {
            target.clone()
        };
        let mut h = tar_header(&l.meta, tar::EntryType::Symlink);
        if let Err(e) = b.append_link(&mut h, &l.path, &target) {
            warnings.push(format!("Skipped the link {}: {e}", l.src.display()));
        }
    }
    let tar = b.into_inner().map_err(|e| io_err(&inputs[0], e))?;
    st.set_fraction(span.at(1.0));
    Ok((tar, warnings))
}

// ---------------------------------------------------------------------------------------------
// Extraction

/// Where an archive's contents go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractMode {
    /// Like a double-click: next to the archive if it holds one item, else in a new folder.
    Smart,
    /// Into the archive's own folder; top-level items that clash get numbered names.
    Here,
    /// Into a new folder named after the archive.
    ToFolder,
}

/// Extracts each archive in turn. If several are selected and some fail, the others still
/// run and the failures are listed in [`Outcome::errors`]; a single failing archive is an
/// error. Non-archives in a mixed selection are ignored.
pub fn extract(paths: &[PathBuf], mode: ExtractMode, st: &Status) -> Result<Outcome, Error> {
    extract_with(paths, mode, false, st)
}

/// [`extract`], letting the codec use the graphics card for large archives when `gpu` is set.
pub fn extract_with(paths: &[PathBuf], mode: ExtractMode, gpu: bool, st: &Status) -> Result<Outcome, Error> {
    let mut archives = absolute_inputs(paths)?;
    let is_pp = |p: &PathBuf| naming::is_archive_name(&naming::display_name(p));
    if archives.iter().any(is_pp) {
        archives.retain(is_pp);
    }
    if archives.is_empty() {
        return Err(Error::Nothing("Select a PiedPiper archive (.pp) to extract.".into()));
    }
    let sizes: Vec<u64> = archives
        .iter()
        .map(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0).max(1))
        .collect();
    let total: u64 = sizes.iter().sum();
    let mut out = Outcome::default();
    let mut start = 0u64;
    let mut single_failure = None;
    for (p, &size) in archives.iter().zip(&sizes) {
        check_cancel(st)?;
        let span = Span {
            start: start as f64 / total as f64,
            len: size as f64 / total as f64,
        };
        start += size;
        match extract_one(p, mode, gpu, st, span) {
            Ok(o) => out.merge(o),
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => {
                out.errors.push(e.to_string());
                single_failure = Some(e);
            }
        }
    }
    if archives.len() == 1 {
        if let Some(e) = single_failure {
            return Err(e);
        }
    }
    st.set_fraction(1.0);
    Ok(out)
}

fn extract_one(path: &Path, mode: ExtractMode, gpu: bool, st: &Status, span: Span) -> Result<Outcome, Error> {
    let name = naming::display_name(path);
    let parent = path
        .parent()
        .ok_or_else(|| Error::Nothing(format!("Cannot extract {}: it is not in a folder.", path.display())))?;
    st.update(span.at(0.0), format!("Reading {}", fmt::quoted(&name)));
    let packed = fs::read(path).map_err(|e| io_err(path, e))?;
    if !piedpiper::is_compressed(&packed) {
        return Err(Error::NotArchive(path.to_path_buf()));
    }
    let unpack = span.sub(0.0, 0.95);
    st.update(unpack.at(0.0), format!("Decompressing {}", fmt::quoted(&name)));
    let dopts = DecompressOptions { gpu: gpu_mode(gpu), ..Default::default() };
    let plain = piedpiper::decompress_with_progress(&packed, &dopts, &mut |done, total| {
        let total = total.max(1);
        st.update(
            unpack.at(done as f64 / total as f64),
            format!(
                "Decompressing {}: {} of {}",
                fmt::quoted(&name),
                fmt::size(done),
                fmt::size(total)
            ),
        );
        !st.is_cancelled()
    })
    .map_err(|e| codec_err(path, e))?;
    check_cancel(st)?;

    let write = span.sub(0.95, 1.0);
    let mut o = if naming::is_tar_archive_name(&name) && looks_like_tar(&plain) {
        extract_tar(&plain, &name, parent, mode, st, write)?
    } else {
        extract_single(&plain, &name, parent, mode)?
    };
    o.bytes_in = plain.len() as u64;
    o.bytes_out = packed.len() as u64;
    st.set_fraction(span.at(1.0));
    Ok(o)
}

/// A POSIX or GNU tar header with a valid checksum at the start.
fn looks_like_tar(b: &[u8]) -> bool {
    if b.len() < 512 || &b[257..262] != b"ustar" {
        return false;
    }
    let field = &b[148..156];
    let digits: String = field
        .iter()
        .map(|&c| c as char)
        .skip_while(|c| *c == ' ')
        .take_while(|c| c.is_digit(8))
        .collect();
    let Ok(stored) = u32::from_str_radix(&digits, 8) else {
        return false;
    };
    let sum: u32 = b[..512]
        .iter()
        .enumerate()
        .map(|(i, &c)| if (148..156).contains(&i) { 32 } else { c as u32 })
        .sum();
    stored == sum
}

fn extract_single(plain: &[u8], archive_name: &str, parent: &Path, mode: ExtractMode) -> Result<Outcome, Error> {
    let file_name = naming::sanitize_component(&naming::single_output_name(archive_name));
    let created = match mode {
        ExtractMode::ToFolder => {
            let dir = create_unique_dir(parent, &naming::folder_name_for(archive_name))?;
            if let Err(e) = write_new_file(&dir, &file_name, plain) {
                remove_created(&dir);
                return Err(e);
            }
            dir
        }
        ExtractMode::Here | ExtractMode::Smart => write_new_file(parent, &file_name, plain)?,
    };
    Ok(Outcome {
        created: vec![created],
        ..Default::default()
    })
}

/// Two names that the local file system treats as the same (case-insensitive on Windows and
/// macOS).
fn name_key(s: &str) -> String {
    if cfg!(any(windows, target_os = "macos")) {
        s.to_lowercase()
    } else {
        s.to_string()
    }
}

fn entry_components<R: io::Read>(entry: &tar::Entry<'_, R>) -> Result<Vec<String>, naming::UnsafeName> {
    naming::sanitize_entry_path(&String::from_utf8_lossy(&entry.path_bytes()))
}

fn damaged(archive: &Path, e: io::Error) -> Error {
    io_err(
        archive,
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("the tarball inside is damaged: {e}"),
        ),
    )
}

fn extract_tar(
    plain: &[u8],
    archive_name: &str,
    parent: &Path,
    mode: ExtractMode,
    st: &Status,
    span: Span,
) -> Result<Outcome, Error> {
    let archive_path = parent.join(archive_name);

    // Pass 1: the top-level names decide where things go.
    let mut roots: Vec<(String, bool)> = Vec::new(); // (name, is a folder)
    let mut ar = tar::Archive::new(plain);
    for entry in ar.entries().map_err(|e| damaged(&archive_path, e))? {
        let entry = entry.map_err(|e| damaged(&archive_path, e))?;
        let Ok(comps) = entry_components(&entry) else { continue };
        let Some(first) = comps.first() else { continue };
        let is_dir = comps.len() > 1 || entry.header().entry_type().is_dir();
        match roots.iter_mut().find(|(r, _)| name_key(r) == name_key(first)) {
            Some(r) => r.1 |= is_dir,
            None => roots.push((first.clone(), is_dir)),
        }
    }
    if roots.is_empty() {
        return Ok(Outcome {
            warnings: vec![format!("{} is empty.", fmt::quoted(archive_name))],
            ..Default::default()
        });
    }

    let here = match mode {
        ExtractMode::Here => true,
        ExtractMode::ToFolder => false,
        ExtractMode::Smart => roots.len() == 1,
    };
    let mut created = Vec::new();
    let mut rename = HashMap::new();
    let base = if here {
        for (r, is_dir) in &roots {
            let free = naming::unique_path(parent, r, *is_dir);
            rename.insert(name_key(r), naming::display_name(&free));
        }
        parent.to_path_buf()
    } else {
        let dir = create_unique_dir(parent, &naming::folder_name_for(archive_name))?;
        created.push(dir.clone());
        dir
    };

    let mut warnings = Vec::new();
    let r = write_entries(
        plain,
        &archive_path,
        &base,
        &rename,
        st,
        span,
        &mut created,
        &mut warnings,
    );
    if let Err(e) = r {
        for c in &created {
            remove_created(c);
        }
        return Err(e);
    }
    let name = fmt::quoted(archive_name);
    Ok(Outcome {
        created,
        warnings: warnings.into_iter().map(|w| format!("{name}: {w}")).collect(),
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn write_entries(
    plain: &[u8],
    archive_path: &Path,
    base: &Path,
    rename: &HashMap<String, String>,
    st: &Status,
    span: Span,
    created: &mut Vec<PathBuf>,
    warnings: &mut Vec<String>,
) -> Result<(), Error> {
    let total = plain.len().max(1) as f64;
    let map = |comps: &[String]| -> PathBuf {
        let mut p = base.to_path_buf();
        for (i, c) in comps.iter().enumerate() {
            match rename.get(&name_key(c)).filter(|_| i == 0) {
                Some(r) => p.push(r),
                None => p.push(c),
            }
        }
        p
    };
    let mut written: HashSet<PathBuf> = HashSet::new();
    let mut dirs: Vec<(PathBuf, u32)> = Vec::new();
    let mut symlinks: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut hardlinks: Vec<(PathBuf, PathBuf)> = Vec::new();
    let (mut unsafe_names, mut special) = (0usize, 0usize);

    let mut ar = tar::Archive::new(plain);
    for entry in ar.entries().map_err(|e| damaged(archive_path, e))? {
        check_cancel(st)?;
        let mut entry = entry.map_err(|e| damaged(archive_path, e))?;
        let comps = match entry_components(&entry) {
            Ok(c) if !c.is_empty() => c,
            Ok(_) => continue,
            Err(_) => {
                unsafe_names += 1;
                continue;
            }
        };
        let target = map(&comps);
        // In "here" mode each top-level item is new; remember it so a failure can undo it.
        let root = map(&comps[..1]);
        if !created.contains(&root) && !created.iter().any(|c| root.starts_with(c)) {
            created.push(root);
        }
        st.update(
            span.at(entry.raw_file_position() as f64 / total),
            format!(
                "Writing {}",
                fmt::quoted(comps.last().map(String::as_str).unwrap_or(""))
            ),
        );
        let kind = entry.header().entry_type();
        let mode = entry.header().mode().unwrap_or(0o644);
        let parent_of = |p: &Path| -> Result<(), Error> {
            match p.parent() {
                Some(d) => fs::create_dir_all(d).map_err(|e| io_err(d, e)),
                None => Ok(()),
            }
        };
        if kind.is_dir() {
            fs::create_dir_all(&target).map_err(|e| io_err(&target, e))?;
            dirs.push((target, mode));
        } else if kind.is_file() || kind.is_contiguous() || kind.is_gnu_sparse() {
            parent_of(&target)?;
            // An existing path that this extraction did not write is a name that differs only
            // in case from an earlier entry (or a folder of that name): keep both.
            let target = if naming::exists(&target) && !written.contains(&target) {
                let dir = target.parent().unwrap_or(base);
                naming::unique_path(dir, &naming::display_name(&target), false)
            } else {
                target
            };
            let mut f = File::create(&target).map_err(|e| io_err(&target, e))?;
            io::copy(&mut entry, &mut f).map_err(|e| io_err(&target, e))?;
            if let Ok(mtime) = entry.header().mtime() {
                let _ = f.set_modified(UNIX_EPOCH + Duration::from_secs(mtime));
            }
            drop(f);
            set_mode(&target, mode | 0o400);
            written.insert(target);
        } else if kind.is_symlink() {
            if let Some(link) = entry.link_name_bytes() {
                symlinks.push((target, PathBuf::from(String::from_utf8_lossy(&link).into_owned())));
            }
        } else if kind.is_hard_link() {
            let source = entry
                .link_name_bytes()
                .map(|l| naming::sanitize_entry_path(&String::from_utf8_lossy(&l)));
            match source {
                Some(Ok(c)) if !c.is_empty() => hardlinks.push((target, map(&c))),
                _ => unsafe_names += 1,
            }
        } else if kind.is_pax_global_extensions()
            || kind.is_pax_local_extensions()
            || kind.is_gnu_longname()
            || kind.is_gnu_longlink()
        {
            // Metadata for the next entry; the tar reader applies it.
        } else {
            special += 1; // devices, FIFOs
        }
    }

    // Hard links point at files written above; copy if the file system refuses the link.
    for (dst, src) in hardlinks {
        parent_dir(&dst)?;
        if fs::hard_link(&src, &dst).is_err() && fs::copy(&src, &dst).is_err() {
            special += 1;
        }
    }
    // Symbolic links last: nothing above can have been written through one.
    let mut failed_links = 0usize;
    for (dst, target) in symlinks {
        parent_dir(&dst)?;
        if naming::exists(&dst) || make_symlink(&target, &dst).is_err() {
            failed_links += 1;
        }
    }
    // Folder permissions last, so a read-only folder does not block the files inside it.
    for (d, mode) in dirs.iter().rev() {
        set_mode(d, mode | 0o700);
    }

    if unsafe_names > 0 {
        warnings.push(format!(
            "skipped {} with a name pointing outside the destination folder",
            fmt::count(unsafe_names, "entry", "entries")
        ));
    }
    if special > 0 {
        warnings.push(format!(
            "skipped {} (devices, pipes or broken links)",
            fmt::count(special, "special file", "special files")
        ));
    }
    if failed_links > 0 {
        let why = if cfg!(windows) {
            " (Windows creates them only in Developer Mode or as administrator)"
        } else {
            ""
        };
        warnings.push(format!(
            "could not create {}{why}",
            fmt::count(failed_links, "symbolic link", "symbolic links")
        ));
    }
    Ok(())
}

fn parent_dir(p: &Path) -> Result<(), Error> {
    match p.parent() {
        Some(d) => fs::create_dir_all(d).map_err(|e| io_err(d, e)),
        None => Ok(()),
    }
}

#[cfg(unix)]
fn set_mode(p: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(p, fs::Permissions::from_mode(mode & 0o777));
}

#[cfg(not(unix))]
fn set_mode(_p: &Path, _mode: u32) {}

#[cfg(unix)]
fn make_symlink(target: &Path, dst: &Path) -> io::Result<()> {
    std::os::unix::fs::symlink(target, dst)
}

#[cfg(windows)]
fn make_symlink(target: &Path, dst: &Path) -> io::Result<()> {
    let target = PathBuf::from(target.to_string_lossy().replace('/', "\\"));
    let points_at_dir = dst.parent().map(|d| d.join(&target).is_dir()).unwrap_or(false);
    if points_at_dir {
        std::os::windows::fs::symlink_dir(&target, dst)
    } else {
        std::os::windows::fs::symlink_file(&target, dst)
    }
}

#[cfg(not(any(unix, windows)))]
fn make_symlink(_target: &Path, _dst: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "symbolic links are not supported here",
    ))
}
