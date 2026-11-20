//! File and folder names: what an archive is called, how clashes with existing files are
//! resolved, and how names stored in an archive are made safe for the local file system.

use std::path::{Path, PathBuf};

/// Extension of every PiedPiper stream.
pub const EXT: &str = ".pp";
/// Extension of a multi-file archive (a tarball inside a PiedPiper stream).
pub const TAR_EXT: &str = ".tar.pp";
/// Name of an archive made from several selected items (the Finder's convention).
pub const MULTI_NAME: &str = "Archive";

fn strip_suffix_ci<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    let cut = s.len().checked_sub(suffix.len())?;
    (s.is_char_boundary(cut) && s[cut..].eq_ignore_ascii_case(suffix)).then(|| &s[..cut])
}

/// `*.pp`, ignoring case.
pub fn is_archive_name(name: &str) -> bool {
    strip_suffix_ci(name, EXT).is_some_and(|stem| !stem.is_empty())
}

/// `*.tar.pp`, ignoring case: a tarball to unpack after decompression.
pub fn is_tar_archive_name(name: &str) -> bool {
    strip_suffix_ci(name, TAR_EXT).is_some_and(|stem| !stem.is_empty())
}

/// Name of the archive for one selected file: `report.docx` -> `report.docx.pp`.
pub fn single_archive_name(file_name: &str) -> String {
    format!("{file_name}{EXT}")
}

/// Name of the archive for one selected folder: `Photos` -> `Photos.tar.pp`.
pub fn tar_archive_name(base: &str) -> String {
    format!("{base}{TAR_EXT}")
}

/// What a single-file archive restores to: `report.docx.pp` -> `report.docx`.
pub fn single_output_name(archive_name: &str) -> String {
    match strip_suffix_ci(archive_name, EXT) {
        Some(stem) if !stem.is_empty() => stem.to_string(),
        _ => format!("{archive_name}.out"),
    }
}

/// Folder that "Extract to folder" creates: `Photos.tar.pp` -> `Photos`,
/// `notes.txt.pp` -> `notes`, `release-1.2.pp` -> `release-1.2`.
pub fn folder_name_for(archive_name: &str) -> String {
    let mut base = strip_suffix_ci(archive_name, EXT).unwrap_or(archive_name);
    // Drop one inner extension (".tar", ".docx", ...) but not version-like tails (".2").
    if let Some(stem) = strip_suffix_ci(base, ".tar") {
        base = stem;
    } else if let Some(dot) = base.rfind('.') {
        let ext = &base[dot + 1..];
        let looks_like_ext = dot > 0
            && (1..=5).contains(&ext.len())
            && ext.bytes().all(|b| b.is_ascii_alphanumeric())
            && ext.bytes().any(|b| b.is_ascii_alphabetic());
        if looks_like_ext {
            base = &base[..dot];
        }
    }
    let base = base.trim_end_matches(['.', ' ']);
    if base.is_empty() {
        MULTI_NAME.to_string()
    } else {
        base.to_string()
    }
}

/// Splits `name` into stem and extension for numbering duplicates; `.tar.pp` counts as one
/// extension and a leading dot (`.profile`) as none.
pub fn split_ext(name: &str) -> (&str, &str) {
    if is_tar_archive_name(name) {
        let cut = name.len() - TAR_EXT.len();
        return (&name[..cut], &name[cut..]);
    }
    match name.rfind('.') {
        Some(i) if i > 0 => (&name[..i], &name[i..]),
        _ => (name, ""),
    }
}

/// True if anything (file, folder, or a dangling link) has this path.
pub fn exists(p: &Path) -> bool {
    p.symlink_metadata().is_ok()
}

/// The suffix for the n-th duplicate: ` (2)` on Windows and Linux, ` 2` on macOS, where the
/// Finder numbers copies that way.
pub fn counter(n: u32) -> String {
    if cfg!(target_os = "macos") {
        format!(" {n}")
    } else {
        format!(" ({n})")
    }
}

/// `dir/name` if that is free, else the first free `dir/name (2)`, `dir/name (3)`, ...
/// The number goes before a file's extension (`report (2).docx`, `Archive (2).tar.pp`) and at
/// the end of a folder name.
pub fn unique_path(dir: &Path, name: &str, is_dir: bool) -> PathBuf {
    let first = dir.join(name);
    if !exists(&first) {
        return first;
    }
    let (stem, ext) = if is_dir { (name, "") } else { split_ext(name) };
    (2u32..)
        .map(|n| dir.join(format!("{stem}{}{ext}", counter(n))))
        .find(|p| !exists(p))
        .expect("some numbered name is free")
}

/// Same as [`unique_path`], but checks a set of names already taken (case-insensitively)
/// instead of the file system. Used for top-level names inside an archive being built.
pub fn unique_name(taken: &mut std::collections::HashSet<String>, name: &str, is_dir: bool) -> String {
    let key = |s: &str| s.to_lowercase();
    if taken.insert(key(name)) {
        return name.to_string();
    }
    let (stem, ext) = if is_dir { (name, "") } else { split_ext(name) };
    for n in 2u32.. {
        let candidate = format!("{stem}{}{ext}", counter(n));
        if taken.insert(key(&candidate)) {
            return candidate;
        }
    }
    unreachable!()
}

/// Why an entry name from an archive cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsafeName {
    /// It climbs out of the extraction folder with `..`.
    ParentDir,
}

/// Splits an entry path stored in an archive into components that are safe to create below
/// the extraction folder. Leading `/` and `.` components are dropped (as tar does); `..` is
/// refused. On Windows, backslashes also separate components and each component is cleaned
/// with [`sanitize_component_windows`]. An empty result means there is nothing to create
/// (for example the entry `./`).
pub fn sanitize_entry_path(raw: &str) -> Result<Vec<String>, UnsafeName> {
    let separators: &[char] = if cfg!(windows) { &['/', '\\'] } else { &['/'] };
    let mut out = Vec::new();
    for comp in raw.split(separators) {
        match comp {
            "" | "." => {}
            ".." => return Err(UnsafeName::ParentDir),
            c => out.push(sanitize_component(c)),
        }
    }
    Ok(out)
}

/// Makes one file name usable on this OS. Only Windows needs changes (see
/// [`sanitize_component_windows`]); macOS and Linux accept everything tar can store.
pub fn sanitize_component(c: &str) -> String {
    if cfg!(windows) {
        sanitize_component_windows(c)
    } else {
        c.to_string()
    }
}

/// Windows rules: `<>:"|?*`, `\` and control characters become `_`, trailing dots and spaces
/// are dropped, and reserved device names (`CON`, `NUL`, `COM1`, `aux.txt`, ...) get a `_`
/// prefix.
pub fn sanitize_component_windows(c: &str) -> String {
    let mut s: String = c
        .chars()
        .map(|ch| {
            if (ch as u32) < 0x20 || "<>:\"|?*\\/".contains(ch) {
                '_'
            } else {
                ch
            }
        })
        .collect();
    while s.ends_with('.') || s.ends_with(' ') {
        s.pop();
    }
    if s.is_empty() {
        return "_".to_string();
    }
    let stem = s.split('.').next().unwrap_or("").trim_end().to_ascii_uppercase();
    let numbered =
        |prefix: &str| stem.len() == 4 && stem.starts_with(prefix) && matches!(stem.as_bytes()[3], b'1'..=b'9');
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$")
        || numbered("COM")
        || numbered("LPT");
    if reserved {
        s.insert(0, '_');
    }
    s
}

/// The last component of `p` as text, for messages and archive entry names.
pub fn display_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_names() {
        assert!(is_archive_name("a.pp"));
        assert!(is_archive_name("Photos.TAR.PP"));
        assert!(!is_archive_name(".pp"));
        assert!(!is_archive_name("a.ppt"));
        assert!(is_tar_archive_name("Photos.tar.pp"));
        assert!(!is_tar_archive_name("notes.txt.pp"));
        assert!(!is_tar_archive_name(".tar.pp"));
        assert_eq!(single_archive_name("report.docx"), "report.docx.pp");
        assert_eq!(tar_archive_name("Photos"), "Photos.tar.pp");
    }

    #[test]
    fn output_names() {
        assert_eq!(single_output_name("report.docx.pp"), "report.docx");
        assert_eq!(single_output_name("REPORT.PP"), "REPORT");
        assert_eq!(single_output_name("weird"), "weird.out");
        assert_eq!(folder_name_for("Photos.tar.pp"), "Photos");
        assert_eq!(folder_name_for("notes.txt.pp"), "notes");
        assert_eq!(folder_name_for("release-1.2.pp"), "release-1.2");
        assert_eq!(folder_name_for("my.photos.tar.pp"), "my.photos");
        assert_eq!(folder_name_for("archive.pp"), "archive");
        assert_eq!(folder_name_for(".tar.pp"), "Archive");
    }

    #[test]
    fn extensions() {
        assert_eq!(split_ext("Archive.tar.pp"), ("Archive", ".tar.pp"));
        assert_eq!(split_ext("report.docx"), ("report", ".docx"));
        assert_eq!(split_ext(".profile"), (".profile", ""));
        assert_eq!(split_ext("Makefile"), ("Makefile", ""));
    }

    #[test]
    fn unique_names_in_a_set() {
        let mut taken = std::collections::HashSet::new();
        assert_eq!(unique_name(&mut taken, "a.txt", false), "a.txt");
        assert_eq!(unique_name(&mut taken, "A.TXT", false), format!("A{}.TXT", counter(2)));
        assert_eq!(unique_name(&mut taken, "a.txt", false), format!("a{}.txt", counter(3)));
        assert_eq!(unique_name(&mut taken, "dir.d", true), "dir.d");
        assert_eq!(unique_name(&mut taken, "dir.d", true), format!("dir.d{}", counter(2)));
    }

    #[test]
    fn unique_paths_on_disk() {
        let dir = std::env::temp_dir().join(format!("ppdesktop-naming-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(unique_path(&dir, "Archive.tar.pp", false), dir.join("Archive.tar.pp"));
        std::fs::write(dir.join("Archive.tar.pp"), b"x").unwrap();
        assert_eq!(
            unique_path(&dir, "Archive.tar.pp", false),
            dir.join(format!("Archive{}.tar.pp", counter(2)))
        );
        std::fs::create_dir(dir.join("Photos")).unwrap();
        assert_eq!(
            unique_path(&dir, "Photos", true),
            dir.join(format!("Photos{}", counter(2)))
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn entry_paths() {
        assert_eq!(sanitize_entry_path("a/b/c.txt").unwrap(), ["a", "b", "c.txt"]);
        assert_eq!(sanitize_entry_path("/etc/passwd").unwrap(), ["etc", "passwd"]);
        assert_eq!(sanitize_entry_path("./x/./y/").unwrap(), ["x", "y"]);
        assert_eq!(sanitize_entry_path("./").unwrap(), Vec::<String>::new());
        assert_eq!(sanitize_entry_path("a/../../evil"), Err(UnsafeName::ParentDir));
        assert_eq!(sanitize_entry_path(".."), Err(UnsafeName::ParentDir));
        assert_eq!(sanitize_entry_path("..."), Ok(vec![sanitize_component("...")]));
    }

    #[test]
    fn windows_component_rules() {
        assert_eq!(sanitize_component_windows("a:b?.txt"), "a_b_.txt");
        assert_eq!(sanitize_component_windows("name. "), "name");
        assert_eq!(sanitize_component_windows("..."), "_");
        assert_eq!(sanitize_component_windows("con"), "_con");
        assert_eq!(sanitize_component_windows("aux.txt"), "_aux.txt");
        assert_eq!(sanitize_component_windows("COM7.log"), "_COM7.log");
        assert_eq!(sanitize_component_windows("COM0"), "COM0");
        assert_eq!(sanitize_component_windows("console"), "console");
        assert_eq!(sanitize_component_windows("tab\there"), "tab_here");
        assert_eq!(sanitize_component_windows("C:"), "C_");
    }
}
