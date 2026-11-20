//! End-to-end tests of the archive engine: compress selections, extract them every way, and
//! compare the trees. Level 1 and small inputs keep the neural codec quick.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ppdesktop::{compress, extract, naming, CompressRequest, Error, ExtractMode, Status};

const LEVEL: u32 = 1;

/// A fresh folder under the system temp folder, deleted when the test ends.
struct TempDir(PathBuf);

impl std::ops::Deref for TempDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn temp_dir(name: &str) -> TempDir {
    let d = std::env::temp_dir()
        .join("ppdesktop-tests")
        .join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    TempDir(d)
}

fn write(p: &Path, data: &[u8]) {
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, data).unwrap();
}

fn st() -> Status {
    Status::new("test")
}

fn compress_paths(paths: &[PathBuf]) -> Result<ppdesktop::Outcome, Error> {
    compress(
        &CompressRequest {
            inputs: paths.to_vec(),
            level: LEVEL,
            dest_dir: None,
        },
        &st(),
    )
}

/// Relative path (with `/`) -> file contents, or `None` for a folder.
fn tree(root: &Path) -> BTreeMap<String, Option<Vec<u8>>> {
    fn walk(base: &Path, p: &Path, out: &mut BTreeMap<String, Option<Vec<u8>>>) {
        for e in fs::read_dir(p).unwrap() {
            let e = e.unwrap();
            let rel = e
                .path()
                .strip_prefix(base)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if e.file_type().unwrap().is_dir() {
                out.insert(rel, None);
                walk(base, &e.path(), out);
            } else {
                out.insert(rel, Some(fs::read(e.path()).unwrap()));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

fn text(n: usize, seed: &str) -> Vec<u8> {
    (0..n)
        .map(|i| format!("{seed} line {i}: the quick brown fox jumps over the lazy dog\n"))
        .collect::<String>()
        .into_bytes()
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
fn single_file_round_trip() {
    let d = temp_dir("single");
    let src = d.join("notes.txt");
    let data = text(40, "notes");
    write(&src, &data);

    let o = compress_paths(std::slice::from_ref(&src)).unwrap();
    assert_eq!(o.created, vec![d.join("notes.txt.pp")]);
    let packed = fs::read(&o.created[0]).unwrap();
    // A plain PiedPiper stream: the command-line tool and every SDK read it.
    assert_eq!(piedpiper::decompress(&packed).unwrap(), data);
    assert!(packed.len() < data.len() / 3, "{} -> {}", data.len(), packed.len());

    // The original is still there, so "here" picks a numbered name.
    let e = extract(&o.created, ExtractMode::Here, &st()).unwrap();
    assert_eq!(e.created, vec![d.join(format!("notes{}.txt", naming::counter(2)))]);
    assert_eq!(fs::read(&e.created[0]).unwrap(), data);

    let e = extract(&o.created, ExtractMode::ToFolder, &st()).unwrap();
    assert_eq!(e.created, vec![d.join("notes")]);
    assert_eq!(fs::read(d.join("notes").join("notes.txt")).unwrap(), data);

    // A second archive of the same file does not replace the first.
    let o2 = compress_paths(&[src]).unwrap();
    assert_eq!(o2.created, vec![d.join(format!("notes.txt{}.pp", naming::counter(2)))]);
}

#[test]
fn folder_round_trip() {
    let d = temp_dir("folder");
    let project = d.join("Project");
    write(&project.join("README.md"), &text(20, "readme"));
    write(
        &project.join("src/main.rs"),
        b"fn main() {\n    println!(\"hello\");\n}\n",
    );
    write(&project.join("src/lib.rs"), &text(10, "lib"));
    write(
        &project.join("data/blob.bin"),
        &(0..3000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect::<Vec<_>>(),
    );
    write(&project.join("empty.txt"), b"");
    write(
        &project.join("\u{00fc}n\u{00ef}c\u{00f8}d\u{00e9} \u{2713}.txt"),
        "Gr\u{00fc}\u{00df}e".as_bytes(),
    );
    write(&project.join("a/b/c/d/e/deep.txt"), b"deep");
    let long = "a-file-name-well-beyond-the-hundred-bytes-that-a-plain-ustar-header-has-room-for-so-tar-needs-an-extension.txt";
    write(&project.join(long), b"long name");
    fs::create_dir_all(project.join("empty-dir")).unwrap();
    write(&project.join(".DS_Store"), b"finder junk");
    let stamp = UNIX_EPOCH + Duration::from_secs(1_000_000_000);
    fs::File::options()
        .write(true)
        .open(project.join("README.md"))
        .unwrap()
        .set_modified(stamp)
        .unwrap();

    let o = compress_paths(std::slice::from_ref(&project)).unwrap();
    assert_eq!(o.created, vec![d.join("Project.tar.pp")]);
    assert!(o.warnings.is_empty(), "{:?}", o.warnings);

    // One top-level folder: a double-click puts it next to the archive (numbered, since the
    // original is still there).
    let e = extract(&o.created, ExtractMode::Smart, &st()).unwrap();
    let out = d.join(format!("Project{}", naming::counter(2)));
    assert_eq!(e.created, vec![out.clone()]);
    let mut expected = tree(&project);
    expected.remove(".DS_Store");
    assert_eq!(tree(&out), expected);
    let mtime = fs::metadata(out.join("README.md")).unwrap().modified().unwrap();
    assert_eq!(mtime.duration_since(UNIX_EPOCH).unwrap().as_secs(), 1_000_000_000);

    // "To folder" wraps it in a folder named after the archive.
    let e = extract(&o.created, ExtractMode::ToFolder, &st()).unwrap();
    let wrapped = d.join(format!("Project{}", naming::counter(3)));
    assert_eq!(e.created, vec![wrapped.clone()]);
    assert_eq!(tree(&wrapped.join("Project")), expected);
}

#[test]
fn multi_selection_round_trip() {
    let d = temp_dir("multi");
    write(&d.join("a.txt"), &text(5, "a"));
    write(&d.join("b.txt"), &text(5, "b"));
    write(&d.join("sub/c.txt"), &text(5, "c"));
    let sel = [d.join("a.txt"), d.join("b.txt"), d.join("sub")];

    let o = compress_paths(&sel).unwrap();
    assert_eq!(o.created, vec![d.join("Archive.tar.pp")]);
    let o2 = compress_paths(&sel).unwrap();
    assert_eq!(
        o2.created,
        vec![d.join(format!("Archive{}.tar.pp", naming::counter(2)))]
    );

    // Several top-level items: a double-click makes a folder.
    let e = extract(&o.created, ExtractMode::Smart, &st()).unwrap();
    assert_eq!(e.created, vec![d.join("Archive")]);
    assert_eq!(names(&d.join("Archive")), ["a.txt", "b.txt", "sub"]);
    assert_eq!(fs::read(d.join("Archive/sub/c.txt")).unwrap(), text(5, "c"));

    // "Here" in another folder puts the three items straight in it.
    let other = d.join("elsewhere");
    fs::create_dir(&other).unwrap();
    fs::copy(&o.created[0], other.join("Archive.tar.pp")).unwrap();
    let e = extract(&[other.join("Archive.tar.pp")], ExtractMode::Here, &st()).unwrap();
    assert_eq!(e.created.len(), 3);
    assert_eq!(names(&other), ["Archive.tar.pp", "a.txt", "b.txt", "sub"]);

    // Again: every clashing top-level item gets a number, nothing is overwritten.
    write(&other.join("a.txt"), b"changed");
    extract(&[other.join("Archive.tar.pp")], ExtractMode::Here, &st()).unwrap();
    assert_eq!(fs::read(other.join("a.txt")).unwrap(), b"changed");
    assert_eq!(
        fs::read(other.join(format!("a{}.txt", naming::counter(2)))).unwrap(),
        text(5, "a")
    );
    assert!(other.join(format!("sub{}", naming::counter(2))).join("c.txt").is_file());
}

#[test]
fn several_archives_with_one_bad() {
    let d = temp_dir("several");
    write(&d.join("good.txt"), &text(5, "good"));
    let good = compress_paths(&[d.join("good.txt")]).unwrap().created.remove(0);
    write(&d.join("bad.pp"), b"this is a Puppet manifest, not an archive");

    let o = extract(&[good.clone(), d.join("bad.pp")], ExtractMode::ToFolder, &st()).unwrap();
    assert_eq!(o.created, vec![d.join("good")]);
    assert_eq!(o.errors.len(), 1);
    assert!(o.errors[0].contains("bad.pp"), "{}", o.errors[0]);

    // On its own, the bad one is an error.
    assert!(matches!(
        extract(&[d.join("bad.pp")], ExtractMode::Here, &st()),
        Err(Error::NotArchive(_))
    ));
    // Non-archives in a mixed selection are ignored.
    let o = extract(&[d.join("good.txt"), good], ExtractMode::ToFolder, &st()).unwrap();
    assert_eq!(o.created, vec![d.join(format!("good{}", naming::counter(2)))]);
}

#[test]
fn cancelled_jobs_leave_nothing() {
    let d = temp_dir("cancel");
    write(&d.join("x.txt"), &text(5, "x"));
    let s = st();
    s.cancel();
    let r = compress(
        &CompressRequest {
            inputs: vec![d.join("x.txt")],
            level: LEVEL,
            dest_dir: None,
        },
        &s,
    );
    assert!(matches!(r, Err(Error::Cancelled)));
    assert_eq!(names(&d), ["x.txt"]);

    let archive = compress_paths(&[d.join("x.txt")]).unwrap().created.remove(0);
    let r = extract(&[archive], ExtractMode::ToFolder, &s);
    assert!(matches!(r, Err(Error::Cancelled)));
    assert_eq!(names(&d), ["x.txt", "x.txt.pp"]);
}

/// A tarball written by hand, so it can hold names that tar libraries refuse to create.
fn raw_tar(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, data) in entries {
        let mut h = [0u8; 512];
        h[..name.len()].copy_from_slice(name.as_bytes());
        h[100..107].copy_from_slice(b"0000644");
        h[108..115].copy_from_slice(b"0000000");
        h[116..123].copy_from_slice(b"0000000");
        h[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
        h[136..147].copy_from_slice(format!("{:011o}", 1_000_000_000u64).as_bytes());
        h[156] = b'0';
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        h[148..156].copy_from_slice(b"        ");
        let sum: u32 = h.iter().map(|&b| b as u32).sum();
        h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(data);
        out.resize(out.len().div_ceil(512) * 512, 0);
    }
    out.extend_from_slice(&[0u8; 1024]);
    out
}

fn archive_of(dir: &Path, name: &str, tar: &[u8]) -> PathBuf {
    let packed = piedpiper::compress(tar, &piedpiper::CompressOptions::level(LEVEL)).unwrap();
    let p = dir.join(name);
    fs::write(&p, packed).unwrap();
    p
}

#[test]
fn unsafe_names_stay_inside() {
    let d = temp_dir("unsafe");
    let inner = d.join("in");
    fs::create_dir(&inner).unwrap();
    let tar = raw_tar(&[
        ("ok.txt", b"fine"),
        ("../evil.txt", b"escaped"),
        ("/abs.txt", b"absolute"),
        ("dir/../../evil2.txt", b"escaped"),
    ]);
    let a = archive_of(&inner, "evil.tar.pp", &tar);
    let o = extract(&[a], ExtractMode::ToFolder, &st()).unwrap();
    assert_eq!(o.created, vec![inner.join("evil")]);
    assert_eq!(names(&inner.join("evil")), ["abs.txt", "ok.txt"]);
    assert!(!d.join("evil.txt").exists() && !inner.join("evil.txt").exists() && !d.join("evil2.txt").exists());
    assert!(
        o.warnings.iter().any(|w| w.contains("skipped 2 entries")),
        "{:?}",
        o.warnings
    );
}

#[test]
fn tarball_named_without_tar_is_one_file() {
    // Only `.tar.pp` archives are unpacked; anything else restores the single file inside.
    let d = temp_dir("plain");
    let tar = raw_tar(&[("x.txt", b"x")]);
    let a = archive_of(&d, "backup.pp", &tar);
    let o = extract(&[a], ExtractMode::Here, &st()).unwrap();
    assert_eq!(o.created, vec![d.join("backup")]);
    assert_eq!(fs::read(d.join("backup")).unwrap(), tar);
}

#[cfg(windows)]
#[test]
fn windows_names_are_cleaned() {
    let d = temp_dir("winnames");
    let tar = raw_tar(&[("a:b?.txt", b"1"), ("con.txt", b"2"), ("trailing. ", b"3")]);
    let a = archive_of(&d, "names.tar.pp", &tar);
    extract(&[a], ExtractMode::ToFolder, &st()).unwrap();
    assert_eq!(names(&d.join("names")), ["_con.txt", "a_b_.txt", "trailing"]);
}

#[cfg(unix)]
#[test]
fn symlinks_round_trip() {
    let d = temp_dir("links");
    let src = d.join("Bundle");
    write(&src.join("Versions/A/lib.txt"), b"lib");
    std::os::unix::fs::symlink("A", src.join("Versions/Current")).unwrap();
    let o = compress_paths(std::slice::from_ref(&src)).unwrap();
    let e = extract(&o.created, ExtractMode::ToFolder, &st()).unwrap();
    let link = e.created[0].join("Bundle/Versions/Current");
    assert_eq!(fs::read_link(&link).unwrap(), PathBuf::from("A"));
    assert_eq!(fs::read(link.join("lib.txt")).unwrap(), b"lib");
}

#[test]
fn modified_time_survives_single_file_tar() {
    let d = temp_dir("mtime");
    let f = d.join("dir/when.txt");
    write(&f, b"when");
    let t = SystemTime::UNIX_EPOCH + Duration::from_secs(1_234_567_890);
    fs::File::options()
        .write(true)
        .open(&f)
        .unwrap()
        .set_modified(t)
        .unwrap();
    let o = compress_paths(&[d.join("dir")]).unwrap();
    let e = extract(&o.created, ExtractMode::ToFolder, &st()).unwrap();
    let got = fs::metadata(e.created[0].join("dir/when.txt"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(got, t);
}
