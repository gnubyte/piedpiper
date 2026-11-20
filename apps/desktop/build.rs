//! Windows builds embed the program icon, the `.pp` file icon, an application manifest
//! (common controls 6 for the task dialog, per-monitor DPI awareness, long paths) and version
//! information. Other targets need nothing.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=resources/windows");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("resources")
        .join("windows");
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let mut parts: Vec<u16> = version
        .split(['.', '-', '+'])
        .take(3)
        .map(|p| p.parse().unwrap_or(0))
        .collect();
    parts.resize(4, 0);
    let numeric = parts.iter().map(u16::to_string).collect::<Vec<_>>().join(",");
    let path = |name: &str| -> String {
        let p: &Path = &dir.join(name);
        p.display().to_string().replace('\\', "\\\\")
    };
    let rc = format!(
        r#"#pragma code_page(65001)
1 ICON "{app_icon}"
2 ICON "{archive_icon}"
1 24 "{manifest}"
1 VERSIONINFO
FILEVERSION {numeric}
PRODUCTVERSION {numeric}
FILEFLAGSMASK 0x3F
FILEFLAGS 0x0
FILEOS 0x40004
FILETYPE 0x1
FILESUBTYPE 0x0
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "The PiedPiper authors"
      VALUE "FileDescription", "PiedPiper Archiver"
      VALUE "FileVersion", "{version}"
      VALUE "InternalName", "piedpiper-archiver"
      VALUE "LegalCopyright", "MIT License"
      VALUE "OriginalFilename", "piedpiper-archiver.exe"
      VALUE "ProductName", "PiedPiper"
      VALUE "ProductVersion", "{version}"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#,
        app_icon = path("piedpiper.ico"),
        archive_icon = path("archive.ico"),
        manifest = path("app.manifest"),
    );
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("piedpiper-archiver.rc");
    fs::write(&out, rc).unwrap();
    embed_resource::compile(&out, embed_resource::NONE)
        .manifest_required()
        .unwrap();
}
