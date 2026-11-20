//! What Windows File Explorer needs to know about PiedPiper: the registry values that add
//! the right-click commands and the `.pp` file type.
//!
//! The commands are "DropTarget" verbs. Explorer starts `piedpiper-archiver.exe --com ACTION`
//! as a COM local server and hands it the whole selection at once, so selecting 500 files
//! starts one process, not 500 (a plain `command` verb would start one per file).
//!
//! The MSI installer (`packaging/windows/PiedPiper.wxs`) writes the same values under
//! `HKLM\Software\Classes`, with `[#ArchiverExe]` for the program's path; a test keeps the two
//! in sync. `piedpiper-archiver register` writes them under `HKCU\Software\Classes` for a
//! per-user setup without the installer.

use crate::job::Action;

/// COM class that Explorer creates for "Compress with PiedPiper".
pub const CLSID_COMPRESS: &str = "{27A4092B-ED3D-4569-9C13-DF2ED29481D9}";
/// COM class for "Extract here with PiedPiper".
pub const CLSID_EXTRACT_HERE: &str = "{58FBBC3B-004A-4BD4-9932-577D0D2CF1C7}";
/// COM class for "Extract to folder with PiedPiper".
pub const CLSID_EXTRACT_TO: &str = "{3298D9A7-1F73-4CB7-88D9-BD7515A43736}";
/// File type of `.pp` archives.
pub const PROGID: &str = "PiedPiper.Archive";

pub const VERB_COMPRESS: &str = "PiedPiper.Compress";
pub const VERB_EXTRACT_HERE: &str = "PiedPiper.ExtractHere";
pub const VERB_EXTRACT_TO: &str = "PiedPiper.ExtractTo";

/// The COM class behind each verb (the double-click "Extract" is a plain command).
pub fn clsid(action: Action) -> Option<&'static str> {
    match action {
        Action::Compress => Some(CLSID_COMPRESS),
        Action::ExtractHere => Some(CLSID_EXTRACT_HERE),
        Action::ExtractToFolder => Some(CLSID_EXTRACT_TO),
        Action::Extract => None,
    }
}

/// One registry value below `Software\Classes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub key: String,
    /// Value name; empty for the key's default value.
    pub name: String,
    pub data: String,
}

fn v(key: impl Into<String>, name: &str, data: impl Into<String>) -> Value {
    Value {
        key: key.into(),
        name: name.to_string(),
        data: data.into(),
    }
}

/// All registry values, for the program at `exe`.
pub fn values(exe: &str) -> Vec<Value> {
    let quoted = format!("\"{exe}\"");
    let app_icon = format!("{exe},0");
    let archive_icon = format!("{exe},1");
    let mut out = Vec::new();

    for (action, clsid, title) in [
        (Action::Compress, CLSID_COMPRESS, "PiedPiper Compress"),
        (Action::ExtractHere, CLSID_EXTRACT_HERE, "PiedPiper Extract Here"),
        (Action::ExtractToFolder, CLSID_EXTRACT_TO, "PiedPiper Extract To Folder"),
    ] {
        let key = format!("CLSID\\{clsid}");
        out.push(v(&key, "", title));
        out.push(v(
            format!("{key}\\LocalServer32"),
            "",
            format!("{quoted} --com {}", action.arg()),
        ));
    }

    // Files and folders, also mixed selections.
    let verbs = [
        (
            format!("AllFilesystemObjects\\shell\\{VERB_COMPRESS}"),
            "Compress with PiedPiper",
            CLSID_COMPRESS,
            &app_icon,
        ),
        (
            format!("SystemFileAssociations\\.pp\\shell\\{VERB_EXTRACT_HERE}"),
            "Extract here with PiedPiper",
            CLSID_EXTRACT_HERE,
            &archive_icon,
        ),
        (
            format!("SystemFileAssociations\\.pp\\shell\\{VERB_EXTRACT_TO}"),
            "Extract to folder with PiedPiper",
            CLSID_EXTRACT_TO,
            &archive_icon,
        ),
    ];
    for (key, label, clsid, icon) in verbs {
        out.push(v(&key, "MUIVerb", label));
        out.push(v(&key, "Icon", icon.as_str()));
        // Without this, Explorer hides the verb when more than 15 items are selected.
        out.push(v(&key, "MultiSelectModel", "Player"));
        out.push(v(format!("{key}\\DropTarget"), "CLSID", clsid));
    }

    // The .pp file type: icon, name, and double-click = Extract.
    out.push(v(".pp", "", PROGID));
    out.push(v(".pp", "Content Type", "application/x-piedpiper"));
    out.push(v(".pp", "PerceivedType", "compressed"));
    out.push(v(PROGID, "", "PiedPiper archive"));
    out.push(v(PROGID, "FriendlyTypeName", "PiedPiper archive"));
    out.push(v(format!("{PROGID}\\DefaultIcon"), "", archive_icon.as_str()));
    out.push(v(format!("{PROGID}\\shell"), "", "open"));
    out.push(v(format!("{PROGID}\\shell\\open"), "MUIVerb", "Extract"));
    out.push(v(
        format!("{PROGID}\\shell\\open\\command"),
        "",
        format!("{quoted} extract \"%1\""),
    ));
    out
}

/// Keys that belong to PiedPiper alone and are deleted whole on unregistration. The `.pp`
/// key is shared with other programs and only cleaned of PiedPiper's values.
pub fn owned_keys() -> Vec<String> {
    vec![
        format!("CLSID\\{CLSID_COMPRESS}"),
        format!("CLSID\\{CLSID_EXTRACT_HERE}"),
        format!("CLSID\\{CLSID_EXTRACT_TO}"),
        format!("AllFilesystemObjects\\shell\\{VERB_COMPRESS}"),
        format!("SystemFileAssociations\\.pp\\shell\\{VERB_EXTRACT_HERE}"),
        format!("SystemFileAssociations\\.pp\\shell\\{VERB_EXTRACT_TO}"),
        PROGID.to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unescape(s: &str) -> String {
        s.replace("&quot;", "\"")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&apos;", "'")
            .replace("&amp;", "&")
    }

    fn attr(element: &str, name: &str) -> Option<String> {
        let pat = format!(" {name}=\"");
        let start = element.find(&pat)? + pat.len();
        let end = start + element[start..].find('"')?;
        Some(unescape(&element[start..end]))
    }

    /// Every value `register` writes is also in the MSI, and the other way round.
    #[test]
    fn installer_matches_registration() {
        let wxs_path = concat!(env!("CARGO_MANIFEST_DIR"), "/packaging/windows/PiedPiper.wxs");
        let wxs = std::fs::read_to_string(wxs_path).expect("PiedPiper.wxs");
        let mut in_msi = Vec::new();
        for chunk in wxs.split("<RegistryValue").skip(1) {
            let element = &chunk[..chunk.find("/>").expect("self-closing RegistryValue")];
            if attr(element, "Root").as_deref() != Some("HKCR") {
                continue;
            }
            in_msi.push(Value {
                key: attr(element, "Key").expect("Key"),
                name: attr(element, "Name").unwrap_or_default(),
                data: attr(element, "Value").expect("Value"),
            });
        }
        let expected = values("[#ArchiverExe]");
        for e in &expected {
            assert!(in_msi.contains(e), "missing from PiedPiper.wxs: {e:?}");
        }
        for m in &in_msi {
            assert!(
                expected.contains(m),
                "PiedPiper.wxs has a value `register` does not write: {m:?}"
            );
        }
    }

    #[test]
    fn every_verb_has_a_class() {
        let vals = values("C:\\x.exe");
        for clsid in [CLSID_COMPRESS, CLSID_EXTRACT_HERE, CLSID_EXTRACT_TO] {
            assert!(vals.iter().any(|v| v.name == "CLSID" && v.data == clsid));
            assert!(vals.iter().any(|v| v.key == format!("CLSID\\{clsid}\\LocalServer32")));
        }
        for key in owned_keys() {
            assert!(
                vals.iter()
                    .any(|v| v.key == key || v.key.starts_with(&format!("{key}\\"))),
                "{key}"
            );
        }
    }
}
