//! `register` / `unregister`: write or remove the Explorer integration under
//! `HKCU\Software\Classes` (this user) or `HKLM\Software\Classes` (`--machine`, all users).
//! The MSI installer writes the same values itself; see `ppdesktop::explorer`.

use std::process::ExitCode;

use ppdesktop::explorer;
use windows::core::{HSTRING, PCWSTR};
use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS, WIN32_ERROR};
use windows::Win32::System::Console::GetConsoleWindow;
use windows::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteKeyValueW, RegDeleteKeyW, RegDeleteTreeW, RegGetValueW, RegOpenKeyExW,
    RegQueryInfoKeyW, RegSetValueExW, HKEY, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ, KEY_SET_VALUE,
    REG_OPTION_NON_VOLATILE, REG_SZ, RRF_RT_REG_SZ,
};
use windows::Win32::UI::Shell::{SHChangeNotify, SHCNE_ASSOCCHANGED, SHCNF_IDLIST};

fn check(r: WIN32_ERROR, what: &str) -> Result<(), String> {
    if r == ERROR_SUCCESS {
        Ok(())
    } else {
        Err(format!(
            "{what}: {}",
            windows::core::Error::from(r.to_hresult()).message()
        ))
    }
}

fn classes(key: &str) -> String {
    format!("Software\\Classes\\{key}")
}

fn set_string(root: HKEY, key: &str, name: &str, data: &str) -> Result<(), String> {
    let mut hkey = HKEY::default();
    unsafe {
        check(
            RegCreateKeyExW(
                root,
                &HSTRING::from(key),
                None,
                PCWSTR::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_SET_VALUE,
                None,
                &mut hkey,
                None,
            ),
            key,
        )?;
        let wide: Vec<u16> = data.encode_utf16().chain(Some(0)).collect();
        let bytes = std::slice::from_raw_parts(wide.as_ptr().cast::<u8>(), wide.len() * 2);
        let r = RegSetValueExW(hkey, &HSTRING::from(name), None, REG_SZ, Some(bytes));
        let _ = RegCloseKey(hkey);
        check(r, key)
    }
}

fn get_string(root: HKEY, key: &str, name: &str) -> Option<String> {
    let mut buf = vec![0u16; 2048];
    let mut len = (buf.len() * 2) as u32;
    let r = unsafe {
        RegGetValueW(
            root,
            &HSTRING::from(key),
            &HSTRING::from(name),
            RRF_RT_REG_SZ,
            None,
            Some(buf.as_mut_ptr().cast()),
            Some(&mut len),
        )
    };
    if r != ERROR_SUCCESS {
        return None;
    }
    let n = (len as usize / 2).saturating_sub(1).min(buf.len());
    Some(String::from_utf16_lossy(&buf[..n]))
}

fn delete_tree(root: HKEY, key: &str) -> Result<(), String> {
    let r = unsafe { RegDeleteTreeW(root, &HSTRING::from(key)) };
    if r == ERROR_FILE_NOT_FOUND {
        return Ok(());
    }
    check(r, key)?;
    // RegDeleteTree empties the key; remove the key itself too.
    let r = unsafe { RegDeleteKeyW(root, &HSTRING::from(key)) };
    if r == ERROR_FILE_NOT_FOUND {
        Ok(())
    } else {
        check(r, key)
    }
}

fn delete_value(root: HKEY, key: &str, name: &str) -> Result<(), String> {
    let r = unsafe { RegDeleteKeyValueW(root, &HSTRING::from(key), &HSTRING::from(name)) };
    if r == ERROR_FILE_NOT_FOUND {
        Ok(())
    } else {
        check(r, key)
    }
}

/// Deletes `key` if it has neither values nor subkeys left (keys shared with other programs).
fn delete_if_empty(root: HKEY, key: &str) {
    unsafe {
        let mut hkey = HKEY::default();
        if RegOpenKeyExW(root, &HSTRING::from(key), None, KEY_READ, &mut hkey) != ERROR_SUCCESS {
            return;
        }
        let (mut subkeys, mut values) = (0u32, 0u32);
        let r = RegQueryInfoKeyW(
            hkey,
            None,
            None,
            None,
            Some(&mut subkeys),
            None,
            None,
            Some(&mut values),
            None,
            None,
            None,
            None,
        );
        let _ = RegCloseKey(hkey);
        if r == ERROR_SUCCESS && subkeys == 0 && values == 0 {
            let _ = RegDeleteKeyW(root, &HSTRING::from(key));
        }
    }
}

fn add(root: HKEY) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find this program: {e}"))?;
    let exe = exe.to_string_lossy();
    for v in explorer::values(&exe) {
        set_string(root, &classes(&v.key), &v.name, &v.data)?;
    }
    Ok(())
}

fn remove(root: HKEY) -> Result<(), String> {
    for key in explorer::owned_keys() {
        delete_tree(root, &classes(&key))?;
    }
    // The .pp key may hold other programs' values: take out only ours.
    let pp = classes(".pp");
    if get_string(root, &pp, "").as_deref() == Some(explorer::PROGID) {
        delete_value(root, &pp, "")?;
        if get_string(root, &pp, "PerceivedType").as_deref() == Some("compressed") {
            delete_value(root, &pp, "PerceivedType")?;
        }
    }
    if get_string(root, &pp, "Content Type").as_deref() == Some("application/x-piedpiper") {
        delete_value(root, &pp, "Content Type")?;
    }
    delete_if_empty(root, &pp);
    // Parent keys registration may have created; removed only when nothing else uses them.
    for key in [
        "SystemFileAssociations\\.pp\\shell",
        "SystemFileAssociations\\.pp",
        "SystemFileAssociations",
        "AllFilesystemObjects\\shell",
        "AllFilesystemObjects",
    ] {
        delete_if_empty(root, &classes(key));
    }
    Ok(())
}

/// Tells Explorer that file associations changed, so icons and menus refresh.
pub fn refresh_shell() {
    unsafe { SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None) };
}

pub fn register(add_it: bool, machine: bool, quiet: bool) -> ExitCode {
    let root = if machine { HKEY_LOCAL_MACHINE } else { HKEY_CURRENT_USER };
    let scope = if machine { "all users" } else { "this user" };
    let result = if add_it { add(root) } else { remove(root) };
    refresh_shell();
    let has_console = unsafe { !GetConsoleWindow().is_invalid() };
    match result {
        Ok(()) => {
            let msg = if add_it {
                format!("PiedPiper's commands are now in File Explorer's right-click menu for {scope}.")
            } else {
                format!("PiedPiper's commands were removed from File Explorer for {scope}.")
            };
            println!("{msg}");
            if !quiet && !has_console {
                super::ui::info_box(&msg);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            let hint = if machine {
                " (registering for all users needs an administrator)"
            } else {
                ""
            };
            let msg = format!("Could not update the registry{hint}: {e}");
            eprintln!("piedpiper-archiver: {msg}");
            if !quiet && !has_console {
                super::ui::error_box(&msg);
            }
            ExitCode::FAILURE
        }
    }
}
