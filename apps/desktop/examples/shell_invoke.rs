//! Test harness: runs a File Explorer context-menu verb on a selection, through the same
//! shell code Explorer uses (`IContextMenu` on an `IShellItemArray`), so the registered
//! DropTarget verbs and the COM server get exercised end to end without clicking.
//!
//!     cargo run --example shell_invoke -- PiedPiper.Compress a.txt b.txt folder
//!     cargo run --example shell_invoke -- PiedPiper.ExtractTo Archive.tar.pp
//!     cargo run --example shell_invoke -- open Archive.tar.pp
//!
//! All items must be in the same folder, as in a real Explorer selection.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    match win::invoke() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("shell_invoke: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("shell_invoke is a Windows test harness");
}

#[cfg(windows)]
mod win {
    use std::ffi::CString;
    use std::path::PathBuf;

    use windows::core::{HSTRING, PCSTR};
    use windows::Win32::System::Com::{CoInitializeEx, CoTaskMemFree, COINIT_APARTMENTTHREADED};
    use windows::Win32::UI::Shell::Common::ITEMIDLIST;
    use windows::Win32::UI::Shell::{
        BHID_SFUIObject, IContextMenu, IShellItemArray, SHCreateShellItemArrayFromIDLists, SHParseDisplayName,
        CMF_NORMAL, CMINVOKECOMMANDINFO,
    };

    const CMIC_MASK_FLAG_NO_UI: u32 = 0x0000_0400;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreatePopupMenu, DestroyMenu, DispatchMessageW, MsgWaitForMultipleObjects, PeekMessageW, TranslateMessage, MSG,
        PM_REMOVE, QS_ALLINPUT, SW_SHOWNORMAL,
    };

    fn step(what: &str) {
        if std::env::var_os("SHELL_INVOKE_TRACE").is_some() {
            eprintln!("shell_invoke: {what}");
        }
    }

    pub fn invoke() -> Result<(), String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (verb, files) = args.split_first().ok_or("usage: shell_invoke VERB PATH...")?;
        if files.is_empty() {
            return Err("no paths given".into());
        }
        unsafe {
            CoInitializeEx(None, COINIT_APARTMENTTHREADED)
                .ok()
                .map_err(|e| e.to_string())?;
            let mut pidls: Vec<*mut ITEMIDLIST> = Vec::new();
            for f in files {
                let abs = std::path::absolute(PathBuf::from(f)).map_err(|e| e.to_string())?;
                let mut pidl = std::ptr::null_mut();
                SHParseDisplayName(&HSTRING::from(abs.as_os_str()), None, &mut pidl, 0, None)
                    .map_err(|e| format!("{}: {e}", abs.display()))?;
                pidls.push(pidl);
            }
            let consts: Vec<*const ITEMIDLIST> = pidls.iter().map(|p| *p as *const _).collect();
            let items: IShellItemArray = SHCreateShellItemArrayFromIDLists(&consts).map_err(|e| e.to_string())?;
            step("item array");
            let menu: IContextMenu = items
                .BindToHandler(None, &BHID_SFUIObject)
                .map_err(|e| format!("context menu: {e}"))?;
            step("context menu");
            let hmenu = CreatePopupMenu().map_err(|e| e.to_string())?;
            menu.QueryContextMenu(hmenu, 0, 1, 0x7fff, CMF_NORMAL)
                .ok()
                .map_err(|e| e.to_string())?;
            step("menu filled");
            let verb_c = CString::new(verb.as_str()).map_err(|e| e.to_string())?;
            let info = CMINVOKECOMMANDINFO {
                cbSize: std::mem::size_of::<CMINVOKECOMMANDINFO>() as u32,
                // No error dialogs: a failure must come back as an error, not wait for a click.
                fMask: CMIC_MASK_FLAG_NO_UI, // = SEE_MASK_FLAG_NO_UI
                lpVerb: PCSTR(verb_c.as_ptr().cast()),
                nShow: SW_SHOWNORMAL.0,
                ..Default::default()
            };
            let r = menu.InvokeCommand(&info).map_err(|e| format!("invoking {verb}: {e}"));
            let _ = DestroyMenu(hmenu);
            for p in pidls {
                CoTaskMemFree(Some(p as *const _));
            }
            r?;
            // Command-line verbs may be started on a shell worker thread; give it time before
            // this process exits and takes the thread with it.
            let until = std::time::Instant::now() + std::time::Duration::from_secs(3);
            while std::time::Instant::now() < until {
                MsgWaitForMultipleObjects(None, false, 50, QS_ALLINPUT);
                let mut msg = MSG::default();
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        }
        println!("invoked {verb} on {} item(s)", files.len());
        Ok(())
    }
}
