//! Docs tooling: opens File Explorer's right-click menu for a selection, built by the shell
//! exactly as Explorer builds it (so it includes PiedPiper's registered commands), over a plain
//! white window, and saves a screenshot of it as a 32-bit BMP. Used for the README.
//!
//!     cargo run --example menu_screenshot -- OUT.bmp PATH...
//!
//! The commands must be registered first (`piedpiper-archiver register`). The menu shows the
//! classic style, which Windows 11 lists under "Show more options".

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    match win::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("menu_screenshot: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("menu_screenshot is a Windows tool");
}

#[cfg(windows)]
mod win {
    use std::cell::RefCell;
    use std::path::PathBuf;
    use std::time::Duration;

    use windows::core::{w, Interface, HSTRING};
    use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
    use windows::Win32::Graphics::Gdi::{
        BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits, GetStockObject,
        ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HBRUSH, SRCCOPY, WHITE_BRUSH,
    };
    use windows::Win32::System::Com::{CoInitializeEx, CoTaskMemFree, COINIT_APARTMENTTHREADED};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
    use windows::Win32::UI::Input::KeyboardAndMouse::SetFocus;
    use windows::Win32::UI::HiDpi::{SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2};
    use windows::Win32::UI::Shell::Common::ITEMIDLIST;
    use windows::Win32::UI::Shell::{
        BHID_SFUIObject, IContextMenu, IContextMenu3, IShellItemArray, SHCreateShellItemArrayFromIDLists,
        SHParseDisplayName, CMF_CANRENAME, CMF_EXPLORE, CMF_NORMAL,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, CreatePopupMenu, CreateWindowExW, DefWindowProcW, DestroyMenu, DestroyWindow, FindWindowW,
        GetForegroundWindow, GetMenuItemCount, GetSystemMetrics, GetWindowRect, GetWindowThreadProcessId, PostMessageW,
        RegisterClassW, SetForegroundWindow, ShowWindow, TrackPopupMenuEx, SM_CXSCREEN,
        SM_CYSCREEN, SW_SHOW, TPM_RETURNCMD, TPM_RIGHTBUTTON, WM_CANCELMODE, WM_DRAWITEM, WM_INITMENUPOPUP,
        WM_MEASUREITEM, WM_MENUCHAR, WNDCLASSW, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_POPUP, WS_VISIBLE,
    };

    thread_local! {
        static MENU: RefCell<Option<IContextMenu3>> = const { RefCell::new(None) };
    }

    /// Owner-drawn parts of the shell menu (icons, "Send to", "Open with") are drawn by the
    /// context menu handler, which gets these messages forwarded.
    unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if matches!(msg, WM_INITMENUPOPUP | WM_DRAWITEM | WM_MEASUREITEM | WM_MENUCHAR) {
            let handled = MENU.with(|m| {
                m.borrow().as_ref().and_then(|cm| {
                    let mut result = LRESULT(0);
                    unsafe { cm.HandleMenuMsg2(msg, wparam, lparam, Some(&mut result)) }.ok().map(|_| result)
                })
            });
            if let Some(r) = handled {
                return r;
            }
        }
        unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
    }

    /// Saves a screen rectangle as a top-down 32-bit BMP.
    fn capture(r: RECT, out: &PathBuf) -> Result<(), String> {
        let (w, h) = (r.right - r.left, r.bottom - r.top);
        unsafe {
            let screen = GetDC(None);
            let mem = CreateCompatibleDC(Some(screen));
            let bmp = CreateCompatibleBitmap(screen, w, h);
            let old = SelectObject(mem, bmp.into());
            BitBlt(mem, 0, 0, w, h, Some(screen), r.left, r.top, SRCCOPY).map_err(|e| e.to_string())?;
            let mut info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: w,
                    biHeight: -h,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut pixels = vec![0u8; (w * h * 4) as usize];
            GetDIBits(mem, bmp, 0, h as u32, Some(pixels.as_mut_ptr().cast()), &mut info, DIB_RGB_COLORS);
            SelectObject(mem, old);
            let _ = DeleteObject(bmp.into());
            let _ = DeleteDC(mem);
            ReleaseDC(None, screen);
            let size = 14 + 40 + pixels.len() as u32;
            let mut file = Vec::with_capacity(size as usize);
            file.extend_from_slice(b"BM");
            file.extend_from_slice(&size.to_le_bytes());
            file.extend_from_slice(&0u32.to_le_bytes());
            file.extend_from_slice(&54u32.to_le_bytes());
            file.extend_from_slice(&40u32.to_le_bytes());
            file.extend_from_slice(&w.to_le_bytes());
            file.extend_from_slice(&(-h).to_le_bytes());
            file.extend_from_slice(&1u16.to_le_bytes());
            file.extend_from_slice(&32u16.to_le_bytes());
            file.extend_from_slice(&[0u8; 24]);
            file.extend_from_slice(&pixels);
            std::fs::write(out, file).map_err(|e| e.to_string())
        }
    }

    pub fn run() -> Result<(), String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let (out, files) = args.split_first().ok_or("usage: menu_screenshot OUT.bmp PATH...")?;
        let out = std::path::absolute(out).map_err(|e| e.to_string())?;
        unsafe {
            let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
            CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok().map_err(|e| e.to_string())?;

            // The selection, as Explorer hands it to the shell.
            let mut pidls: Vec<*mut ITEMIDLIST> = Vec::new();
            for f in files {
                let abs = std::path::absolute(f).map_err(|e| e.to_string())?;
                let mut pidl = std::ptr::null_mut();
                SHParseDisplayName(&HSTRING::from(abs.as_os_str()), None, &mut pidl, 0, None)
                    .map_err(|e| format!("{}: {e}", abs.display()))?;
                pidls.push(pidl);
            }
            let consts: Vec<*const ITEMIDLIST> = pidls.iter().map(|p| *p as *const _).collect();
            let items: IShellItemArray = SHCreateShellItemArrayFromIDLists(&consts).map_err(|e| e.to_string())?;
            let menu: IContextMenu = items.BindToHandler(None, &BHID_SFUIObject).map_err(|e| e.to_string())?;
            MENU.with(|m| *m.borrow_mut() = menu.cast::<IContextMenu3>().ok());

            // A plain white window behind the menu, so the screenshot has a clean background.
            let instance = GetModuleHandleW(None).map_err(|e| e.to_string())?;
            let class = WNDCLASSW {
                lpfnWndProc: Some(wndproc),
                hInstance: instance.into(),
                lpszClassName: w!("PiedPiperMenuShot"),
                hbrBackground: HBRUSH(GetStockObject(WHITE_BRUSH).0),
                ..Default::default()
            };
            RegisterClassW(&class);
            let (sw, sh) = (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
            let (x, y, ww, wh) = (sw / 2 - 600, 20, 1200, sh - 80);
            let owner = CreateWindowExW(
                WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                w!("PiedPiperMenuShot"),
                w!(""),
                WS_POPUP | WS_VISIBLE,
                x,
                y,
                ww,
                wh,
                None,
                None,
                Some(instance.into()),
                None,
            )
            .map_err(|e| e.to_string())?;
            let _ = ShowWindow(owner, SW_SHOW);
            // A menu needs its owner in the foreground, which Windows only grants to the process
            // the user is working in: borrow that input state for a moment.
            let fg = GetForegroundWindow();
            let fg_thread = GetWindowThreadProcessId(fg, None);
            let me = GetCurrentThreadId();
            let attached = fg_thread != 0 && fg_thread != me && AttachThreadInput(me, fg_thread, true).as_bool();
            let _ = BringWindowToTop(owner);
            let _ = SetForegroundWindow(owner);
            let _ = SetFocus(Some(owner));
            if GetForegroundWindow() != owner {
                eprintln!("menu_screenshot: could not bring the window to the front; the menu may not open");
            }

            let hmenu = CreatePopupMenu().map_err(|e| e.to_string())?;
            menu.QueryContextMenu(hmenu, 0, 1, 0x7fff, CMF_NORMAL | CMF_EXPLORE | CMF_CANRENAME)
                .ok()
                .map_err(|e| e.to_string())?;

            // While the menu is open (TrackPopupMenuEx blocks), photograph it from another thread.
            let owner_raw = owner.0 as isize;
            let out_c = out.clone();
            let shooter = std::thread::spawn(move || -> Result<(), String> {
                let delay = std::env::var("MENU_SHOT_DELAY_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
                std::thread::sleep(Duration::from_millis(delay));
                let found = match FindWindowW(w!("#32768"), None) {
                    Ok(h) => h,
                    Err(_) => {
                        let _ = PostMessageW(Some(HWND(owner_raw as *mut _)), WM_CANCELMODE, WPARAM(0), LPARAM(0));
                        return Err("the menu did not open".to_string());
                    }
                };
                let mut r = RECT::default();
                GetWindowRect(found, &mut r).map_err(|e| e.to_string())?;
                // Include the drop shadow around the menu.
                let pad = 24;
                let area = RECT { left: r.left - pad, top: r.top - pad, right: r.right + pad, bottom: r.bottom + pad };
                let res = capture(area, &out_c);
                let _ = PostMessageW(Some(HWND(owner_raw as *mut _)), WM_CANCELMODE, WPARAM(0), LPARAM(0));
                res
            });
            eprintln!("menu_screenshot: {} menu items", GetMenuItemCount(Some(hmenu)));
            let t0 = std::time::Instant::now();
            let picked = TrackPopupMenuEx(hmenu, (TPM_RETURNCMD | TPM_RIGHTBUTTON).0, x + 60, y + 60, owner, None);
            eprintln!("menu_screenshot: menu closed after {} ms (result {})", t0.elapsed().as_millis(), picked.0);
            if !picked.as_bool() {
                let err = windows::core::Error::from_thread();
                if err.code().is_err() {
                    eprintln!("menu_screenshot: TrackPopupMenuEx: {err}");
                }
            }
            if attached {
                let _ = AttachThreadInput(me, fg_thread, false);
            }
            let shot = shooter.join().map_err(|_| "capture thread panicked".to_string())?;
            MENU.with(|m| *m.borrow_mut() = None);
            let _ = DestroyMenu(hmenu);
            let _ = DestroyWindow(owner);
            for p in pidls {
                CoTaskMemFree(Some(p as *const _));
            }
            shot?;
        }
        println!("saved {}", out.display());
        Ok(())
    }
}
