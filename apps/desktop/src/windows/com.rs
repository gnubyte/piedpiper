//! The COM local server behind Explorer's "DropTarget" verbs.
//!
//! For `HKCR\...\shell\PiedPiper.Compress\DropTarget\CLSID = {...}`, Explorer creates that COM
//! class (starting `piedpiper-archiver.exe --com compress -Embedding`), then "drops" the whole
//! selection on it as an `IDataObject`. One process gets every selected item, however many.

use std::ffi::{c_void, OsString};
use std::os::windows::ffi::OsStringExt;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ppdesktop::{explorer, log::log, Action};
use windows::core::{implement, Interface, Ref, Result, BOOL, GUID};
use windows::Win32::Foundation::{CLASS_E_NOAGGREGATION, E_UNEXPECTED, POINTL};
use windows::Win32::System::Com::{
    CoRegisterClassObject, CoRevokeClassObject, IClassFactory, IClassFactory_Impl, IDataObject, CLSCTX_LOCAL_SERVER,
    DVASPECT_CONTENT, FORMATETC, REGCLS_SINGLEUSE, TYMED_HGLOBAL,
};
use windows::Win32::System::Ole::{
    IDropTarget, IDropTarget_Impl, ReleaseStgMedium, CF_HDROP, DROPEFFECT, DROPEFFECT_COPY,
};
use windows::Win32::System::SystemServices::MODIFIERKEYS_FLAGS;
use windows::Win32::UI::Shell::{DragQueryFileW, HDROP};

/// How long to wait for Explorer to drop the selection after starting us.
const WAIT: Duration = Duration::from_secs(60);

/// The dropped selection. A plain slot polled by the waiting loop: a quit message posted
/// from inside the COM call can be swallowed by COM's own modal message loop.
static SELECTION: Mutex<Option<Vec<PathBuf>>> = Mutex::new(None);

fn set_copy(effect: *mut DROPEFFECT) {
    if !effect.is_null() {
        unsafe { *effect = DROPEFFECT_COPY };
    }
}

/// File system paths in a CF_HDROP data object (what Explorer puts on a selection).
fn dropped_paths(data: &IDataObject) -> Result<Vec<PathBuf>> {
    let format = FORMATETC {
        cfFormat: CF_HDROP.0,
        ptd: std::ptr::null_mut(),
        dwAspect: DVASPECT_CONTENT.0,
        lindex: -1,
        tymed: TYMED_HGLOBAL.0 as u32,
    };
    unsafe {
        let mut medium = data.GetData(&format)?;
        let hdrop = HDROP(medium.u.hGlobal.0);
        let count = DragQueryFileW(hdrop, u32::MAX, None);
        let mut paths = Vec::with_capacity(count as usize);
        for i in 0..count {
            let len = DragQueryFileW(hdrop, i, None) as usize;
            let mut buf = vec![0u16; len + 1];
            let got = DragQueryFileW(hdrop, i, Some(&mut buf)) as usize;
            paths.push(PathBuf::from(OsString::from_wide(&buf[..got.min(len)])));
        }
        ReleaseStgMedium(&mut medium);
        Ok(paths)
    }
}

#[implement(IDropTarget)]
struct VerbTarget;

impl IDropTarget_Impl for VerbTarget_Impl {
    fn DragEnter(
        &self,
        _data: Ref<IDataObject>,
        _keys: MODIFIERKEYS_FLAGS,
        _pt: &POINTL,
        effect: *mut DROPEFFECT,
    ) -> Result<()> {
        set_copy(effect);
        Ok(())
    }

    fn DragOver(&self, _keys: MODIFIERKEYS_FLAGS, _pt: &POINTL, effect: *mut DROPEFFECT) -> Result<()> {
        set_copy(effect);
        Ok(())
    }

    fn DragLeave(&self) -> Result<()> {
        Ok(())
    }

    fn Drop(
        &self,
        data: Ref<IDataObject>,
        _keys: MODIFIERKEYS_FLAGS,
        _pt: &POINTL,
        effect: *mut DROPEFFECT,
    ) -> Result<()> {
        let paths = dropped_paths(data.ok()?)?;
        log(format!("selection received: {} item(s)", paths.len()));
        *SELECTION.lock().unwrap_or_else(|e| e.into_inner()) = Some(paths);
        set_copy(effect);
        // Return to Explorer right away; the job starts once the waiting loop sees the slot.
        Ok(())
    }
}

#[implement(IClassFactory)]
struct Factory;

impl IClassFactory_Impl for Factory_Impl {
    fn CreateInstance(
        &self,
        outer: Ref<windows::core::IUnknown>,
        iid: *const GUID,
        object: *mut *mut c_void,
    ) -> Result<()> {
        if object.is_null() || iid.is_null() {
            return Err(E_UNEXPECTED.into());
        }
        unsafe { *object = std::ptr::null_mut() };
        if !outer.is_null() {
            return Err(CLASS_E_NOAGGREGATION.into());
        }
        let target: IDropTarget = VerbTarget.into();
        unsafe { target.query(iid, object).ok() }
    }

    fn LockServer(&self, _lock: BOOL) -> Result<()> {
        Ok(())
    }
}

/// Registers the COM class for `action`, waits for Explorer to drop a selection on it, and
/// returns the selected paths (`None` on timeout). The calling thread must be in a
/// single-threaded apartment; it keeps pumping messages so Explorer's calls get through.
pub fn receive_selection(action: Action) -> Option<Vec<PathBuf>> {
    let Some(clsid) = explorer::clsid(action) else {
        log(format!("no COM class for {}", action.arg()));
        return None;
    };
    let clsid = GUID::try_from(clsid.trim_matches(|c| c == '{' || c == '}')).ok()?;
    let factory: IClassFactory = Factory.into();
    // Single use: a second right-click command while this one runs starts a new process.
    let cookie = match unsafe { CoRegisterClassObject(&clsid, &factory, CLSCTX_LOCAL_SERVER, REGCLS_SINGLEUSE) } {
        Ok(c) => c,
        Err(e) => {
            log(format!("CoRegisterClassObject failed: {e}"));
            return None;
        }
    };
    let deadline = Instant::now() + WAIT;
    let selection = loop {
        if let Some(paths) = SELECTION.lock().unwrap_or_else(|e| e.into_inner()).take() {
            break Some(paths);
        }
        if Instant::now() > deadline {
            log("timed out waiting for Explorer");
            break None;
        }
        // Explorer's calls arrive as window messages on this thread.
        super::ui::pump(50);
    };
    unsafe {
        let _ = CoRevokeClassObject(cookie);
    }
    selection
}
