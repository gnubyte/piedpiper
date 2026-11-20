//! Windows dialogs: the shell's progress dialog (the one Explorer uses for copies, with time
//! remaining and Cancel), message boxes, and the settings task dialog.

use std::any::Any;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;

use ppdesktop::archive::selection_size;
use ppdesktop::settings::{gpu_line, level_label, GPU_ABOUT, GPU_OPTION};
use ppdesktop::{fmt, log::log, Action, Error, Job, Outcome, Settings, Snapshot, Status};
use windows::core::{w, HSTRING, PCWSTR};
use windows::Win32::Foundation::HWND;
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Controls::{
    TaskDialogIndirect, TASKDIALOGCONFIG, TASKDIALOG_BUTTON, TDCBF_CANCEL_BUTTON, TDCBF_OK_BUTTON,
    TDF_ALLOW_DIALOG_CANCELLATION, TDF_EXPAND_FOOTER_AREA, TDF_VERIFICATION_FLAG_CHECKED,
};
use windows::Win32::UI::Shell::{
    CLSID_ProgressDialog, IProgressDialog, PDTIMER_RESET, PROGDLG_AUTOTIME, PROGDLG_NORMAL,
};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, MessageBoxW, MsgWaitForMultipleObjects, PeekMessageW, TranslateMessage, IDOK, IDYES,
    MB_DEFBUTTON2, MB_ICONERROR, MB_ICONINFORMATION, MB_ICONQUESTION, MB_ICONWARNING, MB_OK, MB_SETFOREGROUND,
    MB_TOPMOST, MB_YESNO, MESSAGEBOX_STYLE, MSG, PM_REMOVE, QS_ALLINPUT,
};

/// Selections above this size get a confirmation before compressing.
const LARGE_SELECTION: u64 = 2_000_000_000;
/// At most this many lines of warnings in a message box.
const MAX_LINES: usize = 15;

/// COM initialised for this thread (single-threaded apartment) until dropped.
pub struct Com(bool);

impl Com {
    pub fn init() -> Com {
        Com(unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) }.is_ok())
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        if self.0 {
            unsafe { CoUninitialize() };
        }
    }
}

fn message(text: &str, style: MESSAGEBOX_STYLE) -> windows::Win32::UI::WindowsAndMessaging::MESSAGEBOX_RESULT {
    unsafe {
        MessageBoxW(
            None,
            &HSTRING::from(text),
            w!("PiedPiper"),
            style | MB_SETFOREGROUND | MB_TOPMOST,
        )
    }
}

pub fn error_box(text: &str) {
    message(text, MB_OK | MB_ICONERROR);
}

pub fn info_box(text: &str) {
    message(text, MB_OK | MB_ICONINFORMATION);
}

fn confirm(text: &str) -> bool {
    message(text, MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2) == IDYES
}

/// Handles window messages for up to `ms` milliseconds (COM calls from Explorer arrive as
/// messages on this thread).
pub fn pump(ms: u32) {
    unsafe {
        MsgWaitForMultipleObjects(None, false, ms, QS_ALLINPUT);
        let mut msg = MSG::default();
        while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// The shell's progress dialog. It runs on its own thread; this side only feeds it numbers.
struct ProgressDialog {
    dlg: IProgressDialog,
    detail: String,
}

impl ProgressDialog {
    fn start(headline: &str) -> Option<ProgressDialog> {
        let dlg: IProgressDialog = match unsafe { CoCreateInstance(&CLSID_ProgressDialog, None, CLSCTX_INPROC_SERVER) }
        {
            Ok(d) => d,
            Err(e) => {
                log(format!("no progress dialog: {e}"));
                return None;
            }
        };
        unsafe {
            let _ = dlg.SetTitle(w!("PiedPiper"));
            let _ = dlg.SetCancelMsg(&HSTRING::from("Cancelling\u{2026}"), None);
            let _ = dlg.SetLine(1, &HSTRING::from(headline), false, None);
            if let Err(e) = dlg.StartProgressDialog(None, None, PROGDLG_NORMAL | PROGDLG_AUTOTIME, None) {
                log(format!("progress dialog did not start: {e}"));
                return None;
            }
            let _ = dlg.Timer(PDTIMER_RESET, None);
        }
        Some(ProgressDialog {
            dlg,
            detail: String::new(),
        })
    }

    fn update(&mut self, s: &Snapshot) {
        unsafe {
            let _ = self.dlg.SetProgress64((s.fraction * 1_000_000.0) as u64, 1_000_000);
            if s.detail != self.detail {
                let _ = self.dlg.SetLine(2, &HSTRING::from(s.detail.as_str()), false, None);
                self.detail.clone_from(&s.detail);
            }
        }
    }

    fn cancelled(&self) -> bool {
        unsafe { self.dlg.HasUserCancelled().as_bool() }
    }

    fn stop(self) {
        unsafe {
            let _ = self.dlg.StopProgressDialog();
        }
    }
}

/// Runs a job with a progress dialog and reports problems in a message box.
pub fn run_job(job: Job, settings: &Settings) -> ExitCode {
    let _com = Com::init();
    let headless = settings.headless;
    if job.action == Action::Compress && !headless {
        let size = selection_size(&job.paths);
        if size > LARGE_SELECTION {
            let text = format!(
                "The selection holds {} of data. PiedPiper keeps it in memory while it compresses and learns from the data as it goes, which can take hours for this much.\n\nCompress it anyway?",
                fmt::size(size)
            );
            if !confirm(&text) {
                return ExitCode::SUCCESS;
            }
        }
    }
    let st = Arc::new(Status::new(job.headline()));
    let worker = {
        let st = Arc::clone(&st);
        thread::spawn(move || job.run(&st))
    };
    let mut dialog = if headless {
        None
    } else {
        ProgressDialog::start(&st.snapshot().headline)
    };
    while !worker.is_finished() {
        pump(100);
        if let Some(d) = dialog.as_mut() {
            d.update(&st.snapshot());
            if d.cancelled() {
                st.cancel();
            }
        }
    }
    if let Some(d) = dialog {
        d.stop();
    }
    report(worker.join(), headless)
}

fn panic_text(p: &(dyn Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown error".into())
}

fn report(result: thread::Result<Result<Outcome, Error>>, headless: bool) -> ExitCode {
    match result {
        Ok(Ok(o)) => {
            if let Some(text) = o.problems(MAX_LINES) {
                log(&text);
                if !headless {
                    message(
                        &text,
                        MB_OK
                            | if o.errors.is_empty() {
                                MB_ICONWARNING
                            } else {
                                MB_ICONERROR
                            },
                    );
                }
            }
            if o.errors.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Ok(Err(Error::Cancelled)) => {
            log("cancelled");
            ExitCode::SUCCESS
        }
        Ok(Err(e)) => {
            log(format!("error: {e}"));
            if !headless {
                error_box(&e.to_string());
            }
            ExitCode::FAILURE
        }
        Err(p) => {
            let text = format!("PiedPiper stopped unexpectedly: {}", panic_text(p.as_ref()));
            log(&text);
            if !headless {
                error_box(&text);
            }
            ExitCode::FAILURE
        }
    }
}

/// MAKEINTRESOURCEW: a resource id passed where Windows expects a name.
fn make_int_resource(id: u16) -> PCWSTR {
    PCWSTR(id as usize as *const u16)
}

/// Started from the Start menu (no arguments): explains where the commands are and lets the
/// user pick the compression level and GPU acceleration (with an "About GPU acceleration"
/// section that says which graphics cards work and whether this computer has one).
pub fn settings_dialog(settings: &Settings) -> ExitCode {
    const FIRST_ID: i32 = 100;
    let labels: Vec<HSTRING> = (piedpiper::MIN_LEVEL..=piedpiper::MAX_LEVEL)
        .map(|l| HSTRING::from(level_label(l)))
        .collect();
    let radios: Vec<TASKDIALOG_BUTTON> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| TASKDIALOG_BUTTON {
            nButtonID: FIRST_ID + i as i32 + 1,
            pszButtonText: PCWSTR(label.as_ptr()),
        })
        .collect();
    let title = HSTRING::from("PiedPiper");
    let main = HSTRING::from("Compress and extract from File Explorer");
    let content = HSTRING::from(
        "Select files or folders in File Explorer, right-click them and choose Compress with PiedPiper. \
         Double-click a .pp archive to extract it, or right-click it for Extract here and Extract to folder.\n\n\
         On Windows 11 the commands are under Show more options (or press Shift+F10).\n\n\
         Compression level for new archives:",
    );
    let footer = HSTRING::from(format!(
        "PiedPiper {} \u{00b7} archive format v{}",
        env!("CARGO_PKG_VERSION"),
        piedpiper::FORMAT_VERSION
    ));
    let gpu_text = HSTRING::from(GPU_OPTION);
    let about = HSTRING::from(format!("{GPU_ABOUT}\n\n{}", gpu_line()));
    let about_label = HSTRING::from("About GPU acceleration");
    let mut cfg = TASKDIALOGCONFIG {
        cbSize: std::mem::size_of::<TASKDIALOGCONFIG>() as u32,
        hwndParent: HWND::default(),
        // The "About GPU acceleration" text opens in the footer, under the options.
        dwFlags: TDF_ALLOW_DIALOG_CANCELLATION
            | TDF_EXPAND_FOOTER_AREA
            | if settings.gpu { TDF_VERIFICATION_FLAG_CHECKED } else { Default::default() },
        dwCommonButtons: TDCBF_OK_BUTTON | TDCBF_CANCEL_BUTTON,
        pszWindowTitle: PCWSTR(title.as_ptr()),
        pszMainInstruction: PCWSTR(main.as_ptr()),
        pszContent: PCWSTR(content.as_ptr()),
        cRadioButtons: radios.len() as u32,
        pRadioButtons: radios.as_ptr(),
        nDefaultRadioButton: FIRST_ID + settings.level as i32,
        pszFooter: PCWSTR(footer.as_ptr()),
        // Without a fixed width the long "About GPU acceleration" text would stretch the
        // dialog; this keeps its earlier size (dialog units).
        cxWidth: 300,
        pszVerificationText: PCWSTR(gpu_text.as_ptr()),
        pszExpandedInformation: PCWSTR(about.as_ptr()),
        pszCollapsedControlText: PCWSTR(about_label.as_ptr()),
        pszExpandedControlText: PCWSTR(about_label.as_ptr()),
        ..Default::default()
    };
    unsafe {
        // The program's own icon (resource 1) at the top of the dialog.
        if let Ok(module) = GetModuleHandleW(None) {
            cfg.hInstance = module.into();
            cfg.Anonymous1.pszMainIcon = make_int_resource(1);
        }
    }
    let (mut button, mut radio) = (0i32, 0i32);
    let mut use_gpu = windows::core::BOOL::from(settings.gpu);
    if let Err(e) = unsafe { TaskDialogIndirect(&cfg, Some(&mut button), Some(&mut radio), Some(&mut use_gpu)) } {
        log(format!("settings dialog failed: {e}"));
        error_box(&format!("The settings window could not be opened: {e}"));
        return ExitCode::FAILURE;
    }
    if button == IDOK.0 && radio > FIRST_ID {
        let mut s = settings.clone();
        s.level = (radio - FIRST_ID) as u32;
        s.gpu = use_gpu.as_bool();
        if let Err(e) = s.save() {
            error_box(&format!("The settings could not be saved: {e}"));
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
