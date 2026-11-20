//! Windows front end: the shell's own progress dialog, message boxes for problems, the COM
//! local server behind Explorer's right-click commands, and per-user registration.

mod com;
mod registry;
mod ui;

use std::process::ExitCode;

use ppdesktop::{Action, Job, Settings};
use windows::Win32::System::Console::{AttachConsole, GetStdHandle, ATTACH_PARENT_PROCESS, STD_OUTPUT_HANDLE};

pub use registry::{refresh_shell, register};
pub use ui::{error_box, run_job, settings_dialog};

/// A GUI-subsystem program has no console. For `--cli`, `register` and errors, print to the
/// terminal it was started from, unless output is already redirected (then it works anyway).
pub fn attach_console() {
    unsafe {
        let redirected = matches!(GetStdHandle(STD_OUTPUT_HANDLE), Ok(h) if !h.is_invalid() && !h.0.is_null());
        if !redirected {
            let _ = AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

/// Explorer started us for a right-click command: take the selection it drops on our COM
/// object, then run the job with a progress dialog.
pub fn com_server(action: Action, settings: &Settings) -> ExitCode {
    let _com = ui::Com::init();
    match com::receive_selection(action) {
        Some(paths) if !paths.is_empty() => run_job(Job::new(action, paths, settings.level).with_gpu(settings.gpu), settings),
        _ => ExitCode::SUCCESS,
    }
}
