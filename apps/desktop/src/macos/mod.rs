//! macOS front end: a small AppKit app.
//!
//! * The Finder's right-click menu (Quick Actions / Services) offers "Compress with
//!   PiedPiper", "Extract Here with PiedPiper" and "Extract to Folder with PiedPiper". They
//!   are `NSServices` declared in Info.plist; macOS launches the app if needed and passes the
//!   whole selection on a pasteboard to the methods below.
//! * Double-clicking a `.pp` file (or dropping files on the Dock icon) arrives as
//!   `application:openURLs:`.
//! * Each job gets a small progress window with Cancel; the app quits when the last one ends.
//! * Opened by hand, the app shows a short explanation and the compression level setting.

use std::cell::RefCell;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, sel, DefinedClass, MainThreadMarker, MainThreadOnly, Message};
use objc2_app_kit::{
    NSAlert, NSAlertFirstButtonReturn, NSAlertStyle, NSApplication, NSApplicationActivationPolicy,
    NSApplicationDelegate, NSApplicationLaunchIsDefaultLaunchKey, NSBackingStoreType, NSButton, NSColor, NSFont,
    NSLineBreakMode, NSPasteboard, NSPasteboardTypeFileURL, NSPopUpButton, NSProgressIndicator, NSTextField,
    NSUpdateDynamicServices, NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{
    ns_string, NSArray, NSNotification, NSNumber, NSPoint, NSRect, NSSize, NSString, NSTimer, NSURL,
};
use ppdesktop::archive::selection_size;
use ppdesktop::settings::level_label;
use ppdesktop::{fmt, log::log, naming, Action, Error, Eta, Job, Outcome, Settings, Status};

/// Selections above this size get a confirmation before compressing.
const LARGE_SELECTION: u64 = 2_000_000_000;
/// If macOS launched the app for a service or a file that never arrives, quit after this.
const IDLE_QUIT_SECS: f64 = 20.0;

thread_local! {
    /// Running jobs (main thread only). A controller removes itself when its job ends.
    static JOBS: RefCell<Vec<Retained<JobController>>> = const { RefCell::new(Vec::new()) };
    /// A job given on the command line, started once the app has launched.
    static PENDING: RefCell<Option<Job>> = const { RefCell::new(None) };
    static SETTINGS: RefCell<Settings> = RefCell::new(Settings::default());
}

pub fn run(job: Option<Job>, settings: &Settings) -> ExitCode {
    let Some(mtm) = MainThreadMarker::new() else {
        eprintln!("piedpiper-archiver: must run on the main thread");
        return ExitCode::FAILURE;
    };
    SETTINGS.with(|s| *s.borrow_mut() = settings.clone());
    PENDING.with(|p| *p.borrow_mut() = job);
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    let delegate = AppDelegate::new(mtm);
    app.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    app.run();
    ExitCode::SUCCESS
}

/// Brings the app to the front. `activate` replaces this call only from macOS 14 on, and the
/// app supports macOS 11.
#[allow(deprecated)]
fn activate(mtm: MainThreadMarker) {
    NSApplication::sharedApplication(mtm).activateIgnoringOtherApps(true);
}

fn settings() -> Settings {
    SETTINGS.with(|s| s.borrow().clone())
}

fn job_count() -> usize {
    JOBS.with(|j| j.borrow().len())
}

/// File paths on a services pasteboard (file URLs, also the Finder's file reference URLs).
fn pasteboard_paths(pboard: &NSPasteboard) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Some(items) = pboard.pasteboardItems() else {
        return out;
    };
    let file_url = unsafe { NSPasteboardTypeFileURL };
    for item in items.iter() {
        let Some(s) = item.stringForType(file_url) else {
            continue;
        };
        let Some(url) = NSURL::URLWithString(&s) else { continue };
        let url = url.filePathURL().unwrap_or(url);
        if let Some(path) = url.path() {
            out.push(PathBuf::from(path.to_string()));
        }
    }
    out
}

fn alert(mtm: MainThreadMarker, title: &str, body: &str, style: NSAlertStyle) {
    let a = NSAlert::new(mtm);
    a.setMessageText(&NSString::from_str(title));
    a.setInformativeText(&NSString::from_str(body));
    a.setAlertStyle(style);
    activate(mtm);
    a.runModal();
}

fn confirm(mtm: MainThreadMarker, title: &str, body: &str, ok: &str) -> bool {
    let a = NSAlert::new(mtm);
    a.setMessageText(&NSString::from_str(title));
    a.setInformativeText(&NSString::from_str(body));
    a.addButtonWithTitle(&NSString::from_str(ok));
    a.addButtonWithTitle(ns_string!("Cancel"));
    activate(mtm);
    a.runModal() == NSAlertFirstButtonReturn
}

/// Starts a job with its own progress window.
fn start_job(mtm: MainThreadMarker, job: Job) {
    if job.paths.is_empty() {
        return;
    }
    let headless = settings().headless;
    if job.action == Action::Compress && !headless {
        let size = selection_size(&job.paths);
        if size > LARGE_SELECTION
            && !confirm(
                mtm,
                &format!("Compress {}?", fmt::size(size)),
                "PiedPiper keeps the selection in memory while it compresses and learns from the data as it goes, which can take hours for this much.",
                "Compress",
            )
        {
            quit_if_idle(mtm);
            return;
        }
    }
    let controller = JobController::new(mtm, job, headless);
    JOBS.with(|j| j.borrow_mut().push(controller));
}

/// Quits when no job is running and no window is open.
fn quit_if_idle(mtm: MainThreadMarker) {
    if job_count() == 0 {
        let app = NSApplication::sharedApplication(mtm);
        if app.windows().iter().all(|w| !w.isVisible()) {
            app.terminate(None);
        }
    }
}

/// Opened by hand: how to use it, and the compression level.
fn show_welcome(mtm: MainThreadMarker) {
    let current = settings();
    let a = NSAlert::new(mtm);
    a.setMessageText(ns_string!("PiedPiper"));
    a.setInformativeText(ns_string!(
        "To compress, select files or folders in the Finder, Control-click them and choose \
         Compress with PiedPiper (under Quick Actions or Services). Double-click a .pp archive to \
         extract it, or Control-click it for Extract Here and Extract to Folder.\n\n\
         If the commands are missing, enable them in System Settings › Keyboard › Keyboard \
         Shortcuts › Services › Files and Folders.\n\nCompression level for new archives:"
    ));
    let popup = NSPopUpButton::initWithFrame_pullsDown(
        NSPopUpButton::alloc(mtm),
        NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(320.0, 26.0)),
        false,
    );
    let titles: Vec<Retained<NSString>> = (piedpiper::MIN_LEVEL..=piedpiper::MAX_LEVEL)
        .map(|l| NSString::from_str(&level_label(l)))
        .collect();
    popup.addItemsWithTitles(&NSArray::from_retained_slice(&titles));
    popup.selectItemAtIndex(current.level as isize - 1);
    a.setAccessoryView(Some(&popup));
    a.addButtonWithTitle(ns_string!("Save"));
    a.addButtonWithTitle(ns_string!("Cancel"));
    activate(mtm);
    if a.runModal() == NSAlertFirstButtonReturn {
        let mut s = current;
        s.level = (popup.indexOfSelectedItem() + 1).clamp(1, 9) as u32;
        match s.save() {
            Ok(()) => SETTINGS.with(|c| *c.borrow_mut() = s),
            Err(e) => alert(
                mtm,
                "The setting could not be saved",
                &e.to_string(),
                NSAlertStyle::Warning,
            ),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// App delegate and services provider

#[derive(Default)]
struct AppIvars {}

define_class!(
    // SAFETY: NSObject has no subclassing requirements and AppDelegate does not implement Drop.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "PPArchiverAppDelegate"]
    #[ivars = AppIvars]
    struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn did_finish_launching(&self, notification: &NSNotification) {
            let mtm = self.mtm();
            let app = NSApplication::sharedApplication(mtm);
            // This object answers the NSServices declared in Info.plist.
            unsafe { app.setServicesProvider(Some(self.as_ref())) };
            NSUpdateDynamicServices();
            log("launched");

            if let Some(job) = PENDING.with(|p| p.borrow_mut().take()) {
                start_job(mtm, job);
                return;
            }
            let default_launch = notification
                .userInfo()
                .and_then(|info| info.objectForKey(unsafe { NSApplicationLaunchIsDefaultLaunchKey }.as_ref()))
                .and_then(|v| v.downcast::<NSNumber>().ok())
                .map(|n| n.boolValue())
                .unwrap_or(true);
            if default_launch && job_count() == 0 {
                show_welcome(mtm);
                app.terminate(None);
            } else {
                // Launched for a service or a file: quit if nothing arrives.
                unsafe {
                    let _: () = msg_send![
                        self,
                        performSelector: sel!(quitIfIdle:),
                        withObject: Option::<&AnyObject>::None,
                        afterDelay: IDLE_QUIT_SECS
                    ];
                }
            }
        }

        #[unsafe(method(application:openURLs:))]
        fn open_urls(&self, _application: &NSApplication, urls: &NSArray<NSURL>) {
            let paths: Vec<PathBuf> = urls.iter().filter_map(|u| u.path()).map(|p| PathBuf::from(p.to_string())).collect();
            log(format!("open: {} item(s)", paths.len()));
            let (archives, others): (Vec<PathBuf>, Vec<PathBuf>) =
                paths.into_iter().partition(|p| naming::is_archive_name(&naming::display_name(p)));
            let level = settings().level;
            if !archives.is_empty() {
                start_job(self.mtm(), Job::new(Action::Extract, archives, level));
            }
            if !others.is_empty() {
                start_job(self.mtm(), Job::new(Action::Compress, others, level));
            }
        }

        // Quitting is explicit (`quit_if_idle`): closing an alert before a job's window opens
        // must not end the app.
        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        fn should_terminate_after_last_window_closed(&self, _sender: &NSApplication) -> bool {
            false
        }
    }

    impl AppDelegate {
        #[unsafe(method(compressFiles:userData:error:))]
        fn service_compress(&self, pboard: &NSPasteboard, _user_data: Option<&NSString>, _error: *mut *mut NSString) {
            self.service(Action::Compress, pboard);
        }

        #[unsafe(method(extractHere:userData:error:))]
        fn service_extract_here(&self, pboard: &NSPasteboard, _user_data: Option<&NSString>, _error: *mut *mut NSString) {
            self.service(Action::ExtractHere, pboard);
        }

        #[unsafe(method(extractToFolder:userData:error:))]
        fn service_extract_to_folder(&self, pboard: &NSPasteboard, _user_data: Option<&NSString>, _error: *mut *mut NSString) {
            self.service(Action::ExtractToFolder, pboard);
        }

        #[unsafe(method(quitIfIdle:))]
        fn quit_if_idle_selector(&self, _arg: Option<&AnyObject>) {
            quit_if_idle(self.mtm());
        }
    }
);

impl AppDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(AppIvars::default());
        unsafe { msg_send![super(this), init] }
    }

    fn service(&self, action: Action, pboard: &NSPasteboard) {
        let paths = pasteboard_paths(pboard);
        log(format!("service {}: {} item(s)", action.arg(), paths.len()));
        start_job(self.mtm(), Job::new(action, paths, settings().level));
    }
}

// ---------------------------------------------------------------------------------------------
// Progress window

struct Window {
    window: Retained<NSWindow>,
    detail: Retained<NSTextField>,
    bar: Retained<NSProgressIndicator>,
    cancel: Retained<NSButton>,
}

struct JobState {
    status: Arc<Status>,
    worker: Option<JoinHandle<Result<Outcome, Error>>>,
    window: Option<Window>,
    timer: Option<Retained<NSTimer>>,
    eta: Eta,
    shown: String,
}

#[derive(Default)]
struct JobIvars {
    state: RefCell<Option<JobState>>,
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements and JobController does not implement Drop.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "PPArchiverJobController"]
    #[ivars = JobIvars]
    struct JobController;

    unsafe impl NSObjectProtocol for JobController {}

    impl JobController {
        #[unsafe(method(tick:))]
        fn tick(&self, _timer: &NSTimer) {
            self.on_tick();
        }

        #[unsafe(method(cancel:))]
        fn cancel(&self, _sender: Option<&AnyObject>) {
            if let Some(state) = self.ivars().state.borrow().as_ref() {
                state.status.cancel();
                if let Some(w) = &state.window {
                    w.cancel.setEnabled(false);
                    w.cancel.setTitle(ns_string!("Cancelling\u{2026}"));
                }
            }
        }
    }
);

fn label(mtm: MainThreadMarker, text: &str, frame: NSRect, size: f64, bold: bool) -> Retained<NSTextField> {
    let l = NSTextField::labelWithString(&NSString::from_str(text), mtm);
    l.setFrame(frame);
    let font = if bold {
        NSFont::boldSystemFontOfSize(size)
    } else {
        NSFont::systemFontOfSize(size)
    };
    l.setFont(Some(&font));
    l.setLineBreakMode(NSLineBreakMode::ByTruncatingMiddle);
    l
}

fn rect(x: f64, y: f64, w: f64, h: f64) -> NSRect {
    NSRect::new(NSPoint::new(x, y), NSSize::new(w, h))
}

impl JobController {
    fn new(mtm: MainThreadMarker, job: Job, headless: bool) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(JobIvars::default());
        let this: Retained<Self> = unsafe { msg_send![super(this), init] };
        let headline = job.headline();
        let window = (!headless).then(|| this.build_window(mtm, &headline));
        let status = Arc::new(Status::new(headline));
        let worker = {
            let st = Arc::clone(&status);
            thread::spawn(move || job.run(&st))
        };
        let timer = unsafe {
            NSTimer::scheduledTimerWithTimeInterval_target_selector_userInfo_repeats(
                0.1,
                this.as_ref(),
                sel!(tick:),
                None,
                true,
            )
        };
        *this.ivars().state.borrow_mut() = Some(JobState {
            status,
            worker: Some(worker),
            window,
            timer: Some(timer),
            eta: Eta::new(),
            shown: String::new(),
        });
        this
    }

    fn build_window(&self, mtm: MainThreadMarker, headline: &str) -> Window {
        let style = NSWindowStyleMask::Titled | NSWindowStyleMask::Miniaturizable;
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                rect(0.0, 0.0, 460.0, 132.0),
                style,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        unsafe { window.setReleasedWhenClosed(false) };
        window.setTitle(ns_string!("PiedPiper"));
        let content = window.contentView().expect("window has a content view");

        content.addSubview(&label(mtm, headline, rect(20.0, 92.0, 420.0, 20.0), 13.0, true));
        let bar = NSProgressIndicator::initWithFrame(NSProgressIndicator::alloc(mtm), rect(20.0, 66.0, 420.0, 20.0));
        bar.setIndeterminate(false);
        bar.setMinValue(0.0);
        bar.setMaxValue(1.0);
        content.addSubview(&bar);
        let detail = label(mtm, "", rect(20.0, 44.0, 420.0, 18.0), 11.0, false);
        detail.setTextColor(Some(&NSColor::secondaryLabelColor()));
        content.addSubview(&detail);
        let cancel = unsafe {
            NSButton::buttonWithTitle_target_action(ns_string!("Cancel"), Some(self.as_ref()), Some(sel!(cancel:)), mtm)
        };
        cancel.setFrame(rect(344.0, 10.0, 100.0, 30.0));
        cancel.setKeyEquivalent(ns_string!("\u{1b}"));
        content.addSubview(&cancel);

        // Cascade windows of concurrent jobs instead of stacking them exactly.
        window.center();
        let offset = 24.0 * (job_count() % 8) as f64;
        let frame = window.frame();
        window.setFrameOrigin(NSPoint::new(frame.origin.x + offset, frame.origin.y - offset));
        window.makeKeyAndOrderFront(None);
        activate(mtm);
        Window {
            window,
            detail,
            bar,
            cancel,
        }
    }

    fn on_tick(&self) {
        let finished = {
            let mut guard = self.ivars().state.borrow_mut();
            let Some(state) = guard.as_mut() else { return };
            let snap = state.status.snapshot();
            if let Some(w) = &state.window {
                w.bar.setDoubleValue(snap.fraction);
                let text = match state.eta.update(snap.fraction) {
                    Some(left) if !snap.detail.is_empty() => {
                        format!("{} \u{2014} {} left", snap.detail, fmt::remaining(left))
                    }
                    _ => snap.detail.clone(),
                };
                if text != state.shown {
                    w.detail.setStringValue(&NSString::from_str(&text));
                    state.shown = text;
                }
            }
            state.worker.as_ref().is_none_or(|w| w.is_finished())
        };
        if finished {
            self.finish();
        }
    }

    /// The job ended: stop the timer, report problems, close the window, maybe quit.
    fn finish(&self) {
        // JOBS may hold the last reference to this object; keep it alive until we return.
        let _keep: Retained<JobController> = self.retain();
        let Some(mut state) = self.ivars().state.borrow_mut().take() else {
            return;
        };
        if let Some(t) = state.timer.take() {
            t.invalidate();
        }
        let result = state.worker.take().map(|w| w.join());
        let mtm = self.mtm();
        let headless = state.window.is_none();
        let problem: Option<(String, String, NSAlertStyle)> = match result {
            Some(Ok(Ok(o))) => o.problems(15).map(|text| {
                let title = if o.errors.is_empty() {
                    "Some items were skipped"
                } else {
                    "Some archives could not be extracted"
                };
                let style = if o.errors.is_empty() {
                    NSAlertStyle::Warning
                } else {
                    NSAlertStyle::Critical
                };
                (title.to_string(), text, style)
            }),
            Some(Ok(Err(Error::Cancelled))) => None,
            Some(Ok(Err(e))) => Some((
                "PiedPiper could not finish".to_string(),
                e.to_string(),
                NSAlertStyle::Critical,
            )),
            Some(Err(_)) | None => Some((
                "PiedPiper stopped unexpectedly".to_string(),
                "Internal error.".to_string(),
                NSAlertStyle::Critical,
            )),
        };
        if let Some((title, body, style)) = problem {
            log(format!("{title}: {body}"));
            if !headless {
                alert(mtm, &title, &body, style);
            }
        }
        // Forget this job before closing its window, so closing the last window quits.
        let me: *const JobController = self;
        JOBS.with(|j| j.borrow_mut().retain(|c| !std::ptr::eq(&**c, me)));
        if let Some(w) = state.window {
            w.window.close();
        }
        quit_if_idle(mtm);
    }
}
