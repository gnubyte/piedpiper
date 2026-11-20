//! `piedpiper-archiver`: PiedPiper's right-click Compress / Extract commands for Windows File
//! Explorer and the macOS Finder. Explorer and the Finder start it; it can also be run by hand.

// No console window when Explorer starts it.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod cli;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(windows)]
mod windows;

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use ppdesktop::{naming, Action, Job, Settings};

const USAGE: &str = "\
piedpiper-archiver: PiedPiper's Compress / Extract commands for File Explorer and the Finder.

USAGE:
    piedpiper-archiver compress PATH...      all PATHs into one archive next to them
    piedpiper-archiver extract ARCHIVE...    like a double-click: next to the archive, in a
                                             new folder unless it holds a single item
    piedpiper-archiver extract-here ARCHIVE...
    piedpiper-archiver extract-to ARCHIVE... each into a new folder named after it
    piedpiper-archiver PATH...               extract if all PATHs are .pp archives, else compress
    piedpiper-archiver                       settings (compression level)

OPTIONS:
    -l, --level N    compression level 1-9 for this run (default: the saved setting)
        --gpu        use the graphics card (AMD Radeon or another Vulkan GPU) where it is
                     faster, for this run (default: the saved setting)
        --cli        show progress in the terminal instead of a window
    -h, --help       this help
    -V, --version    version

WINDOWS ONLY:
    register [--machine]      add the Explorer commands for this user (or all users)
    unregister [--machine]    remove them again (the MSI installer does both itself)
";

enum Command {
    Job(Job),
    Settings,
    Help,
    Version,
    /// Started by Explorer as the COM server behind a right-click command.
    #[cfg_attr(not(windows), allow(dead_code))]
    Com(Action),
    #[cfg_attr(not(windows), allow(dead_code))]
    Register {
        machine: bool,
        quiet: bool,
    },
    #[cfg_attr(not(windows), allow(dead_code))]
    Unregister {
        machine: bool,
        quiet: bool,
    },
    /// Tell Explorer that file types changed (run by the installer).
    #[cfg_attr(not(windows), allow(dead_code))]
    RefreshShell,
}

struct Args {
    command: Command,
    cli: bool,
}

fn parse(args: impl Iterator<Item = OsString>, settings: &Settings) -> Result<Args, String> {
    let mut verb: Option<String> = None;
    let mut paths = Vec::new();
    let (mut level, mut cli, mut machine, mut quiet, mut com, mut only_paths) =
        (None, false, false, false, None, false);
    let mut gpu = settings.gpu;
    let mut it = args.peekable();
    while let Some(arg) = it.next() {
        let s = arg.to_string_lossy().into_owned();
        if only_paths {
            paths.push(PathBuf::from(arg));
            continue;
        }
        match s.as_str() {
            "--" => only_paths = true,
            "-h" | "--help" | "/?" => {
                return Ok(Args {
                    command: Command::Help,
                    cli: true,
                })
            }
            "-V" | "--version" => {
                return Ok(Args {
                    command: Command::Version,
                    cli: true,
                })
            }
            "--cli" => cli = true,
            "--gpu" => gpu = true,
            "--machine" => machine = true,
            "-q" | "--quiet" => quiet = true,
            "-l" | "--level" => {
                let v = it.next().ok_or("--level needs a value")?;
                let l: u32 = v
                    .to_string_lossy()
                    .parse()
                    .map_err(|_| "--level needs a number from 1 to 9")?;
                if !(piedpiper::MIN_LEVEL..=piedpiper::MAX_LEVEL).contains(&l) {
                    return Err("--level needs a number from 1 to 9".into());
                }
                level = Some(l);
            }
            "--com" => {
                let v = it.next().ok_or("--com needs an action")?.to_string_lossy().into_owned();
                com = Some(Action::from_arg(&v).ok_or_else(|| format!("unknown action {v}"))?);
            }
            // Added by COM when it starts a local server.
            "-Embedding" | "/Embedding" | "-embedding" | "/embedding" => {}
            // Added by old macOS versions when launching from the Finder.
            s if s.starts_with("-psn_") => {}
            s if verb.is_none() && paths.is_empty() && is_verb(s) => verb = Some(s.to_string()),
            s if s.len() > 1 && s.starts_with('-') => return Err(format!("unknown option {s}")),
            _ => paths.push(PathBuf::from(arg)),
        }
    }
    let level = level.unwrap_or(settings.level);
    let command = if let Some(action) = com {
        Command::Com(action)
    } else {
        match verb.as_deref() {
            Some("register") => Command::Register { machine, quiet },
            Some("unregister") => Command::Unregister { machine, quiet },
            Some("refresh-shell") => Command::RefreshShell,
            Some("settings") => Command::Settings,
            Some(v) => {
                let action = Action::from_arg(v).expect("is_verb");
                if paths.is_empty() {
                    return Err(format!("{v} needs at least one path"));
                }
                Command::Job(Job::new(action, paths, level).with_gpu(gpu))
            }
            None if paths.is_empty() => Command::Settings,
            // Files dropped on the program: extract archives, compress anything else.
            None => {
                let all_archives = paths.iter().all(|p| naming::is_archive_name(&naming::display_name(p)));
                let action = if all_archives {
                    Action::Extract
                } else {
                    Action::Compress
                };
                Command::Job(Job::new(action, paths, level).with_gpu(gpu))
            }
        }
    };
    Ok(Args { command, cli })
}

fn is_verb(s: &str) -> bool {
    Action::from_arg(s).is_some() || matches!(s, "register" | "unregister" | "refresh-shell" | "settings")
}

fn main() -> ExitCode {
    let settings = Settings::load();
    ppdesktop::log::init(settings.log || std::env::var_os("PIEDPIPER_ARCHIVER_LOG").is_some());
    ppdesktop::log::log(format!("start: {:?}", std::env::args_os().collect::<Vec<_>>()));

    let args = match parse(std::env::args_os().skip(1), &settings) {
        Ok(a) => a,
        Err(e) => {
            ppdesktop::log::log(format!("bad arguments: {e}"));
            return usage_error(&e);
        }
    };
    #[cfg(windows)]
    if args.cli || matches!(args.command, Command::Register { .. } | Command::Unregister { .. }) {
        windows::attach_console();
    }

    match args.command {
        Command::Help => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Command::Version => {
            println!(
                "piedpiper-archiver {} (PiedPiper {}, format v{})",
                env!("CARGO_PKG_VERSION"),
                piedpiper::VERSION,
                piedpiper::FORMAT_VERSION
            );
            ExitCode::SUCCESS
        }
        Command::Job(job) if args.cli => cli::run(job),
        Command::Job(job) => run_job(job, &settings),
        Command::Settings => settings_ui(&settings),
        Command::Com(action) => com(action, &settings),
        Command::Register { machine, quiet } => register(true, machine, quiet),
        Command::Unregister { machine, quiet } => register(false, machine, quiet),
        Command::RefreshShell => {
            #[cfg(windows)]
            windows::refresh_shell();
            ExitCode::SUCCESS
        }
    }
}

#[cfg(windows)]
fn usage_error(e: &str) -> ExitCode {
    windows::attach_console();
    eprintln!("piedpiper-archiver: {e}\n\n{USAGE}");
    windows::error_box(&format!("{e}\n\n{USAGE}"));
    ExitCode::from(2)
}

#[cfg(not(windows))]
fn usage_error(e: &str) -> ExitCode {
    eprintln!("piedpiper-archiver: {e}\n\n{USAGE}");
    ExitCode::from(2)
}

#[cfg(windows)]
fn run_job(job: Job, settings: &Settings) -> ExitCode {
    windows::run_job(job, settings)
}

#[cfg(target_os = "macos")]
fn run_job(job: Job, settings: &Settings) -> ExitCode {
    macos::run(Some(job), settings)
}

#[cfg(not(any(windows, target_os = "macos")))]
fn run_job(job: Job, _settings: &Settings) -> ExitCode {
    cli::run(job)
}

#[cfg(windows)]
fn settings_ui(settings: &Settings) -> ExitCode {
    windows::settings_dialog(settings)
}

#[cfg(target_os = "macos")]
fn settings_ui(settings: &Settings) -> ExitCode {
    macos::run(None, settings)
}

#[cfg(not(any(windows, target_os = "macos")))]
fn settings_ui(settings: &Settings) -> ExitCode {
    println!(
        "Compression level: {}. Change it with `level = N` in {}.\n\n{USAGE}",
        settings.level,
        ppdesktop::settings::settings_path()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    );
    ExitCode::SUCCESS
}

#[cfg(windows)]
fn com(action: Action, settings: &Settings) -> ExitCode {
    windows::com_server(action, settings)
}

#[cfg(not(windows))]
fn com(_action: Action, _settings: &Settings) -> ExitCode {
    usage_error("--com is only used on Windows")
}

#[cfg(windows)]
fn register(add: bool, machine: bool, quiet: bool) -> ExitCode {
    windows::register(add, machine, quiet)
}

#[cfg(not(windows))]
fn register(_add: bool, _machine: bool, _quiet: bool) -> ExitCode {
    usage_error(
        "register and unregister are only used on Windows; on macOS the app registers its Finder commands itself",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_strs(a: &[&str]) -> Result<Args, String> {
        parse(a.iter().map(OsString::from), &Settings::default())
    }

    #[test]
    fn commands() {
        let a = parse_strs(&["compress", "a.txt", "dir", "--level", "3"]).unwrap();
        match a.command {
            Command::Job(j) => {
                assert_eq!(j.action, Action::Compress);
                assert_eq!(j.paths.len(), 2);
                assert_eq!(j.level, 3);
            }
            _ => panic!("expected a job"),
        }
        assert!(matches!(
            parse_strs(&["x.pp", "y.PP"]).unwrap().command,
            Command::Job(Job {
                action: Action::Extract,
                ..
            })
        ));
        assert!(matches!(
            parse_strs(&["x.pp", "notes.txt"]).unwrap().command,
            Command::Job(Job {
                action: Action::Compress,
                ..
            })
        ));
        assert!(matches!(
            parse_strs(&["--com", "extract-here", "-Embedding"]).unwrap().command,
            Command::Com(Action::ExtractHere)
        ));
        assert!(matches!(parse_strs(&[]).unwrap().command, Command::Settings));
        assert!(matches!(
            parse_strs(&["register", "--machine"]).unwrap().command,
            Command::Register { machine: true, .. }
        ));
        // A file called "compress" after a verb is a path.
        assert!(matches!(
            parse_strs(&["extract", "compress"]).unwrap().command,
            Command::Job(Job {
                action: Action::Extract,
                ..
            })
        ));
        assert!(parse_strs(&["compress"]).is_err());
        assert!(parse_strs(&["--level", "12", "a"]).is_err());
        assert!(parse_strs(&["--bogus"]).is_err());
    }
}
