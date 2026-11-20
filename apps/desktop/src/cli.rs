//! Terminal front end: `--cli`, and the only front end on Linux. Prints progress on stderr
//! and the paths it created on stdout (one per line), so scripts and CI can use it.

use std::io::{self, IsTerminal};
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use ppdesktop::{fmt, Error, Eta, Job, Status};

pub fn run(job: Job) -> ExitCode {
    let st = Arc::new(Status::new(job.headline()));
    let worker = {
        let st = Arc::clone(&st);
        thread::spawn(move || job.run(&st))
    };
    let tty = io::stderr().is_terminal();
    let mut eta = Eta::new();
    while !worker.is_finished() {
        thread::sleep(Duration::from_millis(200));
        if tty {
            let s = st.snapshot();
            let left = eta
                .update(s.fraction)
                .map(|t| format!(", {} left", fmt::remaining(t)))
                .unwrap_or_default();
            let line = format!("{}: {:.1}%{left}  {}", s.headline, s.fraction * 100.0, s.detail);
            eprint!("\r{line:<110.110}");
        }
    }
    if tty {
        eprint!("\r{:110}\r", "");
    }
    match worker.join() {
        Ok(Ok(o)) => {
            for p in &o.created {
                println!("{}", p.display());
            }
            for w in &o.warnings {
                eprintln!("warning: {w}");
            }
            for e in &o.errors {
                eprintln!("error: {e}");
            }
            if o.errors.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Ok(Err(Error::Cancelled)) => ExitCode::FAILURE,
        Ok(Err(e)) => {
            eprintln!("piedpiper-archiver: {e}");
            ExitCode::FAILURE
        }
        Err(_) => {
            eprintln!("piedpiper-archiver: internal error");
            ExitCode::FAILURE
        }
    }
}
