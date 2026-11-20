//! `piedpiper` — command-line front end (gzip/xz-style).

use piedpiper::{CompressOptions, DecompressOptions, ModelConfig, Prior, TokenizerChoice, TrainOptions};
use std::sync::Arc;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const EXT: &str = "pp";

const HELP: &str = "\
piedpiper — neural lossless compression (TufaZip method: an online-learning
Transformer driving an arithmetic coder). Output is bit-identical on every platform.

USAGE:
    piedpiper [OPTIONS] [FILE]...          compress FILE -> FILE.pp
    piedpiper -d [OPTIONS] [FILE]...       decompress FILE.pp -> FILE
    piedpiper info FILE.pp...              show stream header
    piedpiper bench [OPTIONS] FILE...      compress + decompress + verify, report speed
    piedpiper train -o PRIOR [OPTIONS] FILE...
                                           train a prior (initial weights) on a corpus

    With no FILE, or FILE '-', reads stdin and writes stdout.

OPTIONS:
    -1 ... -9              compression level (default 2). 1-3 start text from built-in
                           priors; 4-9 are bigger networks for large inputs
    -d, --decompress       decompress
    -c, --stdout           write to standard output
    -o, --output PATH      output file (single input only)
    -f, --force            overwrite existing output files
        --rm               delete input files after success (default: keep them)
    -t, --threads N        worker threads (default: all cores; never changes output)
    -T, --tokenizer KIND   auto | bytes | bpe | bpe:VOCAB   (default auto)
        --verify           after compressing, decompress in memory and compare
    -p, --prior FILE       start from (and decompress with) a pretrained prior
        --no-prior         don't use the built-in text prior (levels 1-3)
        --no-match         disable the long-range match model
        --no-deflate       don't recompress deflate streams (PNG/ZIP/gzip/PDF)
        --no-filter        don't pre-filter executables (branch targets) or raw images
                           and PNG pixels (colour + 2-D prediction)
        --no-jpeg          code JPEG files with the neural model instead of Lepton
        --no-video         code H.264, HEVC and VP9 video with the neural model instead of
                           the lossless video recoder
        --no-audio         code WAV files as plain bytes instead of prediction residuals
        --epochs N         training passes for `train` (default 1); WAV inputs train
                           on their prediction residuals (an audio prior)
    -q, --quiet            no progress output
    -v, --verbose          print model details
    -h, --help             this help
    -V, --version          version

ADVANCED (override the level's network / training recipe):
    --d-model N --layers N --heads N --d-ff N --window N --seg N --streams N
    --lr F --lr-min F --warmup N --decay N --beta1 F --beta2 F --clip F --seed N
";

#[derive(Default)]
struct Args {
    mode: Mode,
    files: Vec<String>,
    level: Option<u32>,
    stdout: bool,
    output: Option<String>,
    force: bool,
    rm: bool,
    threads: usize,
    tokenizer: Option<TokenizerChoice>,
    verify: bool,
    quiet: bool,
    verbose: bool,
    model: Vec<(String, String)>,
    prior: Option<String>,
    no_prior: bool,
    no_match: bool,
    no_deflate: bool,
    no_jpeg: bool,
    no_video: bool,
    no_filter: bool,
    no_audio: bool,
    epochs: u32,
}

#[derive(Default, PartialEq, Eq, Clone, Copy)]
enum Mode {
    #[default]
    Compress,
    Decompress,
    Info,
    Bench,
    Train,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1).peekable();
    let mut first = true;
    let mut only_files = false;
    while let Some(arg) = it.next() {
        if first {
            first = false;
            match arg.as_str() {
                "compress" | "c" => continue,
                "decompress" | "d" | "x" => {
                    a.mode = Mode::Decompress;
                    continue;
                }
                "info" | "i" | "list" => {
                    a.mode = Mode::Info;
                    continue;
                }
                "bench" | "b" => {
                    a.mode = Mode::Bench;
                    continue;
                }
                "train" => {
                    a.mode = Mode::Train;
                    continue;
                }
                _ => {}
            }
        }
        if only_files || arg == "-" || !arg.starts_with('-') {
            a.files.push(arg);
            continue;
        }
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--" => only_files = true,
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("piedpiper {} (format v{})", piedpiper::VERSION, piedpiper::FORMAT_VERSION);
                return Ok(None);
            }
            "-d" | "--decompress" => a.mode = Mode::Decompress,
            "-l" | "--list" | "--info" => a.mode = Mode::Info,
            "--bench" => a.mode = Mode::Bench,
            "-c" | "--stdout" => a.stdout = true,
            "-o" | "--output" => a.output = Some(val("--output")?),
            "-f" | "--force" => a.force = true,
            "-k" | "--keep" => a.rm = false,
            "--rm" => a.rm = true,
            "-t" | "--threads" => a.threads = val("--threads")?.parse().map_err(|_| "bad --threads")?,
            "-T" | "--tokenizer" => a.tokenizer = Some(parse_tokenizer(&val("--tokenizer")?)?),
            "--verify" => a.verify = true,
            "-p" | "--prior" => a.prior = Some(val("--prior")?),
            "--no-prior" => a.no_prior = true,
            "--no-match" => a.no_match = true,
            "--no-deflate" => a.no_deflate = true,
            "--no-jpeg" => a.no_jpeg = true,
            "--no-video" => a.no_video = true,
            "--no-filter" => a.no_filter = true,
            "--no-audio" => a.no_audio = true,
            "--epochs" => a.epochs = val("--epochs")?.parse().map_err(|_| "bad --epochs")?,
            "-q" | "--quiet" => a.quiet = true,
            "-v" | "--verbose" => a.verbose = true,
            s if s.len() == 2 && s.as_bytes()[1].is_ascii_digit() && s != "-0" => {
                a.level = Some((s.as_bytes()[1] - b'0') as u32)
            }
            s if s.starts_with("--") => {
                let key = s.trim_start_matches("--").to_string();
                const KEYS: &[&str] = &[
                    "d-model", "layers", "heads", "d-ff", "window", "seg", "streams", "lr", "lr-min", "warmup",
                    "decay", "beta1", "beta2", "clip", "seed", "init-scale", "level",
                ];
                if !KEYS.contains(&key.as_str()) {
                    return Err(format!("unknown option {s} (see --help)"));
                }
                let v = val(s)?;
                if key == "level" {
                    a.level = Some(v.parse().map_err(|_| "bad --level")?);
                } else {
                    a.model.push((key, v));
                }
            }
            s => return Err(format!("unknown option {s} (see --help)")),
        }
    }
    Ok(Some(a))
}

fn parse_tokenizer(s: &str) -> Result<TokenizerChoice, String> {
    match s {
        "auto" => Ok(TokenizerChoice::Auto),
        "bytes" | "byte" => Ok(TokenizerChoice::Bytes),
        "bpe" | "words" | "word" => Ok(TokenizerChoice::Bpe { vocab: 4096 }),
        _ => {
            if let Some(n) = s.strip_prefix("bpe:") {
                let vocab: u32 = n.parse().map_err(|_| format!("bad tokenizer {s}"))?;
                Ok(TokenizerChoice::Bpe { vocab })
            } else {
                Err(format!("unknown tokenizer {s} (auto, bytes, bpe, bpe:N)"))
            }
        }
    }
}

fn load_prior(a: &Args) -> Result<Option<Arc<Prior>>, String> {
    match &a.prior {
        None => Ok(None),
        Some(p) => {
            let bytes = fs::read(p).map_err(|e| format!("{p}: {e}"))?;
            Ok(Some(Arc::new(Prior::from_bytes(&bytes).map_err(|e| format!("{p}: {e}"))?)))
        }
    }
}

fn dopts(a: &Args, prior: &Option<Arc<Prior>>) -> DecompressOptions {
    DecompressOptions { threads: a.threads, prior: prior.clone() }
}

fn build_options(a: &Args) -> Result<CompressOptions, String> {
    let level = a.level.unwrap_or(piedpiper::DEFAULT_LEVEL);
    let mut o = CompressOptions::level(level);
    o.threads = a.threads;
    o.prior = load_prior(a)?;
    o.builtin_prior = !a.no_prior;
    o.match_model = !a.no_match;
    o.deflate = !a.no_deflate;
    o.jpeg = !a.no_jpeg;
    o.media = !a.no_video;
    o.audio = !a.no_audio;
    if a.no_filter {
        o.filter = piedpiper::FilterChoice::Off;
    }
    if let Some(t) = a.tokenizer {
        o.tokenizer = t;
    }
    if !a.model.is_empty() {
        let pre = piedpiper::preset(level);
        let mut mc: ModelConfig = pre.model;
        let mut tc = pre.train;
        let mut model_touched = false;
        for (k, v) in &a.model {
            let u = || v.parse::<u32>().map_err(|_| format!("--{k} needs an integer"));
            let f = || v.parse::<f32>().map_err(|_| format!("--{k} needs a number"));
            match k.as_str() {
                "d-model" => (mc.d_model, model_touched) = (u()?, true),
                "layers" => (mc.n_layer, model_touched) = (u()?, true),
                "heads" => (mc.n_head, model_touched) = (u()?, true),
                "d-ff" => (mc.d_ff, model_touched) = (u()?, true),
                "window" => (mc.window, model_touched) = (u()?, true),
                "seg" => (mc.seg_len, model_touched) = (u()?, true),
                "streams" => (mc.n_streams, model_touched) = (u()?, true),
                "lr" => tc.lr = f()?,
                "lr-min" => tc.lr_min = f()?,
                "warmup" => tc.warmup = u()?,
                "decay" => tc.decay = u()?,
                "beta1" => tc.beta1 = f()?,
                "beta2" => tc.beta2 = f()?,
                "clip" => tc.clip = f()?,
                "init-scale" => tc.init_scale = f()?,
                "seed" => tc.seed = v.parse().map_err(|_| "--seed needs an integer")?,
                _ => unreachable!(),
            }
        }
        if model_touched {
            o.model = Some(mc);
        }
        o.train = Some(tc);
    }
    Ok(o)
}

struct Progress {
    enabled: bool,
    start: Instant,
    last: Instant,
    label: String,
}

impl Progress {
    fn new(enabled: bool, label: &str) -> Self {
        let now = Instant::now();
        Progress { enabled, start: now, last: now - std::time::Duration::from_secs(1), label: label.to_string() }
    }

    fn update(&mut self, done: u64, total: u64) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        if done < total && now.duration_since(self.last).as_millis() < 200 {
            return;
        }
        self.last = now;
        let el = now.duration_since(self.start).as_secs_f64().max(1e-9);
        let rate = done as f64 / el;
        let pct = if total == 0 { 100.0 } else { done as f64 * 100.0 / total as f64 };
        let eta = if rate > 0.0 { (total - done) as f64 / rate } else { 0.0 };
        eprint!(
            "\r{}: {:5.1}%  {}/s  ETA {}   ",
            self.label,
            pct,
            human(rate as u64),
            hms(eta)
        );
        let _ = io::stderr().flush();
    }

    fn finish(&self) {
        if self.enabled {
            eprint!("\r{:70}\r", "");
        }
    }
}

fn human(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.2} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1} KB", b / 1e3)
    } else {
        format!("{b:.0} B")
    }
}

fn hms(s: f64) -> String {
    let s = s.max(0.0) as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s / 60) % 60)
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}

fn read_input(name: &str) -> Result<Vec<u8>, String> {
    if name == "-" {
        let mut v = Vec::new();
        io::stdin().read_to_end(&mut v).map_err(|e| format!("stdin: {e}"))?;
        Ok(v)
    } else {
        fs::read(name).map_err(|e| format!("{name}: {e}"))
    }
}

fn write_output(path: Option<&Path>, data: &[u8], force: bool) -> Result<(), String> {
    match path {
        None => {
            let mut out = io::stdout().lock();
            out.write_all(data).and_then(|_| out.flush()).map_err(|e| format!("stdout: {e}"))
        }
        Some(p) => {
            if p.exists() && !force {
                return Err(format!("{}: already exists (use -f to overwrite)", p.display()));
            }
            let tmp = p.with_extension(format!(
                "{}.part",
                p.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default()
            ));
            fs::write(&tmp, data).map_err(|e| format!("{}: {e}", tmp.display()))?;
            if p.exists() {
                let _ = fs::remove_file(p);
            }
            fs::rename(&tmp, p).map_err(|e| format!("{}: {e}", p.display()))
        }
    }
}

fn out_path(a: &Args, input: &str) -> Result<Option<PathBuf>, String> {
    if a.stdout || input == "-" && a.output.is_none() {
        return Ok(None);
    }
    if let Some(o) = &a.output {
        return Ok(if o == "-" { None } else { Some(PathBuf::from(o)) });
    }
    Ok(Some(match a.mode {
        Mode::Decompress => {
            let p = Path::new(input);
            match p.extension().and_then(|e| e.to_str()) {
                Some(EXT) => p.with_extension(""),
                _ => PathBuf::from(format!("{input}.out")),
            }
        }
        _ => PathBuf::from(format!("{input}.{EXT}")),
    }))
}

fn describe(info: &piedpiper::Info) -> String {
    let s = match (&info.model, &info.train) {
        (Some(m), Some(t)) => format!(
            "tokenizer {} ({} tokens), {} layers x d{} ({} heads, ff {}), window {}, seg {}, {} streams, {:.2}M params, lr {:.1e}",
            info.tokenizer,
            info.n_tokens,
            m.n_layer,
            m.d_model,
            m.n_head,
            m.d_ff,
            m.window,
            m.seg_len,
            m.n_streams,
            info.n_params as f64 / 1e6,
            t.lr
        ),
        _ if info.tokenizer == "h264-recode" => "H.264 video, CABAC/CAVLC layer recompressed losslessly".into(),
        _ if info.tokenizer == "hevc-recode" => "HEVC video, CABAC layer recompressed losslessly".into(),
        _ if info.tokenizer == "vp9-recode" => "VP9 video, entropy-coding layer recompressed losslessly".into(),
        _ if info.tokenizer == "vp8-recode" => "VP8 video, entropy-coding layer recompressed losslessly".into(),
        _ if info.tokenizer == "mpeg2-recode" => "MPEG-2 video, variable-length codes recompressed losslessly".into(),
        _ if info.tokenizer == "h264-ts-recode" => "H.264 transport stream, CABAC/CAVLC layer recompressed losslessly".into(),
        _ if info.tokenizer == "hevc-ts-recode" => "HEVC transport stream, CABAC layer recompressed losslessly".into(),
        _ if info.tokenizer == "jpeg-lepton" => "JPEG, DCT coefficients recompressed losslessly (Lepton)".into(),
        _ if info.tokenizer == "deflate-recode" => "deflate streams (PNG/ZIP/gzip/PDF) expanded and recompressed".into(),
        _ if info.tokenizer == "pcm-audio" => "16-bit PCM audio, adaptive prediction residuals coded by the neural model".into(),
        _ => "stored (incompressible)".into(),
    };
    let s = match &info.filter {
        Some(f) if f == "image" => format!("{s}, image filter (colour + 2-D prediction)"),
        Some(f) => format!("{s}, {f} branch filter"),
        None => s,
    };
    if info.prior_id != 0 {
        match &info.prior_name {
            Some(n) => format!("{s}, built-in prior {n}"),
            None => format!("{s}, prior {:016x}", info.prior_id),
        }
    } else {
        s
    }
}

fn compress_one(a: &Args, opts: &CompressOptions, input: &str) -> Result<(), String> {
    let out = out_path(a, input)?;
    if let Some(p) = &out {
        if p.exists() && !a.force {
            return Err(format!("{}: already exists (use -f to overwrite)", p.display()));
        }
    }
    let data = read_input(input)?;
    let show = !a.quiet && io::stderr().is_terminal();
    let mut prog = Progress::new(show, &short(input));
    let t0 = Instant::now();
    let packed = piedpiper::compress_with_progress(&data, opts, &mut |d, t| {
        prog.update(d, t);
        true
    })
    .map_err(|e| format!("{input}: {e}"))?;
    let secs = t0.elapsed().as_secs_f64();
    prog.finish();
    if a.verify {
        let back = piedpiper::decompress_with_progress(&packed, &dopts(a, &opts.prior), &mut |_, _| true)
            .map_err(|e| format!("{input}: verification failed: {e}"))?;
        if back != data {
            return Err(format!("{input}: verification failed (roundtrip mismatch)"));
        }
    }
    write_output(out.as_deref(), &packed, a.force)?;
    if !a.quiet {
        let ratio = if packed.is_empty() { 0.0 } else { data.len() as f64 / packed.len() as f64 };
        let bpb = if data.is_empty() { 0.0 } else { packed.len() as f64 * 8.0 / data.len() as f64 };
        eprintln!(
            "{}: {} -> {} ({ratio:.3}x, {bpb:.3} bits/byte) in {:.1}s ({}/s){}",
            short(input),
            human(data.len() as u64),
            human(packed.len() as u64),
            secs,
            human((data.len() as f64 / secs.max(1e-9)) as u64),
            if a.verify { ", verified" } else { "" }
        );
        if a.verbose {
            if let Ok(info) = piedpiper::inspect(&packed) {
                eprintln!("    {}", describe(&info));
            }
        }
    }
    if a.rm && input != "-" && out.is_some() {
        fs::remove_file(input).map_err(|e| format!("{input}: {e}"))?;
    }
    Ok(())
}

fn decompress_one(a: &Args, prior: &Option<Arc<Prior>>, input: &str) -> Result<(), String> {
    let out = out_path(a, input)?;
    if let Some(p) = &out {
        if p.exists() && !a.force {
            return Err(format!("{}: already exists (use -f to overwrite)", p.display()));
        }
    }
    let data = read_input(input)?;
    let show = !a.quiet && io::stderr().is_terminal();
    let mut prog = Progress::new(show, &short(input));
    let t0 = Instant::now();
    let plain = piedpiper::decompress_with_progress(&data, &dopts(a, prior), &mut |d, t| {
        prog.update(d, t);
        true
    })
    .map_err(|e| format!("{input}: {e}"))?;
    prog.finish();
    write_output(out.as_deref(), &plain, a.force)?;
    if !a.quiet && out.is_some() {
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "{}: {} -> {} in {:.1}s ({}/s)",
            short(input),
            human(data.len() as u64),
            human(plain.len() as u64),
            secs,
            human((plain.len() as f64 / secs.max(1e-9)) as u64)
        );
    }
    if a.rm && input != "-" && out.is_some() {
        fs::remove_file(input).map_err(|e| format!("{input}: {e}"))?;
    }
    Ok(())
}

fn info_one(input: &str) -> Result<(), String> {
    let data = read_input(input)?;
    let info = piedpiper::inspect(&data).map_err(|e| format!("{input}: {e}"))?;
    let ratio = if info.compressed_size == 0 { 0.0 } else { info.original_size as f64 / info.compressed_size as f64 };
    println!(
        "{}: {} -> {} ({ratio:.3}x), crc32 {:08x}",
        input,
        human(info.compressed_size),
        human(info.original_size),
        info.crc32
    );
    println!("    {}", describe(&info));
    Ok(())
}

fn bench_one(a: &Args, opts: &CompressOptions, input: &str) -> Result<(), String> {
    let data = read_input(input)?;
    let show = !a.quiet && io::stderr().is_terminal();
    let mut prog = Progress::new(show, "compress");
    let t0 = Instant::now();
    let packed = piedpiper::compress_with_progress(&data, opts, &mut |d, t| {
        prog.update(d, t);
        true
    })
    .map_err(|e| e.to_string())?;
    let tc = t0.elapsed().as_secs_f64();
    prog.finish();
    let mut prog = Progress::new(show, "decompress");
    let t1 = Instant::now();
    let back = piedpiper::decompress_with_progress(&packed, &dopts(a, &opts.prior), &mut |d, t| {
        prog.update(d, t);
        true
    })
    .map_err(|e| e.to_string())?;
    let td = t1.elapsed().as_secs_f64();
    prog.finish();
    let ok = back == data;
    let bpb = packed.len() as f64 * 8.0 / data.len().max(1) as f64;
    println!(
        "{}: {} -> {} bytes  {:.4} bpb  ratio {:.3}  comp {:.1}s ({}/s)  decomp {:.1}s ({}/s)  {}",
        short(input),
        data.len(),
        packed.len(),
        bpb,
        data.len() as f64 / packed.len().max(1) as f64,
        tc,
        human((data.len() as f64 / tc.max(1e-9)) as u64),
        td,
        human((data.len() as f64 / td.max(1e-9)) as u64),
        if ok { "OK" } else { "MISMATCH" }
    );
    if a.verbose {
        if let Ok(info) = piedpiper::inspect(&packed) {
            println!("    {}", describe(&info));
        }
    }
    if ok {
        Ok(())
    } else {
        Err("roundtrip mismatch".into())
    }
}

fn train(a: &Args, opts: &CompressOptions) -> Result<(), String> {
    let out = a.output.clone().ok_or("train needs -o PRIOR_FILE")?;
    if Path::new(&out).exists() && !a.force {
        return Err(format!("{out}: already exists (use -f to overwrite)"));
    }
    let mut corpus = Vec::new();
    for f in &a.files {
        let data = read_input(f)?;
        // A prior for audio must learn what the audio mode actually codes: residuals.
        match piedpiper::audio_residuals(&data).filter(|_| !a.no_audio) {
            Some(res) => {
                if !a.quiet {
                    eprintln!("{f}: WAV audio, training on its prediction residuals");
                }
                corpus.extend_from_slice(&res);
            }
            None => corpus.extend_from_slice(&data),
        }
    }
    let topts = TrainOptions {
        level: opts.level,
        tokenizer: a.tokenizer.unwrap_or(TokenizerChoice::Bytes),
        model: opts.model,
        train: opts.train,
        epochs: a.epochs.max(1),
        threads: a.threads,
        init: opts.prior.clone(),
    };
    let show = !a.quiet && io::stderr().is_terminal();
    let mut prog = Progress::new(show, "train");
    let t0 = Instant::now();
    let rep = piedpiper::train_prior(&corpus, &topts, &mut |d, t| {
        prog.update(d, t);
        true
    })
    .map_err(|e| e.to_string())?;
    prog.finish();
    write_output(Some(Path::new(&out)), &rep.prior, a.force)?;
    let p = Prior::from_bytes(&rep.prior).map_err(|e| e.to_string())?;
    if !a.quiet {
        for (i, b) in rep.epoch_bpb.iter().enumerate() {
            eprintln!("epoch {}: {:.4} bits/byte (online)", i + 1, b);
        }
        eprintln!(
            "{out}: {} prior, {:.2}M params, {} tokenizer, id {:016x}, trained on {} in {:.0}s",
            human(rep.prior.len() as u64),
            p.n_params() as f64 / 1e6,
            p.tokenizer_name(),
            p.id(),
            human(corpus.len() as u64),
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn short(p: &str) -> String {
    Path::new(p).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| p.to_string())
}

fn main() -> ExitCode {
    let mut a = match parse_args() {
        Ok(Some(a)) => a,
        Ok(None) => return ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("piedpiper: {e}");
            return ExitCode::from(2);
        }
    };
    if a.files.is_empty() {
        if a.mode == Mode::Info || a.mode == Mode::Bench || a.mode == Mode::Train {
            eprintln!("piedpiper: no input files");
            return ExitCode::from(2);
        }
        if io::stdin().is_terminal() {
            eprintln!("piedpiper: refusing to read from a terminal (see --help)");
            return ExitCode::from(2);
        }
        a.files.push("-".into());
    }
    if a.output.is_some() && a.files.len() > 1 && a.mode != Mode::Train {
        eprintln!("piedpiper: --output needs exactly one input");
        return ExitCode::from(2);
    }
    let opts = match build_options(&a) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("piedpiper: {e}");
            return ExitCode::from(2);
        }
    };
    if a.mode == Mode::Train {
        return match train(&a, &opts) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("piedpiper: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let mut failed = false;
    for f in a.files.clone() {
        let r = match a.mode {
            Mode::Compress => compress_one(&a, &opts, &f),
            Mode::Decompress => decompress_one(&a, &opts.prior, &f),
            Mode::Train => unreachable!(),
            Mode::Info => info_one(&f),
            Mode::Bench => bench_one(&a, &opts, &f),
        };
        if let Err(e) = r {
            eprintln!("piedpiper: {e}");
            failed = true;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
