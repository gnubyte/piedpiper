//! Compress / decompress / train drivers. Encoder and decoder share one loop ([`run`]) so
//! they execute the identical sequence of floating-point operations.

use std::sync::Arc;

use crate::coder::{Decoder, Encoder};
use crate::config::{batch_policy, preset, ModelConfig, TrainConfig, DEFAULT_LEVEL, MAX_LEVEL, MIN_LEVEL};
use crate::error::{Error, Result};
use crate::audio;
use crate::format::{self, crc32, Header, MODE_AUDIO, MODE_DEFLATE, MODE_JPEG, MODE_MEDIA, MODE_NEURAL, MODE_STORED};
use crate::filter::{Filter, FilterChoice};
use crate::matcher::MatchModel;
use crate::fpenv::FpGuard;
use crate::jpeg;
use crate::media;
use crate::model::{prof, Model};
use crate::prior::{builtin_prior, builtin_prior_by_id, Prior};
use crate::tokenizer::{looks_like_text, Bpe, Tokenizer};

/// Where the neural model runs. The output is byte-identical either way: the GPU path
/// reproduces the CPU's arithmetic exactly, so every build decompresses every file. GPU support
/// needs the `gpu` cargo feature and a Vulkan 1.2 GPU (tested on AMD Radeon); anything else runs
/// on the CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuMode {
    /// Always the CPU.
    #[default]
    Off,
    /// The GPU when one is usable and the stream gains from it: compressing 64K tokens or
    /// more, decompressing large inputs (decoding waits on the GPU for every symbol).
    Auto,
    /// The GPU for every neural stream, whenever one is usable (small inputs can then be
    /// slower than on the CPU).
    On,
}

/// Whether this build and machine can run the neural model on a GPU (see [`gpu_status`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuStatus {
    /// A GPU that passed the self-test against the CPU model.
    Available { device: String, driver: String },
    /// Why not: no GPU support in this build, no Vulkan driver, a device without the required
    /// float controls, or a failed self-test.
    Unavailable(String),
}

/// Whether this build can run the neural model on a GPU at all (cargo feature `gpu`).
pub const GPU_SUPPORT: bool = cfg!(feature = "gpu");

static GPU_STREAMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many neural streams this process has run on a GPU so far (for diagnostics, e.g. to tell
/// whether [`GpuMode::Auto`] picked the GPU for a file).
pub fn gpu_stream_count() -> u64 {
    GPU_STREAMS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Opens the GPU (once per process), runs a small network on it and on the CPU, and reports
/// whether they agree bit for bit. [`GpuMode::Auto`] and [`GpuMode::On`] use the GPU only when
/// this says [`GpuStatus::Available`].
pub fn gpu_status() -> GpuStatus {
    #[cfg(feature = "gpu")]
    {
        let _fp = FpGuard::new();
        match crate::gpu::status() {
            Ok((device, driver)) => GpuStatus::Available { device, driver },
            Err(e) => GpuStatus::Unavailable(e),
        }
    }
    #[cfg(not(feature = "gpu"))]
    GpuStatus::Unavailable("this build has no GPU support (cargo feature `gpu`)".into())
}

/// How the input is turned into symbols.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerChoice {
    /// Word-level BPE for large text, raw bytes otherwise.
    Auto,
    /// One symbol per byte (256 symbols).
    Bytes,
    /// Word-aware BPE learned from the input, with up to `vocab` symbols (257..=65535).
    Bpe { vocab: u32 },
}

#[derive(Clone, Debug)]
pub struct CompressOptions {
    /// 1 (fastest) ..= 9 (strongest). Default 5.
    pub level: u32,
    pub tokenizer: TokenizerChoice,
    /// Worker threads (0 = all cores). Never changes the output.
    pub threads: usize,
    /// Override the preset's network shape (used verbatim, including `n_streams` and
    /// `seg_len`).
    pub model: Option<ModelConfig>,
    /// Override the preset's training recipe.
    pub train: Option<TrainConfig>,
    /// Start from a pretrained prior (its shape and tokenizer replace the level's).
    pub prior: Option<Arc<Prior>>,
    /// When no `prior` is given, start text-like input from the level's built-in prior
    /// (levels 1–3; ignored when `model` overrides the network). Default `true`.
    pub builtin_prior: bool,
    /// Fall back to storing the input verbatim when the model cannot beat that.
    pub allow_stored: bool,
    /// Recompress H.264 video in MP4/MOV files with the video recoder (format mode 2)
    /// instead of the neural model, which cannot shrink already-compressed video.
    /// Default `true`.
    pub media: bool,
    /// Recompress DEFLATE streams inside PNG / ZIP (docx, xlsx, jar, ...) / gzip / PDF
    /// files (mode 4). Default `true`.
    pub deflate: bool,
    /// Mix a long-range match model into the network's predictions. Default `true`.
    pub match_model: bool,
    /// Recompress JPEG images with Lepton (format mode 3) instead of the neural model,
    /// which cannot shrink their entropy-coded data. Default `true`.
    pub jpeg: bool,
    /// Reversible pre-transform before neural coding (e.g. branch filters for
    /// executables). Default [`FilterChoice::Auto`].
    pub filter: FilterChoice,
    /// Code 16-bit PCM WAV files with stereo decorrelation and adaptive prediction in front
    /// of the neural model (format mode 5). A `prior` or `model` override then applies to
    /// the prediction residuals. Default `true`.
    pub audio: bool,
    /// Run the neural model on a GPU (Vulkan, e.g. AMD Radeon). Never changes the output.
    /// Default [`GpuMode::Off`].
    pub gpu: GpuMode,
}

impl Default for CompressOptions {
    fn default() -> Self {
        CompressOptions {
            level: DEFAULT_LEVEL,
            tokenizer: TokenizerChoice::Auto,
            threads: 0,
            model: None,
            train: None,
            prior: None,
            builtin_prior: true,
            allow_stored: true,
            media: true,
            deflate: true,
            match_model: true,
            jpeg: true,
            filter: FilterChoice::Auto,
            audio: true,
            gpu: GpuMode::Off,
        }
    }
}

impl CompressOptions {
    pub fn level(level: u32) -> Self {
        CompressOptions { level, ..Default::default() }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DecompressOptions {
    /// Worker threads (0 = all cores).
    pub threads: usize,
    /// The prior the stream was compressed with, if any.
    pub prior: Option<Arc<Prior>>,
    /// Run the neural model on a GPU (Vulkan, e.g. AMD Radeon). Default [`GpuMode::Off`].
    pub gpu: GpuMode,
}

/// Header information of a compressed stream.
#[derive(Clone, Debug)]
pub struct Info {
    pub original_size: u64,
    pub compressed_size: u64,
    pub crc32: u32,
    pub stored: bool,
    pub tokenizer: String,
    pub vocab_size: usize,
    pub n_tokens: u64,
    pub model: Option<ModelConfig>,
    pub train: Option<TrainConfig>,
    pub n_params: usize,
    /// Id of the prior needed to decompress (0 = none).
    pub prior_id: u64,
    /// Name of the built-in prior with that id, if this build has it.
    pub prior_name: Option<String>,
    /// Pre-transform the stream was coded with (`"x86"`, `"arm64"`), if any.
    pub filter: Option<String>,
}

/// Progress callback: `(processed_input_bytes, total_input_bytes) -> keep_going`.
pub type ProgressFn<'a> = dyn FnMut(u64, u64) -> bool + Send + 'a;

trait SymbolCoder {
    /// True when the symbols are known before coding (encoding, measuring): the GPU can then
    /// run whole segments ahead of the coder.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    const KNOWS_SYMBOLS: bool;
    fn code(&mut self, probs: &[f32], sym: u16) -> u16;
}

struct Enc(Encoder);
impl SymbolCoder for Enc {
    const KNOWS_SYMBOLS: bool = true;
    #[inline]
    fn code(&mut self, probs: &[f32], sym: u16) -> u16 {
        self.0.encode_symbol(probs, sym as usize);
        sym
    }
}

struct Dec<'a>(Decoder<'a>);
impl SymbolCoder for Dec<'_> {
    const KNOWS_SYMBOLS: bool = false;
    #[inline]
    fn code(&mut self, probs: &[f32], _sym: u16) -> u16 {
        self.0.decode_symbol(probs) as u16
    }
}

/// Measures the ideal code length instead of coding (prior training).
struct Meter {
    bits: f64,
}
impl SymbolCoder for Meter {
    const KNOWS_SYMBOLS: bool = true;
    #[inline]
    fn code(&mut self, probs: &[f32], sym: u16) -> u16 {
        self.bits -= (probs[sym as usize].max(1e-30) as f64).log2();
        sym
    }
}

/// Stream `b` covers tokens `starts[b]..starts[b] + lens[b]`; earlier streams are at most
/// one token longer than later ones.
pub(crate) fn split_streams(n: usize, ns: usize) -> (Vec<usize>, Vec<usize>) {
    let base = n / ns;
    let extra = n % ns;
    let mut starts = Vec::with_capacity(ns);
    let mut lens = Vec::with_capacity(ns);
    let mut s = 0;
    for b in 0..ns {
        let l = base + usize::from(b < extra);
        starts.push(s);
        lens.push(l);
        s += l;
    }
    (starts, lens)
}

/// The network that drives the coder: the CPU model, or the same model on a GPU (which
/// computes the same bits).
// One per stream; boxing the CPU model would only add an indirection to its hot path.
#[allow(clippy::large_enum_variant)]
enum Net {
    Cpu(Model),
    #[cfg(feature = "gpu")]
    Gpu(Box<crate::gpu::GpuNet>),
}

/// [`GpuMode::Auto`] uses the GPU where it wins: compressing 64K tokens or more (it runs
/// whole segments ahead of the coder), and decompressing only large inputs, whose batches of
/// 256+ rows amortise one GPU round trip per position.
#[cfg(feature = "gpu")]
fn wants_gpu(mode: GpuMode, mc: &ModelConfig, n_tokens: usize, decoding: bool) -> bool {
    let rows = mc.n_streams as usize * mc.seg_len as usize;
    match mode {
        GpuMode::Off => false,
        GpuMode::On => n_tokens < 1 << 31,
        GpuMode::Auto => (1 << 16..1 << 31).contains(&n_tokens) && (!decoding || rows >= 256),
    }
}

impl Net {
    /// The network for a stream of `n_tokens`: on the GPU when `mode` asks for it and a usable
    /// GPU passed its self-test, otherwise the CPU model. `weights` replaces the initial
    /// weights (a prior).
    #[allow(unused_variables)]
    fn new(vocab: usize, mc: &ModelConfig, tc: &TrainConfig, weights: Option<&[f32]>, mode: GpuMode, n_tokens: usize, decoding: bool) -> Net {
        let mut model = Model::new(vocab, mc, tc);
        if let Some(w) = weights {
            model.params_mut().copy_from_slice(w);
            model.params_changed();
        }
        #[cfg(feature = "gpu")]
        if wants_gpu(mode, mc, n_tokens, decoding) {
            if let Ok(g) = crate::gpu::model_for(vocab, mc, tc, model.params()) {
                return Net::Gpu(Box::new(crate::gpu::GpuNet::new(g)));
            }
        }
        Net::Cpu(model)
    }

    /// True when the GPU failed during this stream (the stream must then be coded again on the
    /// CPU, which gives the same bytes).
    fn gpu_failed(&self) -> bool {
        match self {
            Net::Cpu(_) => false,
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.failed,
        }
    }

    /// Counts a stream that ran to the end on the GPU (see [`gpu_stream_count`]).
    fn note_finished(&self) {
        #[cfg(feature = "gpu")]
        if let Net::Gpu(g) = self {
            if !g.failed {
                GPU_STREAMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// The weights of a CPU network (prior training reads them back).
    fn cpu_params(&self) -> Option<&[f32]> {
        match self {
            Net::Cpu(m) => Some(m.params()),
            #[cfg(feature = "gpu")]
            Net::Gpu(_) => None,
        }
    }

    fn n_streams(&self) -> usize {
        match self {
            Net::Cpu(m) => m.n_streams(),
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.model.n_streams(),
        }
    }

    fn segment_len(&self) -> usize {
        match self {
            Net::Cpu(m) => m.segment_len(),
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.model.segment_len(),
        }
    }

    fn bos(&self) -> u32 {
        match self {
            Net::Cpu(m) => m.bos(),
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.model.bos(),
        }
    }

    fn vocab(&self) -> usize {
        match self {
            Net::Cpu(m) => m.vocab,
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.model.vocab,
        }
    }

    fn begin_segment(&mut self, pos: u64) {
        match self {
            Net::Cpu(m) => m.begin_segment(pos),
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.begin_segment(pos),
        }
    }

    fn forward_step(&mut self, t: usize, inputs: &[u32], active: &[bool]) -> Result<()> {
        match self {
            Net::Cpu(m) => {
                m.forward_step(t, inputs, active);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.forward_step(t, inputs, active),
        }
    }

    fn probs_row(&self, t: usize, b: usize) -> &[f32] {
        match self {
            Net::Cpu(m) => m.probs_row(t, b),
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.probs_row(t, b),
        }
    }

    fn set_target(&mut self, t: usize, b: usize, sym: u32) {
        match self {
            Net::Cpu(m) => m.set_target(t, b, sym),
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.set_target(t, b, sym),
        }
    }

    fn train_step(&mut self, lseg: usize) {
        match self {
            Net::Cpu(m) => {
                m.train_step(lseg);
            }
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.train_step(lseg),
        }
    }

    fn end_segment(&mut self, lseg: usize) -> Result<()> {
        match self {
            Net::Cpu(m) => {
                m.end_segment(lseg);
                Ok(())
            }
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => g.end_segment(lseg),
        }
    }
}

/// Match models: up to two per stream (short and long context), tables sized to the stream
/// (<= 16 MB each in total).
fn match_models(lens: &[usize], match_min: usize, match_long: usize) -> (Vec<MatchModel>, Vec<MatchModel>) {
    let ns = lens.len();
    let budget_bits = (22 - (usize::BITS - ns.leading_zeros()).saturating_sub(1)).clamp(12, 20);
    let make = |m: usize| -> Vec<MatchModel> {
        if m > 0 {
            lens.iter().map(|&l| MatchModel::new(m, l, budget_bits)).collect()
        } else {
            Vec::new()
        }
    };
    (make(match_min), make(match_long))
}

/// Codes the symbol at position `p` of stream `b` with the network's distribution `row`, mixed
/// with the stream's match models, updates them, and returns the symbol.
#[allow(clippy::too_many_arguments)]
#[inline]
fn code_symbol<C: SymbolCoder>(
    row: &[f32],
    tokens: &mut [u16],
    starts: &[usize],
    lens: &[usize],
    b: usize,
    p: usize,
    matchers: &mut [MatchModel],
    long_matchers: &mut [MatchModel],
    mixed: &mut [f32],
    coder: &mut C,
) -> u16 {
    let idx = starts[b] + p;
    let stream = &tokens[starts[b]..starts[b] + lens[b]];
    let pred_l = long_matchers.get(b).and_then(|m| m.predict(stream, p));
    let pred_s = matchers.get(b).and_then(|m| m.predict(stream, p));
    let sym = if pred_l.is_some() || pred_s.is_some() {
        // Long context first (more reliable), then the short one.
        mixed.copy_from_slice(row);
        let mut used = false;
        if let Some((tok, bk)) = pred_l {
            used |= long_matchers[b].mix(mixed, tok, bk);
        }
        if let Some((tok, bk)) = pred_s {
            used |= matchers[b].mix(mixed, tok, bk);
        }
        coder.code(if used { &mixed[..] } else { row }, tokens[idx])
    } else {
        coder.code(row, tokens[idx])
    };
    tokens[idx] = sym;
    let stream = &tokens[starts[b]..starts[b] + lens[b]];
    if let Some(m) = long_matchers.get_mut(b) {
        m.update(stream, p);
    }
    if let Some(m) = matchers.get_mut(b) {
        m.update(stream, p);
    }
    sym
}

/// The shared encode/decode/train loop.
fn run<C: SymbolCoder>(
    net: &mut Net,
    tokens: &mut [u16],
    coder: &mut C,
    match_min: usize,
    match_long: usize,
    progress: &mut dyn FnMut(usize, usize) -> bool,
) -> Result<()> {
    #[cfg(feature = "gpu")]
    if C::KNOWS_SYMBOLS {
        if let Net::Gpu(g) = net {
            return run_gpu_encoder(g, tokens, coder, match_min, match_long, progress);
        }
    }
    let ns = net.n_streams();
    let seg = net.segment_len();
    let n = tokens.len();
    let (starts, lens) = split_streams(n, ns);
    let max_len = lens[0];
    let bos = net.bos();
    let mut inputs = vec![bos; ns];
    let mut active = vec![false; ns];
    let (mut matchers, mut long_matchers) = match_models(&lens, match_min, match_long);
    let mut mixed = vec![0f32; net.vocab()];
    let mut pos = 0usize;
    while pos < max_len {
        let lseg = seg.min(max_len - pos);
        net.begin_segment(pos as u64);
        for t in 0..lseg {
            let p = pos + t;
            for b in 0..ns {
                active[b] = p < lens[b];
                inputs[b] = if p == 0 || !active[b] { bos } else { tokens[starts[b] + p - 1] as u32 };
            }
            prof!(0, net.forward_step(t, &inputs, &active))?;
            prof!(4, {
                for b in 0..ns {
                    if active[b] {
                        let sym = code_symbol(net.probs_row(t, b), tokens, &starts, &lens, b, p, &mut matchers, &mut long_matchers, &mut mixed, coder);
                        net.set_target(t, b, sym as u32);
                    }
                }
            });
        }
        pos += lseg;
        if pos < max_len {
            net.train_step(lseg);
        }
        prof!(10, net.end_segment(lseg))?;
        let done: usize = lens.iter().map(|&l| l.min(pos)).sum();
        if !progress(done, n) {
            return Err(Error::Cancelled);
        }
    }
    #[cfg(feature = "profile")]
    if std::env::var_os("PIEDPIPER_PROFILE").is_some() {
        crate::model::profiler::report();
    }
    Ok(())
}

/// [`run`] when the symbols are known in advance and the network is on a GPU: every input of a
/// segment exists before it is coded, so the GPU runs up to `KMAX` segments (forward pass,
/// training step, memory update) per submission, and the coder then reads their probabilities
/// in the usual order. The result is bit-identical to the position-by-position loop.
///
/// Batches grow and shrink so a submission takes roughly 50-400 ms: enough to amortise its
/// fixed cost, and far below the 2 s after which Windows resets a GPU that seems stuck.
#[cfg(feature = "gpu")]
fn run_gpu_encoder<C: SymbolCoder>(
    net: &mut crate::gpu::GpuNet,
    tokens: &mut [u16],
    coder: &mut C,
    match_min: usize,
    match_long: usize,
    progress: &mut dyn FnMut(usize, usize) -> bool,
) -> Result<()> {
    use crate::gpu::{SegIo, KMAX};
    let mut kcur = 1usize;
    let ns = net.model.n_streams();
    let seg = net.model.segment_len();
    let n = tokens.len();
    let (starts, lens) = split_streams(n, ns);
    let max_len = lens[0];
    let bos = net.model.bos();
    let (mut matchers, mut long_matchers) = match_models(&lens, match_min, match_long);
    let mut mixed = vec![0f32; net.model.vocab];
    type Staged = (usize, usize, Vec<u32>, Vec<bool>, Vec<u32>);
    let mut pos = 0usize;
    while pos < max_len {
        let mut batch: Vec<Staged> = Vec::with_capacity(KMAX);
        let mut p0 = pos;
        while p0 < max_len && batch.len() < kcur {
            let lseg = seg.min(max_len - p0);
            let mut inputs = vec![bos; lseg * ns];
            let mut active = vec![false; lseg * ns];
            let mut targets = vec![0u32; lseg * ns];
            for t in 0..lseg {
                let p = p0 + t;
                for b in 0..ns {
                    let r = t * ns + b;
                    active[r] = p < lens[b];
                    if active[r] {
                        inputs[r] = if p == 0 { bos } else { tokens[starts[b] + p - 1] as u32 };
                        targets[r] = tokens[starts[b] + p] as u32;
                    }
                }
            }
            batch.push((p0, lseg, inputs, active, targets));
            p0 += lseg;
        }
        let segs: Vec<SegIo> = batch
            .iter()
            .map(|(p, l, i, a, t)| SegIo { pos: *p as u64, lseg: *l, inputs: i, active: a, targets: t, train: p + l < max_len })
            .collect();
        let t0 = std::time::Instant::now();
        if let Err(e) = net.model.run_segments(&segs) {
            return Err(net.fail(e));
        }
        let secs = t0.elapsed().as_secs_f64();
        if secs < 0.05 && kcur < KMAX {
            kcur *= 2;
        } else if secs > 0.4 && kcur > 1 {
            kcur /= 2;
        }
        let g = &net.model;
        for (k, (sp, lseg, _, _, _)) in batch.iter().enumerate() {
            for t in 0..*lseg {
                let p = sp + t;
                for b in 0..ns {
                    if p < lens[b] {
                        code_symbol(g.probs_row_k(k, t, b), tokens, &starts, &lens, b, p, &mut matchers, &mut long_matchers, &mut mixed, coder);
                    }
                }
            }
            let done: usize = lens.iter().map(|&l| l.min(sp + lseg)).sum();
            if !progress(done, n) {
                return Err(Error::Cancelled);
            }
        }
        pos = p0;
    }
    Ok(())
}

/// Prior and fine-tuning rate for a stream. A lower rate keeps more of what the prior knows:
/// better on text like its training data, worse on text less like it. Alternate priors of the
/// same shape (built-in ones only) suit other kinds of text. Compress a prefix with each
/// candidate and keep the winner. The comparison is between byte counts from the
/// deterministic coder, so every platform makes the same choice; prior id and rate are
/// stored in the header as usual.
#[allow(clippy::too_many_arguments)]
fn pick_prior_lr(
    p: &Arc<Prior>,
    alts: &[Arc<Prior>],
    vocab: usize,
    tokens: &[u16],
    tc: &TrainConfig,
    max_streams: u32,
    match_min: usize,
    match_long: usize,
) -> Result<(Arc<Prior>, TrainConfig)> {
    const MIN_TOKENS: usize = 8192;
    const TRIAL_TOKENS: usize = 16384;
    if tokens.len() < MIN_TOKENS {
        return Ok((p.clone(), *tc));
    }
    let n = TRIAL_TOKENS.min(tokens.len() / 2);
    let mut low = *tc;
    low.lr *= 0.5;
    low.lr_min *= 0.5;
    let size = |p: &Prior, t: &TrainConfig| -> Result<usize> {
        let mut mc = p.model;
        let (ns, seg) = batch_policy(n, max_streams);
        mc.n_streams = ns.min(n as u32).max(1);
        mc.seg_len = seg;
        let mut model = Model::new(vocab, &mc, t);
        model.params_mut().copy_from_slice(&p.weights);
        model.params_changed();
        let mut trial = tokens[..n].to_vec();
        let mut enc = Enc(Encoder::new(Vec::new()));
        run(&mut Net::Cpu(model), &mut trial, &mut enc, match_min, match_long, &mut |_, _| true)?;
        Ok(enc.0.finish().len())
    };
    let mut best = (size(p, tc)?, p.clone(), *tc);
    // The default prior at half the rate, then each alternate at half the rate (they were
    // trained gently on in-domain text). Ties keep the earlier candidate.
    let candidates = std::iter::once(p).chain(alts.iter().filter(|a| a.model == p.model && a.tokenizer.vocab_size() == vocab));
    for q in candidates {
        let s = size(q, &low)?;
        if s < best.0 {
            best = (s, q.clone(), low);
        }
    }
    Ok((best.1, best.2))
}

fn scale(done: usize, n: usize, total: u64) -> u64 {
    if n == 0 {
        total
    } else {
        (done as u128 * total as u128 / n as u128) as u64
    }
}

/// Runs `f` on a worker of a (cached) pool with `threads` threads (0 = all cores), so the
/// kernels' parallel loops dispatch cheaply from inside the pool.
fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    #[cfg(feature = "parallel")]
    {
        use std::collections::HashMap;
        use std::sync::Mutex;
        use std::sync::OnceLock;
        static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
        let n = if threads == 0 {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            threads
        }
        .min(256);
        let pool = {
            let mut map = POOLS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner());
            match map.get(&n) {
                Some(p) => Some(p.clone()),
                None => rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("piedpiper-{i}"))
                    .build()
                    .ok()
                    .map(|p| {
                        let p = Arc::new(p);
                        map.insert(n, p.clone());
                        p
                    }),
            }
        };
        if let Some(pool) = pool {
            return pool.install(|| {
                let _fp = FpGuard::new();
                f()
            });
        }
    }
    let _ = threads;
    let _fp = FpGuard::new();
    f()
}

fn stored(input: &[u8], crc: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + 16);
    format::write_prefix(&mut out, MODE_STORED, input.len() as u64, crc);
    out.extend_from_slice(input);
    out
}

/// Picks the tokenizer for `input`.
fn choose_tokenizer(input: &[u8], choice: TokenizerChoice, max_vocab: u32) -> Tokenizer {
    match choice {
        TokenizerChoice::Bytes => Tokenizer::Bytes,
        TokenizerChoice::Bpe { vocab } => Tokenizer::Bpe(Bpe::train(input, vocab.clamp(257, 65535) as usize, 2)),
        TokenizerChoice::Auto => {
            const MIN_BPE_BYTES: usize = 1 << 20;
            if input.len() < MIN_BPE_BYTES || !looks_like_text(input) {
                return Tokenizer::Bytes;
            }
            // Grow the vocabulary with the input: ~1 symbol per 1 KiB of text.
            let vocab = (256 + input.len() / 1024).clamp(512, max_vocab.max(512) as usize);
            let min_count = (input.len() as u64 / 50_000).max(8);
            Tokenizer::Bpe(Bpe::train(input, vocab, min_count))
        }
    }
}

/// Built-in priors are byte-level text models: use them for text, and for inputs small
/// enough that a word tokenizer trained from scratch would not do better.
fn wants_builtin_prior(input: &[u8], choice: TokenizerChoice) -> bool {
    const MAX_AUTO_PRIOR_BYTES: usize = 8 << 20;
    looks_like_text(input) && (choice == TokenizerChoice::Bytes || input.len() < MAX_AUTO_PRIOR_BYTES)
}

/// Deflate containers nest at most this deep (a ZIP holding a DOCX: 2 levels).
const MAX_NESTING: u32 = 3;

/// Mode 4: expand DEFLATE streams, then compress the metadata, the inflated content and
/// each embedded file as ordinary PiedPiper streams ("parts"). Returns `None` when nothing
/// was expanded or it did not pay.
#[cfg(feature = "deflate")]
fn deflate_mode(input: &[u8], crc: u32, opts: &CompressOptions, progress: &mut ProgressFn<'_>, depth: u32) -> Result<Option<Vec<u8>>> {
    let Some(x) = crate::deflate::expand(input) else { return Ok(None) };
    let mut inner = opts.clone();
    inner.deflate = false;
    inner.media = false;
    inner.jpeg = false;
    inner.allow_stored = true;
    // Embedded files are compressed like files of their own: JPEGs go to Lepton, nested
    // containers are expanded again (up to MAX_NESTING), MP4s go to the video recoder.
    let mut embedded = opts.clone();
    embedded.allow_stored = true;
    // PNG pixels: the inflated scanlines get the image filter.
    let mut pixels = inner.clone();
    pixels.filter = plain_filter(input, &x.plain, &inner.filter);
    // PDF images arrive with the image-filter layout read from their dictionary.
    let per_part: Vec<CompressOptions> = x
        .hints
        .iter()
        .map(|h| match h {
            Some(f) if opts.filter != FilterChoice::Off => CompressOptions { filter: FilterChoice::Force(f.clone()), ..embedded.clone() },
            _ => embedded.clone(),
        })
        .collect();
    let mut parts: Vec<(&[u8], &CompressOptions)> = vec![(x.meta.as_slice(), &inner), (x.plain.as_slice(), &pixels)];
    parts.extend(x.embedded.iter().zip(&per_part).map(|(j, o)| (j.as_slice(), o)));
    let total_in = input.len() as u64;
    let work = parts.iter().map(|p| p.0.len() as u64).sum::<u64>().max(1);
    let mut out = Vec::with_capacity(input.len() / 2 + 64);
    format::write_prefix(&mut out, MODE_DEFLATE, total_in, crc);
    format::put_varint(&mut out, parts.len() as u64);
    let mut before = 0u64;
    for (data, o) in parts {
        let s = compress_at(data, o, &mut |d, _| progress((before + d).min(work) * total_in / work, total_in), depth + 1)?;
        before += data.len() as u64;
        format::put_varint(&mut out, s.len() as u64);
        out.extend_from_slice(&s);
    }
    if out.len() + 16 >= input.len() {
        return Ok(None);
    }
    Ok(Some(out))
}

/// Filter for the inflated content of mode 4. PNG scanlines get the image filter, which
/// unfilters and re-predicts the pixels (and re-filters them exactly on decode).
#[cfg(feature = "deflate")]
fn plain_filter(input: &[u8], plain: &[u8], choice: &FilterChoice) -> FilterChoice {
    match choice {
        FilterChoice::Auto => crate::image::png_layout_for(input, plain)
            .map_or(FilterChoice::Auto, |l| FilterChoice::Force(Filter::Image(l))),
        other => other.clone(),
    }
}

/// Decodes the parts of a mode-4 stream. Metadata and inflated content are plain streams;
/// embedded files may use any mode (nesting is bounded by `MAX_NESTING`).
#[cfg(feature = "deflate")]
fn deflate_restore(body: &[u8], opts: &DecompressOptions, progress: &mut ProgressFn<'_>, depth: u32) -> Result<Vec<u8>> {
    let bad = |m: &str| Error::Format(m.to_string());
    let mut pos = 0;
    let n = format::get_varint(body, &mut pos)?;
    if !(2..=body.len() as u64).contains(&n) {
        return Err(bad("malformed deflate container"));
    }
    let mut streams = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        let len = usize::try_from(format::get_varint(body, &mut pos)?).map_err(|_| bad("bad length"))?;
        let s = body.get(pos..pos.checked_add(len).ok_or_else(|| bad("bad length"))?).ok_or_else(|| bad("truncated deflate container"))?;
        pos += len;
        let ok = i >= 2 || s.len() <= 5 || s[5] == MODE_NEURAL || s[5] == MODE_STORED;
        if !ok {
            return Err(bad("nested container streams are not allowed"));
        }
        streams.push(s);
    }
    if pos != body.len() {
        return Err(bad("trailing data in deflate container"));
    }
    let total = streams.iter().map(|s| s.len() as u64).sum::<u64>().max(1);
    let mut parts = Vec::with_capacity(streams.len());
    let mut before = 0u64;
    for s in streams {
        parts.push(decompress_at(s, opts, &mut |d, t| progress(before + d * s.len() as u64 / t.max(1), total), depth + 1)?);
        before += s.len() as u64;
    }
    let embedded = parts.split_off(2);
    crate::deflate::restore(&parts[0], &parts[1], &embedded)
}

#[cfg(not(feature = "deflate"))]
fn deflate_restore(_: &[u8], _: &DecompressOptions, _: &mut ProgressFn<'_>, _: u32) -> Result<Vec<u8>> {
    Err(Error::Format("this build cannot decompress deflate-recompressed streams (feature `deflate`)".into()))
}

/// Compresses `input` with the given options.
pub fn compress(input: &[u8], opts: &CompressOptions) -> Result<Vec<u8>> {
    compress_with_progress(input, opts, &mut |_, _| true)
}

/// Like [`compress`], reporting progress (and allowing cancellation).
pub fn compress_with_progress(input: &[u8], opts: &CompressOptions, progress: &mut ProgressFn<'_>) -> Result<Vec<u8>> {
    compress_at(input, opts, progress, 0)
}

/// [`compress_with_progress`] for a stream nested `depth` containers deep.
fn compress_at(input: &[u8], opts: &CompressOptions, progress: &mut ProgressFn<'_>, depth: u32) -> Result<Vec<u8>> {
    #[cfg(not(feature = "deflate"))]
    let _ = depth;
    if !(MIN_LEVEL..=MAX_LEVEL).contains(&opts.level) {
        return Err(Error::Options(format!("level must be in {MIN_LEVEL}..={MAX_LEVEL}")));
    }
    let crc = crc32(input);
    if input.is_empty() {
        return Ok(stored(input, crc));
    }
    if opts.media && opts.model.is_none() && opts.prior.is_none() {
        if let Some(out) = media::compress(input, crc, progress)? {
            return Ok(out);
        }
    }
    if opts.jpeg && opts.model.is_none() && opts.prior.is_none() {
        if let Some(out) = jpeg::compress(input, crc, opts.threads, progress)? {
            return Ok(out);
        }
    }
    // Unlike the modes above, audio also runs with a prior or model override: those
    // configure the inner residual stream.
    if opts.audio {
        if let Some(out) = audio::compress(input, crc, opts, progress)? {
            return Ok(out);
        }
    }
    #[cfg(feature = "deflate")]
    if opts.deflate && depth < MAX_NESTING && opts.model.is_none() && opts.prior.is_none() && crate::deflate::candidate(input) {
        if let Some(out) = deflate_mode(input, crc, opts, progress, depth)? {
            return Ok(out);
        }
    }
    // Already-compressed video that mode 2 cannot recode (HEVC, VP9, AV1, MKV/WebM, ...):
    // the neural model would spend hours to save ~0%, so store it right away.
    if opts.media && opts.allow_stored && opts.model.is_none() && opts.prior.is_none() && media::opaque(input) {
        return Ok(stored(input, crc));
    }
    // The same for compressed archives and media (xz, zstd, 7z, MP3, FLAC, WebP, ...).
    if opts.allow_stored && opts.model.is_none() && opts.prior.is_none() && crate::opaque::is_opaque(input) {
        return Ok(stored(input, crc));
    }
    // Reversible pre-transform; everything below codes `data`, the CRC covers `input`.
    let pick = match &opts.filter {
        FilterChoice::Auto => Filter::detect(input),
        FilterChoice::Off => Filter::None,
        FilterChoice::Force(f) => f.clone(),
    };
    let filtered = pick
        .encode(input)
        .filter(|v| v.len() <= input.len() && pick.decode(v.clone()).is_ok_and(|d| d == input));
    let filter = if filtered.is_some() { pick } else { Filter::None };
    let data: &[u8] = filtered.as_deref().unwrap_or(input);
    let pre = preset(opts.level);
    let prior = match &opts.prior {
        Some(p) => Some(p.clone()),
        None if opts.builtin_prior
            && opts.model.is_none()
            && matches!(opts.tokenizer, TokenizerChoice::Auto | TokenizerChoice::Bytes)
            && wants_builtin_prior(data, opts.tokenizer) =>
        {
            builtin_prior(opts.level)
        }
        None => None,
    };
    let mut tc = opts.train.unwrap_or(pre.train);
    if opts.train.is_none() && prior.is_some() {
        tc.lr = pre.prior_lr;
        tc.lr_min = pre.prior_lr * 0.25;
    }
    tc.validate()?;
    let model_override = opts.model;
    let choice = opts.tokenizer;
    let match_model = opts.match_model;
    let auto_lr = opts.train.is_none();
    // Built-in priors (not a caller's own) may be swapped for an alternate of the same level.
    let builtin_chosen = opts.prior.is_none();
    let level = opts.level;
    let gpu = opts.gpu;
    let total = input.len() as u64;
    let out = with_threads(opts.threads, move || -> Result<Vec<u8>> {
        let (tok, base_cfg) = match &prior {
            Some(p) => (p.tokenizer.clone(), p.model),
            None => (choose_tokenizer(data, choice, pre.max_vocab), model_override.unwrap_or(pre.model)),
        };
        let mut tokens = tok.encode(data);
        let n = tokens.len();
        let mut mc = base_cfg;
        match model_override {
            Some(m) => {
                mc.seg_len = m.seg_len;
                mc.n_streams = m.n_streams;
            }
            None => {
                let (ns, seg) = batch_policy(n, pre.model.n_streams);
                mc.n_streams = ns;
                mc.seg_len = seg;
            }
        }
        mc.n_streams = mc.n_streams.min(n as u32).max(1);
        mc.validate(tok.vocab_size())?;
        let mut out = Vec::with_capacity(input.len() / 3 + 256);
        format::write_prefix(&mut out, MODE_NEURAL, total, crc);
        // Match models, context lengths in tokens: a short one (for bytes it acts like a
        // recency-weighted order-3 model) and a long one for repeated records. Tuned on
        // held-out text, logs, CSV and JSON at level 2.
        // Photo residuals (MED prediction) are noise to the short model, so it is left out
        // there; flat images (colour transform only) gain from both.
        let (match_min, match_long): (usize, usize) = match &filter {
            _ if !match_model => (0, 0),
            _ if tok.vocab_size() > 256 => (4, 0),
            Filter::Image(l) if l.predictor == crate::image::Predictor::ColorMed => (0, 12),
            _ => (3, 12),
        };
        // Experiment knobs, only in profiling builds.
        #[cfg(feature = "profile")]
        let match_min: usize = std::env::var("PIEDPIPER_MATCH_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(match_min);
        #[cfg(feature = "profile")]
        let match_long: usize = std::env::var("PIEDPIPER_MATCH_LONG").ok().and_then(|v| v.parse().ok()).unwrap_or(match_long);
        let (prior, tc) = match &prior {
            Some(p) if auto_lr => {
                let alts = if builtin_chosen { crate::prior::builtin_alternates(level) } else { Vec::new() };
                let (q, t) = pick_prior_lr(p, &alts, tok.vocab_size(), &tokens, &tc, pre.model.n_streams, match_min, match_long)?;
                (Some(q), t)
            }
            _ => (prior.clone(), tc),
        };
        format::write_neural(&mut out, &tok, &mc, &tc, n as u64, prior.as_ref().map_or(0, |p| p.id()), match_min as u8, match_long as u8, &filter);
        let weights = prior.as_ref().map(|p| &p.weights[..]);
        let mut net = Net::new(tok.vocab_size(), &mc, &tc, weights, gpu, n, false);
        let header = out.clone();
        let mut enc = Enc(Encoder::new(out));
        match run(&mut net, &mut tokens, &mut enc, match_min, match_long, &mut |d, n| progress(scale(d, n, total), total)) {
            Ok(()) => net.note_finished(),
            // The GPU failed mid-stream (it is not used again): code the stream again on the CPU,
            // which gives the same bytes.
            Err(_) if net.gpu_failed() => {
                drop(net);
                let mut cpu = Net::new(tok.vocab_size(), &mc, &tc, weights, GpuMode::Off, n, false);
                enc = Enc(Encoder::new(header));
                run(&mut cpu, &mut tokens, &mut enc, match_min, match_long, &mut |d, n| progress(scale(d, n, total), total))?;
            }
            Err(e) => return Err(e),
        }
        Ok(enc.0.finish())
    })?;
    if opts.allow_stored && out.len() >= input.len() + 10 {
        return Ok(stored(input, crc));
    }
    Ok(out)
}

/// Decompresses a PiedPiper stream.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    decompress_with_progress(data, &DecompressOptions::default(), &mut |_, _| true)
}

/// Like [`decompress`], with options and progress reporting.
pub fn decompress_with_progress(data: &[u8], opts: &DecompressOptions, progress: &mut ProgressFn<'_>) -> Result<Vec<u8>> {
    decompress_at(data, opts, progress, 0)
}

/// [`decompress_with_progress`] for a stream nested `depth` containers deep.
fn decompress_at(data: &[u8], opts: &DecompressOptions, progress: &mut ProgressFn<'_>, depth: u32) -> Result<Vec<u8>> {
    let hdr: Header = format::read_header(data)?;
    let out = match hdr.mode {
        MODE_STORED => {
            let body = &data[hdr.payload..];
            if (body.len() as u64) < hdr.orig_len {
                return Err(Error::Format("truncated stored data".into()));
            }
            body[..hdr.orig_len as usize].to_vec()
        }
        MODE_MEDIA => media::decompress(&data[hdr.payload..], hdr.orig_len, progress)?,
        MODE_DEFLATE if depth >= MAX_NESTING => return Err(Error::Format("deflate containers nested too deeply".into())),
        MODE_DEFLATE => deflate_restore(&data[hdr.payload..], opts, progress, depth)?,
        MODE_JPEG => jpeg::decompress(&data[hdr.payload..], hdr.orig_len, opts.threads, progress)?,
        MODE_AUDIO => audio::decompress(&data[hdr.payload..], hdr.orig_len, opts, progress)?,
        _ => {
            let prior = if hdr.prior_id != 0 {
                let builtin = builtin_prior_by_id(hdr.prior_id).map(|b| b.prior.clone());
                match opts.prior.as_ref().filter(|p| p.id() == hdr.prior_id).or(builtin.as_ref()).or(opts.prior.as_ref()) {
                    Some(p) if p.id() == hdr.prior_id => {
                        // The whole shape must match (a damaged header could otherwise ask
                        // for a different parameter count than the prior holds).
                        if p.model.d_model != hdr.model.d_model
                            || p.model.n_layer != hdr.model.n_layer
                            || p.model.n_head != hdr.model.n_head
                            || p.model.d_ff != hdr.model.d_ff
                            || p.model.window != hdr.model.window
                            || p.vocab_size() != hdr.tokenizer.vocab_size()
                        {
                            return Err(Error::Prior("prior does not match the stream's network".into()));
                        }
                        Some(p.clone())
                    }
                    Some(p) => {
                        return Err(Error::Prior(format!(
                            "stream needs prior {:016x}, but prior {:016x} was given",
                            hdr.prior_id,
                            p.id()
                        )))
                    }
                    None => {
                        return Err(Error::Prior(format!(
                            "stream was compressed with prior {:016x}; supply that prior to decompress",
                            hdr.prior_id
                        )))
                    }
                }
            } else {
                None
            };
            let n = usize::try_from(hdr.n_tokens).map_err(|_| Error::TooLarge("token count".into()))?;
            let orig = usize::try_from(hdr.orig_len).map_err(|_| Error::TooLarge("output size".into()))?;
            let total = hdr.orig_len;
            with_threads(opts.threads, || -> Result<Vec<u8>> {
                let mut tokens = vec![0u16; n];
                let weights = prior.as_ref().map(|p| &p.weights[..]);
                let mut net = Net::new(hdr.tokenizer.vocab_size(), &hdr.model, &hdr.train, weights, opts.gpu, n, true);
                let mut dec = Dec(Decoder::new(&data[hdr.payload..]));
                let (mm, ml) = (hdr.match_min as usize, hdr.match_long as usize);
                match run(&mut net, &mut tokens, &mut dec, mm, ml, &mut |d, n| progress(scale(d, n, total), total)) {
                    Ok(()) => net.note_finished(),
                    // The GPU failed mid-stream: decode again from the start on the CPU.
                    Err(_) if net.gpu_failed() => {
                        drop(net);
                        let mut cpu = Net::new(hdr.tokenizer.vocab_size(), &hdr.model, &hdr.train, weights, GpuMode::Off, n, true);
                        tokens.iter_mut().for_each(|t| *t = 0);
                        let mut dec = Dec(Decoder::new(&data[hdr.payload..]));
                        run(&mut cpu, &mut tokens, &mut dec, mm, ml, &mut |d, n| progress(scale(d, n, total), total))?;
                    }
                    Err(e) => return Err(e),
                }
                let mut out = Vec::with_capacity(orig);
                hdr.tokenizer.decode_into(&tokens, &mut out);
                hdr.filter.decode(out)
            })?
        }
    };
    if out.len() as u64 != hdr.orig_len {
        return Err(Error::Checksum { expected: hdr.crc, actual: crc32(&out) });
    }
    let actual = crc32(&out);
    if actual != hdr.crc {
        return Err(Error::Checksum { expected: hdr.crc, actual });
    }
    Ok(out)
}

/// Reads the header of a compressed stream without decompressing it.
pub fn inspect(data: &[u8]) -> Result<Info> {
    let hdr = format::read_header(data)?;
    let stored = hdr.mode == MODE_STORED;
    let vocab = hdr.tokenizer.vocab_size();
    let tokenizer = match hdr.mode {
        MODE_STORED => "stored".into(),
        MODE_MEDIA => media::name(&data[hdr.payload..]).into(),
        MODE_DEFLATE => "deflate-recode".into(),
        MODE_JPEG => jpeg::NAME.into(),
        MODE_AUDIO => audio::NAME.into(),
        _ => hdr.tokenizer.name(),
    };
    let stored = stored || hdr.mode == MODE_MEDIA || hdr.mode == MODE_JPEG || hdr.mode == MODE_DEFLATE || hdr.mode == MODE_AUDIO;
    Ok(Info {
        original_size: hdr.orig_len,
        compressed_size: data.len() as u64,
        crc32: hdr.crc,
        stored: hdr.mode == MODE_STORED,
        tokenizer,
        vocab_size: vocab,
        n_tokens: hdr.n_tokens,
        model: (!stored).then_some(hdr.model),
        train: (!stored).then_some(hdr.train),
        n_params: if stored { 0 } else { hdr.model.n_params(vocab) },
        prior_id: hdr.prior_id,
        prior_name: builtin_prior_by_id(hdr.prior_id).map(|b| b.name.to_string()),
        filter: (hdr.filter != Filter::None).then(|| hdr.filter.name().to_string()),
    })
}

/// Options for [`train_prior`].
#[derive(Clone, Debug)]
pub struct TrainOptions {
    /// Network shape comes from this level unless `model` is given.
    pub level: u32,
    pub tokenizer: TokenizerChoice,
    pub model: Option<ModelConfig>,
    pub train: Option<TrainConfig>,
    /// Passes over the corpus.
    pub epochs: u32,
    pub threads: usize,
    /// Continue training an existing prior (its shape and tokenizer are kept).
    pub init: Option<Arc<Prior>>,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions {
            level: DEFAULT_LEVEL,
            tokenizer: TokenizerChoice::Bytes,
            model: None,
            train: None,
            epochs: 1,
            threads: 0,
            init: None,
        }
    }
}

/// Result of [`train_prior`].
pub struct TrainReport {
    /// The serialised prior (load with [`Prior::from_bytes`]).
    pub prior: Vec<u8>,
    /// Online code length of each epoch, in bits per input byte.
    pub epoch_bpb: Vec<f64>,
}

/// Trains a prior on `corpus` with the same online engine (no coding), so that later
/// compressions can start from these weights.
pub fn train_prior(corpus: &[u8], opts: &TrainOptions, progress: &mut ProgressFn<'_>) -> Result<TrainReport> {
    if corpus.is_empty() {
        return Err(Error::Options("empty training corpus".into()));
    }
    let pre = preset(opts.level);
    let tc = opts.train.unwrap_or_else(|| {
        let mut t = pre.train;
        t.decay = 4096;
        t
    });
    tc.validate()?;
    let epochs = opts.epochs.max(1);
    let total = corpus.len() as u64 * epochs as u64;
    with_threads(opts.threads, || -> Result<TrainReport> {
        let (tok, mut mc, init_w) = match &opts.init {
            Some(p) => (p.tokenizer.clone(), p.model, Some(p.weights.clone())),
            None => {
                let tok = match opts.tokenizer {
                    TokenizerChoice::Auto => choose_tokenizer(corpus, TokenizerChoice::Auto, pre.max_vocab),
                    c => choose_tokenizer(corpus, c, pre.max_vocab),
                };
                (tok, opts.model.unwrap_or(pre.model), None)
            }
        };
        let tokens = tok.encode(corpus);
        let n = tokens.len();
        if let Some(m) = &opts.model {
            mc.seg_len = m.seg_len;
            mc.n_streams = m.n_streams;
        } else {
            mc.seg_len = 32;
            mc.n_streams = 32;
        }
        mc.n_streams = mc.n_streams.min(n as u32).max(1);
        mc.validate(tok.vocab_size())?;
        let mut weights = init_w;
        let mut epoch_bpb = Vec::new();
        for e in 0..epochs {
            let mut model = Model::new(tok.vocab_size(), &mc, &tc);
            if let Some(w) = &weights {
                model.params_mut().copy_from_slice(w);
                model.params_changed();
            }
            let mut meter = Meter { bits: 0.0 };
            let mut toks = tokens.clone();
            let base = e as u64 * corpus.len() as u64;
            let mut net = Net::Cpu(model);
            run(&mut net, &mut toks, &mut meter, 0, 0, &mut |d, n| {
                progress(base + scale(d, n, corpus.len() as u64), total)
            })?;
            epoch_bpb.push(meter.bits / corpus.len() as f64);
            weights = net.cpu_params().map(|w| w.to_vec());
        }
        let prior = Prior::encode(&tok, &mc, weights.as_deref().unwrap_or(&[]));
        Ok(TrainReport { prior, epoch_bpb })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_opts() -> CompressOptions {
        CompressOptions {
            level: 1,
            model: Some(ModelConfig { d_model: 32, n_layer: 2, n_head: 2, d_ff: 48, window: 24, seg_len: 8, n_streams: 3 }),
            allow_stored: false,
            ..Default::default()
        }
    }

    fn roundtrip(data: &[u8], opts: &CompressOptions) -> Vec<u8> {
        let c = compress(data, opts).unwrap();
        let d = decompress_with_progress(&c, &DecompressOptions { threads: 0, prior: opts.prior.clone(), ..Default::default() }, &mut |_, _| true)
            .unwrap();
        assert_eq!(d, data);
        c
    }

    #[test]
    fn executable_filter_roundtrips() {
        // A fake x86-64 PE: headers, then code calling four functions over and over.
        let mut exe = vec![0u8; 0x200];
        exe[..2].copy_from_slice(b"MZ");
        exe[0x3c] = 0x80;
        exe[0x80..0x84].copy_from_slice(b"PE\0\0");
        exe[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        let targets = [0x1000u32, 0x2340, 0x5678, 0x9abc];
        for i in 0..1500u32 {
            let pos = exe.len() as u32;
            exe.extend_from_slice(&[0x48, 0x89, 0xc1, 0xE8]);
            exe.extend_from_slice(&targets[(i * 7 % 4) as usize].wrapping_sub(pos + 8).to_le_bytes());
        }
        let on = roundtrip(&exe, &small_opts());
        assert_eq!(inspect(&on).unwrap().filter.as_deref(), Some("x86"));
        let off = roundtrip(&exe, &CompressOptions { filter: FilterChoice::Off, ..small_opts() });
        assert_eq!(inspect(&off).unwrap().filter, None);
        assert!(on.len() < off.len(), "{} vs {}", on.len(), off.len());
        // Forcing a filter that does not fit still round-trips.
        roundtrip(b"just some text, no machine code here", &CompressOptions { filter: FilterChoice::Force(Filter::Arm64), ..small_opts() });
    }

    /// A photo-like RGB test image: smooth gradients plus noise in `0..noise`.
    fn photo_pixels(w: usize, h: usize, noise: u64) -> Vec<u8> {
        let mut rng = crate::rng::Rng::new(21);
        let mut px = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let n = (rng.next_u64() % noise) as usize;
                px.extend_from_slice(&[(x * 3 + n) as u8, (y * 4 + x + n) as u8, (200 + 2 * n).wrapping_sub(x + y / 2) as u8]);
            }
        }
        px
    }

    #[test]
    fn image_filter_roundtrips() {
        let (w, h) = (64, 48);
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        ppm.extend_from_slice(&photo_pixels(w, h, 7));
        let opts = CompressOptions { allow_stored: false, ..CompressOptions::level(1) };
        let on = roundtrip(&ppm, &opts);
        assert_eq!(inspect(&on).unwrap().filter.as_deref(), Some("image"));
        let off = roundtrip(&ppm, &CompressOptions { filter: FilterChoice::Off, ..opts.clone() });
        assert_eq!(inspect(&off).unwrap().filter, None);
        assert!(on.len() < off.len(), "{} vs {}", on.len(), off.len());
    }

    /// PNG pixels reach the neural codec through mode 4; the inflated scanlines get the
    /// image filter, which unfilters them and re-filters exactly on decode.
    #[cfg(feature = "deflate")]
    #[test]
    fn png_pixels_get_the_image_filter() {
        let (w, h) = (64usize, 48usize);
        let px = photo_pixels(w, h, 7);
        let rb = w * 3;
        let mut scan = Vec::new();
        for y in 0..h {
            let kind = if y % 2 == 0 { 1u8 } else { 2 }; // Sub, Up
            scan.push(kind);
            for i in 0..rb {
                let left = if i >= 3 { px[y * rb + i - 3] } else { 0 };
                let up = if y > 0 { px[(y - 1) * rb + i] } else { 0 };
                scan.push(px[y * rb + i].wrapping_sub(if kind == 1 { left } else { up }));
            }
        }
        // A real deflate stream (mode 4 leaves stored blocks alone). Level 6 on this image is
        // one that preflate-rs reconstructs; it rejects some streams.
        let z = miniz_oxide::deflate::compress_to_vec_zlib(&scan, 6);
        let chunk = |png: &mut Vec<u8>, kind: &[u8], data: &[u8]| {
            png.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let start = png.len();
            png.extend_from_slice(kind);
            png.extend_from_slice(data);
            let crc = format::crc32(&png[start..]);
            png.extend_from_slice(&crc.to_be_bytes());
        };
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&(w as u32).to_be_bytes());
        ihdr.extend_from_slice(&(h as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"IDAT", &z);
        chunk(&mut png, b"IEND", &[]);

        let plain = crate::deflate::expand(&png).expect("PNG expands").plain;
        let FilterChoice::Force(Filter::Image(l)) = plain_filter(&png, &plain, &FilterChoice::Auto) else {
            panic!("PNG scanlines did not get the image filter")
        };
        assert!(l.png && l.width == w as u32 && l.height == h as u32 && l.channels == 3);
        assert_eq!(plain_filter(&png, &plain, &FilterChoice::Off), FilterChoice::Off);
        assert_eq!(plain_filter(b"not a png", &plain, &FilterChoice::Auto), FilterChoice::Auto);
        // The filtered inner stream round-trips and beats the unfiltered one. Match models
        // are off to isolate the filter: on this tiny synthetic image the short match model
        // finds structure in the raw scanlines that real photos lack (on a real photo the
        // filter still wins by ~30% with them on).
        let opts = CompressOptions { allow_stored: false, match_model: false, ..CompressOptions::level(1) };
        let on = roundtrip(&plain, &CompressOptions { filter: FilterChoice::Force(Filter::Image(l)), ..opts.clone() });
        assert_eq!(inspect(&on).unwrap().filter.as_deref(), Some("image"));
        let off = roundtrip(&plain, &CompressOptions { filter: FilterChoice::Off, ..opts.clone() });
        assert!(on.len() < off.len(), "{} vs {}", on.len(), off.len());
        // The whole PNG round-trips through compress, whichever mode wins.
        roundtrip(&png, &opts);
    }

    #[test]
    fn roundtrips() {
        let text = "PiedPiper compresses text with a transformer that learns as it goes. ".repeat(60);
        let mut rng = crate::rng::Rng::new(3);
        let random: Vec<u8> = (0..3000).map(|_| rng.next_u64() as u8).collect();
        for data in [&b""[..], b"a", b"ab", b"hello world", text.as_bytes(), &random] {
            roundtrip(data, &small_opts());
        }
        let c = roundtrip(text.as_bytes(), &small_opts());
        assert!(c.len() < text.len() * 3 / 4, "{} vs {}", c.len(), text.len());
        // Default (preset) path, including the batch policy.
        roundtrip(text.as_bytes(), &CompressOptions { allow_stored: false, ..CompressOptions::level(1) });
    }

    #[test]
    fn window_shorter_and_longer_than_segment() {
        let text = b"abracadabra, abracadabra; the quick brown fox. ".repeat(40);
        for (win, seg, ns) in [(3, 8, 1), (40, 5, 2), (1, 1, 4), (16, 16, 7)] {
            let mut o = small_opts();
            o.model = Some(ModelConfig { d_model: 16, n_layer: 1, n_head: 2, d_ff: 16, window: win, seg_len: seg, n_streams: ns });
            roundtrip(&text, &o);
        }
    }

    #[test]
    fn bpe_and_momentum() {
        let text = "It was the best of times, it was the worst of times, it was the age of wisdom. ".repeat(80);
        let mut o = small_opts();
        o.tokenizer = TokenizerChoice::Bpe { vocab: 300 };
        let mut tc = preset(1).train;
        tc.beta1 = 0.9;
        tc.clip = 0.5;
        o.train = Some(tc);
        roundtrip(text.as_bytes(), &o);
    }

    #[test]
    fn thread_count_does_not_change_output() {
        let text = "Determinism across thread counts is a hard requirement. 0123456789 ".repeat(50);
        let mut o = small_opts();
        o.threads = 1;
        let a = compress(text.as_bytes(), &o).unwrap();
        o.threads = 4;
        let b = compress(text.as_bytes(), &o).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn corruption_is_detected() {
        let text = b"some text that will be corrupted after compression ".repeat(20);
        let mut c = compress(&text, &small_opts()).unwrap();
        // Flip a bit in the middle of the coded payload (the last few flush bytes of a
        // range coder can legitimately be "don't care").
        let k = c.len() / 2 + 20;
        c[k] ^= 0x40;
        assert!(decompress(&c).is_err());
        assert!(decompress(b"nope").is_err());
        assert!(decompress(&c[..10]).is_err());
    }

    #[test]
    fn stored_fallback() {
        let mut rng = crate::rng::Rng::new(9);
        let random: Vec<u8> = (0..500).map(|_| rng.next_u64() as u8).collect();
        let mut o = small_opts();
        o.allow_stored = true;
        let c = compress(&random, &o).unwrap();
        assert!(inspect(&c).unwrap().stored);
        assert_eq!(decompress(&c).unwrap(), random);
    }

    #[test]
    fn builtin_priors_are_automatic() {
        let text = b"Built-in priors make small text files compress well from the first byte. ".repeat(8);
        let o = CompressOptions { level: 1, allow_stored: false, ..Default::default() };
        let c = compress(&text, &o).unwrap();
        let info = inspect(&c).unwrap();
        if crate::prior::builtin_priors().is_empty() {
            assert_eq!(info.prior_id, 0);
        } else {
            assert_ne!(info.prior_id, 0);
            assert_eq!(info.prior_name.as_deref(), Some("text-s"));
        }
        assert_eq!(decompress(&c).unwrap(), text);
        // Binary-looking input and explicit opt-out train from scratch.
        let bin: Vec<u8> = (0..2000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        assert_eq!(inspect(&compress(&bin, &o).unwrap()).unwrap().prior_id, 0);
        let off = CompressOptions { builtin_prior: false, ..o.clone() };
        assert_eq!(inspect(&compress(&text, &off).unwrap()).unwrap().prior_id, 0);
    }

    #[test]
    fn prior_train_and_use() {
        let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(100);
        let topts = TrainOptions {
            model: Some(ModelConfig { d_model: 32, n_layer: 1, n_head: 2, d_ff: 32, window: 16, seg_len: 8, n_streams: 4 }),
            epochs: 2,
            ..Default::default()
        };
        let rep = train_prior(corpus.as_bytes(), &topts, &mut |_, _| true).unwrap();
        assert_eq!(rep.epoch_bpb.len(), 2);
        assert!(rep.epoch_bpb[1] < rep.epoch_bpb[0]);
        let prior = Arc::new(Prior::from_bytes(&rep.prior).unwrap());
        let text = b"the cat sat on the log. the dog sat on the mat. ".repeat(10);
        let mut o = CompressOptions::level(1);
        o.allow_stored = false;
        o.builtin_prior = false;
        o.prior = Some(prior.clone());
        let with = roundtrip(&text, &o);
        o.prior = None;
        let without = compress(&text, &o).unwrap();
        assert!(with.len() < without.len(), "{} vs {}", with.len(), without.len());
        // Wrong / missing prior is a clean error.
        o.prior = Some(prior);
        let c = compress(&text, &o).unwrap();
        assert!(matches!(decompress(&c), Err(Error::Prior(_))));
    }
}

