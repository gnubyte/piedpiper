//! WebAssembly exports for the JavaScript package (`bindings/js`).
//!
//! The ABI is deliberately tiny so the JS glue needs no generator:
//! * memory is managed with `pp_alloc` / `pp_free`;
//! * every operation returns a status (0 = ok, 1 = error, 2 = cancelled) and leaves its
//!   output (bytes, JSON or an error message) in a result buffer read with
//!   `pp_result_ptr` / `pp_result_len` and released with `pp_result_free`;
//! * compression options arrive as an array of f64 (see `OPT_*`), which represents every
//!   u32 / f32 option exactly;
//! * progress is reported through the imported `env.pp_progress(done, total) -> i32`
//!   (return 0 to cancel).

use piedpiper::{CompressOptions, DecompressOptions, Error, ModelConfig, Prior, TokenizerChoice, TrainConfig};
use std::cell::RefCell;
use std::sync::Arc;

#[link(wasm_import_module = "env")]
extern "C" {
    fn pp_progress(done: f64, total: f64) -> i32;
}

thread_local! {
    static RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static PRIOR_CACHE: RefCell<Option<(u64, usize, Arc<Prior>)>> = const { RefCell::new(None) };
}

const OK: u32 = 0;
const ERR: u32 = 1;
const CANCELLED: u32 = 2;

// Option slots (f64 array).
const OPT_LEVEL: usize = 0;
const OPT_TOKENIZER: usize = 1; // 0 auto, 1 bytes, 2 bpe
const OPT_BPE_VOCAB: usize = 2;
const OPT_ALLOW_STORED: usize = 3;
const OPT_HAS_MODEL: usize = 4;
const OPT_MODEL: usize = 5; // d_model, n_layer, n_head, d_ff, window, seg_len, n_streams
const OPT_HAS_TRAIN: usize = 12;
const OPT_TRAIN: usize = 13; // lr, lr_min, warmup, decay, beta1, beta2, eps, clip, init_scale, seed_lo, seed_hi
const OPT_PROGRESS: usize = 24; // 1 = call pp_progress
const OPT_BUILTIN_PRIOR: usize = 25; // 1 = use the built-in text prior (levels 1-3)
const OPT_LEN: usize = 26;

fn set_result(v: Vec<u8>) {
    RESULT.with(|r| *r.borrow_mut() = v);
}

fn finish(r: Result<Vec<u8>, Error>) -> u32 {
    match r {
        Ok(v) => {
            set_result(v);
            OK
        }
        Err(Error::Cancelled) => {
            set_result(b"cancelled".to_vec());
            CANCELLED
        }
        Err(e) => {
            set_result(e.to_string().into_bytes());
            ERR
        }
    }
}

unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

fn load_prior(data: &[u8]) -> Result<Option<Arc<Prior>>, Error> {
    if data.is_empty() {
        return Ok(None);
    }
    // Hash only a prefix + length for the cache key; parsing verifies the full CRC anyway.
    let key = {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in data.iter().take(4096).chain(data.iter().rev().take(64)) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3);
        }
        h
    };
    if let Some(p) = PRIOR_CACHE.with(|c| {
        c.borrow().as_ref().filter(|(k, n, _)| *k == key && *n == data.len()).map(|(_, _, p)| p.clone())
    }) {
        return Ok(Some(p));
    }
    let p = Arc::new(Prior::from_bytes(data)?);
    PRIOR_CACHE.with(|c| *c.borrow_mut() = Some((key, data.len(), p.clone())));
    Ok(Some(p))
}

fn progress_fn(enabled: bool) -> impl FnMut(u64, u64) -> bool + Send {
    move |d, t| !enabled || unsafe { pp_progress(d as f64, t as f64) } != 0
}

/// Allocates `len` bytes inside WebAssembly memory for the caller to fill.
#[no_mangle]
pub extern "C" fn pp_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Frees a buffer from [`pp_alloc`].
///
/// # Safety
/// `ptr`/`len` must come from `pp_alloc(len)`.
#[no_mangle]
pub unsafe extern "C" fn pp_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
    }
}

#[no_mangle]
pub extern "C" fn pp_result_ptr() -> *const u8 {
    RESULT.with(|r| r.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn pp_result_len() -> usize {
    RESULT.with(|r| r.borrow().len())
}

#[no_mangle]
pub extern "C" fn pp_result_free() {
    set_result(Vec::new());
}

/// Number of f64 option slots `pp_compress` expects.
#[no_mangle]
pub extern "C" fn pp_options_len() -> usize {
    OPT_LEN
}

fn parse_options(o: &[f64]) -> Result<(CompressOptions, bool), Error> {
    if o.len() < OPT_LEN {
        return Err(Error::Options("options array too short".into()));
    }
    let u = |i: usize| o[i] as u32;
    let mut c = CompressOptions::level(u(OPT_LEVEL));
    c.threads = 1;
    c.tokenizer = match u(OPT_TOKENIZER) {
        0 => TokenizerChoice::Auto,
        1 => TokenizerChoice::Bytes,
        2 => TokenizerChoice::Bpe { vocab: u(OPT_BPE_VOCAB) },
        k => return Err(Error::Options(format!("unknown tokenizer {k}"))),
    };
    c.allow_stored = o[OPT_ALLOW_STORED] != 0.0;
    c.builtin_prior = o[OPT_BUILTIN_PRIOR] != 0.0;
    if o[OPT_HAS_MODEL] != 0.0 {
        let m = &o[OPT_MODEL..OPT_MODEL + 7];
        c.model = Some(ModelConfig {
            d_model: m[0] as u32,
            n_layer: m[1] as u32,
            n_head: m[2] as u32,
            d_ff: m[3] as u32,
            window: m[4] as u32,
            seg_len: m[5] as u32,
            n_streams: m[6] as u32,
        });
    }
    if o[OPT_HAS_TRAIN] != 0.0 {
        let t = &o[OPT_TRAIN..OPT_TRAIN + 11];
        c.train = Some(TrainConfig {
            lr: t[0] as f32,
            lr_min: t[1] as f32,
            warmup: t[2] as u32,
            decay: t[3] as u32,
            beta1: t[4] as f32,
            beta2: t[5] as f32,
            eps: t[6] as f32,
            clip: t[7] as f32,
            init_scale: t[8] as f32,
            seed: (t[9] as u64) | ((t[10] as u64) << 32),
        });
    }
    Ok((c, o[OPT_PROGRESS] != 0.0))
}

/// Compresses `input` → result buffer.
///
/// # Safety
/// Pointers must reference valid memory of the given lengths; `opts` holds `opts_len`
/// little-endian f64 values (any alignment).
#[no_mangle]
pub unsafe extern "C" fn pp_compress(
    input: *const u8,
    input_len: usize,
    opts: *const u8,
    opts_len: usize,
    prior: *const u8,
    prior_len: usize,
) -> u32 {
    let data = bytes(input, input_len);
    let raw = bytes(opts, opts_len * 8);
    let o: Vec<f64> = (0..raw.len() / 8).map(|i| f64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap())).collect();
    let r = (|| {
        let (mut c, prog) = parse_options(&o)?;
        c.prior = load_prior(bytes(prior, prior_len))?;
        let mut f = progress_fn(prog);
        piedpiper::compress_with_progress(data, &c, &mut f)
    })();
    finish(r)
}

/// Decompresses `input` → result buffer.
///
/// # Safety
/// Pointers must reference valid memory of the given lengths.
#[no_mangle]
pub unsafe extern "C" fn pp_decompress(
    input: *const u8,
    input_len: usize,
    prior: *const u8,
    prior_len: usize,
    progress: u32,
) -> u32 {
    let data = bytes(input, input_len);
    let r = (|| {
        let o = DecompressOptions { threads: 1, prior: load_prior(bytes(prior, prior_len))? };
        let mut f = progress_fn(progress != 0);
        piedpiper::decompress_with_progress(data, &o, &mut f)
    })();
    finish(r)
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Header information as JSON → result buffer.
///
/// # Safety
/// Pointer must reference valid memory of the given length.
#[no_mangle]
pub unsafe extern "C" fn pp_inspect(input: *const u8, input_len: usize) -> u32 {
    let r = piedpiper::inspect(bytes(input, input_len)).map(|i| {
        let mut s = format!(
            "{{\"originalSize\":{},\"compressedSize\":{},\"crc32\":{},\"stored\":{},\"tokenizer\":{},\"vocabSize\":{},\"tokens\":{},\"params\":{},\"priorId\":\"{:016x}\",\"priorName\":{}",
            i.original_size,
            i.compressed_size,
            i.crc32,
            i.stored,
            json_str(&i.tokenizer),
            i.vocab_size,
            i.n_tokens,
            i.n_params,
            i.prior_id,
            i.prior_name.as_deref().map_or("null".to_string(), json_str)
        );
        if let Some(m) = i.model {
            s += &format!(
                ",\"model\":{{\"dModel\":{},\"layers\":{},\"heads\":{},\"dFf\":{},\"window\":{},\"segLen\":{},\"streams\":{}}}",
                m.d_model, m.n_layer, m.n_head, m.d_ff, m.window, m.seg_len, m.n_streams
            );
        }
        if let Some(t) = i.train {
            s += &format!(
                ",\"train\":{{\"lr\":{},\"lrMin\":{},\"warmup\":{},\"decay\":{},\"beta1\":{},\"beta2\":{},\"eps\":{},\"clip\":{},\"initScale\":{},\"seed\":\"{}\"}}",
                t.lr, t.lr_min, t.warmup, t.decay, t.beta1, t.beta2, t.eps, t.clip, t.init_scale, t.seed
            );
        }
        s.push('}');
        s.into_bytes()
    });
    finish(r)
}

/// Preset training recipe for `level` as f64 slots (lr, lr_min, warmup, decay, beta1, beta2,
/// eps, clip, init_scale, seed_lo, seed_hi) → result buffer (little-endian f64s).
#[no_mangle]
pub extern "C" fn pp_preset_train(level: u32) -> u32 {
    let t = piedpiper::preset(level).train;
    let vals = [
        t.lr as f64,
        t.lr_min as f64,
        t.warmup as f64,
        t.decay as f64,
        t.beta1 as f64,
        t.beta2 as f64,
        t.eps as f64,
        t.clip as f64,
        t.init_scale as f64,
        (t.seed & 0xFFFF_FFFF) as f64,
        (t.seed >> 32) as f64,
    ];
    set_result(vals.iter().flat_map(|v| v.to_le_bytes()).collect());
    OK
}

/// Library version string → result buffer.
#[no_mangle]
pub extern "C" fn pp_version() -> u32 {
    set_result(piedpiper::VERSION.as_bytes().to_vec());
    OK
}
