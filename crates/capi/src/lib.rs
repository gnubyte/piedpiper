//! C ABI (`include/piedpiper.h`) for using PiedPiper from C, C++, Swift, Go, C#, …
#![allow(non_camel_case_types)]

use piedpiper::{CompressOptions, DecompressOptions, ModelConfig, Prior, TokenizerChoice, TrainConfig};
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

pub const PP_OK: c_int = 0;
pub const PP_ERROR: c_int = 1;
pub const PP_CANCELLED: c_int = 2;

#[repr(C)]
pub struct pp_buffer {
    pub data: *mut u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pp_model {
    pub d_model: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub d_ff: u32,
    pub window: u32,
    pub seg_len: u32,
    pub n_streams: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pp_train {
    pub lr: f32,
    pub lr_min: f32,
    pub warmup: u32,
    pub decay: u32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub clip: f32,
    pub init_scale: f32,
    pub seed: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pp_options {
    pub level: u32,
    pub tokenizer: u32,
    pub bpe_vocab: u32,
    pub threads: u32,
    pub allow_stored: u32,
    pub has_model: u32,
    pub model: pp_model,
    pub has_train: u32,
    pub train: pp_train,
    pub builtin_prior: u32,
}

pub struct pp_prior(Arc<Prior>);

pub type pp_progress_fn = Option<unsafe extern "C" fn(done: u64, total: u64, user: *mut c_void) -> c_int>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

fn to_train(t: &pp_train) -> TrainConfig {
    TrainConfig {
        lr: t.lr,
        lr_min: t.lr_min,
        warmup: t.warmup,
        decay: t.decay,
        beta1: t.beta1,
        beta2: t.beta2,
        eps: t.eps,
        clip: t.clip,
        init_scale: t.init_scale,
        seed: t.seed,
    }
}

fn from_train(t: &TrainConfig) -> pp_train {
    pp_train {
        lr: t.lr,
        lr_min: t.lr_min,
        warmup: t.warmup,
        decay: t.decay,
        beta1: t.beta1,
        beta2: t.beta2,
        eps: t.eps,
        clip: t.clip,
        init_scale: t.init_scale,
        seed: t.seed,
    }
}

unsafe fn slice<'a>(p: *const u8, len: usize) -> &'a [u8] {
    if p.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, len)
    }
}

/// Hands a Vec to C. Every buffer carries one hidden trailing NUL (so JSON / strings can
/// be used directly as C strings); `len` excludes it and `pp_buffer_free` frees `len + 1`.
unsafe fn emit(mut v: Vec<u8>, out: *mut pp_buffer) {
    let len = v.len();
    v.push(0);
    let mut b = v.into_boxed_slice();
    (*out).len = len;
    (*out).data = b.as_mut_ptr();
    std::mem::forget(b);
}

struct UserPtr(*mut c_void);
// SAFETY: the C caller promises the callback/user pointer may be used from worker threads.
unsafe impl Send for UserPtr {}
impl UserPtr {
    // A method, so closures capture the Send wrapper rather than the raw field.
    fn get(&self) -> *mut c_void {
        self.0
    }
}

fn run(
    progress: pp_progress_fn,
    user: *mut c_void,
    out: *mut pp_buffer,
    job: impl FnOnce(&mut piedpiper::ProgressFn<'_>) -> piedpiper::Result<Vec<u8>>,
) -> c_int {
    if out.is_null() {
        set_error("out must not be NULL");
        return PP_ERROR;
    }
    let user = UserPtr(user);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let user = user;
        let mut f = move |d: u64, t: u64| match progress {
            Some(cb) => unsafe { cb(d, t, user.get()) != 0 },
            None => true,
        };
        job(&mut f)
    }));
    match r {
        Ok(Ok(v)) => {
            unsafe { emit(v, out) };
            PP_OK
        }
        Ok(Err(piedpiper::Error::Cancelled)) => {
            set_error("operation cancelled");
            PP_CANCELLED
        }
        Ok(Err(e)) => {
            set_error(&e.to_string());
            PP_ERROR
        }
        Err(_) => {
            set_error("internal error (panic)");
            PP_ERROR
        }
    }
}

/// # Safety
/// `opt` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn pp_options_default(opt: *mut pp_options) {
    if opt.is_null() {
        return;
    }
    let p = piedpiper::preset(piedpiper::DEFAULT_LEVEL);
    *opt = pp_options {
        level: piedpiper::DEFAULT_LEVEL,
        tokenizer: 0,
        bpe_vocab: 4096,
        threads: 0,
        allow_stored: 1,
        has_model: 0,
        model: pp_model {
            d_model: p.model.d_model,
            n_layer: p.model.n_layer,
            n_head: p.model.n_head,
            d_ff: p.model.d_ff,
            window: p.model.window,
            seg_len: p.model.seg_len,
            n_streams: p.model.n_streams,
        },
        has_train: 0,
        train: from_train(&p.train),
        builtin_prior: 1,
    };
}

/// # Safety
/// `train` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn pp_preset_train(level: u32, train: *mut pp_train) {
    if !train.is_null() {
        *train = from_train(&piedpiper::preset(level).train);
    }
}

/// # Safety
/// Pointers must be valid for the given lengths; `opt` and `prior` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn pp_compress(
    data: *const u8,
    len: usize,
    opt: *const pp_options,
    prior: *const pp_prior,
    progress: pp_progress_fn,
    user: *mut c_void,
    out: *mut pp_buffer,
) -> c_int {
    let input = slice(data, len);
    let mut o = CompressOptions::default();
    if let Some(opt) = opt.as_ref() {
        o.level = opt.level;
        o.threads = opt.threads as usize;
        o.allow_stored = opt.allow_stored != 0;
        o.builtin_prior = opt.builtin_prior != 0;
        o.tokenizer = match opt.tokenizer {
            0 => TokenizerChoice::Auto,
            1 => TokenizerChoice::Bytes,
            2 => TokenizerChoice::Bpe { vocab: opt.bpe_vocab },
            k => {
                set_error(&format!("unknown tokenizer {k}"));
                return PP_ERROR;
            }
        };
        if opt.has_model != 0 {
            let m = opt.model;
            o.model = Some(ModelConfig {
                d_model: m.d_model,
                n_layer: m.n_layer,
                n_head: m.n_head,
                d_ff: m.d_ff,
                window: m.window,
                seg_len: m.seg_len,
                n_streams: m.n_streams,
            });
        }
        if opt.has_train != 0 {
            o.train = Some(to_train(&opt.train));
        }
    }
    o.prior = prior.as_ref().map(|p| p.0.clone());
    run(progress, user, out, |f| piedpiper::compress_with_progress(input, &o, f))
}

/// # Safety
/// Pointers must be valid for the given lengths; `prior` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn pp_decompress(
    data: *const u8,
    len: usize,
    threads: u32,
    prior: *const pp_prior,
    progress: pp_progress_fn,
    user: *mut c_void,
    out: *mut pp_buffer,
) -> c_int {
    let input = slice(data, len);
    let o = DecompressOptions { threads: threads as usize, prior: prior.as_ref().map(|p| p.0.clone()) };
    run(progress, user, out, |f| piedpiper::decompress_with_progress(input, &o, f))
}

/// # Safety
/// Pointers must be valid for the given lengths.
#[no_mangle]
pub unsafe extern "C" fn pp_inspect(data: *const u8, len: usize, out: *mut pp_buffer) -> c_int {
    let input = slice(data, len);
    run(None, std::ptr::null_mut(), out, |_| {
        piedpiper::inspect(input).map(|i| {
            let mut s = format!(
                "{{\"original_size\":{},\"compressed_size\":{},\"crc32\":{},\"stored\":{},\"tokenizer\":\"{}\",\"vocab_size\":{},\"tokens\":{},\"params\":{},\"prior_id\":\"{:016x}\",\"prior_name\":{}",
                i.original_size,
                i.compressed_size,
                i.crc32,
                i.stored,
                i.tokenizer,
                i.vocab_size,
                i.n_tokens,
                i.n_params,
                i.prior_id,
                i.prior_name.as_deref().map_or("null".to_string(), |n| format!("\"{n}\""))
            );
            if let Some(m) = i.model {
                s += &format!(
                    ",\"model\":{{\"d_model\":{},\"n_layer\":{},\"n_head\":{},\"d_ff\":{},\"window\":{},\"seg_len\":{},\"n_streams\":{}}}",
                    m.d_model, m.n_layer, m.n_head, m.d_ff, m.window, m.seg_len, m.n_streams
                );
            }
            s.push('}');
            s.into_bytes()
        })
    })
}

/// # Safety
/// `data` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn pp_prior_load(data: *const u8, len: usize) -> *mut pp_prior {
    match Prior::from_bytes(slice(data, len)) {
        Ok(p) => Box::into_raw(Box::new(pp_prior(Arc::new(p)))),
        Err(e) => {
            set_error(&e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `prior` must come from `pp_prior_load` (or be NULL).
#[no_mangle]
pub unsafe extern "C" fn pp_prior_id(prior: *const pp_prior) -> u64 {
    prior.as_ref().map_or(0, |p| p.0.id())
}

/// # Safety
/// `prior` must come from `pp_prior_load` (or be NULL) and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn pp_prior_free(prior: *mut pp_prior) {
    if !prior.is_null() {
        drop(Box::from_raw(prior));
    }
}

/// # Safety
/// `buf` must hold a buffer produced by this library (or be zeroed).
#[no_mangle]
pub unsafe extern "C" fn pp_buffer_free(buf: *mut pp_buffer) {
    if let Some(b) = buf.as_mut() {
        if !b.data.is_null() {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(b.data, b.len + 1)));
        }
        b.data = std::ptr::null_mut();
        b.len = 0;
    }
}

#[no_mangle]
pub extern "C" fn pp_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn pp_version() -> *const c_char {
    static V: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    V.as_ptr() as *const c_char
}
