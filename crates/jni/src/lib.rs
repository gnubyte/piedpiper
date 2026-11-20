//! JNI bindings of PiedPiper for Android.
//!
//! The Java side is the Kotlin object `io.piedpiper.PiedPiper` in `bindings/android/piedpiper`;
//! its `private external` functions bind to the `Java_io_piedpiper_PiedPiper_*` exports below.
//!
//! * Every export runs inside [`std::panic::catch_unwind`], so a panic never unwinds into the
//!   JVM; it surfaces as `io.piedpiper.PiedPiperException` (kind `INTERNAL`).
//! * Core errors become `PiedPiperException(message, code)` (codes in [`code`]); a cancelled
//!   operation throws `io.piedpiper.PiedPiperCancelledException`
//!   (a `java.util.concurrent.CancellationException`).
//! * Progress: the codec calls its progress closure on its own worker threads (rayon). The
//!   closure only holds `Send + Sync` handles (a [`JavaVM`], a [`GlobalRef`] to the listener and
//!   its [`JMethodID`]); each call obtains a `JNIEnv` for the current thread — attaching it to
//!   the VM as a daemon thread the first time — and runs inside its own local-reference frame.
//!   An exception thrown by the listener cancels the operation and is re-thrown, unchanged, from
//!   the original call on the caller's thread.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard};

use jni::errors::Error as JniError;
use jni::objects::{
    AsJArrayRaw, GlobalRef, JByteArray, JClass, JFloatArray, JIntArray, JLongArray, JMethodID, JObject, JThrowable,
    JValue,
};
use jni::signature::{Primitive, ReturnType};
use jni::sys::{jboolean, jbyteArray, jint, jstring, jvalue, JNI_FALSE};
use jni::{JNIEnv, JavaVM};

use piedpiper::{CompressOptions, DecompressOptions, Error, Info, ModelConfig, Prior, TokenizerChoice, TrainConfig};

const EXCEPTION_CLASS: &str = "io/piedpiper/PiedPiperException";
const CANCELLED_CLASS: &str = "io/piedpiper/PiedPiperCancelledException";

/// Error codes carried by `PiedPiperException` (must match `PiedPiperException.Kind` in Kotlin).
pub mod code {
    pub const FORMAT: i32 = 1;
    pub const VERSION: i32 = 2;
    pub const CHECKSUM: i32 = 3;
    pub const OPTIONS: i32 = 4;
    /// Reported as `PiedPiperCancelledException`, never as a `PiedPiperException`.
    pub const CANCELLED: i32 = 5;
    pub const TOO_LARGE: i32 = 6;
    pub const PRIOR: i32 = 7;
    pub const INTERNAL: i32 = 8;
}

/// Number of entries of the `model` array: d_model, n_layer, n_head, d_ff, window, seg_len, n_streams.
const MODEL_FIELDS: usize = 7;
const MODEL_NAMES: [&str; MODEL_FIELDS] = ["dModel", "layers", "heads", "dFf", "window", "segLen", "streams"];
/// Float training fields, mask bits 0..=6: lr, lr_min, beta1, beta2, eps, clip, init_scale.
const TRAIN_F_FIELDS: usize = 7;
/// Integer training fields, mask bits 8..=10: warmup, decay, seed.
const TRAIN_I_FIELDS: usize = 3;
const TRAIN_I_SHIFT: u32 = 8;

/// Why a native call did not produce a value.
enum Fail {
    /// A Java exception is already pending; return to Java and let it propagate.
    Pending,
    /// Throw `PiedPiperException(message, code)`.
    Error(i32, String),
    /// Throw `PiedPiperCancelledException`.
    Cancelled,
    /// Re-throw an exception raised by the progress listener.
    Rethrow(GlobalRef),
}

impl From<JniError> for Fail {
    fn from(e: JniError) -> Self {
        match e {
            JniError::JavaException => Fail::Pending,
            e => Fail::Error(code::INTERNAL, format!("JNI error: {e}")),
        }
    }
}

impl From<Error> for Fail {
    fn from(e: Error) -> Self {
        let c = match &e {
            Error::Format(_) => code::FORMAT,
            Error::Version(_) => code::VERSION,
            Error::Checksum { .. } => code::CHECKSUM,
            Error::Options(_) => code::OPTIONS,
            Error::Cancelled => return Fail::Cancelled,
            Error::TooLarge(_) => code::TOO_LARGE,
            Error::Prior(_) => code::PRIOR,
        };
        Fail::Error(c, e.to_string())
    }
}

fn options_error(msg: impl Into<String>) -> Fail {
    Fail::Error(code::OPTIONS, format!("invalid options: {}", msg.into()))
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn panic_message(p: &(dyn Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Runs `f`, converting failures and panics into Java exceptions; returns `on_error` then.
fn guard<'local, T>(
    env: &mut JNIEnv<'local>,
    on_error: T,
    f: impl FnOnce(&mut JNIEnv<'local>) -> Result<T, Fail>,
) -> T {
    match catch_unwind(AssertUnwindSafe(|| f(env))) {
        Ok(Ok(v)) => v,
        Ok(Err(fail)) => {
            throw(env, fail);
            on_error
        }
        Err(panic) => {
            let msg = format!("internal error (panic): {}", panic_message(panic.as_ref()));
            throw(env, Fail::Error(code::INTERNAL, msg));
            on_error
        }
    }
}

fn throw(env: &mut JNIEnv, fail: Fail) {
    if env.exception_check().unwrap_or(false) {
        // Something more specific (e.g. OutOfMemoryError) is already on its way to Java.
        return;
    }
    let (res, fallback) = match fail {
        Fail::Pending => (Err(JniError::JavaException), "PiedPiper: JNI call failed".to_string()),
        Fail::Error(c, msg) => (throw_piedpiper(env, c, &msg), msg),
        Fail::Cancelled => (env.throw_new(CANCELLED_CLASS, "operation cancelled"), "operation cancelled".to_string()),
        Fail::Rethrow(t) => {
            let obj: &JThrowable = t.as_obj().into();
            (env.throw(obj), "progress listener failed".to_string())
        }
    };
    if res.is_err() {
        // Our exception classes could not be instantiated (e.g. stripped by a shrinker):
        // never return to Java silently.
        if env.exception_check().unwrap_or(false) {
            let _ = env.exception_clear();
        }
        let _ = env.throw_new("java/lang/RuntimeException", fallback);
    }
}

fn throw_piedpiper(env: &mut JNIEnv, c: i32, msg: &str) -> jni::errors::Result<()> {
    let class = env.find_class(EXCEPTION_CLASS)?;
    let jmsg = env.new_string(msg)?;
    let obj = env.new_object(&class, "(Ljava/lang/String;I)V", &[JValue::Object(&jmsg), JValue::Int(c)])?;
    env.throw(JThrowable::from(obj))
}

// ---------------------------------------------------------------------------------------------
// Progress listener bridge

/// A `ProgressListener` the codec can call from any thread.
struct Listener {
    vm: JavaVM,
    obj: GlobalRef,
    on_progress: JMethodID,
    /// Exception thrown by `onProgress`; re-thrown on the caller's thread.
    thrown: Mutex<Option<GlobalRef>>,
    /// JNI failure while calling back (thread attach, local frame, ...).
    failure: Mutex<Option<String>>,
}

impl Listener {
    /// Resolves the listener on the caller's thread (class lookups must happen here: a native
    /// thread attached later only sees the system class loader).
    fn new(env: &mut JNIEnv, listener: &JObject) -> Result<Option<Listener>, Fail> {
        if listener.is_null() {
            return Ok(None);
        }
        let vm = env.get_java_vm()?;
        let obj = env.new_global_ref(listener)?;
        let class = env.get_object_class(listener)?;
        let on_progress = env.get_method_id(&class, "onProgress", "(JJ)Z")?;
        env.delete_local_ref(class)?;
        Ok(Some(Listener { vm, obj, on_progress, thrown: Mutex::new(None), failure: Mutex::new(None) }))
    }

    /// Calls `onProgress(done, total)`; returns whether to keep going.
    fn report(&self, done: u64, total: u64) -> bool {
        // Codec workers are long-lived pool threads: attach each one once, as a daemon, and
        // keep it attached (it detaches automatically when the thread exits). Already-attached
        // threads (e.g. the caller's) are used as they are.
        let mut env = match self.vm.attach_current_thread_as_daemon() {
            Ok(env) => env,
            Err(e) => {
                *lock(&self.failure) = Some(format!("cannot attach codec thread to the JVM: {e}"));
                return false;
            }
        };
        let res = env.with_local_frame(8, |env| -> jni::errors::Result<bool> {
            let args = [jvalue { j: clamp_i64(done) }, jvalue { j: clamp_i64(total) }];
            // SAFETY: `on_progress` was resolved for this object's class with signature (JJ)Z,
            // and the arguments match that signature.
            let r = unsafe {
                env.call_method_unchecked(&self.obj, self.on_progress, ReturnType::Primitive(Primitive::Boolean), &args)
            };
            match r {
                Ok(v) => v.z(),
                Err(JniError::JavaException) => {
                    let ex = env.exception_occurred()?;
                    env.exception_clear()?;
                    let global = env.new_global_ref(&ex)?;
                    *lock(&self.thrown) = Some(global);
                    Ok(false)
                }
                Err(e) => Err(e),
            }
        });
        match res {
            Ok(keep_going) => keep_going,
            Err(e) => {
                // Never leave an exception pending on a codec worker thread.
                if env.exception_check().unwrap_or(false) {
                    let _ = env.exception_clear();
                }
                *lock(&self.failure) = Some(format!("progress callback failed: {e}"));
                false
            }
        }
    }

    /// The failure to raise instead of the codec's own result, if the listener failed.
    fn take_failure(&self) -> Option<Fail> {
        if let Some(t) = lock(&self.thrown).take() {
            return Some(Fail::Rethrow(t));
        }
        lock(&self.failure).take().map(|m| Fail::Error(code::INTERNAL, m))
    }
}

fn clamp_i64(v: u64) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// Runs a codec operation with an optional Java progress listener.
fn with_listener<T>(
    listener: Option<&Listener>,
    op: impl FnOnce(&mut piedpiper::ProgressFn<'_>) -> piedpiper::Result<T>,
) -> Result<T, Fail> {
    let mut cb = |done: u64, total: u64| match listener {
        Some(l) => l.report(done, total),
        None => true,
    };
    let result = op(&mut cb);
    if let Some(fail) = listener.and_then(Listener::take_failure) {
        return Err(fail);
    }
    Ok(result?)
}

// ---------------------------------------------------------------------------------------------
// Argument conversion

fn read_bytes(env: &JNIEnv, array: &JByteArray, what: &str) -> Result<Vec<u8>, Fail> {
    if array.is_null() {
        return Err(options_error(format!("{what} is null")));
    }
    Ok(env.convert_byte_array(array)?)
}

fn read_prior(env: &JNIEnv, array: &JByteArray) -> Result<Option<Arc<Prior>>, Fail> {
    if array.is_null() {
        return Ok(None);
    }
    let bytes = env.convert_byte_array(array)?;
    Ok(Some(Arc::new(Prior::from_bytes(&bytes)?)))
}

fn check_len<'a>(env: &JNIEnv, array: &impl AsJArrayRaw<'a>, want: usize, what: &str) -> Result<(), Fail> {
    if array.as_ref().is_null() {
        return Err(options_error(format!("{what} is null")));
    }
    let len = env.get_array_length(array)?;
    if len < 0 || len as usize != want {
        return Err(options_error(format!("{what} must have {want} entries, got {len}")));
    }
    Ok(())
}

fn read_ints<const N: usize>(env: &JNIEnv, array: &JIntArray, what: &str) -> Result<[i32; N], Fail> {
    check_len(env, array, N, what)?;
    let mut buf = [0i32; N];
    env.get_int_array_region(array, 0, &mut buf)?;
    Ok(buf)
}

fn read_floats<const N: usize>(env: &JNIEnv, array: &JFloatArray, what: &str) -> Result<[f32; N], Fail> {
    check_len(env, array, N, what)?;
    let mut buf = [0f32; N];
    env.get_float_array_region(array, 0, &mut buf)?;
    Ok(buf)
}

fn read_longs<const N: usize>(env: &JNIEnv, array: &JLongArray, what: &str) -> Result<[i64; N], Fail> {
    check_len(env, array, N, what)?;
    let mut buf = [0i64; N];
    env.get_long_array_region(array, 0, &mut buf)?;
    Ok(buf)
}

fn non_negative(v: i64, what: &str) -> Result<u32, Fail> {
    u32::try_from(v).map_err(|_| options_error(format!("{what} must be in 0..={}, got {v}", u32::MAX)))
}

fn tokenizer_choice(kind: i32, bpe_vocab: i32) -> Result<TokenizerChoice, Fail> {
    match kind {
        0 => Ok(TokenizerChoice::Auto),
        1 => Ok(TokenizerChoice::Bytes),
        2 => Ok(TokenizerChoice::Bpe { vocab: non_negative(bpe_vocab.into(), "BPE vocabulary")? }),
        k => Err(options_error(format!("unknown tokenizer code {k}"))),
    }
}

/// Advanced overrides of the level's network / training recipe, as passed from Kotlin.
#[derive(Clone, Copy, Default)]
struct Overrides {
    /// Bit `i` set: `model[i]` replaces model field `i` (see [`MODEL_NAMES`]).
    model_mask: u32,
    model: [i32; MODEL_FIELDS],
    /// Bits 0..=6: `train_f[i]`; bits 8..=10: `train_i[i - 8]`.
    train_mask: u32,
    train_f: [f32; TRAIN_F_FIELDS],
    train_i: [i64; TRAIN_I_FIELDS],
}

/// Builds the compression options exactly like the desktop CLI's `build_options`: with any
/// override, the network and training recipe start from `preset(level)`; a model override
/// pins the whole network (used verbatim, including `seg_len` and `n_streams`), and any
/// override at all pins the training recipe. Identical overrides therefore produce identical
/// bytes on every platform.
fn build_compress_options(
    level: u32,
    tokenizer: TokenizerChoice,
    threads: usize,
    allow_stored: bool,
    builtin_prior: bool,
    prior: Option<Arc<Prior>>,
    ov: &Overrides,
) -> Result<CompressOptions, Fail> {
    let mut o = CompressOptions::level(level);
    o.tokenizer = tokenizer;
    o.threads = threads;
    o.allow_stored = allow_stored;
    o.builtin_prior = builtin_prior;
    o.prior = prior;
    if ov.model_mask >> MODEL_FIELDS != 0 {
        return Err(options_error(format!("bad model override mask {:#x}", ov.model_mask)));
    }
    let train_valid = ((1u32 << TRAIN_F_FIELDS) - 1) | (((1u32 << TRAIN_I_FIELDS) - 1) << TRAIN_I_SHIFT);
    if ov.train_mask & !train_valid != 0 {
        return Err(options_error(format!("bad training override mask {:#x}", ov.train_mask)));
    }
    if ov.model_mask == 0 && ov.train_mask == 0 {
        return Ok(o);
    }
    let pre = piedpiper::preset(level);
    let mut mc: ModelConfig = pre.model;
    let mut tc: TrainConfig = pre.train;
    for (i, name) in MODEL_NAMES.iter().enumerate() {
        if ov.model_mask & (1 << i) == 0 {
            continue;
        }
        let v = non_negative(ov.model[i].into(), name)?;
        match i {
            0 => mc.d_model = v,
            1 => mc.n_layer = v,
            2 => mc.n_head = v,
            3 => mc.d_ff = v,
            4 => mc.window = v,
            5 => mc.seg_len = v,
            _ => mc.n_streams = v,
        }
    }
    for i in 0..TRAIN_F_FIELDS {
        if ov.train_mask & (1 << i) == 0 {
            continue;
        }
        let v = ov.train_f[i];
        match i {
            0 => tc.lr = v,
            1 => tc.lr_min = v,
            2 => tc.beta1 = v,
            3 => tc.beta2 = v,
            4 => tc.eps = v,
            5 => tc.clip = v,
            _ => tc.init_scale = v,
        }
    }
    for i in 0..TRAIN_I_FIELDS {
        if ov.train_mask & (1 << (TRAIN_I_SHIFT + i as u32)) == 0 {
            continue;
        }
        let v = ov.train_i[i];
        match i {
            0 => tc.warmup = non_negative(v, "warmup")?,
            1 => tc.decay = non_negative(v, "decay")?,
            // Any 64-bit pattern is a valid seed (Kotlin has no unsigned Long in this API).
            _ => tc.seed = v as u64,
        }
    }
    if ov.model_mask != 0 {
        o.model = Some(mc);
    }
    o.train = Some(tc);
    Ok(o)
}

// ---------------------------------------------------------------------------------------------
// JSON (header info / presets). Floats are written as the shortest decimal of their exact f64
// value, so `toDouble().toFloat()` on the Java side reproduces the f32 bit pattern exactly.

fn json_f32(x: f32) -> String {
    if x.is_finite() {
        format!("{:e}", x as f64)
    } else {
        "null".to_string()
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn model_json(m: &ModelConfig) -> String {
    format!(
        "{{\"d_model\":{},\"n_layer\":{},\"n_head\":{},\"d_ff\":{},\"window\":{},\"seg_len\":{},\"n_streams\":{}}}",
        m.d_model, m.n_layer, m.n_head, m.d_ff, m.window, m.seg_len, m.n_streams
    )
}

fn train_json(t: &TrainConfig) -> String {
    format!(
        "{{\"lr\":{},\"lr_min\":{},\"warmup\":{},\"decay\":{},\"beta1\":{},\"beta2\":{},\"eps\":{},\"clip\":{},\"init_scale\":{},\"seed\":{}}}",
        json_f32(t.lr),
        json_f32(t.lr_min),
        t.warmup,
        t.decay,
        json_f32(t.beta1),
        json_f32(t.beta2),
        json_f32(t.eps),
        json_f32(t.clip),
        json_f32(t.init_scale),
        t.seed as i64
    )
}

fn info_json(i: &Info) -> String {
    format!(
        "{{\"original_size\":{},\"compressed_size\":{},\"crc32\":{},\"stored\":{},\"tokenizer\":{},\"vocab_size\":{},\"n_tokens\":{},\"n_params\":{},\"prior_id\":\"{:016x}\",\"prior_name\":{},\"model\":{},\"train\":{}}}",
        i.original_size,
        i.compressed_size,
        i.crc32,
        i.stored,
        json_str(&i.tokenizer),
        i.vocab_size,
        i.n_tokens,
        i.n_params,
        i.prior_id,
        i.prior_name.as_deref().map_or("null".to_string(), json_str),
        i.model.as_ref().map_or("null".to_string(), model_json),
        i.train.as_ref().map_or("null".to_string(), train_json)
    )
}

fn preset_json(level: u32) -> String {
    let p = piedpiper::preset(level);
    format!(
        "{{\"level\":{},\"model\":{},\"train\":{},\"min_stream_tokens\":{},\"max_vocab\":{},\"prior_lr\":{}}}",
        level.clamp(piedpiper::MIN_LEVEL, piedpiper::MAX_LEVEL),
        model_json(&p.model),
        train_json(&p.train),
        p.min_stream_tokens,
        p.max_vocab,
        json_f32(p.prior_lr)
    )
}

// ---------------------------------------------------------------------------------------------
// Exports

/// `String nativeVersion()`: library version.
#[no_mangle]
pub extern "system" fn Java_io_piedpiper_PiedPiper_nativeVersion<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
) -> jstring {
    guard(&mut env, ptr::null_mut(), |env| Ok(env.new_string(piedpiper::VERSION)?.into_raw()))
}

/// `int nativeFormatVersion()`: version of the stream format this build writes.
#[no_mangle]
pub extern "system" fn Java_io_piedpiper_PiedPiper_nativeFormatVersion<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
) -> jint {
    guard(&mut env, 0, |_| Ok(jint::from(piedpiper::FORMAT_VERSION)))
}

/// `String nativePreset(int level)`: the level's network and training recipe as JSON.
#[no_mangle]
pub extern "system" fn Java_io_piedpiper_PiedPiper_nativePreset<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    level: jint,
) -> jstring {
    guard(&mut env, ptr::null_mut(), |env| {
        let level = u32::try_from(level)
            .ok()
            .filter(|l| (piedpiper::MIN_LEVEL..=piedpiper::MAX_LEVEL).contains(l))
            .ok_or_else(|| {
                options_error(format!("level must be in {}..={}", piedpiper::MIN_LEVEL, piedpiper::MAX_LEVEL))
            })?;
        Ok(env.new_string(preset_json(level))?.into_raw())
    })
}

/// `String nativeInfo(byte[] data)`: header of a compressed stream as JSON.
#[no_mangle]
pub extern "system" fn Java_io_piedpiper_PiedPiper_nativeInfo<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    input: JByteArray<'local>,
) -> jstring {
    guard(&mut env, ptr::null_mut(), |env| {
        let data = read_bytes(env, &input, "input")?;
        let info = piedpiper::inspect(&data)?;
        Ok(env.new_string(info_json(&info))?.into_raw())
    })
}

/// `byte[] nativeCompress(byte[] input, int level, int tokenizer, int bpeVocab, int threads,
/// boolean allowStored, boolean builtinPrior, byte[] prior, int modelMask, int[] model, int trainMask, float[] trainF,
/// long[] trainI, ProgressListener progress)`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_io_piedpiper_PiedPiper_nativeCompress<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    input: JByteArray<'local>,
    level: jint,
    tokenizer: jint,
    bpe_vocab: jint,
    threads: jint,
    allow_stored: jboolean,
    builtin_prior: jboolean,
    prior: JByteArray<'local>,
    model_mask: jint,
    model: JIntArray<'local>,
    train_mask: jint,
    train_f: JFloatArray<'local>,
    train_i: JLongArray<'local>,
    progress: JObject<'local>,
) -> jbyteArray {
    guard(&mut env, ptr::null_mut(), |env| {
        let data = read_bytes(env, &input, "input")?;
        // Out-of-range levels are rejected by the codec with its own message.
        let level = u32::try_from(level).unwrap_or(0);
        let tokenizer = tokenizer_choice(tokenizer, bpe_vocab)?;
        let threads = non_negative(threads.into(), "threads")? as usize;
        let prior = read_prior(env, &prior)?;
        let mut ov = Overrides { model_mask: model_mask as u32, train_mask: train_mask as u32, ..Default::default() };
        if ov.model_mask != 0 {
            ov.model = read_ints(env, &model, "model overrides")?;
        }
        if ov.train_mask & ((1 << TRAIN_F_FIELDS) - 1) != 0 {
            ov.train_f = read_floats(env, &train_f, "float training overrides")?;
        }
        if ov.train_mask >> TRAIN_I_SHIFT != 0 {
            ov.train_i = read_longs(env, &train_i, "integer training overrides")?;
        }
        let opts =
            build_compress_options(level, tokenizer, threads, allow_stored != JNI_FALSE, builtin_prior != JNI_FALSE, prior, &ov)?;
        let listener = Listener::new(env, &progress)?;
        let out = with_listener(listener.as_ref(), |cb| piedpiper::compress_with_progress(&data, &opts, cb))?;
        drop(data);
        Ok(env.byte_array_from_slice(&out)?.into_raw())
    })
}

/// `byte[] nativeDecompress(byte[] input, int threads, byte[] prior, ProgressListener progress)`.
#[no_mangle]
pub extern "system" fn Java_io_piedpiper_PiedPiper_nativeDecompress<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    input: JByteArray<'local>,
    threads: jint,
    prior: JByteArray<'local>,
    progress: JObject<'local>,
) -> jbyteArray {
    guard(&mut env, ptr::null_mut(), |env| {
        let data = read_bytes(env, &input, "input")?;
        let threads = non_negative(threads.into(), "threads")? as usize;
        let opts = DecompressOptions { threads, prior: read_prior(env, &prior)?, ..Default::default() };
        let listener = Listener::new(env, &progress)?;
        let out = with_listener(listener.as_ref(), |cb| piedpiper::decompress_with_progress(&data, &opts, cb))?;
        drop(data);
        Ok(env.byte_array_from_slice(&out)?.into_raw())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(level: u32, ov: &Overrides) -> CompressOptions {
        match build_compress_options(level, TokenizerChoice::Bytes, 0, true, true, None, ov) {
            Ok(o) => o,
            Err(Fail::Error(_, m)) => panic!("{m}"),
            Err(_) => panic!("unexpected failure"),
        }
    }

    #[test]
    fn no_overrides_leave_the_preset_path() {
        let o = opts(5, &Overrides::default());
        assert!(o.model.is_none() && o.train.is_none());
        assert_eq!(o.level, 5);
    }

    #[test]
    fn overrides_match_the_cli() {
        // --d-model 32 --layers 2 --heads 2 --d-ff 64 --window 32 --seg 4 --streams 3 --lr 3e-3
        let ov = Overrides {
            model_mask: 0x7f,
            model: [32, 2, 2, 64, 32, 4, 3],
            train_mask: 1,
            train_f: [3e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let o = opts(5, &ov);
        let m = o.model.unwrap();
        assert_eq!(
            (m.d_model, m.n_layer, m.n_head, m.d_ff, m.window, m.seg_len, m.n_streams),
            (32, 2, 2, 64, 32, 4, 3)
        );
        let mut want = piedpiper::preset(5).train;
        want.lr = "3e-3".parse::<f32>().unwrap();
        assert_eq!(o.train.unwrap(), want);

        // Training-only overrides pin the recipe but keep the automatic network policy.
        let ov = Overrides {
            train_mask: (1 << 2) | (1 << 5),
            train_f: [0.0, 0.0, 0.9, 0.0, 0.0, 1.0, 0.0],
            ..Default::default()
        };
        let o = opts(3, &ov);
        assert!(o.model.is_none());
        let t = o.train.unwrap();
        assert_eq!((t.beta1, t.clip, t.lr), (0.9, 1.0, piedpiper::preset(3).train.lr));

        // Integer fields, including a seed with the top bit set.
        let ov = Overrides { train_mask: 0b111 << TRAIN_I_SHIFT, train_i: [10, 20, -1], ..Default::default() };
        let t = opts(1, &ov).train.unwrap();
        assert_eq!((t.warmup, t.decay, t.seed), (10, 20, u64::MAX));
    }

    #[test]
    fn bad_overrides_are_rejected() {
        let ov = Overrides { model_mask: 1, model: [-4, 0, 0, 0, 0, 0, 0], ..Default::default() };
        assert!(matches!(
            build_compress_options(5, TokenizerChoice::Auto, 0, true, true, None, &ov),
            Err(Fail::Error(code::OPTIONS, _))
        ));
        let ov = Overrides { model_mask: 1 << 7, ..Default::default() };
        assert!(build_compress_options(5, TokenizerChoice::Auto, 0, true, true, None, &ov).is_err());
        let ov = Overrides { train_mask: 1 << 7, ..Default::default() };
        assert!(build_compress_options(5, TokenizerChoice::Auto, 0, true, true, None, &ov).is_err());
        assert!(tokenizer_choice(3, 0).is_err());
        assert!(tokenizer_choice(2, -1).is_err());
        assert!(matches!(tokenizer_choice(2, 600), Ok(TokenizerChoice::Bpe { vocab: 600 })));
    }

    #[test]
    fn json_floats_round_trip_exactly() {
        for x in [3e-3f32, 1e-3, 0.9999, 1e-8, 0.0, 1.0, 2.5e-4, f32::MIN_POSITIVE, 123456.78] {
            let s = json_f32(x);
            let back = s.parse::<f64>().unwrap() as f32;
            assert_eq!(back.to_bits(), x.to_bits(), "{x} -> {s}");
        }
        assert_eq!(json_f32(f32::NAN), "null");
    }

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(json_str("bpe:600"), "\"bpe:600\"");
        assert_eq!(json_str("a\"b\\c\n\u{1}"), "\"a\\\"b\\\\c\\n\\u0001\"");
    }

    #[test]
    fn info_and_preset_json_shape() {
        let data = b"hello hello hello hello, PiedPiper! ".repeat(20);
        let packed =
            piedpiper::compress(&data, &CompressOptions { allow_stored: false, builtin_prior: false, ..CompressOptions::level(1) })
                .unwrap();
        let j = info_json(&piedpiper::inspect(&packed).unwrap());
        assert!(j.starts_with("{\"original_size\":720,"), "{j}");
        assert!(j.contains("\"tokenizer\":\"bytes\""), "{j}");
        assert!(j.contains("\"prior_id\":\"0000000000000000\""), "{j}");
        assert!(j.contains("\"model\":{\"d_model\":64,"), "{j}");
        let p = preset_json(5);
        assert!(p.starts_with(&format!("{{\"level\":5,\"model\":{{\"d_model\":{},", piedpiper::preset(5).model.d_model)), "{p}");
        assert!(p.contains(&format!("\"seed\":{}", piedpiper::preset(5).train.seed as i64)), "{p}");
    }
}
