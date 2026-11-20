//! `piedpiper._native` — PyO3 bindings. The pythonic API lives in `python/piedpiper`.
//!
//! The GIL is released while the codec runs; progress callbacks re-acquire it at most every
//! 100 ms, which is also when Ctrl+C (KeyboardInterrupt) is noticed and cancels the job.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::sync::Arc;
use std::time::{Duration, Instant};

pyo3::create_exception!(_native, PiedPiperError, pyo3::exceptions::PyException, "Codec error (corrupt data, bad options, missing prior, ...).");
pyo3::create_exception!(_native, CancelledError, PiedPiperError, "The progress callback returned False.");

fn to_py_err(e: piedpiper::Error) -> PyErr {
    match e {
        piedpiper::Error::Cancelled => CancelledError::new_err("operation cancelled"),
        other => PiedPiperError::new_err(other.to_string()),
    }
}

/// A pretrained prior model (initial weights + network shape + tokenizer).
#[pyclass(frozen, module = "piedpiper._native", name = "Prior")]
struct PyPrior {
    inner: Arc<piedpiper::Prior>,
}

#[pymethods]
impl PyPrior {
    #[new]
    fn new(data: &[u8]) -> PyResult<Self> {
        piedpiper::Prior::from_bytes(data).map(|p| PyPrior { inner: Arc::new(p) }).map_err(to_py_err)
    }

    /// Loads a prior file from disk.
    #[staticmethod]
    fn from_file(path: std::path::PathBuf) -> PyResult<Self> {
        let data = std::fs::read(&path)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {e}", path.display())))?;
        Self::new(&data)
    }

    /// 64-bit id recorded in streams compressed with this prior.
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id()
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    #[getter]
    fn tokenizer(&self) -> String {
        self.inner.tokenizer_name()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    #[getter]
    fn file_size(&self) -> usize {
        self.inner.file_size()
    }

    /// Network shape as a dict.
    #[getter]
    fn model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = self.inner.model_config();
        let d = PyDict::new(py);
        d.set_item("d_model", m.d_model)?;
        d.set_item("n_layer", m.n_layer)?;
        d.set_item("n_head", m.n_head)?;
        d.set_item("d_ff", m.d_ff)?;
        d.set_item("window", m.window)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let m = self.inner.model_config();
        format!(
            "Prior(id={:016x}, tokenizer={}, layers={}, d_model={}, params={})",
            self.inner.id(),
            self.inner.tokenizer_name(),
            m.n_layer,
            m.d_model,
            self.inner.n_params()
        )
    }
}

/// Runs `job` with the GIL released, forwarding throttled progress to `cb`.
fn run_job<T: Send>(
    py: Python<'_>,
    cb: Option<Bound<'_, PyAny>>,
    job: impl FnOnce(&mut piedpiper::ProgressFn<'_>) -> piedpiper::Result<T> + Send,
) -> PyResult<T> {
    let cb: Option<Py<PyAny>> = cb.map(|c| c.unbind());
    let mut err: Option<PyErr> = None;
    let res = py.detach(|| {
        let mut last: Option<Instant> = None;
        let mut prog = |d: u64, t: u64| -> bool {
            let now = Instant::now();
            if d < t {
                if let Some(l) = last {
                    if now.duration_since(l) < Duration::from_millis(100) {
                        return true;
                    }
                }
            }
            last = Some(now);
            Python::attach(|py| {
                if let Err(e) = py.check_signals() {
                    err = Some(e);
                    return false;
                }
                match &cb {
                    None => true,
                    Some(cb) => match cb.call1(py, (d, t)) {
                        Ok(v) => !matches!(v.extract::<bool>(py), Ok(false)),
                        Err(e) => {
                            err = Some(e);
                            false
                        }
                    },
                }
            })
        };
        job(&mut prog)
    });
    match res {
        Ok(v) => Ok(v),
        Err(piedpiper::Error::Cancelled) => Err(err.take().unwrap_or_else(|| CancelledError::new_err("operation cancelled"))),
        Err(e) => Err(to_py_err(e)),
    }
}

fn item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(d.get_item(key)?.filter(|v| !v.is_none()))
}

fn bad_type(key: &str, e: impl Into<PyErr>) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("{key}: {}", e.into()))
}

fn get_u32(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u32>> {
    item(d, key)?.map(|v| v.extract::<u32>().map_err(|e| bad_type(key, e))).transpose()
}

fn get_u64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u64>> {
    item(d, key)?.map(|v| v.extract::<u64>().map_err(|e| bad_type(key, e))).transpose()
}

fn get_f32(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f32>> {
    item(d, key)?.map(|v| v.extract::<f32>().map_err(|e| bad_type(key, e))).transpose()
}

fn model_from_dict(d: &Bound<'_, PyDict>) -> PyResult<piedpiper::ModelConfig> {
    let req = |k: &str| -> PyResult<u32> {
        get_u32(d, k)?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("model needs '{k}'")))
    };
    Ok(piedpiper::ModelConfig {
        d_model: req("d_model")?,
        n_layer: req("n_layer")?,
        n_head: req("n_head")?,
        d_ff: req("d_ff")?,
        window: req("window")?,
        seg_len: req("seg_len")?,
        n_streams: req("n_streams")?,
    })
}

fn train_from_dict(level: u32, d: &Bound<'_, PyDict>) -> PyResult<piedpiper::TrainConfig> {
    let mut t = piedpiper::preset(level).train;
    if let Some(v) = get_f32(d, "lr")? {
        t.lr = v;
    }
    if let Some(v) = get_f32(d, "lr_min")? {
        t.lr_min = v;
    }
    if let Some(v) = get_u32(d, "warmup")? {
        t.warmup = v;
    }
    if let Some(v) = get_u32(d, "decay")? {
        t.decay = v;
    }
    if let Some(v) = get_f32(d, "beta1")? {
        t.beta1 = v;
    }
    if let Some(v) = get_f32(d, "beta2")? {
        t.beta2 = v;
    }
    if let Some(v) = get_f32(d, "eps")? {
        t.eps = v;
    }
    if let Some(v) = get_f32(d, "clip")? {
        t.clip = v;
    }
    if let Some(v) = get_f32(d, "init_scale")? {
        t.init_scale = v;
    }
    if let Some(v) = get_u64(d, "seed")? {
        t.seed = v;
    }
    Ok(t)
}

fn tokenizer_choice(tokenizer: &str, bpe_vocab: u32) -> PyResult<piedpiper::TokenizerChoice> {
    match tokenizer {
        "auto" => Ok(piedpiper::TokenizerChoice::Auto),
        "bytes" => Ok(piedpiper::TokenizerChoice::Bytes),
        "bpe" => Ok(piedpiper::TokenizerChoice::Bpe { vocab: bpe_vocab }),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tokenizer must be 'auto', 'bytes' or 'bpe', not {other:?}"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (data, level=2, tokenizer="auto", bpe_vocab=4096, threads=0, prior=None, progress=None, allow_stored=true, model=None, train=None, builtin_prior=true))]
#[allow(clippy::too_many_arguments)]
fn compress<'py>(
    py: Python<'py>,
    data: &[u8],
    level: u32,
    tokenizer: &str,
    bpe_vocab: u32,
    threads: usize,
    prior: Option<PyRef<'py, PyPrior>>,
    progress: Option<Bound<'py, PyAny>>,
    allow_stored: bool,
    model: Option<Bound<'py, PyDict>>,
    train: Option<Bound<'py, PyDict>>,
    builtin_prior: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut o = piedpiper::CompressOptions::level(level);
    o.builtin_prior = builtin_prior;
    o.tokenizer = tokenizer_choice(tokenizer, bpe_vocab)?;
    o.threads = threads;
    o.allow_stored = allow_stored;
    o.prior = prior.map(|p| p.inner.clone());
    if let Some(m) = &model {
        o.model = Some(model_from_dict(m)?);
    }
    if let Some(t) = &train {
        o.train = Some(train_from_dict(level, t)?);
    }
    let out = run_job(py, progress, |p| piedpiper::compress_with_progress(data, &o, p))?;
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
#[pyo3(signature = (data, threads=0, prior=None, progress=None))]
fn decompress<'py>(
    py: Python<'py>,
    data: &[u8],
    threads: usize,
    prior: Option<PyRef<'py, PyPrior>>,
    progress: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let o = piedpiper::DecompressOptions { threads, prior: prior.map(|p| p.inner.clone()) };
    let out = run_job(py, progress, |p| piedpiper::decompress_with_progress(data, &o, p))?;
    Ok(PyBytes::new(py, &out))
}

#[pyfunction]
fn info<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let i = piedpiper::inspect(data).map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("original_size", i.original_size)?;
    d.set_item("compressed_size", i.compressed_size)?;
    d.set_item("crc32", i.crc32)?;
    d.set_item("stored", i.stored)?;
    d.set_item("tokenizer", i.tokenizer)?;
    d.set_item("vocab_size", i.vocab_size)?;
    d.set_item("tokens", i.n_tokens)?;
    d.set_item("params", i.n_params)?;
    d.set_item("prior_id", i.prior_id)?;
    d.set_item("prior_name", i.prior_name)?;
    if let Some(m) = i.model {
        let md = PyDict::new(py);
        md.set_item("d_model", m.d_model)?;
        md.set_item("n_layer", m.n_layer)?;
        md.set_item("n_head", m.n_head)?;
        md.set_item("d_ff", m.d_ff)?;
        md.set_item("window", m.window)?;
        md.set_item("seg_len", m.seg_len)?;
        md.set_item("n_streams", m.n_streams)?;
        d.set_item("model", md)?;
    }
    if let Some(t) = i.train {
        let td = PyDict::new(py);
        td.set_item("lr", t.lr)?;
        td.set_item("lr_min", t.lr_min)?;
        td.set_item("warmup", t.warmup)?;
        td.set_item("decay", t.decay)?;
        td.set_item("beta1", t.beta1)?;
        td.set_item("beta2", t.beta2)?;
        td.set_item("eps", t.eps)?;
        td.set_item("clip", t.clip)?;
        td.set_item("init_scale", t.init_scale)?;
        td.set_item("seed", t.seed)?;
        d.set_item("train", td)?;
    }
    Ok(d)
}

/// Trains a prior; returns `(prior_file_bytes, bits_per_byte_per_epoch)`.
#[pyfunction]
#[pyo3(signature = (corpus, level=2, epochs=1, tokenizer="bytes", bpe_vocab=4096, threads=0, init=None, model=None, train=None, progress=None))]
#[allow(clippy::too_many_arguments)]
fn train_prior<'py>(
    py: Python<'py>,
    corpus: &[u8],
    level: u32,
    epochs: u32,
    tokenizer: &str,
    bpe_vocab: u32,
    threads: usize,
    init: Option<PyRef<'py, PyPrior>>,
    model: Option<Bound<'py, PyDict>>,
    train: Option<Bound<'py, PyDict>>,
    progress: Option<Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyBytes>, Vec<f64>)> {
    let mut o = piedpiper::TrainOptions {
        level,
        tokenizer: tokenizer_choice(tokenizer, bpe_vocab)?,
        epochs,
        threads,
        init: init.map(|p| p.inner.clone()),
        ..Default::default()
    };
    if let Some(m) = &model {
        o.model = Some(model_from_dict(m)?);
    }
    if let Some(t) = &train {
        o.train = Some(train_from_dict(level, t)?);
    }
    let rep = run_job(py, progress, |p| piedpiper::train_prior(corpus, &o, p))?;
    Ok((PyBytes::new(py, &rep.prior), rep.epoch_bpb))
}

/// Preset (network shape, training recipe) for a level, as dicts.
#[pyfunction]
fn preset<'py>(py: Python<'py>, level: u32) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let p = piedpiper::preset(level);
    let m = PyDict::new(py);
    m.set_item("d_model", p.model.d_model)?;
    m.set_item("n_layer", p.model.n_layer)?;
    m.set_item("n_head", p.model.n_head)?;
    m.set_item("d_ff", p.model.d_ff)?;
    m.set_item("window", p.model.window)?;
    m.set_item("seg_len", p.model.seg_len)?;
    m.set_item("n_streams", p.model.n_streams)?;
    let t = PyDict::new(py);
    t.set_item("lr", p.train.lr)?;
    t.set_item("lr_min", p.train.lr_min)?;
    t.set_item("warmup", p.train.warmup)?;
    t.set_item("decay", p.train.decay)?;
    t.set_item("beta1", p.train.beta1)?;
    t.set_item("beta2", p.train.beta2)?;
    t.set_item("eps", p.train.eps)?;
    t.set_item("clip", p.train.clip)?;
    t.set_item("init_scale", p.train.init_scale)?;
    t.set_item("seed", p.train.seed)?;
    Ok((m, t))
}

#[pyfunction]
fn crc32(data: &[u8]) -> u32 {
    piedpiper::crc32(data)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", piedpiper::VERSION)?;
    m.add("FORMAT_VERSION", piedpiper::FORMAT_VERSION)?;
    m.add("MIN_LEVEL", piedpiper::MIN_LEVEL)?;
    m.add("MAX_LEVEL", piedpiper::MAX_LEVEL)?;
    m.add("DEFAULT_LEVEL", piedpiper::DEFAULT_LEVEL)?;
    m.add("PiedPiperError", m.py().get_type::<PiedPiperError>())?;
    m.add("CancelledError", m.py().get_type::<CancelledError>())?;
    m.add_class::<PyPrior>()?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add_function(wrap_pyfunction!(info, m)?)?;
    m.add_function(wrap_pyfunction!(train_prior, m)?)?;
    m.add_function(wrap_pyfunction!(preset, m)?)?;
    m.add_function(wrap_pyfunction!(crc32, m)?)?;
    Ok(())
}
