// PiedPiper for JavaScript — neural lossless compression (the TufaZip method) in
// WebAssembly. Output is bit-identical to the native CLI, Python and Android builds.
//
//   import { init, compress, decompress } from 'piedpiper';
//   await init();
//   const packed = compress(new TextEncoder().encode('hello'), { level: 3 });
//   const plain = decompress(packed);
//
// Compression is CPU-heavy (tens of KB/s). In browsers use `createWorker()` to keep the
// page responsive.

const OPT_LEN = 26;
const TOKENIZERS = { auto: 0, bytes: 1, bpe: 2 };

let wasm = null;
let initPromise = null;
let progressCb = null;

const isNode =
  typeof process !== 'undefined' && process.versions != null && process.versions.node != null;

/** Error raised by the codec (corrupt data, bad options, missing prior, …). */
export class PiedPiperError extends Error {
  constructor(message) {
    super(message);
    this.name = 'PiedPiperError';
  }
}

/** Raised when an `onProgress` callback returned `false`. */
export class CancelledError extends PiedPiperError {
  constructor(message = 'operation cancelled') {
    super(message);
    this.name = 'CancelledError';
  }
}

function imports() {
  return {
    env: {
      pp_progress(done, total) {
        if (!progressCb) return 1;
        try {
          return progressCb(done, total) === false ? 0 : 1;
        } catch {
          return 0;
        }
      },
    },
  };
}

async function instantiate(source) {
  if (source instanceof WebAssembly.Module) {
    return (await WebAssembly.instantiate(source, imports())).exports;
  }
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
    return (await WebAssembly.instantiate(source, imports())).instance.exports;
  }
  const url = source == null ? new URL('./piedpiper.wasm', import.meta.url) : new URL(String(source), import.meta.url);
  if (isNode && url.protocol === 'file:') {
    const { readFile } = await import(/* @vite-ignore */ 'node:fs/promises');
    return (await WebAssembly.instantiate(await readFile(url), imports())).instance.exports;
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new PiedPiperError(`cannot load ${url}: HTTP ${resp.status}`);
  if (typeof WebAssembly.instantiateStreaming === 'function' &&
      (resp.headers.get('content-type') || '').includes('application/wasm')) {
    return (await WebAssembly.instantiateStreaming(resp, imports())).instance.exports;
  }
  return (await WebAssembly.instantiate(await resp.arrayBuffer(), imports())).instance.exports;
}

/**
 * Loads the WebAssembly module. `source` may be a URL/path, the wasm bytes, or a compiled
 * `WebAssembly.Module`; by default `piedpiper.wasm` next to this file is used.
 */
export async function init(source) {
  if (wasm) return;
  if (!initPromise) {
    initPromise = instantiate(source).then((e) => {
      wasm = e;
    }, (err) => {
      initPromise = null;
      throw err;
    });
  }
  await initPromise;
}

/** Synchronous initialisation from wasm bytes or a compiled module. */
export function initSync(source) {
  if (wasm) return;
  const mod = source instanceof WebAssembly.Module ? source : new WebAssembly.Module(source);
  wasm = new WebAssembly.Instance(mod, imports()).exports;
}

function ready() {
  if (!wasm) throw new PiedPiperError('call `await init()` before using piedpiper');
  return wasm;
}

function toBytes(data, what = 'data') {
  if (data instanceof Uint8Array) return data;
  if (typeof data === 'string') return new TextEncoder().encode(data);
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new TypeError(`${what} must be a Uint8Array, ArrayBuffer, typed array or string`);
}

function put(w, bytes) {
  const p = w.pp_alloc(bytes.length);
  new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
  return p;
}

function take(w, status) {
  const p = w.pp_result_ptr();
  const n = w.pp_result_len();
  const out = new Uint8Array(w.memory.buffer, p, n).slice();
  w.pp_result_free();
  if (status === 0) return out;
  const msg = new TextDecoder().decode(out);
  throw status === 2 ? new CancelledError(msg) : new PiedPiperError(msg);
}

function withBuffers(w, buffers, fn) {
  const ptrs = buffers.map((b) => (b && b.length ? put(w, b) : 0));
  try {
    return fn(ptrs);
  } finally {
    buffers.forEach((b, i) => {
      if (ptrs[i]) w.pp_free(ptrs[i], b.length);
    });
  }
}

function presetTrain(w, level) {
  const raw = take(w, w.pp_preset_train(level));
  return new Float64Array(raw.buffer, raw.byteOffset, 11);
}

function encodeOptions(w, opts) {
  const level = opts.level ?? 2;
  const o = new Float64Array(OPT_LEN);
  o[0] = level;
  const tok = opts.tokenizer ?? 'auto';
  if (!(tok in TOKENIZERS)) throw new PiedPiperError(`unknown tokenizer ${tok}`);
  o[1] = TOKENIZERS[tok];
  o[2] = opts.bpeVocab ?? 4096;
  o[3] = opts.allowStored === false ? 0 : 1;
  if (opts.model) {
    const m = opts.model;
    o[4] = 1;
    o.set([m.dModel, m.layers, m.heads, m.dFf, m.window, m.segLen, m.streams].map(Number), 5);
  }
  if (opts.train) {
    const t = opts.train;
    const base = presetTrain(w, level);
    const seed = t.seed != null ? BigInt(t.seed) : BigInt(base[9]) | (BigInt(base[10]) << 32n);
    o[12] = 1;
    o.set([
      t.lr ?? base[0], t.lrMin ?? base[1], t.warmup ?? base[2], t.decay ?? base[3],
      t.beta1 ?? base[4], t.beta2 ?? base[5], t.eps ?? base[6], t.clip ?? base[7],
      t.initScale ?? base[8], Number(seed & 0xffffffffn), Number(seed >> 32n),
    ].map(Number), 13);
  }
  o[24] = opts.onProgress ? 1 : 0;
  o[25] = opts.builtinPrior === false ? 0 : 1;
  return new Uint8Array(o.buffer);
}

/**
 * Compresses `data`. Options: `level` (1–9, default 2), `tokenizer` ('auto' | 'bytes' |
 * 'bpe'), `bpeVocab`, `prior` (Uint8Array prior file), `onProgress(done, total)` (return
 * `false` to cancel), `allowStored`, and advanced `model` / `train` overrides.
 * @returns {Uint8Array}
 */
export function compress(data, opts = {}) {
  const w = ready();
  const input = toBytes(data);
  const optBytes = encodeOptions(w, opts);
  const prior = opts.prior ? toBytes(opts.prior, 'prior') : null;
  progressCb = opts.onProgress || null;
  try {
    return withBuffers(w, [input, optBytes, prior], ([ip, op, pp]) =>
      take(w, w.pp_compress(ip, input.length, op, OPT_LEN, pp, prior ? prior.length : 0)));
  } finally {
    progressCb = null;
  }
}

/**
 * Decompresses a PiedPiper stream. Options: `prior` (required if the stream was
 * compressed with one), `onProgress(done, total)`.
 * @returns {Uint8Array}
 */
export function decompress(data, opts = {}) {
  const w = ready();
  const input = toBytes(data);
  const prior = opts.prior ? toBytes(opts.prior, 'prior') : null;
  progressCb = opts.onProgress || null;
  try {
    return withBuffers(w, [input, prior], ([ip, pp]) =>
      take(w, w.pp_decompress(ip, input.length, pp, prior ? prior.length : 0, opts.onProgress ? 1 : 0)));
  } finally {
    progressCb = null;
  }
}

/** Reads a stream's header (sizes, tokenizer, network shape, …) without decompressing. */
export function info(data) {
  const w = ready();
  const input = toBytes(data);
  const json = withBuffers(w, [input], ([ip]) => take(w, w.pp_inspect(ip, input.length)));
  return JSON.parse(new TextDecoder().decode(json));
}

/** True if `data` starts with the PiedPiper magic bytes. */
export function isCompressed(data) {
  const b = toBytes(data);
  return b.length >= 4 && b[0] === 0x50 && b[1] === 0x49 && b[2] === 0x45 && b[3] === 0x44;
}

/** Library version. */
export function version() {
  const w = ready();
  return new TextDecoder().decode(take(w, w.pp_version()));
}

/**
 * Runs PiedPiper in a module Web Worker (browsers) so the page stays responsive.
 * Returns `{ compress, decompress, info, terminate }`; the first three return Promises and
 * accept the same options as the synchronous functions (`onProgress` is called on the main
 * thread; to cancel, call `terminate()`).
 */
export function createWorker(workerUrl = new URL('./worker.js', import.meta.url)) {
  const worker = new Worker(workerUrl, { type: 'module' });
  let seq = 0;
  const pending = new Map();
  worker.onmessage = (e) => {
    const m = e.data;
    const p = pending.get(m.id);
    if (!p) return;
    if (m.progress) {
      if (p.onProgress) p.onProgress(m.progress[0], m.progress[1]);
      return;
    }
    pending.delete(m.id);
    if (m.error) p.reject(m.cancelled ? new CancelledError(m.error) : new PiedPiperError(m.error));
    else p.resolve(m.result);
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new PiedPiperError(e.message || 'worker error'));
    pending.clear();
  };
  const call = (op, data, opts = {}) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      const { onProgress, ...rest } = opts;
      pending.set(id, { resolve, reject, onProgress });
      const bytes = toBytes(data).slice();
      worker.postMessage({ id, op, data: bytes, opts: { ...rest, progress: !!onProgress } }, [bytes.buffer]);
    });
  return {
    compress: (data, opts) => call('compress', data, opts),
    decompress: (data, opts) => call('decompress', data, opts),
    info: (data) => call('info', data),
    terminate() {
      worker.terminate();
      for (const p of pending.values()) p.reject(new CancelledError('worker terminated'));
      pending.clear();
    },
  };
}
