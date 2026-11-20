/** Bytes accepted wherever input data is expected (strings are UTF-8 encoded). */
export type BytesLike = Uint8Array | ArrayBuffer | ArrayBufferView | string;

/** Return `false` to cancel. `done`/`total` are input bytes. */
export type ProgressCallback = (done: number, total: number) => boolean | void;

/** Network shape override (all fields required). */
export interface ModelOptions {
  dModel: number;
  layers: number;
  heads: number;
  dFf: number;
  /** Attention span in tokens. */
  window: number;
  /** Tokens per stream between weight updates. */
  segLen: number;
  /** Parallel streams. */
  streams: number;
}

/** Online-training overrides; unspecified fields keep the level's preset. */
export interface TrainOptions {
  lr?: number;
  lrMin?: number;
  warmup?: number;
  decay?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  clip?: number;
  initScale?: number;
  seed?: number | bigint;
}

export interface CompressOptions {
  /** 1 (fastest) … 9 (largest network). Default 2. Levels 1–3 start text from built-in priors. */
  level?: number;
  /** Default 'auto' (word-level BPE for large text, bytes otherwise). */
  tokenizer?: 'auto' | 'bytes' | 'bpe';
  /** Maximum BPE vocabulary when `tokenizer` is 'bpe'. Default 4096. */
  bpeVocab?: number;
  /** A pretrained prior file (the same prior is needed to decompress). */
  prior?: BytesLike;
  onProgress?: ProgressCallback;
  /** Store incompressible input verbatim. Default true. */
  allowStored?: boolean;
  /** Start text-like input from the built-in prior (levels 1–3). Default true. */
  builtinPrior?: boolean;
  model?: ModelOptions;
  train?: TrainOptions;
}

export interface DecompressOptions {
  prior?: BytesLike;
  onProgress?: ProgressCallback;
}

export interface StreamInfo {
  originalSize: number;
  compressedSize: number;
  crc32: number;
  stored: boolean;
  tokenizer: string;
  vocabSize: number;
  tokens: number;
  params: number;
  /** Hex id of the prior needed to decompress ("0000000000000000" = none). */
  priorId: string;
  /** Name of the built-in prior, if the stream uses one. */
  priorName: string | null;
  model?: ModelOptions;
  train?: Required<Omit<TrainOptions, 'seed'>> & { seed: string };
}

export class PiedPiperError extends Error {}
export class CancelledError extends PiedPiperError {}

/** Loads the WebAssembly module (default: `piedpiper.wasm` next to the JS file). */
export function init(source?: string | URL | BufferSource | WebAssembly.Module): Promise<void>;
export function initSync(source: BufferSource | WebAssembly.Module): void;
export function compress(data: BytesLike, opts?: CompressOptions): Uint8Array;
export function decompress(data: BytesLike, opts?: DecompressOptions): Uint8Array;
export function info(data: BytesLike): StreamInfo;
export function isCompressed(data: BytesLike): boolean;
export function version(): string;

export interface PiedPiperWorker {
  compress(data: BytesLike, opts?: CompressOptions & { wasmUrl?: string }): Promise<Uint8Array>;
  decompress(data: BytesLike, opts?: DecompressOptions & { wasmUrl?: string }): Promise<Uint8Array>;
  info(data: BytesLike): Promise<StreamInfo>;
  /** Stops the worker; pending calls reject with CancelledError. */
  terminate(): void;
}

/** Runs the codec in a module Web Worker (browsers). */
export function createWorker(workerUrl?: string | URL): PiedPiperWorker;
