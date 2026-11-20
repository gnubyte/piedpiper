//! # PiedPiper
//!
//! Lossless compression with an online-learning Transformer and arithmetic coding — the
//! method behind [TufaZip](https://tufalabs.ai/research/tufazip/), re-engineered to run on
//! ordinary CPUs (desktop, Android, WebAssembly) and to be **bit-exact on every platform**.
//!
//! Encoder and decoder run the same deterministic computation: a small Transformer
//! predicts the next token, the range coder codes it with that distribution, and after
//! every segment both sides take the same Adam step on what was just coded. The weights are
//! never stored — the decoder re-learns them from the data it decodes.
//!
//! ```
//! let data = b"hello hello hello hello, PiedPiper!".repeat(20);
//! let packed = piedpiper::compress(&data, &piedpiper::CompressOptions::level(1)).unwrap();
//! assert_eq!(piedpiper::decompress(&packed).unwrap(), data);
//! ```

mod audio;
mod bcj;
mod codec;
mod coder;
mod config;
#[cfg(feature = "deflate")]
mod deflate;
mod error;
mod filter;
mod format;
mod fpenv;
mod fpmath;
#[cfg(feature = "gpu")]
mod gpu;
mod image;
mod jpeg;
mod kernels;
mod matcher;
mod media;
mod model;
mod opaque;
mod par;
mod prior;
mod rng;
mod simd;
mod tokenizer;

pub use codec::{
    compress, compress_with_progress, decompress, decompress_with_progress, gpu_status, gpu_stream_count, inspect, train_prior, CompressOptions,
    DecompressOptions, GpuMode, GpuStatus, Info, GPU_SUPPORT, ProgressFn, TokenizerChoice, TrainOptions, TrainReport,
};
pub use config::{batch_policy, preset, ModelConfig, Preset, TrainConfig, DEFAULT_LEVEL, MAX_LEVEL, MIN_LEVEL};
pub use prior::{builtin_prior, builtin_prior_by_id, builtin_priors, BuiltinPrior, Prior, PRIOR_MAGIC};
pub use error::{Error, Result};
pub use filter::{Filter, FilterChoice};
pub use format::{MAGIC, VERSION as FORMAT_VERSION};
pub use audio::residuals as audio_residuals;

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// CRC-32 (IEEE) of `data` — the checksum stored in every stream.
pub fn crc32(data: &[u8]) -> u32 {
    format::crc32(data)
}

/// True if `data` starts with the PiedPiper magic bytes.
pub fn is_compressed(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == MAGIC
}
