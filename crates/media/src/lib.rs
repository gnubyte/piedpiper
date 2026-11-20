//! Lossless recompression of H.264 and HEVC (H.265) video for PiedPiper.
//!
//! H.264 already removes almost all redundancy from video, and its CABAC entropy coder leaves
//! the bytes close to random, so byte-level compressors (including PiedPiper's Transformer)
//! cannot shrink it. This crate instead undoes only the final CABAC step: it parses each
//! slice down to its binary decisions ("bins"), codes those bins with a stronger adaptive
//! model that sees more context than CABAC does, and on decompression re-encodes them with
//! CABAC to rebuild the original bitstream bit for bit. The idea is the same as Dropbox's
//! Lepton (JPEG) and avrecode (H.264).

pub mod ac3;
pub mod bits;
pub mod container;
pub mod detect;
pub mod h264;
pub mod hevc;
pub mod model;
pub mod mkv;
pub mod mp4;
pub mod mpeg2;
#[doc(hidden)]
pub mod testgen;
pub mod ts;
pub mod vp8;
pub mod vp9;

pub use detect::is_opaque_video;
pub use container::{
    compress, compress_with, decompress, decompress_with, probe, Cancelled, Codec, Options, Progress, Stats, FAIL_NAMES,
    MAGIC, MAGIC_HEVC,
};
