//! VP8 (WebM `V_VP8`) lossless recompression.
//!
//! VP8 codes everything with the same boolean arithmetic coder as VP9, in one partition for
//! the frame header and macroblock modes plus one to eight partitions for the coefficient
//! tokens (macroblock rows alternate between them). Its probabilities are fixed per frame:
//! the frame header updates them, and they persist to later frames unless the frame asks
//! for them to be restored afterwards. `parse.rs` follows libvpx's decoder
//! (`vp8/decoder/decodeframe.c`, `decodemv.c`, `detokenize.c`) bool for bool; `stream.rs`
//! is the WebM driver and the `PPV8` format, built like the VP9 one.

pub mod parse;
pub mod stream;
pub mod tables;
