//! MPEG-2 video (DVD, broadcast; Matroska `V_MPEG2`) lossless recompression.
//!
//! MPEG-2 codes its syntax with fixed variable-length codes (Huffman-style tables from the
//! standard's annex B), not with an adaptive arithmetic coder, so a context-mixing model has
//! much more to gain than on H.264, HEVC or VP9. The recoder does not re-encode anything:
//! `parse.rs` walks every slice bit by bit (macroblock addresses and types, motion vectors,
//! coded block patterns, DC sizes, run/level coefficient codes with escapes), and each bit is
//! coded by the model in the context of where it sits in the syntax and its code tree. The
//! decoder runs the same parser on the bits the model gives back, so the output is the input
//! bit for bit by construction. Parsing needs only the current picture's headers, never other
//! pictures.

pub mod parse;
pub mod stream;
#[doc(hidden)]
pub mod synth;
