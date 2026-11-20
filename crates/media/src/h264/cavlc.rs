//! H.264 CAVLC slice data (Baseline profile, and other streams that turn CABAC off).
//!
//! CAVLC codes the macroblock layer with Exp-Golomb codes and fixed variable-length tables
//! (coeff_token chosen by the neighbours' coefficient counts, levels with an adaptive suffix
//! length, total_zeros and run_before), not with an arithmetic coder, so it leaves far more
//! redundancy than CABAC. As with MPEG-2, the recoder does not re-encode anything:
//! [`SliceParser`] reads every slice-data bit through a [`BitIo`] together with a [`Hint`]
//! (syntax element, place in its code, neighbour context), the model codes the bits, and the
//! decoder runs the same parser on the bits the model gives back. Parsed slices are therefore
//! rebuilt bit for bit by construction.
//!
//! Two things are coded as decisions rather than bits: `mb_skip_run`, as one "skipped?"
//! decision per macroblock with the neighbours' skip state as context (the decoder writes
//! the Exp-Golomb code back), and the end of the slice data (`more_rbsp_data()`).
//!
//! Only parsing state matters for exactness. The extra context (neighbouring macroblocks'
//! modes and motion, the previous picture's coefficient counts) is model input only, kept so
//! that the encoder and decoder see the same values: neighbours count only inside the current
//! slice, and cross-slice state changes only when a slice is committed.

use super::mb::Hint;
use super::{Pps, SliceHeader, Sps, B, I, P};
use crate::bits::{Bad, R};
use std::sync::OnceLock;

/// What the next bits of the slice code. The real coders ignore it; the synthetic stream
/// generator uses it to produce valid syntax.
#[derive(Clone, Copy, Debug)]
pub enum Elem {
    /// An Exp-Golomb code number no larger than the value.
    Ue(u32),
    /// mb_type (the largest value is I_PCM).
    MbType(u32),
    /// A code from one of the CAVLC tables. The value is valid up to the second field (for
    /// coeff_token, TotalCoeff up to it).
    Vlc(VlcId, u32),
    /// level_prefix: leading zero bits, then a one.
    Prefix,
    /// Bits that may take any value.
    Bits(u32),
    /// Bits that must be zero (I_PCM alignment).
    Zeros(u32),
}

/// One of the CAVLC code tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VlcId {
    /// coeff_token, for nC in 0..2, 2..4, 4..8, 8.. and chroma DC (nC = -1).
    Token(usize),
    /// total_zeros of a 4x4 block with TotalCoeff = index + 1.
    Tz(usize),
    /// total_zeros of a chroma DC block with TotalCoeff = index + 1.
    TzDc(usize),
    /// run_before with zerosLeft = index + 1 (index 6: more than 6).
    Run(usize),
}

/// Where the slice-data bits come from and go to.
pub trait BitIo {
    /// The next bit of the slice data.
    fn bit(&mut self, h: &Hint) -> R<u32>;
    /// Whether more slice data follows (`more_rbsp_data()`). This is a decision, not a bit
    /// of the stream. `at_end`: the picture has no further macroblocks (the answer must be
    /// no; only the generator looks at it).
    fn more(&mut self, h: &Hint, at_end: bool) -> R<bool>;
    /// For a value the parser codes as decisions: the encoder reads the Exp-Golomb code
    /// number from the stream (no bits are coded); the decoder returns `None`.
    fn peek_ue(&mut self) -> R<Option<u32>>;
    /// A decision that is not a bit of the stream: the encoder codes `v`, the decoder
    /// returns the decoded value.
    fn decision(&mut self, h: &Hint, v: u32) -> R<u32>;
    /// The decoder writes the Exp-Golomb code of a value it decoded as decisions.
    fn put_ue(&mut self, v: u32);
    /// Bit position in the slice RBSP (for I_PCM alignment).
    fn pos(&self) -> usize;
    /// Announces the syntax of the next bits (the generator uses it; others ignore it).
    #[inline]
    fn begin(&mut self, _e: Elem) {}
    /// A bit with hint `h` may come next (the parser announces both possible next bits of a
    /// code before it reads the current one). Only a performance hint.
    #[inline]
    fn prefetch(&mut self, _h: &Hint) {}
}

/// Hint kinds. Macroblock-level elements reuse the CABAC parser's kinds (whose contexts the
/// model keeps in its small, cache-resident table); coefficient syntax has kinds of its own.
pub mod kind {
    use crate::h264::mb::kind as k;
    pub const SKIP: u8 = k::SKIP;
    pub const MB_TYPE: u8 = k::MB_TYPE;
    pub const SUB_TYPE: u8 = k::SUB_TYPE;
    pub const MVD: u8 = k::MVD;
    pub const REF: u8 = k::REF;
    pub const DQP: u8 = k::DQP;
    pub const IPRED: u8 = k::IPRED;
    pub const CPRED: u8 = k::CPRED;
    pub const CBP: u8 = k::CBP_L;
    pub const T8: u8 = k::T8;
    pub const MORE: u8 = 21;
    pub const TOKEN: u8 = 22;
    pub const T1_SIGN: u8 = 23;
    pub const LEVEL_PREFIX: u8 = 24;
    pub const LEVEL_SUFFIX: u8 = 25;
    pub const TOTAL_ZEROS: u8 = 26;
    pub const RUN: u8 = 27;
    pub const PCM: u8 = 28;
}

/// Names of the kinds above, for statistics (index = kind).
pub fn kind_name(k: u8) -> &'static str {
    match k {
        kind::SKIP => "skip",
        kind::MB_TYPE => "mb_type",
        kind::SUB_TYPE => "sub_mb_type",
        kind::MVD => "mvd",
        kind::REF => "ref_idx",
        kind::DQP => "mb_qp_delta",
        kind::IPRED => "intra_pred",
        kind::CPRED => "chroma_pred",
        kind::CBP => "cbp",
        kind::T8 => "transform_8x8",
        kind::MORE => "more_data",
        kind::TOKEN => "coeff_token",
        kind::T1_SIGN => "trailing_sign",
        kind::LEVEL_PREFIX => "level_prefix",
        kind::LEVEL_SUFFIX => "level_suffix",
        kind::TOTAL_ZEROS => "total_zeros",
        kind::RUN => "run_before",
        kind::PCM => "pcm",
        _ => "?",
    }
}

/// Mixer contexts (`Hint::ctx`, all below 460; the model is given a neutral probability so
/// that it selects mixers by these).
mod cx {
    pub const SKIP: usize = 0;
    pub const MORE: usize = 12;
    /// + slice class * 12 + place in the code
    pub const MB_TYPE: usize = 16;
    pub const SUB_TYPE: usize = 52;
    pub const T8: usize = 76;
    pub const IPRED: usize = 80;
    pub const CPRED: usize = 88;
    pub const REF: usize = 100;
    /// + component * 12
    pub const MVD: usize = 114;
    pub const CBP: usize = 138;
    pub const DQP: usize = 162;
    /// + table * 16 + depth
    pub const TOKEN: usize = 174;
    pub const T1_SIGN: usize = 254;
    /// + suffixLength * 8 + prefix bit
    pub const LEVEL_PREFIX: usize = 258;
    pub const LEVEL_SUFFIX: usize = 314;
    /// + table * 4 + depth
    pub const TOTAL_ZEROS: usize = 330;
    /// + table * 8 + depth
    pub const RUN: usize = 402;
    pub const PCM: usize = 458;
}

// --- code tables (from FFmpeg's h264_cavlc.c; [TotalCoeff * 4 + TrailingOnes], len 0 = no code)

const TOKEN_LEN: [[u8; 68]; 4] = [
    [
        1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8, 13, 13, 11, 9, 13, 13, 13,
        10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15, 14, 16, 15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16,
        16, 16, 16, 16,
    ],
    [
        2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9, 9, 6, 11, 11, 11, 7, 12,
        11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13, 13, 13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14,
        14, 14,
    ],
    [
        4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6, 6, 4, 8, 7, 7, 5, 8, 8, 7,
        6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    ],
    [
        6, 0, 0, 0, 6, 6, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    ],
];
const TOKEN_BITS: [[u8; 68]; 4] = [
    [
        1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11, 14, 5, 4, 8, 10, 13, 4, 15,
        14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1, 9, 12, 11, 14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8,
    ],
    [
        3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15, 6, 5, 4, 11, 14, 13, 4, 15,
        10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 12, 7, 11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4,
    ],
    [
        15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9, 14, 13, 9, 8, 10, 9, 8, 15,
        14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8, 10, 13, 8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1,
        4, 3, 2,
    ],
    [
        3, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63,
    ],
];
const CDC_TOKEN_LEN: [u8; 20] = [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7];
const CDC_TOKEN_BITS: [u8; 20] = [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0];
const TZ_LEN: [[u8; 16]; 15] = [
    [1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
    [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 0],
    [4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6, 0, 0],
    [5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5, 0, 0, 0],
    [4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5, 0, 0, 0, 0],
    [6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6, 0, 0, 0, 0, 0],
    [6, 5, 3, 3, 3, 2, 3, 4, 3, 6, 0, 0, 0, 0, 0, 0],
    [6, 4, 5, 3, 2, 2, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0],
    [6, 6, 4, 2, 2, 3, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 3, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 3, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];
const TZ_BITS: [[u8; 16]; 15] = [
    [1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
    [7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0, 0],
    [5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0, 0, 0],
    [3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0, 0, 0, 0],
    [5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 5, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];
const CDC_TZ_LEN: [[u8; 4]; 3] = [[1, 2, 3, 3], [1, 2, 2, 0], [1, 1, 0, 0]];
const CDC_TZ_BITS: [[u8; 4]; 3] = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]];
const RUN_LEN: [[u8; 16]; 7] = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
];
const RUN_BITS: [[u8; 16]; 7] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 1, 3, 2, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
];
/// coded_block_pattern (luma | chroma << 4) by code number, for Intra_4x4/Intra_8x8 and
/// inter macroblocks (ChromaArrayType 1 and 2).
const INTRA_CBP: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28, 35, 37, 42, 44, 1, 2, 4,
    8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];
const INTER_CBP: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
    17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// The (length, code) table of `id`, indexed by value.
fn code_table(id: VlcId) -> (&'static [u8], &'static [u8]) {
    match id {
        VlcId::Token(4) => (&CDC_TOKEN_LEN, &CDC_TOKEN_BITS),
        VlcId::Token(t) => (&TOKEN_LEN[t], &TOKEN_BITS[t]),
        VlcId::Tz(i) => (&TZ_LEN[i], &TZ_BITS[i]),
        VlcId::TzDc(i) => (&CDC_TZ_LEN[i], &CDC_TZ_BITS[i]),
        VlcId::Run(i) => (&RUN_LEN[i], &RUN_BITS[i]),
    }
}

/// The codes of `id` as (code, length, value).
pub fn codes(id: VlcId) -> Vec<(u32, u8, u32)> {
    let (lens, bits) = code_table(id);
    lens.iter().zip(bits).enumerate().filter(|(_, (&l, _))| l > 0).map(|(v, (&l, &c))| (c as u32, l, v as u32)).collect()
}

const NONE: i32 = i32::MIN;

/// A prefix code as a binary trie: children are node indices (> 0), leaves are -(value + 1).
struct Vlc {
    nodes: Vec<[i32; 2]>,
}

impl Vlc {
    fn new(id: VlcId) -> Self {
        let mut nodes = vec![[NONE, NONE]];
        for (code, len, val) in codes(id) {
            let mut n = 0usize;
            for i in (0..len).rev() {
                let b = ((code >> i) & 1) as usize;
                if i == 0 {
                    nodes[n][b] = -(val as i32 + 1);
                } else {
                    if nodes[n][b] == NONE {
                        nodes.push([NONE, NONE]);
                        nodes[n][b] = (nodes.len() - 1) as i32;
                    }
                    n = nodes[n][b] as usize;
                }
            }
        }
        Vlc { nodes }
    }
}

struct Tables {
    token: Vec<Vlc>,
    tz: Vec<Vlc>,
    tz_dc: Vec<Vlc>,
    run: Vec<Vlc>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| Tables {
        token: (0..5).map(|i| Vlc::new(VlcId::Token(i))).collect(),
        tz: (0..15).map(|i| Vlc::new(VlcId::Tz(i))).collect(),
        tz_dc: (0..3).map(|i| Vlc::new(VlcId::TzDc(i))).collect(),
        run: (0..7).map(|i| Vlc::new(VlcId::Run(i))).collect(),
    })
}

// --- picture state --------------------------------------------------------------------------

/// Macroblock classes (model context).
const C_SKIP: u8 = 0;
const C_INTER: u8 = 1;
const C_NXN: u8 = 2;
const C_I16: u8 = 3;
const C_PCM: u8 = 4;
/// No macroblock (unavailable neighbour, or not coded in the previous picture).
const C_NONE: u8 = 7;

/// Block categories.
const CAT_I16DC: u8 = 0;
const CAT_I16AC: u8 = 1;
const CAT_L4: u8 = 2;
const CAT_CDC: u8 = 3;
const CAT_CAC: u8 = 4;
const CAT_L8: u8 = 5;

/// Intra prediction mode of a block that has none (DC, as the standard infers).
const IPM_DC: u8 = 2;
const REF_NONE: u8 = 15;

/// What the parser remembers about a macroblock of the current slice.
#[derive(Clone, Copy)]
struct Mb {
    /// Slice serial number (neighbours must belong to the current slice).
    serial: u32,
    class: u8,
    /// TotalCoeff of each 4x4 block (luma in luma4x4BlkIdx order, then Cb, then Cr).
    nz: [u8; 24],
    /// TotalCoeff of the DC blocks (Intra16x16 luma, Cb, Cr).
    dc: [u8; 3],
    /// Scan positions up to the last coefficient of each 4x4 block.
    last: [u8; 24],
    /// Largest |level| of each 4x4 block (clamped).
    mx: [u8; 24],
    /// Intra prediction mode of each luma 4x4 block.
    ipm: [u8; 16],
    cpm: u8,
    /// |mvd| (clamped) of each 4x4 block, per list and component.
    mvd: [[[u8; 2]; 16]; 2],
    /// Reference index of each 8x8 block per list.
    refs: [[u8; 4]; 2],
    cbp: u8,
    t8: bool,
    mbt: u8,
}

impl Default for Mb {
    fn default() -> Self {
        Mb {
            serial: 0,
            class: C_NONE,
            nz: [0; 24],
            dc: [0; 3],
            last: [0; 24],
            mx: [0; 24],
            ipm: [IPM_DC; 16],
            cpm: 0,
            mvd: [[[0; 2]; 16]; 2],
            refs: [[REF_NONE; 4]; 2],
            cbp: 0,
            t8: false,
            mbt: 0,
        }
    }
}

/// What later pictures see of a macroblock (model context only).
#[derive(Clone, Copy)]
struct Colo {
    class: u8,
    nz: [u8; 24],
    dc: [u8; 3],
}

impl Default for Colo {
    fn default() -> Self {
        Colo { class: C_NONE, nz: [0; 24], dc: [0; 3] }
    }
}

/// CAVLC parser state that lives across slices.
#[derive(Default)]
pub struct Picture {
    mbs: Vec<Mb>,
    serial: u32,
    /// (width in macroblocks, macroblocks) of `cur` and `prev`.
    geom: (usize, usize),
    /// Committed macroblocks of the current and the previous picture.
    cur: Vec<Colo>,
    prev: Vec<Colo>,
    /// Macroblocks in the last committed slice.
    last_len: u32,
}

#[derive(Clone, Copy, Default)]
struct MbT {
    intra: bool,
    nxn: bool,
    i16: bool,
    pcm: bool,
    i16_cbp: u8,
    /// 0 = 16x16, 1 = 16x8, 2 = 8x16, 3 = 8x8
    shape: u8,
    /// Per partition: 1 = L0, 2 = L1, 3 = bi-predicted.
    pred: [u8; 2],
    /// P_8x8ref0
    ref0: bool,
    /// B_Direct_16x16
    direct: bool,
}

#[derive(Clone, Copy, Default)]
struct SubT {
    direct: bool,
    mode: u8,
    /// 0 = 8x8, 1 = 8x4, 2 = 4x8, 3 = 4x4
    shape: u8,
}

const B_PAIRS: [(u8, u8); 9] = [(1, 1), (2, 2), (1, 2), (2, 1), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3)];
const B_SUB: [(u8, u8); 13] =
    [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (1, 3), (2, 3), (3, 3)];
/// Partitions as (x, y, w, h) in 4x4 blocks.
const PART_RECTS: [&[(usize, usize, usize, usize)]; 3] =
    [&[(0, 0, 4, 4)], &[(0, 0, 4, 2), (0, 2, 4, 2)], &[(0, 0, 2, 4), (2, 0, 2, 4)]];
const SUB_RECTS: [&[(usize, usize, usize, usize)]; 4] = [
    &[(0, 0, 2, 2)],
    &[(0, 0, 2, 1), (0, 1, 2, 1)],
    &[(0, 0, 1, 2), (1, 0, 1, 2)],
    &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
];

fn intra_t(v: u32) -> MbT {
    match v {
        0 => MbT { intra: true, nxn: true, ..Default::default() },
        25 => MbT { intra: true, pcm: true, ..Default::default() },
        _ => {
            let k = v - 1;
            let luma = if v >= 13 { 15 } else { 0 };
            MbT { intra: true, i16: true, i16_cbp: luma | (((k / 4) % 3) as u8) << 4, ..Default::default() }
        }
    }
}

fn p_type(v: u32) -> MbT {
    match v {
        0 => MbT { shape: 0, pred: [1, 0], ..Default::default() },
        1 => MbT { shape: 1, pred: [1, 1], ..Default::default() },
        2 => MbT { shape: 2, pred: [1, 1], ..Default::default() },
        3 => MbT { shape: 3, ..Default::default() },
        _ => MbT { shape: 3, ref0: true, ..Default::default() },
    }
}

fn b_type(v: u32) -> MbT {
    match v {
        0 => MbT { direct: true, ..Default::default() },
        1..=3 => MbT { shape: 0, pred: [v as u8, 0], ..Default::default() },
        4..=21 => {
            let (m0, m1) = B_PAIRS[((v - 4) / 2) as usize];
            MbT { shape: if v & 1 == 0 { 1 } else { 2 }, pred: [m0, m1], ..Default::default() }
        }
        _ => MbT { shape: 3, ..Default::default() },
    }
}

/// (x, y) in 4x4 units of luma4x4BlkIdx.
#[inline]
fn blk_xy(blk: usize) -> (usize, usize) {
    (((blk >> 2) & 1) * 2 + (blk & 1), ((blk >> 3) & 1) * 2 + ((blk >> 1) & 1))
}

#[inline]
fn xy_blk(x: usize, y: usize) -> usize {
    (y / 2) * 8 + (x / 2) * 4 + (y % 2) * 2 + (x % 2)
}

#[inline]
fn hint(ctx: usize, kind: u8, a: u32, b: u32, c: u32, d: u32) -> Hint {
    Hint { ctx: ctx.min(459) as u16, kind, a, b, c, d }
}

#[inline]
fn bucket(e: u32) -> u32 {
    if e < 3 {
        e
    } else {
        (32 - (e - 1).leading_zeros()).min(7)
    }
}

/// Reads ue(v) (code number at most `max`). Each bit is coded with `h`, with its place in the
/// code (prefix bit, or suffix bit and the bits before it) in the low 11 bits of `a` and `c`,
/// and the mixer context `ctx` + place (12 contexts).
fn ue<T: BitIo>(io: &mut T, h: Hint, ctx: usize, max: u32) -> R<u32> {
    ue_as(io, Elem::Ue(max), h, ctx, max)
}

/// [`ue`], announcing the element as `e`.
fn ue_as<T: BitIo>(io: &mut T, e: Elem, h: Hint, ctx: usize, max: u32) -> R<u32> {
    io.begin(e);
    // Hints of prefix bit `z` and of suffix bit `j` (after the suffix bits `v`) of a code with
    // `z` leading zeros.
    let pre = |z: u32| {
        let mut hh = h;
        hh.ctx = (ctx + (z as usize).min(5)).min(459) as u16;
        hh.a |= z;
        hh.c |= z;
        hh
    };
    let suf = |z: u32, j: u32, v: u32| {
        let place = if z <= 6 { 64 + (z << 7 | 1 << j | v) } else { 1024 + z * 32 + j };
        let mut hh = h;
        hh.ctx = (ctx + 6 + (z as usize - 1).min(5)).min(459) as u16;
        hh.a |= place;
        hh.c |= place;
        hh
    };
    let mut zeros = 0u32;
    loop {
        let hh = pre(zeros);
        if zeros < 31 {
            io.prefetch(&pre(zeros + 1));
        }
        if zeros > 0 {
            io.prefetch(&suf(zeros, 0, 0));
        }
        if io.bit(&hh)? == 1 {
            break;
        }
        zeros += 1;
        if zeros > 31 {
            return Err(Bad);
        }
    }
    let mut v = 0u64;
    for j in 0..zeros {
        let hh = suf(zeros, j, v as u32);
        if j + 1 < zeros {
            io.prefetch(&suf(zeros, j + 1, (v << 1) as u32));
            io.prefetch(&suf(zeros, j + 1, (v << 1 | 1) as u32));
        }
        v = v << 1 | io.bit(&hh)? as u64;
    }
    let n = (1u64 << zeros) - 1 + v;
    if n > max as u64 {
        return Err(Bad);
    }
    Ok(n as u32)
}

/// se(v) in `lo..=hi` (`lo` < 0 < `hi`).
fn se<T: BitIo>(io: &mut T, h: Hint, ctx: usize, lo: i32, hi: i32) -> R<i32> {
    let max = (2 * hi - 1).max(-2 * lo) as u32;
    // every code number up to `safe` is in range (what the generator may pick)
    let safe = (2 * hi - 1).min(-2 * lo) as u32;
    let k = ue_as(io, Elem::Ue(safe), h, ctx, max)?;
    let v = if k & 1 == 1 { k.div_ceil(2) as i32 } else { -((k / 2) as i32) };
    if v < lo || v > hi {
        return Err(Bad);
    }
    Ok(v)
}

/// Reads one code of table `id` (coding each bit with its trie node in the low bits of `a`
/// and `c`).
fn vlc<T: BitIo>(io: &mut T, id: VlcId, max: u32, h: Hint, ctx: usize, max_depth: usize) -> R<u32> {
    let t = tables();
    let v = match id {
        VlcId::Token(i) => &t.token[i],
        VlcId::Tz(i) => &t.tz[i],
        VlcId::TzDc(i) => &t.tz_dc[i],
        VlcId::Run(i) => &t.run[i],
    };
    io.begin(Elem::Vlc(id, max));
    let mut n = 0usize;
    let mut depth = 0usize;
    let node = |n: usize, depth: usize| {
        let mut hh = h;
        hh.ctx = (ctx + depth.min(max_depth)).min(459) as u16;
        hh.a |= n as u32;
        hh.c |= n as u32;
        hh
    };
    loop {
        let hh = node(n, depth);
        // Both children that are inner nodes may be read next.
        for &c in &v.nodes[n] {
            if c >= 0 {
                io.prefetch(&node(c as usize, depth + 1));
            }
        }
        let b = io.bit(&hh)? as usize;
        let c = v.nodes[n][b];
        if c == NONE {
            return Err(Bad);
        }
        if c < 0 {
            return Ok((-c - 1) as u32);
        }
        n = c as usize;
        depth += 1;
    }
}

/// Reads `n` plain bits (each coded with the bits before it in the low bits of `a`).
fn bits<T: BitIo>(io: &mut T, n: u32, h: Hint, ctx: usize) -> R<u32> {
    io.begin(Elem::Bits(n));
    let at = |j: u32, v: u32| {
        let mut hh = h;
        hh.ctx = (ctx + (j as usize).min(7)).min(459) as u16;
        hh.a |= (1 << j.min(7)) | (v & 0x7F);
        hh
    };
    let mut v = 0u32;
    for j in 0..n {
        let hh = at(j, v);
        if j + 1 < n {
            io.prefetch(&at(j + 1, v << 1));
            io.prefetch(&at(j + 1, v << 1 | 1));
        }
        v = v << 1 | io.bit(&hh)?;
    }
    Ok(v)
}

/// Parses the macroblock layer of one CAVLC slice (frame pictures, 4:2:0, 8 bits).
pub struct SliceParser<'a> {
    sps: &'a Sps,
    pps: &'a Pps,
    sh: &'a SliceHeader,
    pic: &'a mut Picture,
    w: usize,
    size: usize,
    serial: u32,
    /// The slice starts a picture (its committed state becomes the previous picture's).
    new_pic: bool,
    colo_ok: bool,
    first: usize,
    end: usize,
    qp: i32,
    prev_dqp: i32,
    prev_mbt: u32,
    addr: usize,
    cur: Mb,
    /// Blocks with coefficients so far in the current macroblock.
    nzb: u32,
    /// Sum of |mvd| of the current macroblock.
    mot: u32,
}

impl<'a> SliceParser<'a> {
    pub fn new(sps: &'a Sps, pps: &'a Pps, sh: &'a SliceHeader, pic: &'a mut Picture) -> R<Self> {
        if pps.cabac || !sps.supported() || pps.num_slice_groups != 1 {
            return Err(Bad);
        }
        let w = sps.width_mbs as usize;
        let size = w * sps.height_mbs() as usize;
        let first = sh.first_mb as usize;
        if first >= size {
            return Err(Bad);
        }
        if pic.mbs.len() < size {
            pic.mbs.resize(size, Mb::default());
        }
        pic.serial = pic.serial.wrapping_add(1);
        if pic.serial == 0 {
            for m in pic.mbs.iter_mut() {
                m.serial = 0;
            }
            pic.serial = 1;
        }
        let serial = pic.serial;
        let colo_ok = pic.geom == (w, size);
        Ok(SliceParser {
            sps,
            pps,
            sh,
            pic,
            w,
            size,
            serial,
            new_pic: first == 0,
            colo_ok,
            first,
            end: first,
            qp: sh.slice_qp,
            prev_dqp: 0,
            prev_mbt: 0,
            addr: first,
            cur: Mb::default(),
            nzb: 0,
            mot: 0,
        })
    }

    /// Makes a fully parsed slice part of the cross-slice state.
    pub fn commit(self) {
        let pic = self.pic;
        if pic.geom != (self.w, self.size) {
            pic.geom = (self.w, self.size);
            pic.cur = vec![Colo::default(); self.size];
            pic.prev = vec![Colo::default(); self.size];
        } else if self.new_pic {
            std::mem::swap(&mut pic.cur, &mut pic.prev);
            pic.cur.fill(Colo::default());
        }
        for a in self.first..self.end {
            let m = &pic.mbs[a];
            pic.cur[a] = Colo { class: m.class, nz: m.nz, dc: m.dc };
        }
        pic.last_len = (self.end - self.first) as u32;
    }

    /// Macroblocks of the slice (skipped ones included), once parsed.
    pub fn mbs(&self) -> usize {
        self.end - self.first
    }

    fn colo(&self, addr: usize) -> Option<&Colo> {
        if !self.colo_ok {
            return None;
        }
        let v = if self.new_pic { &self.pic.cur } else { &self.pic.prev };
        v.get(addr).filter(|c| c.class != C_NONE)
    }

    fn colo_class(&self, addr: usize) -> u32 {
        self.colo(addr).map_or(C_NONE, |c| c.class) as u32
    }

    /// The left and upper neighbours in the current slice.
    fn nbrs(&self, addr: usize) -> (Option<usize>, Option<usize>) {
        let a = (!addr.is_multiple_of(self.w) && self.pic.mbs[addr - 1].serial == self.serial).then(|| addr - 1);
        let b = (addr >= self.w && self.pic.mbs[addr - self.w].serial == self.serial).then(|| addr - self.w);
        (a, b)
    }

    fn class_of(&self, n: Option<usize>) -> u32 {
        n.map_or(C_NONE, |a| self.pic.mbs[a].class) as u32
    }

    pub fn run<T: BitIo>(&mut self, io: &mut T) -> R<()> {
        let st = self.sh.slice_type;
        let mut addr = self.first;
        loop {
            if st != I {
                let run = self.skip_run(io, addr)?;
                addr += run;
                if run > 0 {
                    let at_end = addr == self.size;
                    if !io.more(&self.more_hint(addr, true), at_end)? {
                        break;
                    }
                    if at_end {
                        return Err(Bad);
                    }
                }
            }
            self.macroblock(io, addr)?;
            addr += 1;
            let at_end = addr == self.size;
            if !io.more(&self.more_hint(addr, false), at_end)? {
                break;
            }
            if at_end {
                return Err(Bad);
            }
        }
        self.end = addr;
        Ok(())
    }

    fn more_hint(&self, next: usize, after_skip: bool) -> Hint {
        let at_end = (next == self.size) as u32;
        let n = (next - self.first) as u32;
        let same = (n == self.pic.last_len) as u32;
        hint(
            cx::MORE + at_end as usize * 2 + after_skip as usize,
            kind::MORE,
            at_end | (after_skip as u32) << 1 | same << 2 | self.sh.slice_type << 3,
            n.min(1 << 16),
            same | (next.is_multiple_of(self.w) as u32) << 1,
            0,
        )
    }

    fn skip_run<T: BitIo>(&mut self, io: &mut T, addr: usize) -> R<usize> {
        let left = self.size - addr;
        let known = io.peek_ue()?;
        if known.is_some_and(|k| k as usize > left) {
            return Err(Bad);
        }
        let mut run = 0usize;
        while run < left {
            let a = addr + run;
            let h = self.skip_hint(a, run);
            let want = known.map_or(0, |k| (run < k as usize) as u32);
            if io.decision(&h, want)? == 0 {
                break;
            }
            self.pic.mbs[a] = Mb { serial: self.serial, class: C_SKIP, refs: [[0; 4], [REF_NONE; 4]], ..Default::default() };
            run += 1;
        }
        io.put_ue(run as u32);
        Ok(run)
    }

    fn skip_hint(&self, a: usize, run: usize) -> Hint {
        let (na, nb) = self.nbrs(a);
        let skip = |n: Option<usize>| n.is_some_and(|i| self.pic.mbs[i].class == C_SKIP) as u32;
        let c = (a >= self.w && a % self.w != self.w - 1 && self.pic.mbs[a + 1 - self.w].serial == self.serial)
            .then(|| a + 1 - self.w);
        let colo = self.colo_class(a);
        let colo_skip = (colo == C_SKIP as u32) as u32;
        let around = [a + 1, a + self.w].iter().filter(|&&i| i < self.size && self.colo_class(i) == C_SKIP as u32).count() as u32;
        let (ls, us) = (skip(na), skip(nb));
        let is_b = (self.sh.slice_type == B) as u32;
        hint(
            cx::SKIP + (ls + us) as usize + 3 * colo_skip as usize + 6 * is_b as usize,
            kind::SKIP,
            ls | us << 1 | skip(c) << 2 | colo_skip << 3 | is_b << 4 | (na.is_some() as u32) << 5 | (nb.is_some() as u32) << 6,
            (run as u32).min(15) | around << 4 | colo << 6,
            colo | self.class_of(na) << 3 | self.class_of(nb) << 6 | is_b << 9,
            (a.is_multiple_of(self.w) as u32) | ((self.qp as u32) >> 2) << 1,
        )
    }

    fn macroblock<T: BitIo>(&mut self, io: &mut T, addr: usize) -> R<()> {
        let st = self.sh.slice_type;
        let (na, nb) = self.nbrs(addr);
        self.addr = addr;
        self.cur = Mb { serial: self.serial, class: C_INTER, ..Default::default() };
        self.nzb = 0;
        self.mot = 0;
        let (max, stc) = match st {
            I => (25, 0),
            P => (30, 1),
            _ => (48, 2),
        };
        let colo = self.colo_class(addr);
        let h = hint(
            0,
            kind::MB_TYPE,
            st << 11,
            self.class_of(na) | self.class_of(nb) << 3 | colo << 6,
            colo << 11 | st << 14,
            self.prev_mbt.min(63) | ((self.qp as u32) >> 2) << 6,
        );
        let v = ue_as(io, Elem::MbType(max), h, cx::MB_TYPE + stc * 12, max)?;
        self.prev_mbt = v;
        self.cur.mbt = v as u8;
        let t = match st {
            I => intra_t(v),
            P if v < 5 => p_type(v),
            P => intra_t(v - 5),
            _ if v < 23 => b_type(v),
            _ => intra_t(v - 23),
        };
        self.cur.class = if t.pcm {
            C_PCM
        } else if t.i16 {
            C_I16
        } else if t.intra {
            C_NXN
        } else {
            C_INTER
        };
        if t.pcm {
            return self.pcm(io, addr);
        }
        let mut t8 = false;
        let mut no_small = true;
        if !t.intra && t.shape == 3 {
            let is_b = st == B;
            let mut sub = [SubT::default(); 4];
            let mut prev = 15u32;
            for i in 0..4 {
                let h = hint(
                    0,
                    kind::SUB_TYPE,
                    (is_b as u32) << 11 | (i as u32) << 12,
                    prev | self.class_of(na) << 4 | self.class_of(nb) << 7,
                    (i as u32) << 11 | (is_b as u32) << 13 | (t.ref0 as u32) << 14,
                    0,
                );
                let v = ue(io, h, cx::SUB_TYPE + is_b as usize * 12, if is_b { 12 } else { 3 })?;
                prev = v;
                sub[i] = if is_b {
                    let (mode, shape) = B_SUB[v as usize];
                    SubT { direct: v == 0, mode, shape }
                } else {
                    SubT { direct: false, mode: 1, shape: v as u8 }
                };
                if sub[i].direct {
                    if !self.sps.direct_8x8_inference {
                        no_small = false;
                    }
                } else if sub[i].shape > 0 {
                    no_small = false;
                }
            }
            self.sub_mb_pred(io, &t, &sub)?;
        } else {
            if self.pps.transform_8x8_mode && t.nxn {
                t8 = self.t8_flag(io, na, nb, true)?;
            }
            self.mb_pred(io, &t, t8, na, nb)?;
        }
        let cbp = if t.i16 {
            t.i16_cbp
        } else {
            let cbp = self.cbp(io, t.intra, t8, na, nb, colo)?;
            if cbp & 15 != 0
                && self.pps.transform_8x8_mode
                && !t.intra
                && no_small
                && (!t.direct || self.sps.direct_8x8_inference)
            {
                t8 = self.t8_flag(io, na, nb, false)?;
            }
            cbp
        };
        self.cur.cbp = cbp;
        self.cur.t8 = t8;
        if cbp != 0 || t.i16 {
            let h = hint(
                0,
                kind::DQP,
                (self.cur.class as u32) << 11,
                (self.prev_dqp + 26).clamp(0, 63) as u32 | (cbp as u32) << 6,
                ((self.qp as u32) >> 2) << 11,
                0,
            );
            let dqp = se(io, h, cx::DQP, -26, 25)?;
            self.qp = (self.qp + dqp + 52) % 52;
            self.prev_dqp = dqp;
            self.residual(io, &t, cbp, t8)?;
        } else {
            self.prev_dqp = 0;
        }
        self.pic.mbs[addr] = self.cur;
        Ok(())
    }

    /// coded_block_pattern, coded as CABAC binarizes it (a decision per luma 8x8 block, two
    /// for chroma) with the neighbouring blocks' flags as context; the decoder writes the
    /// code number back.
    fn cbp<T: BitIo>(&mut self, io: &mut T, intra: bool, t8: bool, na: Option<usize>, nb: Option<usize>, colo: u32) -> R<u8> {
        let known = io.peek_ue()?;
        let want = match known {
            Some(k) if k <= 47 => (if intra { INTRA_CBP[k as usize] } else { INTER_CBP[k as usize] }) as u32,
            Some(_) => return Err(Bad),
            None => 0,
        };
        // neighbour flags: 2 = unavailable
        let left = |b8: usize, cbp: u32| -> u32 {
            if b8 & 1 == 1 {
                (cbp >> (b8 - 1)) & 1
            } else {
                na.map_or(2, |a| (self.pic.mbs[a].cbp as u32 >> (b8 + 1)) & 1)
            }
        };
        let up = |b8: usize, cbp: u32| -> u32 {
            if b8 >= 2 {
                (cbp >> (b8 - 2)) & 1
            } else {
                nb.map_or(2, |a| (self.pic.mbs[a].cbp as u32 >> (b8 + 2)) & 1)
            }
        };
        let qb = (self.qp as u32) >> 2;
        let mbt = (self.cur.mbt as u32).min(31);
        let colo_nz = self.colo(self.addr).map_or(17, |c| c.nz[..16].iter().filter(|&&v| v > 0).count() as u32);
        let mut cbp = 0u32;
        for b8 in 0..4 {
            let (l, u) = (left(b8, cbp), up(b8, cbp));
            let h = hint(
                cx::CBP + (l.min(1) + 2 * u.min(1)) as usize + 4 * intra as usize,
                kind::CBP,
                b8 as u32 | l << 2 | u << 4 | (intra as u32) << 6 | (t8 as u32) << 7,
                cbp | mbt << 4,
                b8 as u32 | colo << 2 | colo_nz << 5 | (intra as u32) << 10,
                qb | (cbp.count_ones()) << 4 | (b8 as u32) << 7,
            );
            cbp |= io.decision(&h, (want >> b8) & 1)? << b8;
        }
        let chroma_of = |n: Option<usize>| n.map_or(3, |a| self.pic.mbs[a].cbp as u32 >> 4);
        let (cl, cu) = (chroma_of(na), chroma_of(nb));
        let wc = want >> 4;
        let h = hint(
            cx::CBP + 8 + ((cl != 0 && cl != 3) as usize + 2 * (cu != 0 && cu != 3) as usize),
            kind::CBP,
            0x100 | cl << 2 | cu << 4 | (intra as u32) << 6,
            cbp | mbt << 4,
            colo | qb << 3,
            qb | cbp.count_ones() << 4,
        );
        if io.decision(&h, (wc != 0) as u32)? == 1 {
            let h = hint(
                cx::CBP + 12 + ((cl == 2) as usize + 2 * (cu == 2) as usize),
                kind::CBP,
                0x200 | cl << 2 | cu << 4 | (intra as u32) << 6,
                cbp | mbt << 4,
                colo | qb << 3,
                qb | cbp.count_ones() << 4,
            );
            cbp |= (1 + io.decision(&h, (wc == 2) as u32)?) << 4;
        }
        let table = if intra { &INTRA_CBP } else { &INTER_CBP };
        let code = table.iter().position(|&v| v as u32 == cbp).ok_or(Bad)?;
        if known.is_some_and(|k| k as usize != code) {
            return Err(Bad);
        }
        io.put_ue(code as u32);
        Ok(cbp as u8)
    }

    fn t8_flag<T: BitIo>(&mut self, io: &mut T, na: Option<usize>, nb: Option<usize>, intra: bool) -> R<bool> {
        let t8_of = |n: Option<usize>| n.map_or(2, |a| self.pic.mbs[a].t8 as u32);
        let h = hint(
            cx::T8 + intra as usize,
            kind::T8,
            0,
            t8_of(na) | t8_of(nb) << 2 | (intra as u32) << 4,
            (self.cur.cbp as u32) | (self.cur.mbt as u32) << 6,
            0,
        );
        Ok(bits(io, 1, h, cx::T8 + intra as usize)? == 1)
    }

    fn pcm<T: BitIo>(&mut self, io: &mut T, addr: usize) -> R<()> {
        let pad = (8 - io.pos() % 8) % 8;
        if pad > 0 {
            io.begin(Elem::Zeros(pad as u32));
            for _ in 0..pad {
                if io.bit(&hint(cx::PCM, kind::PCM, 1 << 12, 0, 0, 0))? != 0 {
                    return Err(Bad);
                }
            }
        }
        let mut prev = 0u32;
        for i in 0..384u32 {
            let h = hint(cx::PCM + 1, kind::PCM, ((i >= 256) as u32) << 9, prev, 0, 0);
            prev = bits(io, 8, h, cx::PCM + 1)?;
        }
        self.cur.nz = [16; 24];
        self.cur.dc = [16; 3];
        self.cur.last = [16; 24];
        self.cur.mx = [255; 24];
        self.cur.cbp = 47;
        self.prev_dqp = 0;
        self.pic.mbs[addr] = self.cur;
        Ok(())
    }

    /// The intra modes of the 4x4 blocks left of and above luma block (x, y): 9 when the
    /// neighbour is outside the slice.
    fn ipm_nbrs(&self, x: usize, y: usize) -> (u8, u8) {
        let (na, nb) = self.nbrs(self.addr);
        let a = if x > 0 {
            self.cur.ipm[xy_blk(x - 1, y)]
        } else {
            na.map_or(9, |i| self.pic.mbs[i].ipm[xy_blk(3, y)])
        };
        let b = if y > 0 {
            self.cur.ipm[xy_blk(x, y - 1)]
        } else {
            nb.map_or(9, |i| self.pic.mbs[i].ipm[xy_blk(x, 3)])
        };
        (a, b)
    }

    fn mb_pred<T: BitIo>(&mut self, io: &mut T, t: &MbT, t8: bool, na: Option<usize>, nb: Option<usize>) -> R<()> {
        if t.intra {
            if t.nxn {
                let n = if t8 { 4 } else { 16 };
                let mut hits = 0u32;
                for i in 0..n {
                    let blk = if t8 { i * 4 } else { i };
                    let (x, y) = blk_xy(blk);
                    let (ma, mb) = self.ipm_nbrs(x, y);
                    let pred = if ma == 9 || mb == 9 { IPM_DC } else { ma.min(mb) };
                    let h = hint(
                        cx::IPRED + (ma == mb) as usize,
                        kind::IPRED,
                        (ma as u32) << 8 | (mb as u32) << 12 | (t8 as u32) << 16,
                        blk as u32 | hits.min(3) << 4,
                        pred as u32 | (t8 as u32) << 4,
                        0,
                    );
                    let mode = if bits(io, 1, h, cx::IPRED + (ma == mb) as usize)? == 1 {
                        hits += 1;
                        pred
                    } else {
                        let h = hint(0, kind::IPRED, 0x100 | (pred as u32) << 9 | (ma as u32) << 13, mb as u32 | (t8 as u32) << 4, 0, 0);
                        let rem = bits(io, 3, h, cx::IPRED + 2)? as u8;
                        if rem < pred {
                            rem
                        } else {
                            rem + 1
                        }
                    };
                    if t8 {
                        self.cur.ipm[blk..blk + 4].fill(mode);
                    } else {
                        self.cur.ipm[blk] = mode;
                    }
                }
            }
            let cpm_of = |n: Option<usize>| {
                n.map_or(4, |a| {
                    let m = &self.pic.mbs[a];
                    if m.class == C_NXN || m.class == C_I16 {
                        m.cpm as u32
                    } else {
                        5
                    }
                })
            };
            let h = hint(0, kind::CPRED, cpm_of(na) << 11 | cpm_of(nb) << 14 | (t.nxn as u32) << 17, 0, (self.cur.mbt as u32) << 11, 0);
            self.cur.cpm = ue(io, h, cx::CPRED, 3)? as u8;
            return Ok(());
        }
        if t.direct {
            return Ok(());
        }
        let parts = PART_RECTS[t.shape as usize];
        for list in 0..2usize {
            let nref = if list == 0 { self.sh.num_ref_l0 } else { self.sh.num_ref_l1 };
            for (p, &(x, y, w, h)) in parts.iter().enumerate() {
                let m = t.pred[p];
                if m != 3 && m as usize != list + 1 {
                    continue;
                }
                let r = if nref > 1 { self.ref_idx(io, list, x, y, nref - 1, p)? } else { 0 };
                self.set_ref(list, x, y, w, h, r);
            }
        }
        for list in 0..2usize {
            for (p, &(x, y, w, h)) in parts.iter().enumerate() {
                let m = t.pred[p];
                if m == 3 || m as usize == list + 1 {
                    self.mvd(io, list, x, y, w, h, t.shape as u32)?;
                }
            }
        }
        Ok(())
    }

    fn sub_mb_pred<T: BitIo>(&mut self, io: &mut T, t: &MbT, sub: &[SubT; 4]) -> R<()> {
        for list in 0..2usize {
            let nref = if list == 0 { self.sh.num_ref_l0 } else { self.sh.num_ref_l1 };
            for (i, s) in sub.iter().enumerate() {
                if s.direct || (s.mode != 3 && s.mode as usize != list + 1) {
                    continue;
                }
                let (x, y) = ((i & 1) * 2, (i >> 1) * 2);
                let r = if nref > 1 && !(list == 0 && t.ref0) { self.ref_idx(io, list, x, y, nref - 1, i)? } else { 0 };
                self.set_ref(list, x, y, 2, 2, r);
            }
        }
        for list in 0..2usize {
            for (i, s) in sub.iter().enumerate() {
                if s.direct || (s.mode != 3 && s.mode as usize != list + 1) {
                    continue;
                }
                let (x8, y8) = ((i & 1) * 2, (i >> 1) * 2);
                for &(dx, dy, w, h) in SUB_RECTS[s.shape as usize] {
                    self.mvd(io, list, x8 + dx, y8 + dy, w, h, 4 + s.shape as u32)?;
                }
            }
        }
        Ok(())
    }

    fn set_ref(&mut self, list: usize, x: usize, y: usize, w: usize, h: usize, r: u32) {
        for yy in (y..y + h).step_by(2) {
            for xx in (x..x + w).step_by(2) {
                self.cur.refs[list][(yy / 2) * 2 + xx / 2] = r.min(14) as u8;
            }
        }
    }

    /// Reference indices of the 8x8 blocks left of and above 4x4 block (x, y).
    fn ref_nbrs(&self, list: usize, x: usize, y: usize) -> (u32, u32) {
        let (na, nb) = self.nbrs(self.addr);
        let (x8, y8) = (x / 2, y / 2);
        let a = if x8 > 0 {
            self.cur.refs[list][y8 * 2]
        } else {
            na.map_or(REF_NONE, |i| self.pic.mbs[i].refs[list][y8 * 2 + 1])
        };
        let b = if y8 > 0 {
            self.cur.refs[list][x8]
        } else {
            nb.map_or(REF_NONE, |i| self.pic.mbs[i].refs[list][2 + x8])
        };
        (a as u32, b as u32)
    }

    fn ref_idx<T: BitIo>(&mut self, io: &mut T, list: usize, x: usize, y: usize, range: u32, part: usize) -> R<u32> {
        let (ra, rb) = self.ref_nbrs(list, x, y);
        if range == 1 {
            let h = hint(cx::REF, kind::REF, (list as u32) << 8 | ra << 9 | rb << 13, part as u32, 0, 0);
            return Ok(1 - bits(io, 1, h, cx::REF)?);
        }
        let h = hint(
            0,
            kind::REF,
            (list as u32) << 11 | range.min(15) << 12,
            ra | rb << 4 | (part as u32) << 8,
            (list as u32) << 11 | ra << 12,
            0,
        );
        ue(io, h, cx::REF + 2, range)
    }

    /// |mvd| of the 4x4 blocks left of and above (x, y).
    fn mvd_nbrs(&self, list: usize, comp: usize, x: usize, y: usize) -> (u32, u32) {
        let (na, nb) = self.nbrs(self.addr);
        let a = if x > 0 {
            self.cur.mvd[list][xy_blk(x - 1, y)][comp]
        } else {
            na.map_or(0, |i| self.pic.mbs[i].mvd[list][xy_blk(3, y)][comp])
        };
        let b = if y > 0 {
            self.cur.mvd[list][xy_blk(x, y - 1)][comp]
        } else {
            nb.map_or(0, |i| self.pic.mbs[i].mvd[list][xy_blk(x, 3)][comp])
        };
        (a as u32, b as u32)
    }

    #[allow(clippy::too_many_arguments)]
    fn mvd<T: BitIo>(&mut self, io: &mut T, list: usize, x: usize, y: usize, w: usize, h: usize, shape: u32) -> R<()> {
        let mut other = 0u32;
        for comp in 0..2usize {
            let (a, b) = self.mvd_nbrs(list, comp, x, y);
            let sum = bucket(a + b);
            let hh = hint(
                0,
                kind::MVD,
                (comp as u32) << 11 | (list as u32) << 12 | sum << 13,
                a.min(127) | b.min(127) << 7 | (comp as u32) << 14,
                (comp as u32) << 11 | shape << 12 | other << 16,
                self.cur.refs[list][(y / 2) * 2 + x / 2] as u32,
            );
            let v = se(io, hh, cx::MVD + comp * 12, -(1 << 15), 1 << 15)?;
            self.mot += v.unsigned_abs();
            let m = v.unsigned_abs().min(255) as u8;
            other = bucket(m as u32);
            for yy in y..y + h {
                for xx in x..x + w {
                    self.cur.mvd[list][xy_blk(xx, yy)][comp] = m;
                }
            }
        }
        Ok(())
    }

    /// TotalCoeff of the luma blocks left of and above `blk` (None = unavailable).
    fn luma_nn(&self, blk: usize) -> (Option<u8>, Option<u8>) {
        let (na, nb) = self.nbrs(self.addr);
        let (x, y) = blk_xy(blk);
        let a = if x > 0 { Some(self.cur.nz[xy_blk(x - 1, y)]) } else { na.map(|i| self.pic.mbs[i].nz[xy_blk(3, y)]) };
        let b = if y > 0 { Some(self.cur.nz[xy_blk(x, y - 1)]) } else { nb.map(|i| self.pic.mbs[i].nz[xy_blk(x, 3)]) };
        (a, b)
    }

    /// TotalCoeff of the chroma blocks left of and above block `b4` of component `c`.
    fn chroma_nn(&self, c: usize, b4: usize) -> (Option<u8>, Option<u8>) {
        let (na, nb) = self.nbrs(self.addr);
        let (x, y) = (b4 & 1, b4 >> 1);
        let base = 16 + c * 4;
        let a = if x > 0 { Some(self.cur.nz[base + y * 2]) } else { na.map(|i| self.pic.mbs[i].nz[base + y * 2 + 1]) };
        let b = if y > 0 { Some(self.cur.nz[base + x]) } else { nb.map(|i| self.pic.mbs[i].nz[base + 2 + x]) };
        (a, b)
    }

    /// The average of a per-block value over the blocks left of and above block `idx` (< 24),
    /// or `none` when neither is in the slice.
    fn nb_avg(&self, idx: usize, sel: fn(&Mb) -> &[u8; 24], none: u32) -> u32 {
        let (na, nb) = self.nbrs(self.addr);
        let (l, u) = if idx < 16 {
            let (x, y) = blk_xy(idx);
            let l = if x > 0 { Some(sel(&self.cur)[xy_blk(x - 1, y)]) } else { na.map(|i| sel(&self.pic.mbs[i])[xy_blk(3, y)]) };
            let u = if y > 0 { Some(sel(&self.cur)[xy_blk(x, y - 1)]) } else { nb.map(|i| sel(&self.pic.mbs[i])[xy_blk(x, 3)]) };
            (l, u)
        } else {
            let base = 16 + ((idx - 16) / 4) * 4;
            let b4 = (idx - 16) % 4;
            let (x, y) = (b4 & 1, b4 >> 1);
            let l = if x > 0 { Some(sel(&self.cur)[base + y * 2]) } else { na.map(|i| sel(&self.pic.mbs[i])[base + y * 2 + 1]) };
            let u = if y > 0 { Some(sel(&self.cur)[base + x]) } else { nb.map(|i| sel(&self.pic.mbs[i])[base + 2 + x]) };
            (l, u)
        };
        match (l, u) {
            (Some(a), Some(b)) => (a as u32 + b as u32).div_ceil(2),
            (Some(a), None) | (None, Some(a)) => a as u32,
            (None, None) => none,
        }
    }

    fn residual<T: BitIo>(&mut self, io: &mut T, t: &MbT, cbp: u8, t8: bool) -> R<()> {
        if t.i16 {
            let (a, b) = self.luma_nn(0);
            let tc = self.block(io, CAT_I16DC, 26, a, b, 16)?;
            self.cur.dc[0] = tc;
        }
        for i8 in 0..4 {
            for i4 in 0..4 {
                let blk = i8 * 4 + i4;
                if cbp & (1 << i8) == 0 {
                    continue;
                }
                let (a, b) = self.luma_nn(blk);
                let (cat, max) = if t.i16 {
                    (CAT_I16AC, 15)
                } else if t8 {
                    (CAT_L8, 16)
                } else {
                    (CAT_L4, 16)
                };
                let tc = self.block(io, cat, blk, a, b, max)?;
                self.cur.nz[blk] = tc;
            }
        }
        let cc = cbp >> 4;
        if cc != 0 {
            for c in 0..2 {
                let tc = self.block(io, CAT_CDC, 24 + c, None, None, 4)?;
                self.cur.dc[1 + c] = tc;
            }
        }
        if cc & 2 != 0 {
            for c in 0..2 {
                for b4 in 0..4 {
                    let idx = 16 + c * 4 + b4;
                    let (a, b) = self.chroma_nn(c, b4);
                    let tc = self.block(io, CAT_CAC, idx, a, b, 15)?;
                    self.cur.nz[idx] = tc;
                }
            }
        }
        Ok(())
    }

    /// TotalCoeff of the co-located block of the previous picture (17 = none).
    fn colo_nz(&self, cat: u8, idx: usize) -> u32 {
        match self.colo(self.addr) {
            None => 17,
            Some(c) => match cat {
                CAT_I16DC => c.dc[0] as u32,
                CAT_CDC => c.dc[1 + (idx - 24)] as u32,
                _ => c.nz[idx] as u32,
            },
        }
    }

    /// residual_block_cavlc(): returns TotalCoeff. `idx`: block index in `nz` (24 + c for
    /// chroma DC, 26 for Intra16x16 DC).
    #[allow(clippy::too_many_arguments)]
    fn block<T: BitIo>(&mut self, io: &mut T, cat: u8, idx: usize, na: Option<u8>, nb: Option<u8>, max: usize) -> R<u8> {
        let nc: i32 = if cat == CAT_CDC {
            -1
        } else {
            match (na, nb) {
                (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
                (Some(a), None) => a as i32,
                (None, Some(b)) => b as i32,
                (None, None) => 0,
            }
        };
        let ti = match nc {
            -1 => 4,
            0 | 1 => 0,
            2 | 3 => 1,
            4..=7 => 2,
            _ => 3,
        };
        let nn = |o: Option<u8>| o.map_or(17u32, |v| v.min(16) as u32);
        let colo = self.colo_nz(cat, idx);
        let cls = self.cur.class as u32;
        let qb = (self.qp as u32) >> 2;
        // place in the 8x8 block, blocks with coefficients before it there, and the DC
        // block's count for AC blocks
        let sub = match cat {
            CAT_L4 | CAT_L8 | CAT_I16AC => {
                let base = idx & !3;
                let nzq = self.cur.nz[base..idx].iter().filter(|&&v| v > 0).count() as u32;
                let dc = if cat == CAT_I16AC { bucket(self.cur.dc[0] as u32).min(3) } else { 0 };
                (idx & 3) as u32 | nzq << 2 | dc << 4
            }
            CAT_CAC => {
                let base = idx & !3;
                let nzq = self.cur.nz[base..idx].iter().filter(|&&v| v > 0).count() as u32;
                ((idx - 16) & 3) as u32 | nzq << 2 | bucket(self.cur.dc[1 + (idx - 16) / 4] as u32).min(3) << 4
            }
            _ => 0,
        };
        let h = hint(
            0,
            kind::TOKEN,
            (ti as u32) << 8 | (cat as u32) << 11,
            nn(na) | nn(nb) << 5 | (idx as u32).min(31) << 10,
            (cat as u32) << 8 | colo << 11 | cls << 16 | (ti as u32) << 19,
            qb | self.nzb.min(7) << 4 | colo << 7 | sub << 12,
        );
        let tok = vlc(io, VlcId::Token(ti), max as u32, h, cx::TOKEN + ti * 16, 15)? as usize;
        let (tc, t1) = (tok >> 2, tok & 3);
        if tc > max {
            return Err(Bad);
        }
        if idx < 24 {
            self.cur.last[idx] = 0;
        }
        if tc == 0 {
            return Ok(0);
        }
        self.nzb += 1;
        let ncb = nc.clamp(0, 16) as u32;
        // levels, highest frequency first
        let mut levels = [0u32; 16];
        let mut suffix_len = (tc > 10 && t1 < 3) as u32;
        let mut prev_abs = 0u32;
        let mut prev_sign = 2u32;
        let nbmax = if idx < 24 { bucket(self.nb_avg(idx, |m| &m.mx, 0)).min(7) | ((self.nbrs(self.addr) == (None, None)) as u32) << 3 } else { 8 };
        for i in 0..tc {
            if i < t1 {
                let h = hint(0, kind::T1_SIGN, (i as u32) << 8 | (t1 as u32) << 10 | (cat as u32) << 12, tc as u32 | prev_sign << 5, 0, 0);
                let s = bits(io, 1, h, cx::T1_SIGN + i)?;
                prev_sign = s;
                levels[i] = 1;
                prev_abs = 1;
                continue;
            }
            io.begin(Elem::Prefix);
            let first = (i == t1 && t1 < 3) as u32;
            let mut prefix = 0u32;
            loop {
                let k = prefix.min(15);
                let h = hint(
                    cx::LEVEL_PREFIX + suffix_len as usize * 8 + (prefix as usize).min(7),
                    kind::LEVEL_PREFIX,
                    k | suffix_len << 4 | first << 7 | (cat as u32) << 8,
                    tc as u32 | (t1 as u32) << 5 | (i as u32) << 7,
                    k | suffix_len << 4 | bucket(prev_abs) << 7 | ncb << 11,
                    qb | cls << 4 | first << 7 | nbmax << 8,
                );
                if io.bit(&h)? == 1 {
                    break;
                }
                prefix += 1;
                if prefix > 28 {
                    return Err(Bad);
                }
            }
            let mut code = (prefix.min(15) << suffix_len) as i64;
            let size = if prefix == 14 && suffix_len == 0 {
                4
            } else if prefix >= 15 {
                prefix - 3
            } else {
                suffix_len
            };
            if size > 0 {
                io.begin(Elem::Bits(size));
                let mut v = 0u32;
                for j in 0..size {
                    let last = (j + 1 == size) as u32;
                    let h = hint(
                        cx::LEVEL_SUFFIX + (size as usize).min(7) * 2 + last as usize,
                        kind::LEVEL_SUFFIX,
                        j | size << 5 | prefix.min(31) << 10 | suffix_len << 15,
                        (v & 0xFF) | j << 8,
                        tc as u32 | (i as u32) << 5 | last << 9,
                        qb | cls << 4,
                    );
                    v = v << 1 | io.bit(&h)?;
                }
                code += v as i64;
            }
            if prefix >= 15 && suffix_len == 0 {
                code += 15;
            }
            if prefix >= 16 {
                code += (1i64 << (prefix - 3)) - 4096;
            }
            if first == 1 {
                code += 2;
            }
            let mag = ((code + 2) >> 1) as u32;
            levels[i] = mag;
            prev_abs = mag;
            if suffix_len == 0 {
                suffix_len = 1;
            }
            if mag > (3 << (suffix_len - 1)) && suffix_len < 6 {
                suffix_len += 1;
            }
        }
        // total_zeros
        let mut zl = 0usize;
        if tc < max {
            let (id, tzi) = if cat == CAT_CDC { (VlcId::TzDc(tc - 1), 15 + tc - 1) } else { (VlcId::Tz(tc - 1), tc - 1) };
            let maxabs = levels[..tc].iter().copied().max().unwrap_or(0);
            let colo_tc = colo.min(17);
            let h = hint(
                0,
                kind::TOTAL_ZEROS,
                (tzi as u32) << 6 | (cat as u32) << 11,
                ncb | bucket(maxabs) << 5 | (t1 as u32) << 9,
                (tzi as u32) << 6 | (if idx < 24 { self.nb_avg(idx, |m| &m.last, 17) } else { 17 }) << 11,
                qb | cls << 4 | colo_tc << 7,
            );
            let tz = vlc(io, id, (max - tc) as u32, h, cx::TOTAL_ZEROS + tzi * 4, 3)? as usize;
            if tz + tc > max {
                return Err(Bad);
            }
            zl = tz;
        }
        let total_zeros = zl as u32;
        if idx < 24 {
            self.cur.last[idx] = (tc + zl) as u8;
            self.cur.mx[idx] = levels[..tc].iter().copied().max().unwrap_or(0).min(255) as u8;
        }
        // run_before
        let mut prev_run = 15u32;
        for (i, &lv) in levels.iter().enumerate().take(tc - 1) {
            if zl == 0 {
                break;
            }
            let ri = zl.min(7) - 1;
            let lb = bucket(lv);
            let pos = (tc - 1 - i + zl) as u32;
            let h = hint(
                0,
                kind::RUN,
                (zl.min(15) as u32) << 4 | (cat as u32) << 12,
                i as u32 | (tc as u32) << 4 | total_zeros << 9 | lb << 14,
                (zl.min(15) as u32) << 4 | prev_run << 8 | lb << 12,
                qb | cls << 4 | pos << 8,
            );
            let rb = vlc(io, VlcId::Run(ri), zl as u32, h, cx::RUN + ri * 8, 7)? as usize;
            if rb > zl {
                return Err(Bad);
            }
            zl -= rb;
            prev_run = rb as u32;
        }
        Ok(tc as u8)
    }
}

/// Synthetic CAVLC slice data for tests: drives [`SliceParser`] with random but valid syntax
/// and writes the bits it reads.
#[doc(hidden)]
pub mod synth {
    use super::{codes, BitIo, Elem};
    use crate::bits::{BitWriter, R};
    use crate::h264::mb::Hint;
    use std::collections::VecDeque;

    pub struct Rng(pub u64);

    impl Rng {
        pub fn step(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }
        pub fn below(&mut self, n: u32) -> u32 {
            self.step() % n
        }
    }

    /// Writes random valid slice data into `w` (which already holds the slice header).
    pub struct GenIo<'a> {
        pub rng: &'a mut Rng,
        pub w: &'a mut BitWriter,
        /// Probability (of 1000) that a macroblock is skipped, and that the slice goes on
        /// after a macroblock.
        pub skip: u32,
        pub go_on: u32,
        /// Probability (of 1000) of an I_PCM macroblock when a macroblock type is chosen.
        pub pcm: u32,
        queue: VecDeque<u32>,
    }

    impl<'a> GenIo<'a> {
        pub fn new(rng: &'a mut Rng, w: &'a mut BitWriter) -> Self {
            GenIo { rng, w, skip: 300, go_on: 990, pcm: 0, queue: VecDeque::new() }
        }

        fn push_bits(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                self.queue.push_back((v >> i) & 1);
            }
        }

        fn push_ue(&mut self, v: u32) {
            let x = v as u64 + 1;
            let n = 64 - x.leading_zeros();
            for _ in 0..n - 1 {
                self.queue.push_back(0);
            }
            self.push_bits(x as u32, n);
        }

        /// A small random Exp-Golomb code number up to `max` (shorter codes more likely).
        fn small_ue(&mut self, max: u32) -> u32 {
            loop {
                let mut zeros = 0;
                while zeros < 8 && self.rng.below(100) < 45 {
                    zeros += 1;
                }
                let v = (1u32 << zeros) - 1 + self.rng.below(1 << zeros);
                if v <= max {
                    return v;
                }
            }
        }
    }

    impl BitIo for GenIo<'_> {
        fn bit(&mut self, _h: &Hint) -> R<u32> {
            let b = self.queue.pop_front().unwrap_or_else(|| self.rng.step() & 1);
            self.w.put(b);
            Ok(b)
        }

        fn more(&mut self, _h: &Hint, at_end: bool) -> R<bool> {
            Ok(!at_end && self.rng.below(1000) < self.go_on)
        }

        fn peek_ue(&mut self) -> R<Option<u32>> {
            Ok(None)
        }

        fn decision(&mut self, _h: &Hint, _v: u32) -> R<u32> {
            Ok((self.rng.below(1000) < self.skip) as u32)
        }

        fn put_ue(&mut self, v: u32) {
            let x = v as u64 + 1;
            let n = 64 - x.leading_zeros();
            self.w.put_bits(0, n - 1);
            for i in (0..n).rev() {
                self.w.put(((x >> i) & 1) as u32);
            }
        }

        fn pos(&self) -> usize {
            self.w.bit_len()
        }

        fn begin(&mut self, e: Elem) {
            debug_assert!(self.queue.is_empty());
            match e {
                Elem::Ue(max) => {
                    let v = self.small_ue(max);
                    self.push_ue(v);
                }
                Elem::MbType(max) => {
                    let v = if self.rng.below(1000) < self.pcm { max } else { self.small_ue(max - 1) };
                    self.push_ue(v);
                }
                Elem::Vlc(id, max) => {
                    let token = matches!(id, super::VlcId::Token(_));
                    let ok: Vec<(u32, u8, u32)> =
                        codes(id).into_iter().filter(|&(_, _, v)| if token { v >> 2 <= max } else { v <= max }).collect();
                    // shorter codes are likelier, as random bits would make them
                    let total: u64 = ok.iter().map(|&(_, l, _)| 1u64 << (16 - l)).sum();
                    let mut x = (self.rng.step() as u64 * 65_537) % total;
                    let mut pick = ok[0];
                    for &c in &ok {
                        let wgt = 1u64 << (16 - c.1);
                        if x < wgt {
                            pick = c;
                            break;
                        }
                        x -= wgt;
                    }
                    self.push_bits(pick.0, pick.1 as u32);
                }
                Elem::Prefix => {
                    // mostly small levels, sometimes the escapes (prefix 14, 15, and 16 as
                    // High profiles allow)
                    let mut p = 0;
                    if self.rng.below(100) < 3 {
                        p = 14 + self.rng.below(3);
                    } else {
                        while p < 13 && self.rng.below(100) < 40 {
                            p += 1;
                        }
                    }
                    for _ in 0..p {
                        self.queue.push_back(0);
                    }
                    self.queue.push_back(1);
                }
                Elem::Bits(n) => {
                    for _ in 0..n {
                        let b = self.rng.step() & 1;
                        self.queue.push_back(b);
                    }
                }
                Elem::Zeros(n) => {
                    for _ in 0..n {
                        self.queue.push_back(0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_tables_are_prefix_codes() {
        let mut ids = vec![];
        ids.extend((0..5).map(VlcId::Token));
        ids.extend((0..15).map(VlcId::Tz));
        ids.extend((0..3).map(VlcId::TzDc));
        ids.extend((0..7).map(VlcId::Run));
        for id in ids {
            let c = codes(id);
            let kraft: f64 = c.iter().map(|&(_, l, _)| 0.5f64.powi(l as i32)).sum();
            assert!(kraft <= 1.0 + 1e-12, "{id:?}: Kraft sum {kraft}");
            for (i, a) in c.iter().enumerate() {
                for b in &c[i + 1..] {
                    let (s, l) = if a.1 <= b.1 { (a, b) } else { (b, a) };
                    assert_ne!(l.0 >> (l.1 - s.1), s.0, "{id:?}: {s:?} is a prefix of {l:?}");
                }
            }
        }
        // total_zeros tables cover every count of zeros a block can have
        for tc in 1..16 {
            assert_eq!(codes(VlcId::Tz(tc - 1)).len(), 17 - tc);
        }
        assert_eq!(codes(VlcId::Token(0)).len(), 62);
        assert_eq!(codes(VlcId::Token(4)).len(), 14);
    }

    #[test]
    fn cbp_maps_are_permutations() {
        for t in [INTRA_CBP, INTER_CBP] {
            let mut seen = [false; 48];
            for v in t {
                let k = ((v >> 4) * 16 + (v & 15)) as usize;
                assert!(k < 48 && !seen[k]);
                seen[k] = true;
            }
        }
    }
}
