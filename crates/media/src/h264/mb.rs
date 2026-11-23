//! CABAC macroblock-layer parser (7.3.4, 7.3.5, 9.3) for progressive 4:2:0 8-bit slices.
//!
//! The parser computes every bin's context exactly as the standard does, but it does not
//! read bits itself: it asks a [`BinIo`] for each bin. During compression the `BinIo`
//! decodes the original CABAC payload; during decompression it decodes bins from
//! PiedPiper's own model and re-encodes them with CABAC. Either way the parser state, and
//! therefore every context, is identical on both sides.
//!
//! Each bin also carries a [`Hint`]: extra context (neighbour coefficient counts, the
//! co-located macroblock of the previous picture, positions, ...) that the standard does
//! not use but that lets PiedPiper's model predict the bin better than CABAC does.

use super::cabac::{init_states, next_state, NUM_CTX};
use super::recon::{self, Edge, Motion};
use super::tables::{LAST_8X8, SIG_8X8};
use super::{Pps, SliceHeader, Sps, B, I};
use crate::bits::{Bad, R};

/// Syntax-element classes, used to select model contexts.
pub mod kind {
    pub const SKIP: u8 = 0;
    pub const MB_TYPE: u8 = 1;
    pub const SUB_TYPE: u8 = 2;
    pub const MVD: u8 = 3;
    pub const REF: u8 = 4;
    pub const DQP: u8 = 5;
    pub const IPRED: u8 = 6;
    pub const CPRED: u8 = 7;
    pub const CBP_L: u8 = 8;
    pub const CBP_C: u8 = 9;
    pub const T8: u8 = 10;
    pub const CBF: u8 = 11;
    pub const SIG: u8 = 12;
    pub const LAST: u8 = 13;
    pub const ABS: u8 = 14;
    pub const SIGN: u8 = 15;
    pub const ABS_SUF: u8 = 16;
    pub const MVD_SUF: u8 = 17;
    pub const MVD_SIGN: u8 = 18;
    pub const END: u8 = 19;
    pub const PCM: u8 = 20;
    pub const COUNT: usize = 21;
}

/// Context for one bin, beyond the CABAC state.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Hint {
    /// CABAC ctxIdx (0 for bypass bins, 276 for terminating bins).
    pub ctx: u16,
    pub kind: u8,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
}

/// Signed coefficient levels of a macroblock's blocks, in scan order (clamped to i8).
/// Used as context for the neighbouring blocks' coefficients.
#[derive(Clone, Copy)]
pub struct Levels {
    /// Luma 4x4 blocks, raster order; I16x16 AC levels sit at their scan positions 1..15.
    pub l4: [[i8; 16]; 16],
    /// Luma 8x8 blocks.
    pub l8: [[i8; 64]; 4],
    /// Chroma AC blocks (Cb 0..3, Cr 0..3), at scan positions 1..15.
    pub c: [[i8; 16]; 8],
    /// Intra 16x16 luma DC.
    pub dc: [i8; 16],
    /// Chroma DC (Cb, Cr).
    pub cdc: [[i8; 4]; 2],
}

impl Default for Levels {
    fn default() -> Self {
        Levels { l4: [[0; 16]; 16], l8: [[0; 64]; 4], c: [[0; 16]; 8], dc: [0; 16], cdc: [[0; 4]; 2] }
    }
}

#[inline]
fn lvq(v: i8, max: u32) -> u32 {
    (v.unsigned_abs() as u32).min(max)
}

#[inline]
fn sgn32(v: i32) -> u32 {
    match v.signum() {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

/// Contexts of a motion vector difference component (see `SliceParser::mvd_ctx`).
#[derive(Clone, Copy)]
struct MvdCtx {
    /// The first prefix bin.
    first: Hint,
    /// ctxIdxOffset of the component.
    base: usize,
    /// Suffix hint.
    hs: u32,
    /// Sign hint fields (without the x component's sign).
    sign_c: u32,
    sign_d: u32,
}

#[inline]
fn sgn(v: i8) -> u32 {
    match v.signum() {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

/// Source/sink of bins.
pub trait BinIo {
    /// A context-coded bin; `state` is the CABAC context state before the bin.
    fn decision(&mut self, state: u8, h: &Hint) -> R<u32>;
    fn bypass(&mut self, h: &Hint) -> R<u32>;
    fn terminate(&mut self, h: &Hint) -> R<u32>;
    /// A bin with hint `h` may come soon (the parser announces the possible next bins
    /// before it asks for the current one). Only a performance hint.
    #[inline]
    fn prefetch(&mut self, _h: &Hint) {}
    /// Ends a CABAC substream and starts the next one (HEVC tiles and wavefront rows): the
    /// terminating end_of_subset_one_bit, which is always 1 and therefore not modelled,
    /// byte_alignment(), and a re-initialised arithmetic coder.
    fn end_substream(&mut self) -> R<()> {
        Err(Bad)
    }
}

/// Everything later macroblocks need to know about a decoded macroblock.
#[derive(Clone, Copy)]
pub struct MbInfo {
    pub slice: u32,
    pub skip: bool,
    pub intra: bool,
    pub i_nxn: bool,
    pub i16: bool,
    /// B_Skip or B_Direct_16x16.
    pub direct16: bool,
    /// Luma CBP in bits 0..3, chroma CBP (0..2) in bits 4..5.
    pub cbp: u8,
    pub cpred_nz: bool,
    pub t8: bool,
    /// coded_block_flag bits: 0..15 luma 4x4 blocks (by blkIdx), 16 luma DC,
    /// 17..18 chroma DC, 19..26 chroma AC (Cb 0..3, Cr 0..3).
    pub cbf: u32,
    /// 8x8 partitions predicted in direct mode.
    pub direct8: u8,
    pub refi: [[i8; 4]; 2],
    /// |mvd| per 4x4 block (raster order) and list.
    pub mvd: [[[u16; 2]; 16]; 2],
    /// Signed mvd per 4x4 block and list (model context only).
    pub mvs: [[[i16; 2]; 16]; 2],
    /// Non-zero coefficients per luma 4x4 block (raster order).
    pub nnz: [u8; 16],
    /// Non-zero coefficients per chroma AC block (Cb 0..3, Cr 0..3).
    pub nnzc: [u8; 8],
    pub class: u8,
    /// Motion vectors per 4x4 block and list, and reference indices per 8x8 block, as used
    /// for prediction (also for skipped and direct blocks; reconstruction only).
    pub mv: [[[i16; 2]; 16]; 2],
    pub rf: [[i8; 4]; 2],
    /// Intra 4x4/8x8 prediction mode per 4x4 block (reconstruction only).
    pub ipm: [u8; 16],
}

impl Default for MbInfo {
    fn default() -> Self {
        MbInfo {
            slice: 0,
            skip: false,
            intra: false,
            i_nxn: false,
            i16: false,
            direct16: false,
            cbp: 0,
            cpred_nz: false,
            t8: false,
            cbf: 0,
            direct8: 0,
            refi: [[-1; 4]; 2],
            mvd: [[[0; 2]; 16]; 2],
            mvs: [[[0; 2]; 16]; 2],
            nnz: [0; 16],
            nnzc: [0; 8],
            class: 0,
            mv: [[[0; 2]; 16]; 2],
            rf: [[-1; 4]; 2],
            ipm: [2; 16],
        }
    }
}

/// What the model remembers about each macroblock of the previous picture.
#[derive(Clone, Copy, Default)]
pub struct Summary {
    pub class: u8,
    pub cbp: u8,
    pub nnz: [u8; 16],
    pub mvd: [u8; 2],
}

/// Per-picture state shared by the slices of a stream.
///
/// `mbs` is scratch space for neighbour lookups inside a slice: entries carry the id of the
/// slice that wrote them and are only read back by that slice, so a slice that fails to
/// parse leaves nothing behind that matters. The per-picture summaries the model uses
/// (`cur`, `prev`) only change in [`SliceParser::commit`], i.e. for slices that were
/// actually recoded, so the compressor and decompressor always see the same history.
#[derive(Default)]
pub struct Picture {
    pub w: usize,
    pub h: usize,
    pub mbs: Vec<MbInfo>,
    /// Coefficient levels, parallel to `mbs` (same scratch semantics).
    pub lv: Vec<Levels>,
    pub cur: Vec<Summary>,
    pub prev: Vec<Summary>,
    next_slice: u32,
    /// Reconstructed pictures (model context), when enabled.
    pub rec: Option<Box<recon::Recon>>,
}

static EMPTY: Summary = Summary { class: 0, cbp: 0, nnz: [0; 16], mvd: [0; 2] };

#[derive(Clone, Copy, Default)]
struct MbT {
    intra: bool,
    i_nxn: bool,
    i16: bool,
    /// Intra16x16 prediction mode.
    i16_mode: u8,
    pcm: bool,
    i16_cbp: u8,
    direct16: bool,
    /// 0 = 16x16, 1 = 16x8, 2 = 8x16, 3 = 8x8.
    shape: u8,
    /// Lists used by each partition: 1 = L0, 2 = L1, 3 = both.
    pred: [u8; 2],
}

#[derive(Clone, Copy, Default)]
struct SubT {
    direct: bool,
    mode: u8,
    /// 0 = 8x8, 1 = 8x4, 2 = 4x8, 3 = 4x4.
    shape: u8,
}

const B_PAIRS: [(u8, u8); 9] = [(1, 1), (2, 2), (1, 2), (2, 1), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3)];
const B_SUB: [(u8, u8); 13] =
    [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (1, 3), (2, 3), (3, 3)];
const SUB_RECTS: [&[(usize, usize, usize, usize)]; 4] = [
    &[(0, 0, 2, 2)],
    &[(0, 0, 2, 1), (0, 1, 2, 1)],
    &[(0, 0, 1, 2), (1, 0, 1, 2)],
    &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
];
const PART_RECTS: [&[(usize, usize, usize, usize)]; 3] =
    [&[(0, 0, 4, 4)], &[(0, 0, 4, 2), (0, 2, 4, 2)], &[(0, 0, 2, 4), (2, 0, 2, 4)]];
const CBF_OFF: [usize; 5] = [0, 4, 8, 12, 16];
const SIG_OFF: [usize; 5] = [0, 15, 29, 44, 47];
const ABS_OFF: [usize; 5] = [0, 10, 20, 30, 39];

/// (x, y) in 4x4 units of luma4x4BlkIdx.
#[inline]
fn blk_xy(blk: usize) -> (usize, usize) {
    (((blk >> 2) & 1) * 2 + (blk & 1), ((blk >> 3) & 1) * 2 + ((blk >> 1) & 1))
}

#[inline]
fn xy_blk(x: usize, y: usize) -> usize {
    (y / 2) * 8 + (x / 2) * 4 + (y % 2) * 2 + (x % 2)
}

fn intra_t(v: u32) -> MbT {
    match v {
        0 => MbT { intra: true, i_nxn: true, ..Default::default() },
        25 => MbT { intra: true, pcm: true, ..Default::default() },
        _ => {
            let k = v - 1;
            let chroma = ((k / 4) % 3) as u8;
            let luma = if v >= 13 { 15 } else { 0 };
            MbT { intra: true, i16: true, i16_cbp: luma | (chroma << 4), i16_mode: (k % 4) as u8, ..Default::default() }
        }
    }
}

fn b_type(v: u32) -> MbT {
    match v {
        1..=3 => MbT { shape: 0, pred: [v as u8, 0], ..Default::default() },
        4..=21 => {
            let (m0, m1) = B_PAIRS[((v - 4) / 2) as usize];
            MbT { shape: if v & 1 == 0 { 1 } else { 2 }, pred: [m0, m1], ..Default::default() }
        }
        _ => MbT { shape: 3, ..Default::default() },
    }
}

#[derive(Clone, Copy)]
enum Nb {
    None,
    Cur,
    Mb(usize),
}

#[inline]
fn q(v: u32, max: u32) -> u32 {
    v.min(max)
}

/// Hint of significant_coeff_flag at scan position `i` with `n` coefficients found so far.
/// `la`/`lb` are the left/above blocks' levels at `i`, `ra`/`rb` their non-zero counts at or
/// after `i`, `nbh` the block's neighbourhood estimate and `ex` [`SliceParser::extra`].
#[allow(clippy::too_many_arguments)]
#[inline]
fn sig_hint(cat: usize, i: usize, n: usize, nbh: u32, ex: u32, la: i8, lb: i8, ra: u8, rb: u8) -> Hint {
    let sig = if cat == 5 { 402 } else { 105 + SIG_OFF[cat] };
    let si = if cat == 5 {
        SIG_8X8[i] as usize
    } else if cat == 3 {
        i.min(2)
    } else {
        i
    };
    let nbi = lvq(la, 3) | lvq(lb, 3) << 2;
    Hint {
        ctx: (sig + si) as u16,
        kind: kind::SIG,
        a: (cat as u32) << 8 | i as u32,
        b: nbh | q(n as u32, 15) << 12,
        c: nbi | ((ra > 0) as u32) << 4 | ((rb > 0) as u32) << 5,
        d: q(ra as u32 + rb as u32, 15) | q(n as u32, 15) << 4 | ex << 8,
    }
}

/// Parses one slice. Create it with [`SliceParser::new`], then call [`SliceParser::run`].
pub struct SliceParser<'a> {
    sps: &'a Sps,
    pps: &'a Pps,
    sh: &'a SliceHeader,
    pic: &'a mut Picture,
    st: [u8; NUM_CTX],
    sid: u32,
    cur: MbInfo,
    curlv: Levels,
    addr: usize,
    a: Option<usize>,
    b: Option<usize>,
    last_dqp_nz: bool,
    last_dqp: u32,
    /// QP of the current macroblock (8-bit video: 0..=51).
    qp: i32,
    first: usize,
    /// This slice starts a new picture, so the picture now in `cur` is the previous one.
    newpic: bool,
    /// The summaries have the wrong size (first slice, or a resolution change).
    fresh: bool,
    pending: Vec<Summary>,
    /// Number of macroblocks parsed.
    pub count: usize,
    rc: Box<RecState>,
}

/// Per-macroblock reconstruction state.
struct RecState {
    on: bool,
    mbx: usize,
    mby: usize,
    /// Prediction: luma 16x16, then Cb and Cr 8x8 (raster).
    pred: [i32; 384],
    ipsyn: [(bool, u8); 16],
    cpm: u32,
    i16m: u32,
    /// Exact levels of the block just parsed (scan order).
    cx: [i32; 64],
    l4: [[i32; 16]; 16],
    l8: [[i32; 64]; 4],
    dc: [i32; 16],
    cdc: [[i32; 4]; 2],
    cac: [[i32; 16]; 8],
    /// Prediction features of the next block: per scan position, whole block, and what
    /// remains after each scan position.
    feat: [u8; 64],
    ftex: u32,
    frem: [u8; 65],
    luma_done: bool,
    /// The chroma samples are already written (static skipped macroblocks).
    chroma_done: bool,
    /// Boundary mismatch of the next block, x128: top row 0..4, left column 4..8
    /// (prediction minus the neighbouring reconstructed samples), and which sides exist.
    bnd: [i32; 16],
    bnd_ok: u8,
    /// QP of the block's dequantisation (boundary features).
    bqp: i32,
    /// Difference of the two predictions of bi-predicted luma samples (0 elsewhere).
    dis: [i32; 256],
    fdis: [u8; 16],
    dtex: u32,
    /// Log2 variance (x2) of the macroblock's prediction, and of the last macroblock that
    /// coded mb_qp_delta (0 = unknown).
    lv_last: u32,
    mc: recon::McScratch,
    pl: [[i32; 256]; 2],
    pc: [[[i32; 64]; 2]; 2],
}

impl Default for RecState {
    fn default() -> Self {
        RecState {
            on: false,
            mbx: 0,
            mby: 0,
            pred: [128; 384],
            ipsyn: [(true, 0); 16],
            cpm: 0,
            i16m: 0,
            cx: [0; 64],
            l4: [[0; 16]; 16],
            l8: [[0; 64]; 4],
            dc: [0; 16],
            cdc: [[0; 4]; 2],
            cac: [[0; 16]; 8],
            feat: [0; 64],
            ftex: 0,
            frem: [0; 65],
            luma_done: false,
            chroma_done: false,
            bnd: [0; 16],
            bnd_ok: 0,
            bqp: 0,
            dis: [0; 256],
            fdis: [0; 16],
            dtex: 0,
            lv_last: 0,
            mc: recon::McScratch::default(),
            pl: [[0; 256]; 2],
            pc: [[[0; 64]; 2]; 2],
        }
    }
}

/// 2x the 4x4 inverse transform basis: CI2[x][u] (sample x, frequency u).
const CI2: [[i32; 4]; 4] = [[2, 2, 2, 1], [2, 1, -2, -2], [2, -1, -2, 2], [2, -2, 2, -1]];

/// Dequantised value of level `c` at raster position `r` of a 4x4 block (flat matrix).
#[inline]
fn deq1(c: i32, qp: i32, r: usize) -> i32 {
    let (m, e) = ((qp % 6) as usize, qp / 6);
    let v = c.wrapping_mul(recon::level_scale4(m, r));
    if e >= 4 {
        v.wrapping_shl((e - 4) as u32)
    } else {
        v.wrapping_add(1 << (3 - e)) >> (4 - e)
    }
}

/// Projection of the boundary mismatch `e` (top row at 0.., left column at 8..) on the
/// basis function of raster position `r` of a 4x4 block.
#[inline]
fn bproj(e: &[i32; 16], ok: u8, r: usize) -> i64 {
    let (u, v) = (r & 3, r >> 2);
    let mut p = 0i64;
    if ok & 1 != 0 {
        for x in 0..4 {
            p += e[x] as i64 * CI2[x][u] as i64;
        }
    }
    if ok & 2 != 0 {
        for y in 0..4 {
            p += e[8 + y] as i64 * CI2[y][v] as i64;
        }
    }
    p
}

/// The same for an 8x8 block (mismatch scaled by 4096, basis C8[x][u] * C8[0][v]).
#[inline]
fn bproj8(e: &[i32; 16], ok: u8, r: usize) -> i64 {
    let (u, v) = (r & 7, r >> 3);
    let mut p = 0i64;
    if ok & 1 != 0 {
        let c0v = recon::C8[0][v] as i64;
        for x in 0..8 {
            p += e[x] as i64 * recon::C8[x][u] as i64 * c0v;
        }
    }
    if ok & 2 != 0 {
        let c0u = recon::C8[0][u] as i64;
        for y in 0..8 {
            p += e[8 + y] as i64 * c0u * recon::C8[y][v] as i64;
        }
    }
    p
}

/// Dequantised value of level `c` at raster position `r` of an 8x8 block (flat matrix).
#[inline]
fn deq8(c: i32, qp: i32, r: usize) -> i32 {
    let (m, e) = ((qp % 6) as usize, qp / 6);
    let v = c.wrapping_mul(recon::level_scale8(m, r));
    if e >= 6 {
        v.wrapping_shl((e - 6) as u32)
    } else {
        v.wrapping_add(1 << (5 - e)) >> (6 - e)
    }
}


#[inline]
fn bucket(v: u32) -> u8 {
    if v < 4 {
        0
    } else {
        (30 - v.leading_zeros()).min(7) as u8
    }
}

#[inline]
fn med3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

#[inline]
fn minpos(a: i8, b: i8) -> i8 {
    if a >= 0 && b >= 0 {
        a.min(b)
    } else {
        a.max(b)
    }
}

impl<'a> SliceParser<'a> {
    pub fn new(sps: &'a Sps, pps: &'a Pps, sh: &'a SliceHeader, pic: &'a mut Picture) -> R<Self> {
        let w = sps.width_mbs as usize;
        let h = sps.height_mbs() as usize;
        if sh.first_mb as usize >= w * h {
            return Err(Bad);
        }
        if pic.w != w || pic.h != h || pic.mbs.len() != w * h {
            pic.w = w;
            pic.h = h;
            pic.mbs = vec![MbInfo::default(); w * h];
            pic.lv = vec![Levels::default(); w * h];
        }
        let fresh = pic.cur.len() != w * h || pic.prev.len() != w * h;
        let model = if sh.slice_type == I { 0 } else { 1 + sh.cabac_init_idc as usize };
        let mut st = [0u8; NUM_CTX];
        init_states(&mut st, sh.slice_qp, model);
        pic.next_slice = pic.next_slice.wrapping_add(1).max(1);
        let sid = pic.next_slice;
        let pic_has_rec = pic.rec.as_ref().is_some_and(|r| r.have_cur && r.cur.y.w == w * 16 && r.cur.y.h == h * 16);
        Ok(SliceParser {
            sps,
            pps,
            sh,
            pic,
            st,
            sid,
            cur: MbInfo::default(),
            curlv: Levels::default(),
            addr: 0,
            a: None,
            b: None,
            last_dqp_nz: false,
            last_dqp: 0,
            qp: sh.slice_qp,
            first: sh.first_mb as usize,
            newpic: sh.first_mb == 0,
            fresh,
            pending: Vec::new(),
            count: 0,
            rc: Box::new(RecState { on: pic_has_rec, ..Default::default() }),
        })
    }

    /// The co-located macroblock of the previous picture.
    #[inline]
    fn colo(&self, addr: usize) -> &Summary {
        if self.fresh {
            &EMPTY
        } else if self.newpic {
            &self.pic.cur[addr]
        } else {
            &self.pic.prev[addr]
        }
    }

    /// Makes this slice part of the history the model sees. Call only for slices that were
    /// recoded, in stream order, on both the compressing and the decompressing side.
    pub fn commit(self) {
        let n = self.pic.w * self.pic.h;
        if self.rc.on {
            if let Some(rec) = self.pic.rec.as_mut() {
                for m in self.first..(self.first + self.count).min(rec.covered.len()) {
                    rec.covered[m] = true;
                }
            }
        }
        if self.fresh {
            self.pic.cur = vec![Summary::default(); n];
            self.pic.prev = vec![Summary::default(); n];
        }
        if self.newpic {
            std::mem::swap(&mut self.pic.cur, &mut self.pic.prev);
            self.pic.cur.iter_mut().for_each(|s| *s = Summary::default());
        }
        for (i, s) in self.pending.into_iter().enumerate() {
            self.pic.cur[self.first + i] = s;
        }
    }

    #[inline]
    fn dec<T: BinIo>(&mut self, io: &mut T, ctx: usize, kind: u8, a: u32, b: u32) -> R<u32> {
        let s = self.st[ctx];
        let bin = io.decision(s, &Hint { ctx: ctx as u16, kind, a, b, c: 0, d: 0 })?;
        self.st[ctx] = next_state(s, bin);
        Ok(bin)
    }


    pub fn run<T: BinIo>(&mut self, io: &mut T) -> R<()> {
        let n = self.pic.w * self.pic.h;
        let is_i = self.sh.slice_type == I;
        let is_b = self.sh.slice_type == B;
        let mut addr = self.first;
        loop {
            if addr >= n {
                return Err(Bad);
            }
            self.begin(addr);
            let mut skip = false;
            if !is_i {
                let h = self.skip_hint(addr, self.a, self.b, is_b);
                io.prefetch(&self.mb_type_hint(is_b));
                skip = self.dech(io, &h)? == 1;
            }
            if skip {
                self.skip_mb(is_b);
            } else {
                self.macroblock(io)?;
            }
            self.finish_mb();
            let last = (addr + 1 == n) as u32;
            if addr + 1 < n {
                let (a, b) = self.nbrs(addr + 1);
                let h = if is_i { self.mb_type_i_hint(addr + 1, a, b) } else { self.skip_hint(addr + 1, a, b, is_b) };
                io.prefetch(&h);
            }
            if io.terminate(&Hint { ctx: 276, kind: kind::END, a: last, b: 0, c: 0, d: 0 })? == 1 {
                break;
            }
            addr += 1;
        }
        self.count = addr + 1 - self.first;
        Ok(())
    }

    /// Neighbours (left, above) of macroblock `addr` inside this slice.
    #[inline]
    fn nbrs(&self, addr: usize) -> (Option<usize>, Option<usize>) {
        let w = self.pic.w;
        let col = addr % w;
        (
            (col > 0 && self.pic.mbs[addr - 1].slice == self.sid).then(|| addr - 1),
            (addr >= w && self.pic.mbs[addr - w].slice == self.sid).then(|| addr - w),
        )
    }

    /// Hint of the mb_skip_flag of macroblock `addr` (P and B slices).
    fn skip_hint(&self, addr: usize, a: Option<usize>, b: Option<usize>, is_b: bool) -> Hint {
        let n = self.pic.w * self.pic.h;
        let ca = a.map_or(0, |i| !self.pic.mbs[i].skip as usize);
        let cb = b.map_or(0, |i| !self.pic.mbs[i].skip as usize);
        let base = if is_b { 24 } else { 11 };
        let colo = self.colo(addr).class as u32;
        let right = if addr + 1 < n { self.colo(addr + 1).class as u32 } else { 8 };
        let ha = a.map_or(8, |i| self.pic.mbs[i].class as u32);
        let hb = b.map_or(8, |i| self.pic.mbs[i].class as u32);
        Hint { ctx: (base + ca + cb) as u16, kind: kind::SKIP, a: ha | hb << 4, b: colo | right << 4, c: 0, d: 0 }
    }

    /// Hint of the first mb_type bin of the current macroblock in a P or B slice.
    fn mb_type_hint(&self, is_b: bool) -> Hint {
        let h = self.mb_class_hint();
        let ctx = if is_b {
            let ca = self.a.map_or(0, |i| !self.pic.mbs[i].direct16 as usize);
            let cb = self.b.map_or(0, |i| !self.pic.mbs[i].direct16 as usize);
            27 + ca + cb
        } else {
            14
        };
        Hint { ctx: ctx as u16, kind: kind::MB_TYPE, a: 0, b: h, c: 0, d: 0 }
    }

    /// Hint of the first mb_type bin of macroblock `addr` in an I slice.
    fn mb_type_i_hint(&self, addr: usize, a: Option<usize>, b: Option<usize>) -> Hint {
        let ca = a.map_or(0, |i| !self.pic.mbs[i].i_nxn as usize);
        let cb = b.map_or(0, |i| !self.pic.mbs[i].i_nxn as usize);
        let ha = a.map_or(8, |i| self.pic.mbs[i].class as u32);
        let hb = b.map_or(8, |i| self.pic.mbs[i].class as u32);
        let h = ha | hb << 4 | (self.colo(addr).class as u32) << 8;
        Hint { ctx: (3 + ca + cb) as u16, kind: kind::MB_TYPE, a: 16, b: h, c: 0, d: 0 }
    }

    fn begin(&mut self, addr: usize) {
        let w = self.pic.w;
        self.addr = addr;
        let col = addr % w;
        self.a = (col > 0 && self.pic.mbs[addr - 1].slice == self.sid).then(|| addr - 1);
        self.b = (addr >= w && self.pic.mbs[addr - w].slice == self.sid).then(|| addr - w);
        self.cur = MbInfo { slice: self.sid, ..Default::default() };
        self.curlv = Levels::default();
        if self.rc.on {
            let rc = &mut *self.rc;
            rc.mbx = col;
            rc.mby = addr / w;
            rc.luma_done = false;
            rc.chroma_done = false;
            rc.feat = [0; 64];
            rc.ftex = 0;
            rc.frem = [0; 65];
        }
    }

    fn skip_mb(&mut self, is_b: bool) {
        self.cur.skip = true;
        self.cur.class = 0;
        if is_b {
            self.cur.direct16 = true;
            self.cur.direct8 = 0xF;
        } else {
            self.cur.refi[0] = [0; 4];
        }
        self.last_dqp_nz = false;
        self.last_dqp = 0;
        if self.rc.on {
            if is_b {
                let (rf, mv) = self.direct_motion();
                self.cur.rf = rf;
                self.cur.mv = mv;
            } else {
                self.pskip_motion();
            }
            if !self.static_skip() {
                self.inter_pred();
            }
        }
    }

    /// Reconstruction of a skipped macroblock whose motion is zero everywhere (static
    /// content) straight from the reference pictures; false when it is not such a
    /// macroblock. Gives the same samples as [`Self::inter_pred`] plus [`Self::recon_finish`].
    fn static_skip(&mut self) -> bool {
        if self.sh.ri.explicit_wp {
            return false;
        }
        let rf = self.cur.rf;
        for l in 0..2 {
            if rf[l].iter().any(|&r| r != rf[l][0]) || (rf[l][0] >= 0 && self.cur.mv[l].iter().any(|&m| m != [0, 0])) {
                return false;
            }
        }
        let Some(rec) = self.pic.rec.as_mut() else { return false };
        let (w, h) = (rec.cur.y.w, rec.cur.y.h);
        let pick = |l: usize| -> Option<usize> {
            let r = rf[l][0];
            let i = *rec.list[l].get(usize::try_from(r).ok()?)?;
            (i != usize::MAX && rec.dpb[i].y.w == w && rec.dpb[i].y.h == h).then_some(i)
        };
        let (i0, i1) = (pick(0), pick(1));
        let weights = match (rf[0][0] >= 0, rf[1][0] >= 0, i0, i1) {
            (true, false, Some(_), _) | (false, true, _, Some(_)) => None,
            (true, true, Some(a), Some(b)) => {
                if self.pps.weighted_bipred_idc == 2 {
                    let (p0, p1) = (&rec.dpb[a], &rec.dpb[b]);
                    let tb = (rec.cur.poc - p0.poc).clamp(-128, 127);
                    let td = (p1.poc - p0.poc).clamp(-128, 127);
                    Some(if td == 0 || p0.long_term || p1.long_term {
                        (32, 32)
                    } else {
                        let tx = (16384 + (td / 2).abs()) / td;
                        let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
                        if !(-64..=128).contains(&(dsf >> 2)) {
                            (32, 32)
                        } else {
                            (64 - (dsf >> 2), dsf >> 2)
                        }
                    })
                } else {
                    Some((0, 0)) // plain average
                }
            }
            _ => return false,
        };
        let (mbx, mby) = (self.rc.mbx, self.rc.mby);
        let cur = std::mem::take(&mut rec.cur);
        let mut cur = cur;
        {
            let a = &rec.dpb[i0.or(i1).expect("one reference")];
            let b = i1.map(|i| &rec.dpb[i]);
            for (dst, pa, pb, n, x0, y0) in [
                (&mut cur.y, &a.y, b.map(|q| &q.y), 16usize, mbx * 16, mby * 16),
                (&mut cur.u, &a.u, b.map(|q| &q.u), 8, mbx * 8, mby * 8),
                (&mut cur.v, &a.v, b.map(|q| &q.v), 8, mbx * 8, mby * 8),
            ] {
                let st = dst.w;
                for j in 0..n {
                    let o = (y0 + j) * st + x0;
                    match (weights, pb) {
                        (Some((w0, w1)), Some(pb)) => {
                            let (ra, rb) = (&pa.d[o..o + n], &pb.d[o..o + n]);
                            for (i, d) in dst.d[o..o + n].iter_mut().enumerate() {
                                let (p, q) = (ra[i] as i32, rb[i] as i32);
                                *d = if w0 == 0 && w1 == 0 { ((p + q + 1) >> 1) as u8 } else { recon::clip1((p * w0 + q * w1 + 32) >> 6) };
                            }
                        }
                        _ => dst.d[o..o + n].copy_from_slice(&pa.d[o..o + n]),
                    }
                }
            }
        }
        rec.cur = cur;
        self.rc.luma_done = true;
        self.rc.chroma_done = true;
        true
    }

    fn finish_mb(&mut self) {
        if self.rc.on {
            self.recon_finish();
        }
        let c = self.cur;
        self.pic.mbs[self.addr] = c;
        self.pic.lv[self.addr] = self.curlv;
        let mx = c.mvd[0].iter().map(|v| v[0] as u32).max().unwrap_or(0).min(255) as u8;
        let my = c.mvd[0].iter().map(|v| v[1] as u32).max().unwrap_or(0).min(255) as u8;
        self.pending.push(Summary { class: c.class, cbp: c.cbp, nnz: c.nnz, mvd: [mx, my] });
    }

    fn nbmb(&self, nb: Nb) -> Option<&MbInfo> {
        match nb {
            Nb::None => None,
            Nb::Cur => Some(&self.cur),
            Nb::Mb(i) => Some(&self.pic.mbs[i]),
        }
    }

    /// Resolves a 4x4 luma block position relative to the current macroblock.
    fn locate(&self, x: i32, y: i32) -> (Nb, usize, usize) {
        if x >= 0 && y >= 0 {
            (Nb::Cur, x as usize, y as usize)
        } else if x < 0 {
            match self.a {
                Some(i) => (Nb::Mb(i), 3, y as usize),
                None => (Nb::None, 0, 0),
            }
        } else {
            match self.b {
                Some(i) => (Nb::Mb(i), x as usize, 3),
                None => (Nb::None, 0, 0),
            }
        }
    }

    fn mb_class_hint(&self) -> u32 {
        let ha = self.a.map_or(8, |i| self.pic.mbs[i].class as u32);
        let hb = self.b.map_or(8, |i| self.pic.mbs[i].class as u32);
        ha | hb << 4 | (self.colo(self.addr).class as u32) << 8
    }

    fn macroblock<T: BinIo>(&mut self, io: &mut T) -> R<()> {
        let t = self.mb_type(io)?;
        if t.pcm {
            return Err(Bad);
        }
        self.cur.intra = t.intra;
        self.cur.i_nxn = t.i_nxn;
        self.cur.i16 = t.i16;
        self.cur.direct16 = t.direct16;
        // 0 skip, 1 direct 16x16, 2..5 inter 16x16/16x8/8x16/8x8, 6 I16x16, 7 I_NxN.
        self.cur.class = if t.intra {
            if t.i16 {
                6
            } else {
                7
            }
        } else if t.direct16 {
            1
        } else {
            2 + t.shape
        };
        let is_b = self.sh.slice_type == B;
        let mut no_sub_lt8 = true;
        if !t.intra && t.shape == 3 {
            let mut sub = [SubT::default(); 4];
            for i in 0..4 {
                sub[i] = self.sub_mb_type(io, is_b, i as u32)?;
                if sub[i].direct {
                    self.cur.direct8 |= 1 << i;
                }
            }
            for list in 0..2 {
                let nref = if list == 0 { self.sh.num_ref_l0 } else { self.sh.num_ref_l1 };
                for i in 0..4 {
                    let uses = !sub[i].direct && sub[i].mode & (1 << list) != 0;
                    self.cur.refi[list][i] = if uses { 0 } else { -1 };
                }
                for i in 0..4 {
                    if nref > 1 && !sub[i].direct && sub[i].mode & (1 << list) != 0 {
                        let r = self.ref_idx(io, list, (i & 1) * 2, (i >> 1) * 2, nref)?;
                        self.cur.refi[list][i] = r.min(127) as i8;
                    }
                }
            }
            if self.rc.on {
                self.cur.rf = self.cur.refi;
                self.cur.mv = [[[0; 2]; 16]; 2];
                if sub.iter().any(|s| s.direct) {
                    let (rf, mv) = self.direct_motion();
                    for (i, s) in sub.iter().enumerate() {
                        if s.direct {
                            let (x0, y0) = ((i & 1) * 2, (i >> 1) * 2);
                            for l in 0..2 {
                                self.cur.rf[l][i] = rf[l][i];
                                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                                    let k = (y0 + dy) * 4 + x0 + dx;
                                    self.cur.mv[l][k] = mv[l][k];
                                }
                            }
                        }
                    }
                }
            }
            for list in 0..2 {
                for (i, s) in sub.iter().enumerate() {
                    if !s.direct && s.mode & (1 << list) != 0 {
                        let (x0, y0) = ((i & 1) * 2, (i >> 1) * 2);
                        for &(dx, dy, w, h) in SUB_RECTS[s.shape as usize] {
                            self.mvd(io, list, x0 + dx, y0 + dy, w, h, s.shape as u32)?;
                        }
                    }
                }
            }
            for s in &sub {
                if !s.direct {
                    if s.shape != 0 {
                        no_sub_lt8 = false;
                    }
                } else if !self.sps.direct_8x8_inference {
                    no_sub_lt8 = false;
                }
            }
            if self.rc.on {
                self.inter_pred();
            }
        } else {
            if self.pps.transform_8x8_mode && t.i_nxn {
                self.cur.t8 = self.t8flag(io)? == 1;
            }
            if t.intra {
                if t.i_nxn {
                    let n = if self.cur.t8 { 4 } else { 16 };
                    for k in 0..n {
                        let h = (self.cur.t8 as u32) << 5 | k;
                        let mut rem = 0u8;
                        let prev = self.dec(io, 68, kind::IPRED, h, 0)? == 1;
                        if !prev {
                            for j in 0..3 {
                                rem |= (self.dec(io, 69, kind::IPRED, h, 1 + j)? as u8) << j;
                            }
                        }
                        self.rc.ipsyn[k as usize] = (prev, rem);
                    }
                }
                let m = self.chroma_pred_mode(io)?;
                self.cur.cpred_nz = m != 0;
                self.rc.cpm = m;
                self.rc.i16m = t.i16_mode as u32;
            } else if !t.direct16 {
                let rects = PART_RECTS[t.shape as usize];
                for list in 0..2 {
                    let nref = if list == 0 { self.sh.num_ref_l0 } else { self.sh.num_ref_l1 };
                    for (p, &(x, y, w, h)) in rects.iter().enumerate() {
                        let v = if t.pred[p] & (1 << list) != 0 { 0 } else { -1 };
                        self.set_ref(list, x, y, w, h, v);
                    }
                    for (p, &(x, y, w, h)) in rects.iter().enumerate() {
                        if nref > 1 && t.pred[p] & (1 << list) != 0 {
                            let r = self.ref_idx(io, list, x, y, nref)?;
                            self.set_ref(list, x, y, w, h, r.min(127) as i8);
                        }
                    }
                }
                if self.rc.on {
                    self.cur.rf = self.cur.refi;
                    self.cur.mv = [[[0; 2]; 16]; 2];
                }
                for list in 0..2 {
                    for (p, &(x, y, w, h)) in rects.iter().enumerate() {
                        if t.pred[p] & (1 << list) != 0 {
                            self.mvd(io, list, x, y, w, h, 4 + t.shape as u32)?;
                        }
                    }
                }
            } else {
                self.cur.direct8 = 0xF;
            }
            if self.rc.on {
                if t.intra {
                    self.intra_pred_mb();
                } else {
                    if t.direct16 {
                        let (rf, mv) = self.direct_motion();
                        self.cur.rf = rf;
                        self.cur.mv = mv;
                    }
                    self.inter_pred();
                }
            }
        }
        if !t.i16 {
            let cbp = self.cbp(io)?;
            self.cur.cbp = cbp;
            if cbp & 15 != 0
                && self.pps.transform_8x8_mode
                && !t.i_nxn
                && no_sub_lt8
                && (!t.direct16 || self.sps.direct_8x8_inference)
            {
                self.cur.t8 = self.t8flag(io)? == 1;
            }
        } else {
            self.cur.cbp = t.i16_cbp;
        }
        let cbp = self.cur.cbp;
        if cbp & 15 != 0 || cbp >> 4 != 0 || t.i16 {
            self.dqp(io)?;
            self.residual(io)?;
        } else {
            self.last_dqp_nz = false;
            self.last_dqp = 0;
            if self.rc.on && self.cur.i_nxn {
                self.intra_nxn_luma_norez();
            }
        }
        Ok(())
    }

    fn set_ref(&mut self, list: usize, x: usize, y: usize, w: usize, h: usize, v: i8) {
        for yy in (y..y + h).step_by(2) {
            for xx in (x..x + w).step_by(2) {
                self.cur.refi[list][(yy / 2) * 2 + xx / 2] = v;
            }
        }
    }

    fn mb_type<T: BinIo>(&mut self, io: &mut T) -> R<MbT> {
        let h = self.mb_class_hint();
        match self.sh.slice_type {
            I => {
                let v = self.mb_type_i(io, 3, true, h)?;
                Ok(intra_t(v))
            }
            B => {
                let ca = self.a.map_or(0, |i| !self.pic.mbs[i].direct16 as usize);
                let cb = self.b.map_or(0, |i| !self.pic.mbs[i].direct16 as usize);
                if self.dec(io, 27 + ca + cb, kind::MB_TYPE, 0, h)? == 0 {
                    return Ok(MbT { direct16: true, ..Default::default() });
                }
                let v = if self.dec(io, 30, kind::MB_TYPE, 1, h)? == 0 {
                    1 + self.dec(io, 32, kind::MB_TYPE, 2, h)?
                } else {
                    let mut bits = self.dec(io, 31, kind::MB_TYPE, 3, h)? << 3;
                    bits |= self.dec(io, 32, kind::MB_TYPE, 4 | bits << 4, h)? << 2;
                    bits |= self.dec(io, 32, kind::MB_TYPE, 5 | bits << 4, h)? << 1;
                    bits |= self.dec(io, 32, kind::MB_TYPE, 6 | bits << 4, h)?;
                    if bits < 8 {
                        bits + 3
                    } else if bits == 13 {
                        let v = self.mb_type_i(io, 32, false, h)?;
                        return Ok(intra_t(v));
                    } else if bits == 14 {
                        11
                    } else if bits == 15 {
                        22
                    } else {
                        ((bits << 1) | self.dec(io, 32, kind::MB_TYPE, 7 | bits << 4, h)?) - 4
                    }
                };
                Ok(b_type(v))
            }
            _ => {
                if self.dec(io, 14, kind::MB_TYPE, 0, h)? == 0 {
                    let t = if self.dec(io, 15, kind::MB_TYPE, 1, h)? == 0 {
                        if self.dec(io, 16, kind::MB_TYPE, 2, h)? == 0 {
                            0
                        } else {
                            3
                        }
                    } else if self.dec(io, 17, kind::MB_TYPE, 3, h)? == 1 {
                        1
                    } else {
                        2
                    };
                    Ok(match t {
                        0 => MbT { shape: 0, pred: [1, 0], ..Default::default() },
                        1 => MbT { shape: 1, pred: [1, 1], ..Default::default() },
                        2 => MbT { shape: 2, pred: [1, 1], ..Default::default() },
                        _ => MbT { shape: 3, ..Default::default() },
                    })
                } else {
                    let v = self.mb_type_i(io, 17, false, h)?;
                    Ok(intra_t(v))
                }
            }
        }
    }

    /// I macroblock types: 0 = I_NxN, 1..=24 = I_16x16, 25 = I_PCM.
    fn mb_type_i<T: BinIo>(&mut self, io: &mut T, off: usize, islice: bool, h: u32) -> R<u32> {
        let b0 = if islice {
            let ca = self.a.map_or(0, |i| !self.pic.mbs[i].i_nxn as usize);
            let cb = self.b.map_or(0, |i| !self.pic.mbs[i].i_nxn as usize);
            self.dec(io, off + ca + cb, kind::MB_TYPE, 16, h)?
        } else {
            self.dec(io, off, kind::MB_TYPE, 16, h)?
        };
        if b0 == 0 {
            return Ok(0);
        }
        if io.terminate(&Hint { ctx: 276, kind: kind::PCM, a: 0, b: 0, c: 0, d: 0 })? == 1 {
            return Ok(25);
        }
        let (cl, cc0, cc1, cp0, cp1) =
            if islice { (off + 3, off + 4, off + 5, off + 6, off + 7) } else { (off + 1, off + 2, off + 2, off + 3, off + 3) };
        let mut v = 1 + 12 * self.dec(io, cl, kind::MB_TYPE, 17, h)?;
        if self.dec(io, cc0, kind::MB_TYPE, 18, h)? == 1 {
            v += 4 + 4 * self.dec(io, cc1, kind::MB_TYPE, 19, h)?;
        }
        v += 2 * self.dec(io, cp0, kind::MB_TYPE, 20, h)?;
        v += self.dec(io, cp1, kind::MB_TYPE, 21 | (v & 2) << 4, h)?;
        Ok(v)
    }

    fn sub_mb_type<T: BinIo>(&mut self, io: &mut T, is_b: bool, idx: u32) -> R<SubT> {
        let h = idx;
        if !is_b {
            let v = if self.dec(io, 21, kind::SUB_TYPE, 0, h)? == 1 {
                0
            } else if self.dec(io, 22, kind::SUB_TYPE, 1, h)? == 0 {
                1
            } else if self.dec(io, 23, kind::SUB_TYPE, 2, h)? == 1 {
                2
            } else {
                3
            };
            return Ok(SubT { direct: false, mode: 1, shape: v });
        }
        if self.dec(io, 36, kind::SUB_TYPE, 8, h)? == 0 {
            return Ok(SubT { direct: true, mode: 0, shape: 0 });
        }
        let v = if self.dec(io, 37, kind::SUB_TYPE, 9, h)? == 0 {
            1 + self.dec(io, 39, kind::SUB_TYPE, 10, h)?
        } else if self.dec(io, 38, kind::SUB_TYPE, 11, h)? == 1 {
            if self.dec(io, 39, kind::SUB_TYPE, 12, h)? == 1 {
                11 + self.dec(io, 39, kind::SUB_TYPE, 13, h)?
            } else {
                let mut t = 7 + 2 * self.dec(io, 39, kind::SUB_TYPE, 14, h)?;
                t += self.dec(io, 39, kind::SUB_TYPE, 15, h)?;
                t
            }
        } else {
            let mut t = 3 + 2 * self.dec(io, 39, kind::SUB_TYPE, 16, h)?;
            t += self.dec(io, 39, kind::SUB_TYPE, 17, h)?;
            t
        };
        let (mode, shape) = B_SUB[v as usize];
        Ok(SubT { direct: false, mode, shape })
    }

    fn ref_cond(&self, list: usize, x: i32, y: i32, is_b: bool) -> (usize, u32) {
        let (nb, bx, by) = self.locate(x, y);
        match self.nbmb(nb) {
            None => (0, 3),
            Some(m) => {
                let b8 = (by / 2) * 2 + bx / 2;
                let direct = is_b && (m.direct8 >> b8) & 1 == 1;
                let r = m.refi[list][b8];
                ((r > 0 && !direct) as usize, if direct { 2 } else { r.clamp(-1, 1) as i32 as u32 & 3 })
            }
        }
    }

    fn ref_idx<T: BinIo>(&mut self, io: &mut T, list: usize, x: usize, y: usize, nref: u32) -> R<u32> {
        let is_b = self.sh.slice_type == B;
        let (ca, ha) = self.ref_cond(list, x as i32 - 1, y as i32, is_b);
        let (cb, hb) = self.ref_cond(list, x as i32, y as i32 - 1, is_b);
        let mut ctx = ca + 2 * cb;
        let mut r = 0u32;
        let h = ha | hb << 2 | (list as u32) << 4 | q(nref, 8) << 5;
        while self.dec(io, 54 + ctx, kind::REF, q(r, 4), h)? == 1 {
            r += 1;
            ctx = if ctx < 4 { 4 } else { 5 };
            if r >= 32 {
                return Err(Bad);
            }
        }
        Ok(r)
    }

    fn mvd_nb(&self, list: usize, comp: usize, x: i32, y: i32) -> (u32, i32) {
        let (nb, bx, by) = self.locate(x, y);
        self.nbmb(nb).map_or((0, 0), |m| (m.mvd[list][by * 4 + bx][comp] as u32, m.mvs[list][by * 4 + bx][comp] as i32))
    }

    /// Contexts of one motion vector difference component: `v0` is the magnitude of the
    /// partition's x component when `comp` is 1.
    #[allow(clippy::too_many_arguments)]
    fn mvd_ctx(&self, list: usize, comp: usize, x: usize, y: usize, shape: u32, v0: u16) -> MvdCtx {
        let lg = |v: u32| -> u32 { 32 - v.leading_zeros() };
        let (a, sa) = self.mvd_nb(list, comp, x as i32 - 1, y as i32);
        let (b, sb) = self.mvd_nb(list, comp, x as i32, y as i32 - 1);
        let amvd = a + b;
        let base = if comp == 0 { 40 } else { 47 };
        let inc = if amvd < 3 {
            0
        } else if amvd <= 32 {
            1
        } else {
            2
        };
        let colo = self.colo(self.addr).mvd[comp] as u32;
        // The other component of this partition (y is coded after x) and the other list.
        let other = if comp == 1 { q(v0 as u32, 62) } else { 63 };
        let (ol, ols) = if list == 1 {
            (self.cur.mvd[0][y * 4 + x][comp] as u32, self.cur.mvs[0][y * 4 + x][comp] as i32)
        } else {
            (0, 0)
        };
        let hb = q(a, 31) | q(b, 31) << 5 | q(colo, 15) << 10 | shape << 14;
        let hc = (comp as u32) << 4 | (list as u32) << 5;
        let hcc = other | q(ol, 31) << 6;
        let hd = lg(amvd).min(15) | lg(other) << 4 | (comp as u32) << 8;
        MvdCtx {
            first: Hint { ctx: (base + inc) as u16, kind: kind::MVD, a: hc, b: hb, c: hcc, d: hd },
            base,
            hs: lg(amvd) | lg(other) << 5 | (comp as u32) << 10,
            sign_c: sgn32(sa) | sgn32(sb) << 2,
            sign_d: sgn32(ols) | q(ol, 3) << 2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mvd<T: BinIo>(&mut self, io: &mut T, list: usize, x: usize, y: usize, w: usize, h: usize, shape: u32) -> R<()> {
        let mut vals = [0u16; 2];
        let mut svals = [0i16; 2];
        let (mvp, cand, cand2, spread) = if self.rc.on { self.mv_cands(list, x, y, w, h, shape) } else { ([0, 0], None, None, [0, 0]) };
        for comp in 0..2 {
            let mut c = self.mvd_ctx(list, comp, x, y, shape, vals[0]);
            if self.rc.on {
                // The differences the candidates predict for this component: the co-located
                // motion (scaled), and for list 1 the partition's own list-0 vector (scaled).
                // Classes: 0 none, 1 zero, 2..=7 positive by magnitude, 8.. negative.
                let class = |cand: Option<[i16; 2]>| -> u32 {
                    cand.map_or(0, |cv| {
                        let d = cv[comp] as i32 - mvp[comp] as i32;
                        let m = d.unsigned_abs();
                        let mag = if m == 0 { 0 } else { (32 - m.leading_zeros()).min(6) };
                        1 + mag + if d < 0 { 7 } else { 0 }
                    })
                };
                let sign = |t: u32| -> u32 {
                    match t {
                        0 => 0,
                        1 => 3,
                        8.. => 2,
                        _ => 1,
                    }
                };
                let (tf, tf2) = (class(cand), class(cand2));
                c.first.a |= tf << 8 | tf2 << 12;
                c.first.d |= spread[comp] << 16;
                c.sign_c |= sign(tf) << 6 | sign(tf2) << 8;
                c.hs |= tf << 12;
            }
            // Prefix bin `m` (1..=8), suffix bins, sign: also announced to the model ahead.
            let un = |m: u32| Hint { ctx: (c.base + 2 + m.min(4) as usize) as u16, a: c.first.a | m, ..c.first };
            let suf = |a: u32, cc: u32| Hint { ctx: 0, kind: kind::MVD_SUF, a, b: c.hs, c: cc, d: 0 };
            let own = if comp == 1 { sgn32(svals[0] as i32) } else { 3 };
            let sign_h = |m: u32| Hint {
                ctx: 0,
                kind: kind::MVD_SIGN,
                a: comp as u32 | (list as u32) << 1,
                b: q(m, 15),
                c: c.sign_c | own << 4,
                d: c.sign_d,
            };
            io.prefetch(&un(1));
            if comp == 0 {
                io.prefetch(&self.mvd_ctx(list, 1, x, y, shape, 0).first);
            }
            let mut m: u32 = 0;
            let mut neg = false;
            if self.dech(io, &c.first)? == 1 {
                m = 1;
                while m < 9 {
                    let h = un(m);
                    io.prefetch(&if m + 1 < 9 { un(m + 1) } else { suf(3, 0) });
                    io.prefetch(&sign_h(m));
                    if self.dech(io, &h)? == 0 {
                        break;
                    }
                    m += 1;
                }
                if m >= 9 {
                    let mut k = 3u32;
                    loop {
                        io.prefetch(&suf(k + 1, 0));
                        io.prefetch(&suf(32 + k - 1, 1));
                        if io.bypass(&suf(k, 0))? == 0 {
                            break;
                        }
                        m += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err(Bad);
                        }
                    }
                    let mut acc = 1u32;
                    while k > 0 {
                        k -= 1;
                        if k > 0 {
                            io.prefetch(&suf(32 + k - 1, (acc << 1).min(63)));
                            io.prefetch(&suf(32 + k - 1, (acc << 1 | 1).min(63)));
                        } else {
                            io.prefetch(&sign_h(m));
                            io.prefetch(&sign_h(m + 1));
                        }
                        let bit = io.bypass(&suf(32 + k, acc.min(63)))?;
                        acc = (acc << 1) | bit;
                        m += bit << k;
                    }
                }
                if comp == 0 {
                    // The y component's first bin does not depend on this sign.
                    io.prefetch(&self.mvd_ctx(list, 1, x, y, shape, m.min(65535) as u16).first);
                }
                neg = io.bypass(&sign_h(m))? == 1;
            }
            vals[comp] = m.min(65535) as u16;
            let sv = m.min(32767) as i16;
            svals[comp] = if neg { -sv } else { sv };
        }
        for yy in y..y + h {
            for xx in x..x + w {
                self.cur.mvd[list][yy * 4 + xx] = vals;
                self.cur.mvs[list][yy * 4 + xx] = svals;
            }
        }
        if self.rc.on {
            let mv = [mvp[0].wrapping_add(svals[0]), mvp[1].wrapping_add(svals[1])];
            self.set_motion(list, x, y, w, h, mv);
        }
        Ok(())
    }

    /// Motion vector predictor of a partition being parsed, and a temporal candidate (the
    /// co-located motion, scaled to this partition's reference).
    #[allow(clippy::type_complexity)]
    fn mv_cands(&self, l: usize, x: usize, y: usize, w: usize, h: usize, shape: u32) -> ([i16; 2], Option<[i16; 2]>, Option<[i16; 2]>, [u32; 2]) {
        let b8 = (y / 2) * 2 + x / 2;
        let refidx = self.cur.rf[l][b8];
        let mut done = 0u16;
        let rect_mask = |x: usize, y: usize, w: usize, h: usize| -> u16 {
            let mut m = 0u16;
            for yy in y..y + h {
                for xx in x..x + w {
                    m |= 1 << (yy * 4 + xx);
                }
            }
            m
        };
        let dir = if shape >= 4 {
            let s = (shape - 4) as usize;
            let rects = PART_RECTS[s.min(2)];
            let p = rects.iter().position(|&(rx, ry, _, _)| rx == x && ry == y).unwrap_or(0);
            for &(rx, ry, rw, rh) in &rects[..p] {
                done |= rect_mask(rx, ry, rw, rh);
            }
            match s {
                1 => 1 + p,
                2 => 3 + p,
                _ => 0,
            }
        } else {
            let i = b8;
            for j in 0..i {
                done |= rect_mask((j & 1) * 2, (j >> 1) * 2, 2, 2);
            }
            let (x0, y0) = ((i & 1) * 2, (i >> 1) * 2);
            for &(dx, dy, rw, rh) in SUB_RECTS[shape as usize] {
                if x0 + dx == x && y0 + dy == y {
                    break;
                }
                done |= rect_mask(x0 + dx, y0 + dy, rw, rh);
            }
            0
        };
        let _ = h;
        let mvp = self.pred_mv(l, x as i32, y as i32, w as i32, refidx, dir, done);
        // Spread of the neighbours' vectors around the predictor (same reference only).
        let mut spread = [0u32; 2];
        for n in [self.mv_nb(l, x as i32 - 1, y as i32, done), self.mv_nb(l, x as i32, y as i32 - 1, done), self.mv_nb(l, x as i32 + w as i32, y as i32 - 1, done)].into_iter().flatten() {
            if n.0 == refidx {
                for c in 0..2 {
                    let d = (n.1[c] as i32 - mvp[c] as i32).unsigned_abs();
                    spread[c] = spread[c].max(if d == 0 { 0 } else { (33 - d.leading_zeros()).min(7) });
                }
            }
        }
        let Some(rec) = self.pic.rec.as_ref() else { return (mvp, None, None, spread) };
        // The partition's list-0 vector scaled to its list-1 reference (bi-prediction).
        let cand2 = if l == 1 && self.cur.rf[0][b8] >= 0 {
            let mv0 = self.cur.mv[0][y * 4 + x];
            match (rec.refpic(0, self.cur.rf[0][b8] as i32), rec.refpic(1, refidx as i32)) {
                (Some(p0), Some(p1)) => {
                    let tb = (rec.cur.poc - p1.poc).clamp(-128, 127);
                    let td = (rec.cur.poc - p0.poc).clamp(-128, 127);
                    if td == 0 {
                        Some(mv0)
                    } else {
                        let tx = (16384 + (td / 2).abs()) / td;
                        let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
                        Some([((dsf * mv0[0] as i32 + 128) >> 8) as i16, ((dsf * mv0[1] as i32 + 128) >> 8) as i16])
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        let col = if self.sh.slice_type == B { rec.refpic(1, 0) } else { rec.refpic(0, 0) };
        let cand = col.and_then(|cp| {
            if cp.y.w != rec.cur.y.w || cp.y.h != rec.cur.y.h {
                return None;
            }
            let m = *cp.mf.get((self.rc.mby * 4 + y) * cp.w4 + self.rc.mbx * 4 + x)?;
            let lu = if m.refi[l] >= 0 {
                l
            } else if m.refi[1 - l] >= 0 {
                1 - l
            } else {
                return None;
            };
            let mv = m.mv[lu];
            let cur_ref = rec.refpic(l, refidx as i32)?;
            let tb = (rec.cur.poc - cur_ref.poc).clamp(-128, 127);
            let td = (cp.poc - m.rpoc[lu]).clamp(-128, 127);
            if td == 0 || m.rpoc[lu] == i32::MIN {
                return Some(mv);
            }
            let tx = (16384 + (td / 2).abs()) / td;
            let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
            Some([((dsf * mv[0] as i32 + 128) >> 8) as i16, ((dsf * mv[1] as i32 + 128) >> 8) as i16])
        });
        (mvp, cand, cand2, spread)
    }

    fn cbp_luma_cond(&self, nb: Option<usize>, b8: usize) -> usize {
        match nb {
            None => 0,
            Some(i) => {
                let m = &self.pic.mbs[i];
                if m.skip {
                    1
                } else {
                    ((m.cbp >> b8) & 1 == 0) as usize
                }
            }
        }
    }

    fn cbp_chroma_cond(&self, nb: Option<usize>) -> (usize, usize) {
        match nb {
            None => (0, 0),
            Some(i) => {
                let m = &self.pic.mbs[i];
                if m.skip {
                    (0, 0)
                } else {
                    let c = m.cbp >> 4;
                    ((c != 0) as usize, (c == 2) as usize)
                }
            }
        }
    }

    /// Prediction texture (and bi-prediction disagreement) of each 8x8 luma block and of
    /// the chroma, for coded_block_pattern; zeros where no prediction exists yet.
    fn cbp_feats(&self) -> ([u32; 4], u32) {
        if !self.rc.on || self.cur.intra {
            return ([0; 4], 0);
        }
        let mut f = [0u32; 4];
        for (b8, fv) in f.iter_mut().enumerate() {
            let (x0, y0) = ((b8 & 1) * 8, (b8 >> 1) * 8);
            let (mut tex, mut dis) = (0u32, 0u32);
            for sb in 0..4 {
                let (sx, sy) = (x0 + (sb & 1) * 4, y0 + (sb >> 1) * 4);
                let mut s = [0i32; 16];
                for j in 0..4 {
                    for i in 0..4 {
                        s[j * 4 + i] = self.rc.pred[(sy + j) * 16 + sx + i];
                        dis += self.rc.dis[(sy + j) * 16 + sx + i].unsigned_abs();
                    }
                }
                let mut qm = [0u32; 16];
                recon::qmag4(&s, self.qp, &mut qm);
                tex += qm[1..].iter().sum::<u32>();
            }
            *fv = bucket(tex) as u32 | (bucket(dis * 4) as u32) << 3;
        }
        let mut ct = 0u32;
        let qpc = recon::chroma_qp(self.qp, self.pps.chroma_qp_offset);
        for c in 0..2 {
            for b4 in 0..4 {
                let (x, y) = ((b4 & 1) * 4, (b4 >> 1) * 4);
                let mut s = [0i32; 16];
                for j in 0..4 {
                    for i in 0..4 {
                        s[j * 4 + i] = self.rc.pred[256 + c * 64 + (y + j) * 8 + x + i];
                    }
                }
                let mut qm = [0u32; 16];
                recon::qmag4(&s, qpc, &mut qm);
                ct += qm[1..].iter().sum::<u32>();
            }
        }
        (f, bucket(ct) as u32)
    }

    fn cbp<T: BinIo>(&mut self, io: &mut T) -> R<u8> {
        let colo = self.colo(self.addr).cbp as u32;
        let cls = self.cur.class as u32;
        let mut cbp = 0u8;
        let (fl, fc) = self.cbp_feats();
        for b8 in 0..4usize {
            let (x8, y8) = (b8 & 1, b8 >> 1);
            let ca = if x8 == 1 { ((cbp >> (b8 - 1)) & 1 == 0) as usize } else { self.cbp_luma_cond(self.a, y8 * 2 + 1) };
            let cb = if y8 == 1 { ((cbp >> (b8 - 2)) & 1 == 0) as usize } else { self.cbp_luma_cond(self.b, 2 + x8) };
            let ctx = 73 + ca + 2 * cb;
            let st = self.st[ctx];
            let h = Hint {
                ctx: ctx as u16,
                kind: kind::CBP_L,
                a: b8 as u32 | (cbp as u32) << 2 | cls << 6,
                b: colo,
                c: fl[b8],
                d: 0,
            };
            let bit = io.decision(st, &h)?;
            self.st[ctx] = next_state(st, bit);
            cbp |= (bit as u8) << b8;
        }
        let (a1, a2) = self.cbp_chroma_cond(self.a);
        let (b1, b2) = self.cbp_chroma_cond(self.b);
        let hl = (cbp as u32) << 2 | cls << 6;
        let cc = if self.rc.on { fc | (fl.iter().map(|v| v & 7).max().unwrap_or(0)) << 4 } else { 0 };
        let dc = |me: &mut Self, io: &mut T, ctx: usize, a: u32| -> R<u32> {
            let st = me.st[ctx];
            let bin = io.decision(st, &Hint { ctx: ctx as u16, kind: kind::CBP_C, a, b: colo >> 4, c: cc, d: 0 })?;
            me.st[ctx] = next_state(st, bin);
            Ok(bin)
        };
        if dc(self, io, 77 + a1 + 2 * b1, hl)? == 1 {
            let c = 1 + dc(self, io, 81 + a2 + 2 * b2, hl | 1)?;
            cbp |= (c as u8) << 4;
        }
        Ok(cbp)
    }

    fn chroma_pred_mode<T: BinIo>(&mut self, io: &mut T) -> R<u32> {
        let ca = self.a.map_or(0, |i| (self.pic.mbs[i].intra && self.pic.mbs[i].cpred_nz) as usize);
        let cb = self.b.map_or(0, |i| (self.pic.mbs[i].intra && self.pic.mbs[i].cpred_nz) as usize);
        let h = self.cur.class as u32;
        if self.dec(io, 64 + ca + cb, kind::CPRED, 0, h)? == 0 {
            return Ok(0);
        }
        if self.dec(io, 67, kind::CPRED, 1, h)? == 0 {
            return Ok(1);
        }
        if self.dec(io, 67, kind::CPRED, 2, h)? == 0 {
            return Ok(2);
        }
        Ok(3)
    }

    fn t8flag<T: BinIo>(&mut self, io: &mut T) -> R<u32> {
        let ca = self.a.map_or(0, |i| self.pic.mbs[i].t8 as usize);
        let cb = self.b.map_or(0, |i| self.pic.mbs[i].t8 as usize);
        let h = self.cur.class as u32 | (self.cur.cbp as u32) << 3;
        self.dec(io, 399 + ca + cb, kind::T8, 0, h)
    }

    fn dqp<T: BinIo>(&mut self, io: &mut T) -> R<()> {
        let mut ctx = 60 + self.last_dqp_nz as usize;
        let h = q(self.last_dqp, 7) | (self.cur.class as u32) << 3;
        let mut v = 0u32;
        let (fc, fd) = if self.rc.on {
            let lv = self.mb_logvar();
            let last = self.rc.lv_last;
            self.rc.lv_last = lv;
            let d = if lv == 0 || last == 0 { 0 } else { (lv as i32 - last as i32).clamp(-7, 7) + 8 };
            (lv, d as u32)
        } else {
            (0, 0)
        };
        let dq = |me: &mut Self, io: &mut T, ctx: usize, a: u32| -> R<u32> {
            let s = me.st[ctx];
            let bin = io.decision(s, &Hint { ctx: ctx as u16, kind: kind::DQP, a, b: h, c: fc, d: fd })?;
            me.st[ctx] = next_state(s, bin);
            Ok(bin)
        };
        if dq(self, io, ctx, 0)? == 1 {
            v = 1;
            ctx = 62;
            while dq(self, io, ctx, q(v, 7))? == 1 {
                ctx = 63;
                v += 1;
                if v > 200 {
                    return Err(Bad);
                }
            }
        }
        self.last_dqp_nz = v != 0;
        self.last_dqp = v;
        let d = if v & 1 == 1 { (v as i32 + 1) / 2 } else { -(v as i32 / 2) };
        self.qp = (self.qp + d + 52).rem_euclid(52);
        Ok(())
    }

    fn residual<T: BinIo>(&mut self, io: &mut T) -> R<()> {
        let cbp = self.cur.cbp;
        let on = self.rc.on;
        let nxn = self.cur.i_nxn;
        if self.cur.i16 {
            self.clear_feat();
            self.block(io, 0, 0, 16)?;
        }
        let mut dcs = [0i32; 16];
        if on && self.cur.i16 && (self.cur.cbf >> 16) & 1 == 1 {
            recon::luma_dc(&self.rc.dc, self.qp, &mut dcs);
        }
        for b8 in 0..4usize {
            let coded = (cbp >> b8) & 1 == 1;
            if !coded && !on {
                continue;
            }
            if !self.cur.t8 {
                for b4 in 0..4usize {
                    let blk = b8 * 4 + b4;
                    if on {
                        let (x, y) = blk_xy(blk);
                        if nxn {
                            self.intra4_pred(blk);
                        }
                        if coded {
                            self.feat_luma4(x, y);
                            self.bnd_luma4(x, y);
                        }
                    }
                    if coded {
                        if self.cur.i16 {
                            self.block(io, 1, blk, 15)?;
                        } else {
                            self.block(io, 2, blk, 16)?;
                        }
                    }
                    if on {
                        let (x, y) = blk_xy(blk);
                        self.recon_l4(x, y, self.cur.i16.then(|| dcs[y * 4 + x]));
                    }
                }
            } else {
                if on {
                    if nxn {
                        self.intra8_pred(b8);
                    }
                    self.clear_feat();
                    if coded {
                        self.feat_luma8(b8);
                        self.bnd_luma8(b8);
                    }
                }
                if coded {
                    self.block8x8(io, b8)?;
                }
                if on {
                    self.recon_l8(b8);
                }
            }
        }
        if on {
            self.rc.luma_done = true;
        }
        let cc = cbp >> 4;
        if cc != 0 {
            for c in 0..2 {
                if on {
                    self.clear_feat();
                }
                self.block(io, 3, c, 4)?;
            }
        }
        if cc == 2 {
            for c in 0..2 {
                for b4 in 0..4 {
                    if on {
                        self.feat_chroma4(c, b4);
                        self.rc.bnd_ok = 0;
                                        self.rc.fdis = [0; 16];
                        self.rc.dtex = 0;
                    }
                    self.block(io, 4, c * 4 + b4, 15)?;
                }
            }
        }
        Ok(())
    }

    /// condTermFlagN for luma 4x4 coded_block_flag (ctxBlockCat 1 and 2), plus the
    /// neighbour block's coefficient count for the model.
    fn luma_cbf_cond(&self, x: i32, y: i32) -> (usize, u32) {
        let (nb, bx, by) = self.locate(x, y);
        match self.nbmb(nb) {
            None => (self.cur.intra as usize, 17),
            Some(m) => {
                let n = m.nnz[by * 4 + bx] as u32;
                if m.skip {
                    return (0, 0);
                }
                let b8 = (by / 2) * 2 + bx / 2;
                if (m.cbp >> b8) & 1 == 0 {
                    (0, 0)
                } else if m.t8 {
                    (1, n)
                } else {
                    (((m.cbf >> xy_blk(bx, by)) & 1) as usize, n)
                }
            }
        }
    }

    fn chroma_ac_cond(&self, c: usize, b4: usize) -> ((usize, u32), (usize, u32)) {
        let (x, y) = (b4 & 1, b4 >> 1);
        let intra = self.cur.intra as usize;
        let get = |m: &MbInfo, blk: usize| -> (usize, u32) {
            if m.skip || m.cbp >> 4 != 2 {
                (0, 0)
            } else {
                (((m.cbf >> (19 + c * 4 + blk)) & 1) as usize, m.nnzc[c * 4 + blk] as u32)
            }
        };
        let a = if x == 1 {
            get(&self.cur, y * 2)
        } else {
            self.a.map_or((intra, 17), |i| get(&self.pic.mbs[i], y * 2 + 1))
        };
        let b = if y == 1 { get(&self.cur, x) } else { self.b.map_or((intra, 17), |i| get(&self.pic.mbs[i], 2 + x)) };
        (a, b)
    }

    fn dc_cond(&self, nb: Option<usize>, luma: bool, c: usize) -> usize {
        let intra = self.cur.intra as usize;
        match nb {
            None => intra,
            Some(i) => {
                let m = &self.pic.mbs[i];
                if luma {
                    if m.i16 {
                        ((m.cbf >> 16) & 1) as usize
                    } else {
                        0
                    }
                } else if !m.skip && m.cbp >> 4 != 0 {
                    ((m.cbf >> (17 + c)) & 1) as usize
                } else {
                    0
                }
            }
        }
    }

    fn block<T: BinIo>(&mut self, io: &mut T, cat: usize, idx: usize, max: usize) -> R<()> {
        // coded_block_flag context, plus a neighbourhood coefficient estimate for the model.
        let (inc, nc) = match cat {
            0 => (self.dc_cond(self.a, true, 0) + 2 * self.dc_cond(self.b, true, 0), 16),
            1 | 2 => {
                let (x, y) = blk_xy(idx);
                let (ca, na) = self.luma_cbf_cond(x as i32 - 1, y as i32);
                let (cb, nb) = self.luma_cbf_cond(x as i32, y as i32 - 1);
                let nc = match (na, nb) {
                    (17, 17) => 17,
                    (17, v) | (v, 17) => v,
                    (a, b) => (a + b).div_ceil(2),
                };
                (ca + 2 * cb, nc)
            }
            3 => (self.dc_cond(self.a, false, idx) + 2 * self.dc_cond(self.b, false, idx), 16),
            _ => {
                let ((ca, na), (cb, nb)) = self.chroma_ac_cond(idx / 4, idx % 4);
                let nc = match (na, nb) {
                    (17, 17) => 17,
                    (17, v) | (v, 17) => v,
                    (a, b) => (a + b).div_ceil(2),
                };
                (ca + 2 * cb, nc)
            }
        };
        let colo = if cat == 1 || cat == 2 {
            let (x, y) = blk_xy(idx);
            self.colo(self.addr).nnz[y * 4 + x] as u32
        } else {
            0
        };
        let hb = q(nc, 17) | q(colo, 15) << 5 | (self.cur.class as u32) << 9;
        // Coefficients of the left and above blocks, for the model.
        let (na, nb) = self.nb_levels(cat, idx);
        let ea = na.map_or(0, |l| l.iter().filter(|&&v| v != 0).count() as u32);
        let eb = nb.map_or(0, |l| l.iter().filter(|&&v| v != 0).count() as u32);
        let hc = q(ea, 15) | q(eb, 15) << 4 | (na.is_some() as u32) << 8 | (nb.is_some() as u32) << 9;
        let off = (cat == 1 || cat == 4) as usize;
        if max > 1 {
            // A coded block starts with significant_coeff_flag[0].
            let la0 = na.map_or(0, |l| l[off]);
            let lb0 = nb.map_or(0, |l| l[off]);
            let cnt = |l: &Option<[i8; 64]>| l.map_or(0, |l| l[off..off + max].iter().filter(|&&v| v != 0).count() as u8);
            io.prefetch(&sig_hint(cat, 0, 0, q(nc, 17) | q(colo, 63) << 6, self.extra(), la0, lb0, cnt(&na), cnt(&nb)));
        }
        let cbf_d = self.extra() | if self.rc.on { self.rc.ftex << 16 | self.rc.dtex << 20 } else { 0 };
        let cbf = self.dec4(io, 85 + CBF_OFF[cat] + inc, kind::CBF, cat as u32 | (self.cur.intra as u32) << 4, hb, hc, cbf_d)?;
        if cbf == 0 {
            return Ok(());
        }
        let bit = match cat {
            0 => 16,
            1 | 2 => idx,
            3 => 17 + idx,
            _ => 19 + idx,
        };
        self.cur.cbf |= 1 << bit;
        let mut out = [0i8; 64];
        let la = na.unwrap_or([0; 64]);
        let lb = nb.unwrap_or([0; 64]);
        let n = self.coeffs(io, cat, max, q(nc, 17), colo, &la, &lb, off, &mut out)?;
        if self.rc.on {
            let rc = &mut *self.rc;
            match cat {
                0 => rc.dc.copy_from_slice(&rc.cx[..16]),
                1 | 2 => {
                    let (x, y) = blk_xy(idx);
                    rc.l4[y * 4 + x].copy_from_slice(&rc.cx[..16]);
                }
                3 => rc.cdc[idx].copy_from_slice(&rc.cx[..4]),
                _ => rc.cac[idx].copy_from_slice(&rc.cx[..16]),
            }
        }
        match cat {
            0 => self.curlv.dc.copy_from_slice(&out[..16]),
            1 | 2 => {
                let (x, y) = blk_xy(idx);
                self.cur.nnz[y * 4 + x] = n;
                self.curlv.l4[y * 4 + x].copy_from_slice(&out[..16]);
            }
            3 => self.curlv.cdc[idx].copy_from_slice(&out[..4]),
            _ => {
                self.cur.nnzc[idx] = n;
                self.curlv.c[idx].copy_from_slice(&out[..16]);
            }
        }
        Ok(())
    }

    /// Levels of a luma 4x4 block position relative to the current macroblock, widened to
    /// 64 entries (8x8-transformed neighbours contribute their first 16 scan positions).
    fn nb_l4(&self, x: i32, y: i32) -> Option<[i8; 64]> {
        let (nb, bx, by) = self.locate(x, y);
        let (m, lv) = match nb {
            Nb::None => return None,
            Nb::Cur => (&self.cur, &self.curlv),
            Nb::Mb(i) => (&self.pic.mbs[i], &self.pic.lv[i]),
        };
        let mut t = [0i8; 64];
        if !m.skip {
            if m.t8 {
                t.copy_from_slice(&lv.l8[(by / 2) * 2 + bx / 2]);
            } else {
                t[..16].copy_from_slice(&lv.l4[by * 4 + bx]);
            }
        }
        Some(t)
    }

    fn nb_levels(&self, cat: usize, idx: usize) -> (Option<[i8; 64]>, Option<[i8; 64]>) {
        let wide = |s: &[i8]| {
            let mut t = [0i8; 64];
            t[..s.len()].copy_from_slice(s);
            t
        };
        match cat {
            0 => {
                let f = |nb: Option<usize>| nb.map(|i| if self.pic.mbs[i].i16 { wide(&self.pic.lv[i].dc) } else { [0; 64] });
                (f(self.a), f(self.b))
            }
            1 | 2 => {
                let (x, y) = blk_xy(idx);
                (self.nb_l4(x as i32 - 1, y as i32), self.nb_l4(x as i32, y as i32 - 1))
            }
            3 => {
                let f = |nb: Option<usize>| nb.map(|i| wide(&self.pic.lv[i].cdc[idx]));
                (f(self.a), f(self.b))
            }
            _ => {
                let (c, b4) = (idx / 4, idx % 4);
                let (x, y) = (b4 & 1, b4 >> 1);
                let a = if x == 1 {
                    Some(wide(&self.curlv.c[c * 4 + y * 2]))
                } else {
                    self.a.map(|i| wide(&self.pic.lv[i].c[c * 4 + y * 2 + 1]))
                };
                let b = if y == 1 { Some(wide(&self.curlv.c[c * 4 + x])) } else { self.b.map(|i| wide(&self.pic.lv[i].c[c * 4 + 2 + x])) };
                (a, b)
            }
        }
    }

    fn block8x8<T: BinIo>(&mut self, io: &mut T, b8: usize) -> R<()> {
        let (x0, y0) = ((b8 & 1) * 2, (b8 >> 1) * 2);
        let (_, na) = self.luma_cbf_cond(x0 as i32 - 1, y0 as i32);
        let (_, nb) = self.luma_cbf_cond(x0 as i32, y0 as i32 - 1);
        let nc = match (na, nb) {
            (17, 17) => 17,
            (17, v) | (v, 17) => v,
            (a, b) => (a + b).div_ceil(2),
        };
        let p = *self.colo(self.addr);
        let colo = p.nnz[y0 * 4 + x0] as u32 + p.nnz[y0 * 4 + x0 + 1] as u32 + p.nnz[(y0 + 1) * 4 + x0] as u32
            + p.nnz[(y0 + 1) * 4 + x0 + 1] as u32;
        let la = self.nb_l4(x0 as i32 - 1, y0 as i32).unwrap_or([0; 64]);
        let lb = self.nb_l4(x0 as i32, y0 as i32 - 1).unwrap_or([0; 64]);
        let mut out = [0i8; 64];
        let n = self.coeffs(io, 5, 64, q(nc * 4, 63), q(colo, 63), &la, &lb, 0, &mut out)?;
        if self.rc.on {
            let rc = &mut *self.rc;
            rc.l8[b8] = rc.cx;
        }
        self.curlv.l8[b8] = out;
        for b4 in 0..4 {
            self.cur.cbf |= 1 << (b8 * 4 + b4);
        }
        let share = (n as u32).div_ceil(4) as u8;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            self.cur.nnz[(y0 + dy) * 4 + x0 + dx] = share;
        }
        Ok(())
    }

    // --- reconstruction (model context) ------------------------------------------------

    fn clear_feat(&mut self) {
        self.rc.fdis = [0; 16];
        self.rc.dtex = 0;
        self.rc.feat = [0; 64];
        self.rc.ftex = 0;
        self.rc.frem = [0; 65];
        self.rc.bnd_ok = 0;
    }

    fn feat_luma8(&mut self, b8: usize) {
        let (x0, y0) = ((b8 & 1) * 8, (b8 >> 1) * 8);
        let mut s = [0i32; 64];
        for j in 0..8 {
            for i in 0..8 {
                s[j * 8 + i] = self.rc.pred[(y0 + j) * 16 + x0 + i];
            }
        }
        let mut qm = [0u32; 64];
        recon::qmag8(&s, self.qp, &mut qm);
        let rc = &mut *self.rc;
        let mut tex = 0u32;
        for i in 0..64 {
            let r = recon::ZZ8[i] as usize;
            rc.feat[i] = bucket(qm[r]);
            if i > 0 {
                tex = tex.saturating_add(qm[r]);
            }
        }
        rc.ftex = bucket(tex) as u32;
        let mut acc = 0u32;
        rc.frem[64] = 0;
        for i in (0..64).rev() {
            rc.frem[i + 1] = bucket(acc);
            acc = acc.saturating_add(qm[recon::ZZ8[i] as usize]);
        }
        rc.frem[0] = bucket(acc);
        rc.fdis = [0; 16];
        rc.dtex = 0;
    }

    /// Boundary mismatch of luma 8x8 block `b8` (x4096) and the magnitude estimates.
    fn bnd_luma8(&mut self, b8: usize) {
        let (x8, y8) = (b8 & 1, b8 >> 1);
        let top = y8 > 0 || self.b.is_some();
        let left = x8 > 0 || self.a.is_some();
        let (px, py) = (self.rc.mbx * 16 + x8 * 8, self.rc.mby * 16 + y8 * 8);
        let rec = self.pic.rec.as_ref().expect("reconstruction on");
        let pl = &rec.cur.y;
        let w = pl.w;
        let rc = &mut *self.rc;
        rc.bnd = [0; 16];
        rc.bnd_ok = top as u8 | (left as u8) << 1;
        rc.bqp = self.qp;
        if top {
            for i in 0..8 {
                rc.bnd[i] = 4096 * (rc.pred[(y8 * 8) * 16 + x8 * 8 + i] - pl.d[(py - 1) * w + px + i] as i32);
            }
        }
        if left {
            for j in 0..8 {
                rc.bnd[8 + j] = 4096 * (rc.pred[(y8 * 8 + j) * 16 + x8 * 8] - pl.d[(py + j) * w + px - 1] as i32);
            }
        }
    }

    /// Boundary mismatch between the prediction of luma 4x4 block (`x`, `y`) and the
    /// reconstructed samples above and to the left of it.
    fn bnd_luma4(&mut self, x: usize, y: usize) {
        let top = y > 0 || self.b.is_some();
        let left = x > 0 || self.a.is_some();
        let (px, py) = (self.rc.mbx * 16 + x * 4, self.rc.mby * 16 + y * 4);
        let rec = self.pic.rec.as_ref().expect("reconstruction on");
        let pl = &rec.cur.y;
        let w = pl.w;
        let rc = &mut *self.rc;
        rc.bnd = [0; 16];
        rc.bnd_ok = top as u8 | (left as u8) << 1;
        rc.bqp = self.qp;
        if top {
            for i in 0..4 {
                rc.bnd[i] = 128 * (rc.pred[(y * 4) * 16 + x * 4 + i] - pl.d[(py - 1) * w + px + i] as i32);
            }
        }
        if left {
            for j in 0..4 {
                rc.bnd[8 + j] = 128 * (rc.pred[(y * 4 + j) * 16 + x * 4] - pl.d[(py + j) * w + px - 1] as i32);
            }
        }
    }

    fn set_feat(&mut self, s: &[i32; 16], qp: i32) {
        let mut qm = [0u32; 16];
        recon::qmag4(s, qp, &mut qm);
        let rc = &mut *self.rc;
        let mut tex = 0u32;
        for i in 0..16 {
            let r = recon::ZZ4[i] as usize;
            rc.feat[i] = bucket(qm[r]);
            if i > 0 {
                tex += qm[r];
            }
        }
        rc.ftex = bucket(tex) as u32;
        let mut acc = 0u32;
        rc.frem[16] = 0;
        for i in (0..16).rev() {
            rc.frem[i + 1] = bucket(acc);
            acc += qm[recon::ZZ4[i] as usize];
        }
        rc.frem[0] = bucket(acc);
    }

    fn feat_luma4(&mut self, x: usize, y: usize) {
        let mut s = [0i32; 16];
        let mut dd = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                s[j * 4 + i] = self.rc.pred[(y * 4 + j) * 16 + x * 4 + i];
                dd[j * 4 + i] = self.rc.dis[(y * 4 + j) * 16 + x * 4 + i];
            }
        }
        self.set_feat(&s, self.qp);
        let mut qm = [0u32; 16];
        if dd.iter().any(|&v| v != 0) {
            // Half the difference: the error of the average if one prediction were right.
            recon::qmag4(&dd, self.qp, &mut qm);
            let mut tot = 0u32;
            for i in 0..16 {
                let r = recon::ZZ4[i] as usize;
                self.rc.fdis[i] = 1 + bucket(qm[r] / 2).min(6);
                tot += qm[r] / 2;
            }
            self.rc.dtex = 1 + bucket(tot).min(6) as u32;
        } else {
            self.rc.fdis = [0; 16];
            self.rc.dtex = 0;
        }
    }

    fn feat_chroma4(&mut self, c: usize, b4: usize) {
        let (x, y) = ((b4 & 1) * 4, (b4 >> 1) * 4);
        let base = 256 + c * 64;
        let mut s = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                s[j * 4 + i] = self.rc.pred[base + (y + j) * 8 + x + i];
            }
        }
        let off = if c == 0 { self.pps.chroma_qp_offset } else { self.pps.chroma_qp_offset2 };
        self.set_feat(&s, recon::chroma_qp(self.qp, off));
    }

    /// Log2 of the luma variance (x2, plus one; 0 = unknown) of the current macroblock's
    /// prediction, or for intra macroblocks of the co-located area of the latest reference.
    fn mb_logvar(&self) -> u32 {
        let mut s = [0i32; 256];
        if self.cur.intra {
            let Some(rec) = self.pic.rec.as_ref() else { return 0 };
            let Some(p) = rec.dpb.last().filter(|p| p.y.w == rec.cur.y.w && p.y.h == rec.cur.y.h) else { return 0 };
            let (px, py) = (self.rc.mbx * 16, self.rc.mby * 16);
            for j in 0..16 {
                for i in 0..16 {
                    s[j * 16 + i] = p.y.d[(py + j) * p.y.w + px + i] as i32;
                }
            }
        } else {
            s.copy_from_slice(&self.rc.pred[..256]);
        }
        let sum: i64 = s.iter().map(|&v| v as i64).sum();
        let sq: i64 = s.iter().map(|&v| (v * v) as i64).sum();
        let var = (sq - sum * sum / 256).max(0) as u64;
        if var == 0 {
            return 1;
        }
        let lg2 = 63 - var.leading_zeros();
        let half = (var >> lg2.saturating_sub(1)) & 1;
        (2 + 2 * lg2 + half as u32).min(48)
    }

    fn nb_c(&self) -> Option<usize> {
        let w = self.pic.w;
        let col = self.addr % w;
        (self.addr >= w && col + 1 < w && self.pic.mbs[self.addr - w + 1].slice == self.sid).then(|| self.addr - w + 1)
    }

    fn nb_d(&self) -> Option<usize> {
        let w = self.pic.w;
        let col = self.addr % w;
        (self.addr >= w && col > 0 && self.pic.mbs[self.addr - w - 1].slice == self.sid).then(|| self.addr - w - 1)
    }

    fn intra_ok(&self, n: Option<usize>) -> bool {
        n.is_some_and(|i| !self.pps.constrained_intra_pred || self.pic.mbs[i].intra)
    }

    /// Neighbouring samples of an `n`x`n` block at (`px`, `py`) of a plane of the current
    /// picture (`plane` 0 luma, 1 Cb, 2 Cr).
    #[allow(clippy::too_many_arguments)]
    fn edge(&self, plane: usize, px: usize, py: usize, n: usize, top: bool, left: bool, corner: bool, tr: bool) -> Edge {
        let rec = self.pic.rec.as_ref().expect("reconstruction on");
        let p = match plane {
            0 => &rec.cur.y,
            1 => &rec.cur.u,
            _ => &rec.cur.v,
        };
        let w = p.w;
        let mut e = Edge { top: [128; 32], left: [128; 16], corner: 128, has_top: top, has_left: left, has_corner: corner };
        if top {
            for i in 0..n {
                e.top[i] = p.d[(py - 1) * w + px + i] as i32;
            }
            for i in n..2 * n {
                e.top[i] = if tr && px + i < w { p.d[(py - 1) * w + px + i] as i32 } else { e.top[n - 1] };
            }
        }
        if left {
            for j in 0..n {
                e.left[j] = p.d[(py + j) * w + px - 1] as i32;
            }
        }
        if corner {
            e.corner = p.d[(py - 1) * w + px - 1] as i32;
        }
        e
    }

    fn ipm_nb(&self, x: i32, y: i32) -> Option<u8> {
        let (nb, bx, by) = self.locate(x, y);
        match nb {
            Nb::None => None,
            Nb::Cur => Some(self.cur.ipm[by * 4 + bx]),
            Nb::Mb(i) => {
                let m = &self.pic.mbs[i];
                if !m.intra && self.pps.constrained_intra_pred {
                    None
                } else if m.i_nxn {
                    Some(m.ipm[by * 4 + bx])
                } else {
                    Some(2)
                }
            }
        }
    }

    fn derive_ipm(&self, k: usize, x: usize, y: usize) -> u8 {
        let pred = match (self.ipm_nb(x as i32 - 1, y as i32), self.ipm_nb(x as i32, y as i32 - 1)) {
            (Some(a), Some(b)) => a.min(b),
            _ => 2,
        };
        let (prev, rem) = self.rc.ipsyn[k];
        if prev {
            pred
        } else if rem < pred {
            rem
        } else {
            rem + 1
        }
    }

    fn intra4_pred(&mut self, blk: usize) {
        let (x, y) = blk_xy(blk);
        let mode = self.derive_ipm(blk, x, y);
        self.cur.ipm[y * 4 + x] = mode;
        let (a_ok, b_ok, c_ok, d_ok) = (self.intra_ok(self.a), self.intra_ok(self.b), self.intra_ok(self.nb_c()), self.intra_ok(self.nb_d()));
        let left = x > 0 || a_ok;
        let top = y > 0 || b_ok;
        let corner = match (x > 0, y > 0) {
            (true, true) => true,
            (true, false) => b_ok,
            (false, true) => a_ok,
            _ => d_ok,
        };
        let tr = if y == 0 {
            if x < 3 {
                b_ok
            } else {
                c_ok
            }
        } else {
            x < 3 && xy_blk(x + 1, y - 1) < xy_blk(x, y)
        };
        let (px, py) = (self.rc.mbx * 16 + x * 4, self.rc.mby * 16 + y * 4);
        let e = self.edge(0, px, py, 4, top, left, corner, tr);
        let mut out = [0i32; 16];
        recon::pred4x4(&e, mode, &mut out);
        for j in 0..4 {
            for i in 0..4 {
                self.rc.pred[(y * 4 + j) * 16 + x * 4 + i] = out[j * 4 + i];
            }
        }
    }

    fn intra8_pred(&mut self, b8: usize) {
        let (x8, y8) = (b8 & 1, b8 >> 1);
        let mode = self.derive_ipm(b8, x8 * 2, y8 * 2);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            self.cur.ipm[(y8 * 2 + dy) * 4 + x8 * 2 + dx] = mode;
        }
        let (a_ok, b_ok, c_ok, d_ok) = (self.intra_ok(self.a), self.intra_ok(self.b), self.intra_ok(self.nb_c()), self.intra_ok(self.nb_d()));
        let left = x8 > 0 || a_ok;
        let top = y8 > 0 || b_ok;
        let corner = match (x8 > 0, y8 > 0) {
            (true, true) => true,
            (true, false) => b_ok,
            (false, true) => a_ok,
            _ => d_ok,
        };
        let tr = if y8 == 0 {
            if x8 == 0 {
                b_ok
            } else {
                c_ok
            }
        } else {
            x8 == 0
        };
        let (px, py) = (self.rc.mbx * 16 + x8 * 8, self.rc.mby * 16 + y8 * 8);
        let e = self.edge(0, px, py, 8, top, left, corner, tr);
        let mut out = [0i32; 64];
        recon::pred8x8(&e, mode, &mut out);
        for j in 0..8 {
            for i in 0..8 {
                self.rc.pred[(y8 * 8 + j) * 16 + x8 * 8 + i] = out[j * 8 + i];
            }
        }
    }

    /// Intra16x16 luma and intra chroma predictions of the current macroblock.
    fn intra_pred_mb(&mut self) {
        self.rc.dis = [0; 256];
        let (a_ok, b_ok, d_ok) = (self.intra_ok(self.a), self.intra_ok(self.b), self.intra_ok(self.nb_d()));
        let (mbx, mby) = (self.rc.mbx, self.rc.mby);
        if self.cur.i16 {
            let e = self.edge(0, mbx * 16, mby * 16, 16, b_ok, a_ok, d_ok, false);
            let mut out = [0i32; 256];
            recon::pred16x16(&e, self.rc.i16m, &mut out);
            self.rc.pred[..256].copy_from_slice(&out);
        }
        let mode = self.rc.cpm;
        for c in 0..2 {
            let e = self.edge(1 + c, mbx * 8, mby * 8, 8, b_ok, a_ok, d_ok, false);
            let mut out = [0i32; 64];
            recon::pred_chroma(&e, mode, &mut out);
            self.rc.pred[256 + c * 64..320 + c * 64].copy_from_slice(&out);
        }
    }

    /// Writes prediction plus residual (`None`: no residual) of an `n`x`n` luma block.
    fn put_luma(&mut self, x: usize, y: usize, n: usize, res: Option<&[i32]>) {
        let (px, py) = (self.rc.mbx * 16 + x, self.rc.mby * 16 + y);
        let rc = &*self.rc;
        let rec = self.pic.rec.as_mut().expect("reconstruction on");
        let w = rec.cur.y.w;
        for j in 0..n {
            let dst = &mut rec.cur.y.d[(py + j) * w + px..][..n];
            let pr = &rc.pred[(y + j) * 16 + x..][..n];
            match res {
                // Predictions are already within 0..=255.
                None => dst.iter_mut().zip(pr).for_each(|(d, &p)| *d = p as u8),
                Some(r) => {
                    let r = &r[j * n..][..n];
                    for i in 0..n {
                        dst[i] = recon::clip1(pr[i].saturating_add(r[i]));
                    }
                }
            }
        }
    }

    /// Reconstructs luma 4x4 block (`x`, `y`) of the current macroblock from its prediction
    /// and stored levels (`dc`: the scaled DC of Intra16x16 blocks).
    fn recon_l4(&mut self, x: usize, y: usize, dc: Option<i32>) {
        let coded = (self.cur.cbf >> xy_blk(x, y)) & 1 == 1;
        if !coded && dc.is_none_or(|d| d == 0) {
            self.put_luma(x * 4, y * 4, 4, None);
            return;
        }
        let lv = if coded { self.rc.l4[y * 4 + x] } else { [0; 16] };
        let mut res = [0i32; 16];
        let mut d = [0i32; 16];
        recon::dequant4(&lv, self.qp, dc, &mut d);
        recon::idct4(&d, &mut res);
        self.put_luma(x * 4, y * 4, 4, Some(&res));
    }

    fn recon_l8(&mut self, b8: usize) {
        if (self.cur.cbf >> (b8 * 4)) & 1 == 0 {
            self.put_luma((b8 & 1) * 8, (b8 >> 1) * 8, 8, None);
            return;
        }
        let lv = self.rc.l8[b8];
        let mut res = [0i32; 64];
        let mut d = [0i32; 64];
        recon::dequant8(&lv, self.qp, &mut d);
        recon::idct8(&d, &mut res);
        self.put_luma((b8 & 1) * 8, (b8 >> 1) * 8, 8, Some(&res));
    }

    /// Intra NxN macroblock without residual: every block is its prediction.
    fn intra_nxn_luma_norez(&mut self) {
        if self.cur.t8 {
            for b8 in 0..4 {
                self.intra8_pred(b8);
                self.recon_l8(b8);
            }
        } else {
            for blk in 0..16 {
                self.intra4_pred(blk);
                let (x, y) = blk_xy(blk);
                self.recon_l4(x, y, None);
            }
        }
        self.rc.luma_done = true;
    }

    /// Writes the rest of the macroblock into the picture and records its motion.
    fn recon_finish(&mut self) {
        let cbp = self.cur.cbp;
        if !self.rc.luma_done && !self.cur.i16 && cbp & 15 == 0 {
            self.put_luma(0, 0, 16, None);
            self.rc.luma_done = true;
        }
        if !self.rc.luma_done {
            if self.cur.i16 {
                let mut dcs = [0i32; 16];
                if (self.cur.cbf >> 16) & 1 == 1 {
                    recon::luma_dc(&self.rc.dc, self.qp, &mut dcs);
                }
                for k in 0..16 {
                    self.recon_l4(k % 4, k / 4, Some(dcs[k]));
                }
            } else if self.cur.t8 {
                for b8 in 0..4 {
                    self.recon_l8(b8);
                }
            } else {
                for k in 0..16 {
                    self.recon_l4(k % 4, k / 4, None);
                }
            }
        }
        // Chroma.
        for c in 0..2 {
            if self.rc.chroma_done {
                break;
            }
            if cbp >> 4 == 0 {
                let (cx0, cy0) = (self.rc.mbx * 8, self.rc.mby * 8);
                let rc = &*self.rc;
                let rec = self.pic.rec.as_mut().expect("reconstruction on");
                let p = if c == 0 { &mut rec.cur.u } else { &mut rec.cur.v };
                let w = p.w;
                for j in 0..8 {
                    let dst = &mut p.d[(cy0 + j) * w + cx0..][..8];
                    let src = &rc.pred[256 + c * 64 + j * 8..][..8];
                    dst.iter_mut().zip(src).for_each(|(d, &s)| *d = s as u8);
                }
                continue;
            }
            let off = if c == 0 { self.pps.chroma_qp_offset } else { self.pps.chroma_qp_offset2 };
            let qpc = recon::chroma_qp(self.qp, off);
            let mut dcs = [0i32; 4];
            if (self.cur.cbf >> (17 + c)) & 1 == 1 {
                recon::chroma_dc(&self.rc.cdc[c], qpc, &mut dcs);
            }
            let (cx0, cy0) = (self.rc.mbx * 8, self.rc.mby * 8);
            for b4 in 0..4 {
                let coded = (self.cur.cbf >> (19 + c * 4 + b4)) & 1 == 1;
                let mut res = [0i32; 16];
                if dcs[b4] != 0 || coded {
                    let lv = if coded { self.rc.cac[c * 4 + b4] } else { [0; 16] };
                    let mut d = [0i32; 16];
                    recon::dequant4(&lv, qpc, Some(dcs[b4]), &mut d);
                    recon::idct4(&d, &mut res);
                }
                let (x, y) = ((b4 & 1) * 4, (b4 >> 1) * 4);
                let rc = &*self.rc;
                let rec = self.pic.rec.as_mut().expect("reconstruction on");
                let p = if c == 0 { &mut rec.cur.u } else { &mut rec.cur.v };
                let w = p.w;
                for j in 0..4 {
                    for i in 0..4 {
                        p.d[(cy0 + y + j) * w + cx0 + x + i] = recon::clip1(rc.pred[256 + c * 64 + (y + j) * 8 + x + i].saturating_add(res[j * 4 + i]));
                    }
                }
            }
        }
        // Motion field (direct prediction in later pictures) and deblocking facts.
        let mut nz = 0u16;
        for k in 0..16 {
            if self.cur.nnz[k] > 0 {
                nz |= 1 << k;
            }
        }
        let qpc = [recon::chroma_qp(self.qp, self.pps.chroma_qp_offset) as u8, recon::chroma_qp(self.qp, self.pps.chroma_qp_offset2) as u8];
        let ri = &self.sh.ri;
        let dbi = recon::DbInfo {
            done: true,
            qp: self.qp as u8,
            qpc,
            intra: self.cur.intra,
            t8: self.cur.t8,
            nz,
            slice: self.sid,
            idc: ri.dbk_idc.min(2) as u8,
            off_a: ri.dbk_a.clamp(-12, 12) as i8,
            off_b: ri.dbk_b.clamp(-12, 12) as i8,
            uni: false,
        };
        let uni = !self.cur.intra
            && nz == 0
            && (0..2).all(|l| {
                let r = self.cur.rf[l][0];
                self.cur.rf[l].iter().all(|&v| v == r) && (r < 0 || self.cur.mv[l].iter().all(|&v| v == self.cur.mv[l][0]))
            });
        let dbi = recon::DbInfo { uni, ..dbi };
        let rec = self.pic.rec.as_mut().expect("reconstruction on");
        let w4 = rec.cur.w4;
        let wmb = w4 / 4;
        rec.cur.db[self.rc.mby * wmb + self.rc.mbx] = dbi;
        let mut rpoc = [[0i32; 4]; 2];
        for l in 0..2 {
            for b8 in 0..4 {
                let r = self.cur.rf[l][b8];
                rpoc[l][b8] = rec.refpic(l, r as i32).map_or(i32::MIN, |p| p.poc);
            }
        }
        for k in 0..16 {
            let (x, y) = (k % 4, k / 4);
            let b8 = (y / 2) * 2 + x / 2;
            let pos = (self.rc.mby * 4 + y) * w4 + self.rc.mbx * 4 + x;
            rec.cur.mf[pos] = Motion {
                mv: [self.cur.mv[0][k], self.cur.mv[1][k]],
                refi: [self.cur.rf[0][b8], self.cur.rf[1][b8]],
                rpoc: [rpoc[0][b8], rpoc[1][b8]],
            };
        }
    }

    /// Motion of a neighbouring 4x4 block (x, y relative to the macroblock, in 4x4 units)
    /// for list `l`: None when not available; refidx -1 when intra or the list is unused.
    fn mv_nb(&self, l: usize, x: i32, y: i32, done: u16) -> Option<(i8, [i16; 2])> {
        if x >= 0 && y >= 0 {
            if x >= 4 || y >= 4 {
                return None;
            }
            let k = (y * 4 + x) as usize;
            if (done >> k) & 1 == 0 {
                return None;
            }
            let r = self.cur.rf[l][(y as usize / 2) * 2 + x as usize / 2];
            return Some((r, if r >= 0 { self.cur.mv[l][k] } else { [0, 0] }));
        }
        let (addr, bx, by) = if x < 0 && y < 0 {
            (self.nb_d()?, 3, 3)
        } else if x < 0 {
            (self.a?, 3, y as usize)
        } else if x < 4 {
            (self.b?, x as usize, 3)
        } else {
            (self.nb_c()?, 0, 3)
        };
        let m = &self.pic.mbs[addr];
        let r = m.rf[l][(by / 2) * 2 + bx / 2];
        Some((r, if r >= 0 { m.mv[l][by * 4 + bx] } else { [0, 0] }))
    }

    /// Motion vector predictor (8.4.1.3); `dir` 1/2: 16x8 top/bottom, 3/4: 8x16 left/right.
    #[allow(clippy::too_many_arguments)]
    fn pred_mv(&self, l: usize, x: i32, y: i32, w: i32, refidx: i8, dir: usize, done: u16) -> [i16; 2] {
        let na = self.mv_nb(l, x - 1, y, done);
        let mut nb = self.mv_nb(l, x, y - 1, done);
        let mut nc = self.mv_nb(l, x + w, y - 1, done).or_else(|| self.mv_nb(l, x - 1, y - 1, done));
        let r = |n: Option<(i8, [i16; 2])>| n.map_or(-1, |v| v.0);
        let mv = |n: Option<(i8, [i16; 2])>| n.map_or([0, 0], |v| v.1);
        match dir {
            1 if r(nb) == refidx => return mv(nb),
            2 | 3 if r(na) == refidx => return mv(na),
            4 if r(nc) == refidx => return mv(nc),
            _ => {}
        }
        if nb.is_none() && nc.is_none() && na.is_some() {
            nb = na;
            nc = na;
        }
        let (ra, rb, rc) = (r(na), r(nb), r(nc));
        let hits = (ra == refidx) as u32 + (rb == refidx) as u32 + (rc == refidx) as u32;
        if hits == 1 {
            return if ra == refidx {
                mv(na)
            } else if rb == refidx {
                mv(nb)
            } else {
                mv(nc)
            };
        }
        let (a, b, c) = (mv(na), mv(nb), mv(nc));
        [med3(a[0], b[0], c[0]), med3(a[1], b[1], c[1])]
    }

    fn pskip_motion(&mut self) {
        let a = self.mv_nb(0, -1, 0, 0);
        let b = self.mv_nb(0, 0, -1, 0);
        let zero = match (a, b) {
            (Some(a), Some(b)) => (a.0 == 0 && a.1 == [0, 0]) || (b.0 == 0 && b.1 == [0, 0]),
            _ => true,
        };
        self.cur.rf = [[0; 4], [-1; 4]];
        self.cur.mv = [[[0; 2]; 16]; 2];
        let mv = if zero { [0, 0] } else { self.pred_mv(0, 0, 0, 4, 0, 0, 0) };
        self.cur.mv[0] = [mv; 16];
    }

    /// Motion of B_Skip / B_Direct_16x16 / direct 8x8 blocks (8.4.1.2).
    fn direct_motion(&self) -> ([[i8; 4]; 2], [[[i16; 2]; 16]; 2]) {
        let mut rf = [[-1i8; 4]; 2];
        let mut mv = [[[0i16; 2]; 16]; 2];
        let Some(rec) = self.pic.rec.as_ref() else { return (rf, mv) };
        let col = rec.refpic(1, 0);
        let (mbx, mby) = (self.rc.mbx, self.rc.mby);
        let colmot = |x: usize, y: usize| -> (i8, [i16; 2], i32) {
            let Some(p) = col.filter(|p| p.y.w == rec.cur.y.w && p.y.h == rec.cur.y.h) else { return (-1, [0, 0], 0) };
            let Some(&m) = p.mf.get((mby * 4 + y) * p.w4 + mbx * 4 + x) else { return (-1, [0, 0], 0) };
            if m.refi[0] >= 0 {
                (m.refi[0], m.mv[0], m.rpoc[0])
            } else if m.refi[1] >= 0 {
                (m.refi[1], m.mv[1], m.rpoc[1])
            } else {
                (-1, [0, 0], 0)
            }
        };
        let inf = self.sps.direct_8x8_inference;
        let cpos = |k: usize| -> (usize, usize) {
            let (x, y) = (k % 4, k / 4);
            if inf {
                ((x / 2) * 3, (y / 2) * 3)
            } else {
                (x, y)
            }
        };
        if self.sh.direct_spatial {
            let mut r = [-1i8; 2];
            for (l, rl) in r.iter_mut().enumerate() {
                let a = self.mv_nb(l, -1, 0, 0).map_or(-1, |v| v.0);
                let b = self.mv_nb(l, 0, -1, 0).map_or(-1, |v| v.0);
                let c = self.mv_nb(l, 4, -1, 0).or_else(|| self.mv_nb(l, -1, -1, 0)).map_or(-1, |v| v.0);
                *rl = minpos(a, minpos(b, c));
            }
            let zero = r[0] < 0 && r[1] < 0;
            if zero {
                r = [0, 0];
            }
            let mvp = [0, 1].map(|l| if !zero && r[l] >= 0 { self.pred_mv(l, 0, 0, 4, r[l], 0, 0) } else { [0, 0] });
            let lt = col.is_none_or(|p| p.long_term);
            for k in 0..16 {
                let (cx, cy) = cpos(k);
                let (ri, cmv, _) = colmot(cx, cy);
                let colzero = !lt && ri == 0 && cmv[0].abs() <= 1 && cmv[1].abs() <= 1;
                for l in 0..2 {
                    if r[l] >= 0 && !(zero || (r[l] == 0 && colzero)) {
                        mv[l][k] = mvp[l];
                    }
                }
            }
            rf = [[r[0]; 4], [r[1]; 4]];
        } else {
            // Temporal direct.
            let p1poc = col.map_or(0, |p| p.poc);
            for k in 0..16 {
                let (x, y) = (k % 4, k / 4);
                let b8 = (y / 2) * 2 + x / 2;
                let (cx, cy) = cpos(k);
                let (ri, cmv, rpoc) = colmot(cx, cy);
                let r0 = if ri < 0 {
                    0
                } else {
                    rec.list[0].iter().position(|&i| i != usize::MAX && rec.dpb[i].poc == rpoc).unwrap_or(0) as i8
                };
                rf[0][b8] = r0;
                rf[1][b8] = 0;
                let p0 = rec.refpic(0, r0 as i32);
                let p0poc = p0.map_or(0, |p| p.poc);
                let tb = (rec.cur.poc - p0poc).clamp(-128, 127);
                let td = (p1poc - p0poc).clamp(-128, 127);
                if td == 0 || p0.is_none_or(|p| p.long_term) {
                    mv[0][k] = cmv;
                } else {
                    let tx = (16384 + (td / 2).abs()) / td;
                    let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
                    let m0 = [((dsf * cmv[0] as i32 + 128) >> 8) as i16, ((dsf * cmv[1] as i32 + 128) >> 8) as i16];
                    mv[0][k] = m0;
                    mv[1][k] = [m0[0].wrapping_sub(cmv[0]), m0[1].wrapping_sub(cmv[1])];
                }
            }
        }
        (rf, mv)
    }

    fn set_motion(&mut self, l: usize, x: usize, y: usize, w: usize, h: usize, mv: [i16; 2]) {
        for yy in y..y + h {
            for xx in x..x + w {
                self.cur.mv[l][yy * 4 + xx] = mv;
            }
        }
    }

    /// Inter prediction of the current macroblock from its motion.
    fn inter_pred(&mut self) {
        let Some(rec) = self.pic.rec.as_ref() else { return };
        let rc = &mut *self.rc;
        let (mbx, mby) = (rc.mbx, rc.mby);
        let implicit = self.pps.weighted_bipred_idc == 2 && self.sh.slice_type == B;
        let explicit = self.sh.ri.explicit_wp;
        // Every sample of both buffers is written below (the groups cover the macroblock).
        let (out, dis) = (&mut rc.pred, &mut rc.dis);
        // Blocks with the same motion are predicted together (the samples do not depend on
        // the block size).
        let b8 = |k: usize| ((k / 4) / 2) * 2 + (k % 4) / 2;
        let same = |a: usize, b: usize| -> bool {
            (0..2).all(|l| {
                let (ra, rb) = (self.cur.rf[l][b8(a)], self.cur.rf[l][b8(b)]);
                ra == rb && (ra < 0 || self.cur.mv[l][a] == self.cur.mv[l][b])
            })
        };
        let mut groups = [(0usize, 0usize, 0usize); 16];
        let mut ng = 0;
        if (1..16).all(|k| same(0, k)) {
            groups[0] = (0, 0, 4);
            ng = 1;
        } else {
            for q in 0..4 {
                let (x0, y0) = ((q & 1) * 2, (q >> 1) * 2);
                let k0 = y0 * 4 + x0;
                if [k0 + 1, k0 + 4, k0 + 5].iter().all(|&k| same(k0, k)) {
                    groups[ng] = (x0, y0, 2);
                    ng += 1;
                } else {
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        groups[ng] = (x0 + dx, y0 + dy, 1);
                        ng += 1;
                    }
                }
            }
        }
        for &(x, y, n) in &groups[..ng] {
            let k = y * 4 + x;
            let b8 = b8(k);
            let (lw, cw) = (n * 4, n * 2);
            let (pl, pc, sc) = (&mut rc.pl, &mut rc.pc, &mut rc.mc);
            let mut used = [false; 2];
            let mut pocs = [0i32; 2];
            let mut lts = [false; 2];
            for l in 0..2 {
                let r = self.cur.rf[l][b8];
                if r < 0 {
                    continue;
                }
                used[l] = true;
                let mv = self.cur.mv[l][k];
                let (mvx, mvy) = (mv[0] as i32, mv[1] as i32);
                match rec.refpic(l, r as i32) {
                    Some(p) => {
                        pocs[l] = p.poc;
                        lts[l] = p.long_term;
                        let (px, py) = ((mbx * 16 + x * 4) as i32, (mby * 16 + y * 4) as i32);
                        recon::mc_luma(sc, &p.y, px, py, mvx, mvy, lw, lw, &mut pl[l], 16);
                        let (cx, cy) = ((mbx * 8 + x * 2) as i32, (mby * 8 + y * 2) as i32);
                        recon::mc_chroma(&p.u, cx, cy, mvx, mvy, cw, cw, &mut pc[l][0], 8);
                        recon::mc_chroma(&p.v, cx, cy, mvx, mvy, cw, cw, &mut pc[l][1], 8);
                    }
                    None => {
                        pl[l] = [128; 256];
                        pc[l] = [[128; 64]; 2];
                    }
                }
            }
            let (w0, w1) = if implicit && used[0] && used[1] {
                let tb = (rec.cur.poc - pocs[0]).clamp(-128, 127);
                let td = (pocs[1] - pocs[0]).clamp(-128, 127);
                if td == 0 || lts[0] || lts[1] {
                    (32, 32)
                } else {
                    let tx = (16384 + (td / 2).abs()) / td;
                    let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
                    if !(-64..=128).contains(&(dsf >> 2)) {
                        (32, 32)
                    } else {
                        (64 - (dsf >> 2), dsf >> 2)
                    }
                }
            } else {
                (32, 32)
            };
            // How the two predictions combine, decided once per block (8.4.2.3): per component
            // (Y, Cb, Cr) the weights, offsets and log2 denominator, or the default average.
            let bi = used[0] && used[1];
            let ri = &self.sh.ri;
            let wts = |c: usize| -> [i32; 5] {
                let ld = ri.wp_denom[(c > 0) as usize] as i32;
                let ew = |l: usize| -> (i32, i32) {
                    let r = self.cur.rf[l][b8];
                    ri.wp[l].get(r.max(0) as usize).map_or((1 << ld, 0), |w| w[c])
                };
                let ((wa, oa), (wb, ob)) = (ew(0), ew(1));
                [wa, oa, wb, ob, ld]
            };
            // Writes `n`x`n` combined samples from `a`/`b` (stride `sa`) to `dst` (stride `sd`).
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            fn rows(dst: &mut [i32], sd: usize, a: &[i32], b: &[i32], sa: usize, n: usize, f: impl Fn(i32, i32) -> i32) {
                for j in 0..n {
                    let (ra, rb) = (&a[j * sa..j * sa + n], &b[j * sa..j * sa + n]);
                    for (i, o) in dst[j * sd..j * sd + n].iter_mut().enumerate() {
                        *o = f(ra[i], rb[i]);
                    }
                }
            }
            let combine = |dst: &mut [i32], sd: usize, a: &[i32], b: &[i32], sa: usize, n: usize, c: usize| {
                if explicit {
                    let [wa, oa, wb, ob, ld] = wts(c);
                    match (used[0], used[1]) {
                        (true, true) => rows(dst, sd, a, b, sa, n, |p, q| recon::clip1(((p * wa + q * wb + (1 << ld)) >> (ld + 1)) + ((oa + ob + 1) >> 1)) as i32),
                        (true, false) | (false, true) => {
                            let (w, o, first) = if used[0] { (wa, oa, true) } else { (wb, ob, false) };
                            if ld >= 1 {
                                rows(dst, sd, a, b, sa, n, |p, q| recon::clip1(((if first { p } else { q } * w + (1 << (ld - 1))) >> ld) + o) as i32)
                            } else {
                                rows(dst, sd, a, b, sa, n, |p, q| recon::clip1(if first { p } else { q } * w + o) as i32)
                            }
                        }
                        _ => rows(dst, sd, a, b, sa, n, |_, _| 128),
                    }
                } else {
                    match (used[0], used[1]) {
                        (true, true) if implicit => rows(dst, sd, a, b, sa, n, |p, q| recon::clip1((p * w0 + q * w1 + 32) >> 6) as i32),
                        (true, true) => {
                            for j in 0..n {
                                let (ra, rb) = (&a[j * sa..j * sa + n], &b[j * sa..j * sa + n]);
                                for (i, o) in dst[j * sd..j * sd + n].iter_mut().enumerate() {
                                    *o = (ra[i] + rb[i] + 1) >> 1;
                                }
                            }
                        }
                        (true, false) => {
                            for j in 0..n {
                                dst[j * sd..j * sd + n].copy_from_slice(&a[j * sa..j * sa + n]);
                            }
                        }
                        (false, true) => {
                            for j in 0..n {
                                dst[j * sd..j * sd + n].copy_from_slice(&b[j * sa..j * sa + n]);
                            }
                        }
                        _ => {
                            for j in 0..n {
                                dst[j * sd..j * sd + n].fill(128);
                            }
                        }
                    }
                }
            };
            let o = (y * 4) * 16 + x * 4;
            combine(&mut out[o..], 16, &pl[0], &pl[1], 16, lw, 0);
            for j in 0..lw {
                let row = &mut dis[o + j * 16..o + j * 16 + lw];
                if bi {
                    let (ra, rb) = (&pl[0][j * 16..j * 16 + lw], &pl[1][j * 16..j * 16 + lw]);
                    for (i, d) in row.iter_mut().enumerate() {
                        *d = ra[i] - rb[i];
                    }
                } else {
                    row.fill(0);
                }
            }
            for c in 0..2 {
                let o = 256 + c * 64 + (y * 2) * 8 + x * 2;
                combine(&mut out[o..], 8, &pc[0][c], &pc[1][c], 8, cw, 1 + c);
            }
        }
    }

    /// Slice type, intra flag and QP bucket, as extra model context.
    #[inline]
    fn extra(&self) -> u32 {
        self.sh.slice_type | (self.cur.intra as u32) << 2 | ((self.qp as u32) >> 2) << 3
    }

    /// Decodes a context-coded bin described by `h` (its CABAC context is `h.ctx`).
    #[inline]
    fn dech<T: BinIo>(&mut self, io: &mut T, h: &Hint) -> R<u32> {
        let ctx = h.ctx as usize;
        let s = self.st[ctx];
        let bin = io.decision(s, h)?;
        self.st[ctx] = next_state(s, bin);
        Ok(bin)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn dec4<T: BinIo>(&mut self, io: &mut T, ctx: usize, kind: u8, a: u32, b: u32, c: u32, d: u32) -> R<u32> {
        let s = self.st[ctx];
        let bin = io.decision(s, &Hint { ctx: ctx as u16, kind, a, b, c, d })?;
        self.st[ctx] = next_state(s, bin);
        Ok(bin)
    }

    /// Significance map and levels of one block; returns the number of non-zero
    /// coefficients and stores the signed levels in `out` (coefficient `i` at `i + off`).
    #[allow(clippy::too_many_arguments)]
    fn coeffs<T: BinIo>(
        &mut self,
        io: &mut T,
        cat: usize,
        max: usize,
        nc: u32,
        colo: u32,
        la: &[i8; 64],
        lb: &[i8; 64],
        off: usize,
        out: &mut [i8; 64],
    ) -> R<u8> {
        let (sig, last, abs) = if cat == 5 { (402, 417, 426) } else { (105 + SIG_OFF[cat], 166 + SIG_OFF[cat], 227 + ABS_OFF[cat]) };
        // Neighbours' non-zero counts at or after each position.
        let mut rem_a = [0u8; 65];
        let mut rem_b = [0u8; 65];
        for j in (0..max).rev() {
            rem_a[j] = rem_a[j + 1] + (la[j + off] != 0) as u8;
            rem_b[j] = rem_b[j + 1] + (lb[j + off] != 0) as u8;
        }
        let mut pos = [0u8; 64];
        let mut n = 0usize;
        let mut i = 0usize;
        let mut found_last = false;
        let cat_h = (cat as u32) << 8;
        let ex = self.extra();
        let nbh = nc | q(colo, 63) << 6;
        let on = self.rc.on;
        let feat = self.rc.feat;
        let frem = self.rc.frem;
        let ftex = self.rc.ftex;
        let bok = if self.rc.on && (cat == 1 || cat == 2 || cat == 5) { self.rc.bnd_ok } else { 0 };
        let n8 = cat == 5;
        let mut bnd = self.rc.bnd;
        let bqp = self.rc.bqp;
        let fdis = self.rc.fdis;
        let fd = |i: usize| -> u32 { if (cat == 1 || cat == 2) && i < 16 { fdis[i] as u32 } else { 0 } };
        if self.rc.on {
            self.rc.cx = [0; 64];
        }
        // Hints of the significance-map bins (with `n` coefficients found so far, or including
        // the one at `i` for last_significant_coeff_flag) and of the first level bin of a
        // coefficient. Besides coding, they announce the possible next bins to the model.
        let _ = sig;
        let sig_h = |i: usize, n: usize| {
            let mut h = sig_hint(cat, i, n, nbh, ex, la[i + off], lb[i + off], rem_a[i], rem_b[i]);
            if on {
                h.a |= (feat[i + off] as u32) << 16;
                h.c |= fd(i + off) << 12;
                h.d |= ftex << 24;
            }
            h
        };
        let last_h = |i: usize, n: usize| -> Hint {
            let li = if cat == 5 {
                LAST_8X8[i] as usize
            } else if cat == 3 {
                i.min(2)
            } else {
                i
            };
            let nbi = lvq(la[i + off], 3) | lvq(lb[i + off], 3) << 2;
            let after = rem_a[i + 1] as u32 + rem_b[i + 1] as u32;
            let mut h = Hint {
                ctx: (last + li) as u16,
                kind: kind::LAST,
                a: cat_h | i as u32,
                b: nbh | q(n as u32, 15) << 12,
                c: nbi | ((rem_a[i + 1] > 0) as u32) << 4 | ((rem_b[i + 1] > 0) as u32) << 5,
                d: q(after, 15) | q(n as u32, 15) << 4 | ex << 8,
            };
            if on {
                h.a |= (frem[i + off + 1] as u32) << 16;
            }
            h
        };
        #[allow(clippy::too_many_arguments)]
        let lvl_h = |p: usize, k: usize, n: usize, prev: u32, maxv: u32, gt1: usize, eq1: usize| -> Hint {
            let inc0 = if gt1 != 0 { 0 } else { (1 + eq1).min(4) };
            let (va, vb) = (la[p + off], lb[p + off]);
            let mut h = Hint {
                ctx: (abs + inc0) as u16,
                kind: kind::ABS,
                a: cat_h | p as u32,
                b: q(prev, 15) | (q(k as u32, 15)) << 4 | q(nc, 17) << 8 | (q(n as u32, 31)) << 13,
                c: lvq(va, 15) | lvq(vb, 15) << 4 | q(prev, 15) << 8,
                d: q(maxv, 15) | q(gt1 as u32, 7) << 4 | q(eq1 as u32, 7) << 7 | ex << 10,
            };
            if on {
                h.d |= (feat[p + off] as u32) << 24;
            }
            h
        };
        // The bin after the map when it ends with `n` coefficients, the last one at `p`.
        let first_lvl = |n: usize, p: usize| lvl_h(p, n - 1, n, 0, 0, 0, 0);
        while i + 1 < max {
            let h = sig_h(i, n);
            io.prefetch(&last_h(i, n + 1));
            io.prefetch(&if i + 2 < max { sig_h(i + 1, n) } else { first_lvl(n + 1, max - 1) });
            if self.dech(io, &h)? == 1 {
                pos[n] = i as u8;
                n += 1;
                let h = last_h(i, n);
                io.prefetch(&first_lvl(n, i));
                io.prefetch(&if i + 2 < max { sig_h(i + 1, n) } else { first_lvl(n + 1, max - 1) });
                if self.dech(io, &h)? == 1 {
                    found_last = true;
                    break;
                }
            }
            i += 1;
        }
        if !found_last {
            pos[n] = (max - 1) as u8;
            n += 1;
        }
        let mut gt1 = 0usize;
        let mut eq1 = 0usize;
        let mut prev = 0u32;
        let mut maxv = 0u32;
        let mut prev_sign = 0u32;
        for k in (0..n).rev() {
            let p = pos[k] as usize;
            let h0 = lvl_h(p, k, n, prev, maxv, gt1, eq1);
            let (va, vb) = (la[p + off], lb[p + off]);
            let inc1 = 5 + gt1.min(4 - (cat == 3) as usize);
            // Boundary sign prediction: the sign whose basis function reduces the mismatch.
            let r4 = if n8 { recon::ZZ8[p] as usize } else { recon::ZZ4[(p + off).min(15)] as usize };
            let proj = if bok == 0 {
                0
            } else if n8 {
                bproj8(&bnd, bok, r4)
            } else {
                bproj(&bnd, bok, r4)
            };
            let sp = |v: u32| -> u32 {
                if bok == 0 {
                    return 0;
                }
                let (d, sh) = if n8 { (deq8(v as i32 + 1, bqp, r4) as u64, 25) } else { (deq1(v as i32 + 1, bqp, r4) as u64, 14) };
                let conf = bucket(((proj.unsigned_abs() * d) >> sh).min(1 << 30) as u32) as u32;
                if conf == 0 {
                    1
                } else {
                    2 + (proj > 0) as u32 + 2 * (conf - 1)
                }
            };
            let sign_h = |v: u32| {
                let mut h = Hint {
                    ctx: 0,
                    kind: kind::SIGN,
                    a: cat as u32,
                    b: q(v, 3) | (p as u32) << 2,
                    c: sgn(va) | sgn(vb) << 2,
                    d: prev_sign | (p as u32) << 2,
                };
                let x = sp(v);
                if x != 0 {
                    h.a |= x << 8;
                    h.c |= x << 4;
                }
                h
            };
            let hs = lvq(va, 63) | lvq(vb, 63) << 6;
            let suf0 = Hint { ctx: 0, kind: kind::ABS_SUF, a: 0, b: q(maxv, 63), c: hs, d: 0 };
            let more = |v: u32| Hint { ctx: (abs + inc1) as u16, a: h0.a | v << 16, ..h0 };
            io.prefetch(&more(1));
            io.prefetch(&sign_h(0));
            let mut v = 0u32;
            if self.dech(io, &h0)? == 1 {
                v = 1;
                while v < 14 {
                    let h = more(v);
                    io.prefetch(&if v + 1 < 14 { more(v + 1) } else { suf0 });
                    io.prefetch(&sign_h(v));
                    if self.dech(io, &h)? == 0 {
                        break;
                    }
                    v += 1;
                }
                if v >= 14 {
                    let mut kk = 0u32;
                    loop {
                        let h = Hint { a: kk, ..suf0 };
                        io.prefetch(&Hint { a: kk + 1, ..h });
                        io.prefetch(&if kk > 0 { Hint { a: 32 + kk - 1, b: 1, ..h } } else { sign_h(14) });
                        if io.bypass(&h)? == 0 {
                            break;
                        }
                        v += 1 << kk;
                        kk += 1;
                        if kk > 28 {
                            return Err(Bad);
                        }
                    }
                    let mut acc = 1u32;
                    while kk > 0 {
                        kk -= 1;
                        let h = Hint { a: 32 + kk, b: acc.min(255), ..suf0 };
                        if kk > 0 {
                            io.prefetch(&Hint { a: 32 + kk - 1, b: (acc << 1).min(255), ..h });
                            io.prefetch(&Hint { a: 32 + kk - 1, b: (acc << 1 | 1).min(255), ..h });
                        } else {
                            io.prefetch(&sign_h(14));
                        }
                        let b = io.bypass(&h)?;
                        acc = (acc << 1) | b;
                        v += b << kk;
                    }
                }
            }
            let sh = sign_h(v);
            if v == 0 {
                eq1 += 1;
            } else {
                gt1 += 1;
            }
            prev = v + 1;
            maxv = maxv.max(v + 1);
            // The sign does not change the next coefficient's first level bin.
            if k > 0 {
                io.prefetch(&lvl_h(pos[k - 1] as usize, k - 1, n, prev, maxv, gt1, eq1));
            }
            let s = io.bypass(&sh)?;
            prev_sign = 1 + s;
            let mag = (v + 1).min(127) as i8;
            out[p + off] = if s == 1 { -mag } else { mag };
            if self.rc.on {
                // Conforming streams stay far inside this range; the clamp keeps the
                // reconstruction arithmetic of corrupt ones in range.
                let lv = (v + 1).min(1 << 15) as i32;
                self.rc.cx[p + off] = if s == 1 { -lv } else { lv };
                if bok != 0 {
                    let sv = if s == 1 { -lv } else { lv };
                    if n8 {
                        let d = deq8(sv, bqp, r4);
                        let (u, vv) = (r4 & 7, r4 >> 3);
                        for x in 0..8 {
                            bnd[x] = bnd[x].wrapping_add(d.wrapping_mul(recon::C8[x][u] * recon::C8[0][vv]));
                            bnd[8 + x] = bnd[8 + x].wrapping_add(d.wrapping_mul(recon::C8[0][u] * recon::C8[x][vv]));
                        }
                    } else {
                        let d = deq1(sv, bqp, r4);
                        let (u, vv) = (r4 & 3, r4 >> 2);
                        for x in 0..4 {
                            bnd[x] = bnd[x].wrapping_add(d.wrapping_mul(CI2[x][u]));
                            bnd[8 + x] = bnd[8 + x].wrapping_add(d.wrapping_mul(CI2[x][vv]));
                        }
                    }
                }
            }
        }
        Ok(n.min(255) as u8)
    }
}
