//! HEVC CABAC slice-segment-data parser (7.3.8, 9.3).
//!
//! Like the H.264 parser it derives every bin's context exactly as the standard does and asks
//! a [`BinIo`] for the bin, so the same code drives verification (decode the original payload
//! and re-encode it), compression and decompression. Each bin carries a [`Hint`] with extra
//! context the standard does not use: neighbouring and co-located block statistics, the
//! coefficients already decoded around a position, QP, slice type and temporal layer.

use super::tables::{self as t, init_states, init_type, CTX_IDX_MAP, NUM_CTX, SCAN, SCAN_POS};
use super::{Geometry, Pps, SliceHeader, Sps, B, I};
use crate::bits::{Bad, R};
use crate::h264::cabac::next_state;
use crate::h264::mb::{BinIo, Hint};

/// Syntax-element classes, used to select model contexts.
pub mod kind {
    pub const SAO: u8 = 0;
    pub const SPLIT_CU: u8 = 1;
    pub const TQ_BYPASS: u8 = 2;
    pub const SKIP: u8 = 3;
    pub const PRED_MODE: u8 = 4;
    pub const PART_MODE: u8 = 5;
    pub const PREV_INTRA: u8 = 6;
    /// mpm_idx and rem_intra_luma_pred_mode (bypass bins).
    pub const INTRA_MODE: u8 = 7;
    pub const CHROMA_PRED: u8 = 8;
    pub const MERGE_FLAG: u8 = 9;
    pub const MERGE_IDX: u8 = 10;
    pub const INTER_DIR: u8 = 11;
    pub const REF_IDX: u8 = 12;
    pub const MVP_FLAG: u8 = 13;
    pub const MVD: u8 = 14;
    pub const MVD_SUF: u8 = 15;
    pub const MVD_SIGN: u8 = 16;
    pub const RQT_ROOT_CBF: u8 = 17;
    pub const SPLIT_TU: u8 = 18;
    /// end_of_slice_segment_flag. Same value as the H.264 END kind: the model codes
    /// terminating bins with a small adaptive counter.
    pub const END: u8 = 19;
    /// pcm_flag (terminating bin).
    pub const PCM: u8 = 20;
    pub const CBF_LUMA: u8 = 21;
    pub const CBF_CHROMA: u8 = 22;
    pub const DQP: u8 = 23;
    pub const TSKIP: u8 = 24;
    pub const LAST: u8 = 25;
    pub const CSBF: u8 = 26;
    pub const SIG: u8 = 27;
    pub const GT1: u8 = 28;
    pub const GT2: u8 = 29;
    /// coeff_abs_level_remaining (bypass bins).
    pub const REM: u8 = 30;
    pub const SIGN: u8 = 31;
    pub const COUNT: usize = 32;
    pub const NAMES: [&str; COUNT] = [
        "sao",
        "split_cu",
        "tq_bypass",
        "skip",
        "pred_mode",
        "part_mode",
        "prev_intra",
        "intra_mode",
        "chroma_pred",
        "merge_flag",
        "merge_idx",
        "inter_dir",
        "ref_idx",
        "mvp_flag",
        "mvd",
        "mvd_suffix",
        "mvd_sign",
        "rqt_root_cbf",
        "split_tu",
        "end",
        "pcm",
        "cbf_luma",
        "cbf_chroma",
        "cu_qp_delta",
        "tskip",
        "last_pos",
        "csbf",
        "sig",
        "gt1",
        "gt2",
        "remaining",
        "sign",
    ];
}

const _: () = assert!(kind::END == crate::h264::mb::kind::END && kind::PCM == crate::h264::mb::kind::PCM);

// PartMode values (Table 7-10).
const P2NX2N: u8 = 0;
const P2NXN: u8 = 1;
const PNX2N: u8 = 2;
const PNXN: u8 = 3;
const P2NXNU: u8 = 4;
const P2NXND: u8 = 5;
const PNLX2N: u8 = 6;
const PNRX2N: u8 = 7;

// Blk flags.
const F_SKIP: u8 = 1;
const F_INTRA: u8 = 2;
const F_MERGE: u8 = 4;
const F_CODED: u8 = 8;

/// What later blocks of the same slice need to know about a 4x4 luma block.
#[derive(Clone, Copy, Default)]
struct Blk {
    /// Slice that wrote this entry; entries of other slices (or pictures) are unavailable.
    sid: u32,
    depth: u8,
    flags: u8,
    /// IntraPredModeY (INTRA_DC for blocks that are not intra coded).
    ipm: u8,
    qp: i8,
    /// Non-zero luma coefficients per 4x4 area of the transform block, capped at 15.
    nz: u8,
    /// The same for chroma (Cb and Cr together).
    nzc: u8,
    /// Size class of the motion vector difference of the prediction block (0 = none).
    mvd: u8,
    /// Signs (0 = none, 1 = positive, 2 = negative) of the low-frequency coefficients of the
    /// transform blocks covering this block, 2 bits each: luma (0,0) (1,0) (0,1) (2,0) (1,1)
    /// (0,2) in bits 0..12, Cb (0,0) (1,0) (0,1) in bits 12..18.
    sg: u32,
}

/// What the model remembers about each 8x8 block of the previous picture.
#[derive(Clone, Copy, Default)]
pub struct Summary {
    /// 0 = unknown, 1 = skip, 2 = merge, 3 = inter with motion vector differences, 4 = intra.
    pub class: u8,
    pub depth: u8,
    pub nz: u8,
    pub mvd: u8,
}

/// Per-stream parser state.
///
/// `blk` is scratch space for neighbour lookups: entries carry the id of the slice that wrote
/// them and are only read back by that slice. The summaries the model uses (`cur`, `prev`) and
/// the context-variable stores that dependent slice segments and wavefront rows resume from
/// are shared between slices, but a slice only ever reads what earlier segments of the same
/// slice wrote, and the recoder only recodes a dependent segment when every earlier segment of
/// its slice was recoded. The summaries change only in [`SliceParser::commit`].
pub struct Picture {
    w: u32,
    h: u32,
    w4: usize,
    w8: usize,
    blk: Vec<Blk>,
    next_sid: u32,
    /// Slice id and tile-scan address following the last parsed segment.
    cur_sid: u32,
    next_ts: usize,
    wpp: [u8; NUM_CTX],
    ds: [u8; NUM_CTX],
    cur: Vec<Summary>,
    prev: Vec<Summary>,
}

impl Default for Picture {
    fn default() -> Self {
        Picture {
            w: 0,
            h: 0,
            w4: 0,
            w8: 0,
            blk: Vec::new(),
            next_sid: 0,
            cur_sid: 0,
            next_ts: usize::MAX,
            wpp: [0; NUM_CTX],
            ds: [0; NUM_CTX],
            cur: Vec::new(),
            prev: Vec::new(),
        }
    }
}

static EMPTY: Summary = Summary { class: 0, depth: 0, nz: 0, mvd: 0 };

#[derive(Clone, Copy, Default)]
struct Cu {
    x0: u32,
    y0: u32,
    log2: u32,
    depth: u32,
    intra: bool,
    tq_bypass: bool,
    part: u8,
    intra_split: bool,
    max_trafo_depth: u32,
    ipm: [u8; 4],
    ipm_c: u8,
    merge0: bool,
    /// Summary class (see [`Summary`]).
    class: u8,
    nz: u32,
    mvd: u32,
}

#[inline]
fn q(v: u32, max: u32) -> u32 {
    v.min(max)
}

/// Coarse logarithmic class of a count: 0, 1, 2, 3-4, 5-8, 9-16, more (16 = unavailable -> 7).
#[inline]
fn ql(v: u32) -> u32 {
    match v {
        0..=2 => v,
        3..=4 => 3,
        5..=8 => 4,
        9..=15 => 5,
        16 => 7,
        _ => 6,
    }
}

/// Class of a sum of neighbouring levels (0..=12).
#[inline]
fn qsum(v: u32) -> u32 {
    match v {
        0..=4 => v,
        5..=6 => 5,
        7..=8 => 6,
        9..=12 => 7,
        13..=16 => 8,
        17..=24 => 9,
        25..=32 => 10,
        33..=48 => 11,
        _ => 12,
    }
}

/// Class of a position on the anti-diagonal x + y of a transform block.
#[inline]
fn dclass(d: u32) -> u32 {
    match d {
        0 => 0,
        1..=2 => 1,
        3..=4 => 2,
        5..=7 => 3,
        8..=12 => 4,
        _ => 5,
    }
}

/// Index of the low-frequency positions (x + y <= 2) whose signs are kept as context, by
/// `y * 3 + x`.
const LF_POS: [u8; 9] = [0, 1, 3, 2, 4, 9, 5, 9, 9];

/// Size class of a motion vector difference component.
#[inline]
fn mvd_class(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        (33 - v.leading_zeros()).min(15)
    }
}

/// Parses one slice segment. Create it with [`SliceParser::new`], then call [`SliceParser::run`].
pub struct SliceParser<'a> {
    sps: &'a Sps,
    pps: &'a Pps,
    sh: &'a SliceHeader,
    geo: &'a Geometry,
    pic: &'a mut Picture,
    st: [u8; NUM_CTX],
    sid: u32,
    init_type: usize,
    chroma: u32,
    w_ctb: usize,
    n_ctb: usize,
    first_ts: usize,
    /// CTB being parsed: raster address and tile.
    rs: usize,
    tile: u16,
    cu: Cu,
    // QP tracking (8.6.1), used only as model context.
    qp: i32,
    qp_prev: i32,
    log2_qg: u32,
    is_dqp_coded: bool,
    dqp: i32,
    /// Slice type, temporal layer and QP bucket, as model context.
    ctx_slice: u32,
    // Coefficients of the transform block being parsed, for model context.
    tu_abs: [u16; 1024],
    tu_sgn: [u8; 1024],
    tu_nz: [u32; 2],
    /// Low-frequency signs of the current transform unit's Cb block (context for Cr).
    tu_cb_sg: u32,
    last_prev: [[(u8, u8); 4]; 2],
    // Model history.
    newpic: bool,
    fresh: bool,
    pending: Vec<(usize, Summary)>,
    /// Number of coding tree units parsed.
    pub count: usize,
}

impl<'a> SliceParser<'a> {
    pub fn new(sps: &'a Sps, pps: &'a Pps, sh: &'a SliceHeader, geo: &'a Geometry, pic: &'a mut Picture) -> R<Self> {
        let n_ctb = geo.w_ctb * geo.h_ctb;
        let addr = sh.segment_address as usize;
        if addr >= n_ctb || (sh.slice_addr as usize) > addr {
            return Err(Bad);
        }
        let first_ts = geo.rs_to_ts[addr] as usize;
        if pic.w != sps.width || pic.h != sps.height || pic.blk.is_empty() {
            pic.w = sps.width;
            pic.h = sps.height;
            pic.w4 = sps.width.div_ceil(4) as usize;
            pic.w8 = sps.width.div_ceil(8) as usize;
            pic.blk = vec![Blk::default(); pic.w4 * sps.height.div_ceil(4) as usize];
            pic.next_ts = usize::MAX;
        }
        let n8 = pic.w8 * sps.height.div_ceil(8) as usize;
        let fresh = pic.cur.len() != n8 || pic.prev.len() != n8;
        let sid = if sh.dependent {
            // A dependent segment continues the slice the previous segment belonged to.
            if pic.next_ts != first_ts {
                return Err(Bad);
            }
            pic.cur_sid
        } else {
            pic.next_sid = pic.next_sid.wrapping_add(1).max(1);
            pic.next_sid
        };
        let it = init_type(sh.slice_type, sh.cabac_init_flag);
        let log2_qg = sps.log2_ctb - pps.diff_cu_qp_delta_depth.min(sps.log2_ctb - sps.log2_min_cb);
        let qp_b = ((sh.slice_qp + 12).clamp(0, 63) as u32) >> 2;
        Ok(SliceParser {
            sps,
            pps,
            sh,
            geo,
            pic,
            st: [0; NUM_CTX],
            sid,
            init_type: it,
            chroma: sps.chroma_array_type(),
            w_ctb: geo.w_ctb,
            n_ctb,
            first_ts,
            rs: addr,
            tile: 0,
            cu: Cu::default(),
            qp: sh.slice_qp,
            qp_prev: sh.slice_qp,
            log2_qg,
            is_dqp_coded: false,
            dqp: 0,
            ctx_slice: sh.slice_type | q(sh.temporal_id, 7) << 2 | qp_b << 5,
            tu_abs: [0; 1024],
            tu_sgn: [0; 1024],
            tu_nz: [0; 2],
            tu_cb_sg: 0,
            last_prev: [[(255, 255); 4]; 2],
            newpic: sh.first_in_pic,
            fresh,
            pending: Vec::new(),
            count: 0,
        })
    }

    /// Makes this segment part of the history the model sees. Call only for segments that
    /// were recoded, in stream order, on both the compressing and the decompressing side.
    pub fn commit(self) {
        let n = self.pic.w8 * self.sps.height.div_ceil(8) as usize;
        if self.fresh {
            self.pic.cur = vec![Summary::default(); n];
            self.pic.prev = vec![Summary::default(); n];
        }
        if self.newpic {
            std::mem::swap(&mut self.pic.cur, &mut self.pic.prev);
            self.pic.cur.iter_mut().for_each(|s| *s = Summary::default());
        }
        for (i, s) in self.pending {
            self.pic.cur[i] = s;
        }
    }

    /// Co-located 8x8 block of the previous picture.
    #[inline]
    fn colo(&self, x: u32, y: u32) -> &Summary {
        if self.fresh || x >= self.pic.w || y >= self.pic.h {
            return &EMPTY;
        }
        let i = (y >> 3) as usize * self.pic.w8 + (x >> 3) as usize;
        if self.newpic {
            &self.pic.cur[i]
        } else {
            &self.pic.prev[i]
        }
    }

    // --- bins ----------------------------------------------------------------------------

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn dec<T: BinIo>(&mut self, io: &mut T, ctx: usize, kind: u8, a: u32, b: u32, c: u32, d: u32) -> R<u32> {
        let s = self.st[ctx];
        let bin = io.decision(s, &Hint { ctx: ctx as u16, kind, a, b, c, d })?;
        self.st[ctx] = next_state(s, bin);
        Ok(bin)
    }

    #[inline]
    fn byp<T: BinIo>(&mut self, io: &mut T, kind: u8, a: u32, b: u32, c: u32, d: u32) -> R<u32> {
        io.bypass(&Hint { ctx: 0, kind, a, b, c, d })
    }

    /// k-th order Exp-Golomb bypass bins (9.3.3.3).
    fn eg_k<T: BinIo>(&mut self, io: &mut T, k: u32, kind: u8, b: u32, c: u32) -> R<u32> {
        let mut k = k;
        let mut v = 0u32;
        let mut n = 0;
        while self.byp(io, kind, n, b, c, 0)? == 1 {
            v = v.checked_add(1 << k).ok_or(Bad)?;
            k += 1;
            n += 1;
            if k > 24 {
                return Err(Bad);
            }
        }
        let mut acc = 1u32;
        while k > 0 {
            k -= 1;
            let bit = self.byp(io, kind, 32 + k, b, c, acc.min(255))?;
            acc = (acc << 1) | bit;
            v += bit << k;
        }
        Ok(v)
    }

    // --- neighbours ----------------------------------------------------------------------

    /// The 4x4 block covering luma sample (x, y) if it is available to the current block
    /// (6.4.1: inside the picture, same slice, same tile, already decoded).
    #[inline]
    fn nb(&self, x: i32, y: i32) -> Option<&Blk> {
        if x < 0 || y < 0 || x as u32 >= self.pic.w || y as u32 >= self.pic.h {
            return None;
        }
        let b = &self.pic.blk[(y as usize >> 2) * self.pic.w4 + (x as usize >> 2)];
        if b.sid != self.sid {
            return None;
        }
        if self.pps.tiles {
            let l = self.sps.log2_ctb;
            let rs = (y as usize >> l) * self.w_ctb + (x as usize >> l);
            if self.geo.tile_rs[rs] != self.tile {
                return None;
            }
        }
        Some(b)
    }

    /// Writes `f` into the 4x4 blocks of a luma rectangle.
    fn set_blocks(&mut self, x0: u32, y0: u32, w: u32, h: u32, f: impl Fn(&mut Blk)) {
        let x1 = (x0 + w).min(self.pic.w).div_ceil(4) as usize;
        let y1 = (y0 + h).min(self.pic.h).div_ceil(4) as usize;
        for by in (y0 >> 2) as usize..y1 {
            let row = by * self.pic.w4;
            for bx in (x0 >> 2) as usize..x1 {
                f(&mut self.pic.blk[row + bx]);
            }
        }
    }

    /// Neighbour class for the model: 0..4 as in [`Summary`], 5 = unavailable.
    #[inline]
    fn nb_class(&self, x: i32, y: i32) -> u32 {
        match self.nb(x, y) {
            None => 5,
            Some(b) => {
                if b.flags & F_INTRA != 0 {
                    4
                } else if b.flags & F_SKIP != 0 {
                    1
                } else if b.flags & F_MERGE != 0 {
                    2
                } else {
                    3
                }
            }
        }
    }

    // --- slice segment data --------------------------------------------------------------

    /// Context variables at the start of a substream (9.3.1, 9.3.2.2 - 9.3.2.4).
    fn init_contexts(&mut self, ts: usize, segment_start: bool) {
        let rs = self.geo.ts_to_rs[ts] as usize;
        let first_in_tile = ts == 0 || self.geo.tile_ts(ts) != self.geo.tile_ts(ts - 1);
        if !first_in_tile && self.pps.entropy_coding_sync && rs.is_multiple_of(self.w_ctb) {
            // Wavefront: resume from the state after the second CTB of the row above, if the
            // CTB above and to the right belongs to this slice.
            let t_avail = self.w_ctb > 1 && rs >= self.w_ctb && rs + 1 - self.w_ctb >= self.sh.slice_addr as usize;
            if t_avail {
                self.st = self.pic.wpp;
                return;
            }
        } else if !first_in_tile && segment_start && self.sh.dependent {
            self.st = self.pic.ds;
            return;
        }
        init_states(&mut self.st, self.sh.slice_qp, self.init_type);
    }

    pub fn run<T: BinIo>(&mut self, io: &mut T) -> R<()> {
        let mut ts = self.first_ts;
        self.init_contexts(ts, true);
        loop {
            let rs = self.geo.ts_to_rs[ts] as usize;
            self.rs = rs;
            self.tile = self.geo.tile_rs[rs];
            let first_in_tile = ts == 0 || self.geo.tile_ts(ts) != self.geo.tile_ts(ts - 1);
            if ts == self.first_ts || first_in_tile || (self.pps.entropy_coding_sync && rs.is_multiple_of(self.w_ctb)) {
                self.qp_prev = self.sh.slice_qp;
            }
            self.ctu(io, rs)?;
            if self.pps.entropy_coding_sync && self.w_ctb > 1 && rs % self.w_ctb == 1 {
                self.pic.wpp = self.st;
            }
            self.count += 1;
            let last = (ts + 1 == self.n_ctb) as u32;
            let end = io.terminate(&Hint { ctx: NUM_CTX as u16, kind: kind::END, a: last, b: 0, c: 0, d: 0 })?;
            ts += 1;
            if end == 1 {
                break;
            }
            if ts >= self.n_ctb {
                return Err(Bad);
            }
            let rs2 = self.geo.ts_to_rs[ts] as usize;
            let new_tile = self.pps.tiles && self.geo.tile_ts(ts) != self.geo.tile_ts(ts - 1);
            let new_row = self.pps.entropy_coding_sync && rs2.is_multiple_of(self.w_ctb);
            if new_tile || new_row {
                io.end_substream()?;
                self.init_contexts(ts, false);
            }
        }
        if self.pps.dependent_slice_segments {
            self.pic.ds = self.st;
        }
        self.pic.cur_sid = self.sid;
        self.pic.next_ts = ts;
        Ok(())
    }

    fn ctu<T: BinIo>(&mut self, io: &mut T, rs: usize) -> R<()> {
        let l = self.sps.log2_ctb;
        let (rx, ry) = (rs % self.w_ctb, rs / self.w_ctb);
        if self.sh.sao_luma || self.sh.sao_chroma {
            self.sao(io, rx, ry)?;
        }
        self.quadtree(io, (rx as u32) << l, (ry as u32) << l, l, 0)
    }

    fn sao<T: BinIo>(&mut self, io: &mut T, rx: usize, ry: usize) -> R<()> {
        let rs = self.rs;
        let slice_addr = self.sh.slice_addr as usize;
        let tile_of = |g: &Geometry, a: usize| g.tile_rs[a];
        let colo = self.colo((rx as u32) << self.sps.log2_ctb, (ry as u32) << self.sps.log2_ctb).class as u32;
        let cs = self.ctx_slice;
        let mut merge = false;
        if rx > 0 && rs > slice_addr && tile_of(self.geo, rs) == tile_of(self.geo, rs - 1) {
            merge = self.dec(io, t::SAO_MERGE, kind::SAO, 0, colo, cs, 0)? == 1;
        }
        if !merge
            && ry > 0
            && rs - self.w_ctb >= slice_addr
            && tile_of(self.geo, rs) == tile_of(self.geo, rs - self.w_ctb)
        {
            merge = self.dec(io, t::SAO_MERGE, kind::SAO, 1, colo, cs, 0)? == 1;
        }
        if merge {
            return Ok(());
        }
        let mut typ = [0u32; 3];
        for c in 0..if self.chroma != 0 { 3 } else { 1 } {
            if !((self.sh.sao_luma && c == 0) || (self.sh.sao_chroma && c > 0)) {
                continue;
            }
            let ci = (c as u32).min(1) << 4;
            if c < 2 {
                typ[c] = if self.dec(io, t::SAO_TYPE, kind::SAO, 2 | ci, colo, cs, 0)? == 0 {
                    0
                } else if self.byp(io, kind::SAO, 3 | ci, 0, 0, 0)? == 0 {
                    1
                } else {
                    2
                };
                if c == 1 {
                    typ[2] = typ[1];
                }
            }
            if typ[c] == 0 {
                continue;
            }
            let bd = if c == 0 { self.sps.bit_depth_luma } else { self.sps.bit_depth_chroma };
            let cmax = (1u32 << (bd.min(10) - 5)) - 1;
            let mut abs = [0u32; 4];
            for (i, a) in abs.iter_mut().enumerate() {
                while *a < cmax && self.byp(io, kind::SAO, 4 | ci | (i as u32) << 5, q(*a, 7), typ[c], 0)? == 1 {
                    *a += 1;
                }
            }
            if typ[c] == 1 {
                for (i, &a) in abs.iter().enumerate() {
                    if a != 0 {
                        self.byp(io, kind::SAO, 5 | ci | (i as u32) << 5, q(a, 7), 0, 0)?;
                    }
                }
                let mut acc = 1u32;
                for _ in 0..5 {
                    acc = (acc << 1) | self.byp(io, kind::SAO, 6 | ci, acc, 0, 0)?;
                }
            } else if c < 2 {
                let mut acc = 1u32;
                for _ in 0..2 {
                    acc = (acc << 1) | self.byp(io, kind::SAO, 7 | ci, acc, 0, 0)?;
                }
            }
        }
        Ok(())
    }

    fn quadtree<T: BinIo>(&mut self, io: &mut T, x0: u32, y0: u32, log2: u32, depth: u32) -> R<()> {
        let size = 1u32 << log2;
        let split = if x0 + size <= self.pic.w && y0 + size <= self.pic.h && log2 > self.sps.log2_min_cb {
            let l = self.nb(x0 as i32 - 1, y0 as i32).map(|b| b.depth as u32);
            let a = self.nb(x0 as i32, y0 as i32 - 1).map(|b| b.depth as u32);
            let inc = l.is_some_and(|d| d > depth) as usize + a.is_some_and(|d| d > depth) as usize;
            let co = self.colo(x0 + size / 2, y0 + size / 2);
            let ha = depth | l.map_or(4, |d| d.min(3)) << 2 | a.map_or(4, |d| d.min(3)) << 5;
            let hb = co.depth as u32 | (co.class as u32) << 3;
            // Depth of the co-located blocks at the four quadrant centres.
            let h = size / 4;
            let mut cd = 0;
            for (dx, dy) in [(h, h), (3 * h, h), (h, 3 * h), (3 * h, 3 * h)] {
                cd = cd * 5 + (self.colo(x0 + dx, y0 + dy).depth as u32).min(4);
            }
            self.dec(io, t::SPLIT_CU + inc, kind::SPLIT_CU, ha, hb, self.ctx_slice, cd)? == 1
        } else {
            log2 > self.sps.log2_min_cb
        };
        if self.pps.cu_qp_delta && log2 >= self.log2_qg {
            self.is_dqp_coded = false;
            self.dqp = 0;
            self.start_qg(x0, y0);
        }
        if split {
            let h = size / 2;
            self.quadtree(io, x0, y0, log2 - 1, depth + 1)?;
            if x0 + h < self.pic.w {
                self.quadtree(io, x0 + h, y0, log2 - 1, depth + 1)?;
            }
            if y0 + h < self.pic.h {
                self.quadtree(io, x0, y0 + h, log2 - 1, depth + 1)?;
                if x0 + h < self.pic.w {
                    self.quadtree(io, x0 + h, y0 + h, log2 - 1, depth + 1)?;
                }
            }
            Ok(())
        } else {
            self.coding_unit(io, x0, y0, log2, depth)
        }
    }

    /// Predicted QP of a quantization group (8.6.1); model context only.
    fn start_qg(&mut self, x: u32, y: u32) {
        let l = self.sps.log2_ctb;
        let same_ctb = |xn: u32, yn: u32| (xn >> l) == (x >> l) && (yn >> l) == (y >> l);
        let qa = if x > 0 && same_ctb(x - 1, y) {
            self.nb(x as i32 - 1, y as i32).map_or(self.qp_prev, |b| b.qp as i32)
        } else {
            self.qp_prev
        };
        let qb = if y > 0 && same_ctb(x, y - 1) {
            self.nb(x as i32, y as i32 - 1).map_or(self.qp_prev, |b| b.qp as i32)
        } else {
            self.qp_prev
        };
        self.qp = (qa + qb + 1) >> 1;
    }

    fn coding_unit<T: BinIo>(&mut self, io: &mut T, x0: u32, y0: u32, log2: u32, depth: u32) -> R<()> {
        let size = 1u32 << log2;
        self.cu = Cu { x0, y0, log2, depth, ..Default::default() };
        let cl = self.nb_class(x0 as i32 - 1, y0 as i32);
        let ca = self.nb_class(x0 as i32, y0 as i32 - 1);
        let co = *self.colo(x0 + size / 2, y0 + size / 2);
        let hn = cl | ca << 3 | depth << 6 | log2 << 8;
        let hco = co.class as u32 | (co.depth as u32) << 3 | (co.mvd as u32).min(15) << 6;
        if self.pps.transquant_bypass {
            self.cu.tq_bypass = self.dec(io, t::TQ_BYPASS, kind::TQ_BYPASS, 0, 0, 0, 0)? == 1;
        }
        let mut skip = false;
        if self.sh.slice_type != I {
            let inc = (cl == 1) as usize + (ca == 1) as usize;
            // The co-located block one CU to the right, as a hint of motion continuity.
            let cr = self.colo(x0 + size + size / 2, y0 + size / 2).class as u32;
            skip = self.dec(io, t::SKIP + inc, kind::SKIP, hn, hco, self.ctx_slice, cr)? == 1;
        }
        let qp = self.qp.clamp(-64, 63) as i8;
        let sid = self.sid;
        let d8 = depth as u8;
        if skip {
            self.cu.class = 1;
            self.set_blocks(x0, y0, size, size, |b| {
                *b = Blk { sid, depth: d8, flags: F_SKIP | F_MERGE, ipm: 1, qp, nz: 0, nzc: 0, mvd: 0, sg: 0 };
            });
            self.prediction_unit(io, x0, y0, size, size, 0, true)?;
            self.finish_cu();
            return Ok(());
        }
        let intra = if self.sh.slice_type != I {
            self.dec(io, t::PRED_MODE, kind::PRED_MODE, hn, hco, self.ctx_slice, 0)? == 1
        } else {
            true
        };
        self.cu.intra = intra;
        let flags = if intra { F_INTRA } else { 0 };
        self.set_blocks(x0, y0, size, size, |b| {
            *b = Blk { sid, depth: d8, flags, ipm: 1, qp, nz: 0, nzc: 0, mvd: 0, sg: 0 };
        });
        let mut part = P2NX2N;
        if !intra || log2 == self.sps.log2_min_cb {
            part = self.part_mode(io, intra, log2, hn, hco)?;
        }
        self.cu.part = part;
        if intra {
            self.cu.class = 4;
            if part == P2NX2N && self.sps.pcm && log2 >= self.sps.log2_min_pcm && log2 <= self.sps.log2_max_pcm {
                let h = Hint { ctx: NUM_CTX as u16, kind: kind::PCM, a: 0, b: 0, c: 0, d: 0 };
                if io.terminate(&h)? == 1 {
                    return Err(Bad); // PCM samples are not supported; the slice stays raw.
                }
            }
            self.intra_modes(io, x0, y0, log2, part)?;
        } else {
            self.cu.class = 3;
            let (s, h2, q4) = (size, size / 2, size / 4);
            let pus: &[(u32, u32, u32, u32)] = match part {
                P2NX2N => &[(0, 0, s, s)],
                P2NXN => &[(0, 0, s, h2), (0, h2, s, h2)],
                PNX2N => &[(0, 0, h2, s), (h2, 0, h2, s)],
                P2NXNU => &[(0, 0, s, q4), (0, q4, s, s - q4)],
                P2NXND => &[(0, 0, s, s - q4), (0, s - q4, s, q4)],
                PNLX2N => &[(0, 0, q4, s), (q4, 0, s - q4, s)],
                PNRX2N => &[(0, 0, s - q4, s), (s - q4, 0, q4, s)],
                _ => &[(0, 0, h2, h2), (h2, 0, h2, h2), (0, h2, h2, h2), (h2, h2, h2, h2)],
            };
            for (i, &(dx, dy, w, h)) in pus.iter().enumerate() {
                let merge = self.prediction_unit(io, x0 + dx, y0 + dy, w, h, i as u32, false)?;
                if i == 0 {
                    self.cu.merge0 = merge;
                }
            }
            if part == P2NX2N && self.cu.merge0 {
                self.cu.class = 2;
            }
        }
        let mut rqt_root_cbf = true;
        if !intra && !(part == P2NX2N && self.cu.merge0) {
            let hb = part as u32 | (self.cu.merge0 as u32) << 3 | log2 << 4;
            rqt_root_cbf = self.dec(io, t::RQT_ROOT_CBF, kind::RQT_ROOT_CBF, hb, co.nz as u32, self.ctx_slice, 0)? == 1;
        }
        if rqt_root_cbf {
            self.cu.intra_split = intra && part == PNXN;
            self.cu.max_trafo_depth = if intra {
                self.sps.max_th_depth_intra + self.cu.intra_split as u32
            } else {
                self.sps.max_th_depth_inter
            };
            self.transform_tree(io, x0, y0, x0, y0, log2, 0, 0, false, false)?;
        }
        self.finish_cu();
        Ok(())
    }

    fn finish_cu(&mut self) {
        let c = self.cu;
        let size = 1u32 << c.log2;
        let qp = self.qp.clamp(-64, 63) as i8;
        self.set_blocks(c.x0, c.y0, size, size, |b| b.qp = qp);
        self.qp_prev = self.qp;
        let area4 = (size * size / 16).max(1);
        let s = Summary {
            class: c.class,
            depth: c.depth as u8,
            nz: (c.nz * 4).div_ceil(area4).min(255) as u8,
            mvd: mvd_class(c.mvd) as u8,
        };
        let w8 = self.pic.w8;
        let x1 = (c.x0 + size).min(self.pic.w).div_ceil(8);
        let y1 = (c.y0 + size).min(self.pic.h).div_ceil(8);
        for by in c.y0 / 8..y1 {
            for bx in c.x0 / 8..x1 {
                self.pending.push((by as usize * w8 + bx as usize, s));
            }
        }
    }

    fn part_mode<T: BinIo>(&mut self, io: &mut T, intra: bool, log2: u32, hn: u32, hco: u32) -> R<u8> {
        let cs = self.ctx_slice;
        let bin = |p: &mut Self, io: &mut T, ctx: usize, i: u32| {
            p.dec(io, t::PART_MODE + ctx, kind::PART_MODE, i | (intra as u32) << 3, hn, hco, cs)
        };
        if intra {
            return Ok(if bin(self, io, 0, 0)? == 1 { P2NX2N } else { PNXN });
        }
        if bin(self, io, 0, 0)? == 1 {
            return Ok(P2NX2N);
        }
        if log2 == self.sps.log2_min_cb {
            if bin(self, io, 1, 1)? == 1 {
                return Ok(P2NXN);
            }
            if log2 == 3 {
                return Ok(PNX2N);
            }
            return Ok(if bin(self, io, 2, 2)? == 1 { PNX2N } else { PNXN });
        }
        if !self.sps.amp {
            return Ok(if bin(self, io, 1, 1)? == 1 { P2NXN } else { PNX2N });
        }
        let hor = bin(self, io, 1, 1)? == 1;
        if bin(self, io, 3, 2 | (hor as u32) << 2)? == 1 {
            return Ok(if hor { P2NXN } else { PNX2N });
        }
        let second = self.byp(io, kind::PART_MODE, 5 | (hor as u32) << 2, hn, 0, 0)? == 1;
        Ok(match (hor, second) {
            (true, false) => P2NXNU,
            (true, true) => P2NXND,
            (false, false) => PNLX2N,
            (false, true) => PNRX2N,
        })
    }

    /// candModeList of 8.4.2.
    fn mpm(&self, x: u32, y: u32) -> [u8; 3] {
        let a = match self.nb(x as i32 - 1, y as i32) {
            Some(b) if b.flags & F_INTRA != 0 => b.ipm,
            _ => 1,
        };
        let l = self.sps.log2_ctb;
        let b = if y == 0 || (y - 1) >> l != y >> l {
            1
        } else {
            match self.nb(x as i32, y as i32 - 1) {
                Some(b) if b.flags & F_INTRA != 0 => b.ipm,
                _ => 1,
            }
        };
        if a == b {
            if a < 2 {
                [0, 1, 26]
            } else {
                [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
            }
        } else {
            let c = if a != 0 && b != 0 {
                0
            } else if a != 1 && b != 1 {
                1
            } else {
                26
            };
            [a, b, c]
        }
    }

    fn intra_modes<T: BinIo>(&mut self, io: &mut T, x0: u32, y0: u32, log2: u32, part: u8) -> R<()> {
        let n = if part == PNXN { 4 } else { 1 };
        let pb = if part == PNXN { 1u32 << (log2 - 1) } else { 1u32 << log2 };
        let cs = self.ctx_slice;
        let mut prev = [false; 4];
        for (i, p) in prev.iter_mut().enumerate().take(n) {
            let (px, py) = (x0 + (i as u32 & 1) * pb, y0 + (i as u32 >> 1) * pb);
            let cand = self.mpm(px, py);
            let same = (cand[0] == cand[1]) as u32 | ((cand[0] < 2) as u32) << 1;
            let hb = cand[0] as u32 | (cand[1] as u32) << 6;
            *p = self.dec(io, t::PREV_INTRA, kind::PREV_INTRA, i as u32 | same << 2 | log2 << 4, hb, cs, 0)? == 1;
        }
        let sid = self.sid;
        for (i, &p) in prev.iter().enumerate().take(n) {
            let (px, py) = (x0 + (i as u32 & 1) * pb, y0 + (i as u32 >> 1) * pb);
            let cand = self.mpm(px, py);
            let hb = cand[0] as u32 | (cand[1] as u32) << 6 | (cand[2] as u32) << 12;
            let mode = if p {
                let idx = if self.byp(io, kind::INTRA_MODE, 0, hb, log2, 0)? == 0 {
                    0
                } else if self.byp(io, kind::INTRA_MODE, 1, hb, log2, 0)? == 0 {
                    1
                } else {
                    2
                };
                cand[idx]
            } else {
                let mut sorted = cand;
                sorted.sort_unstable();
                let hs = sorted[0] as u32 | (sorted[1] as u32) << 6 | (sorted[2] as u32) << 12;
                let mut acc = 1u32;
                for _ in 0..5 {
                    acc = (acc << 1) | self.byp(io, kind::INTRA_MODE, 2 | acc << 2, hs, log2, 0)?;
                }
                let mut m = (acc & 31) as u8;
                for c in sorted {
                    if m >= c {
                        m += 1;
                    }
                }
                m
            };
            self.cu.ipm[i] = mode;
            self.set_blocks(px, py, pb, pb, |b| {
                b.ipm = mode;
                b.sid = sid;
            });
        }
        if self.chroma != 0 {
            let lm = self.cu.ipm[0] as u32;
            let c = if self.dec(io, t::CHROMA_PRED, kind::CHROMA_PRED, 0, lm, cs, 0)? == 0 {
                4
            } else {
                let b1 = self.byp(io, kind::CHROMA_PRED, 1, lm, 0, 0)?;
                let b2 = self.byp(io, kind::CHROMA_PRED, 2 | b1 << 2, lm, 0, 0)?;
                b1 << 1 | b2
            };
            self.cu.ipm_c = if c == 4 {
                lm as u8
            } else {
                let m = [0u8, 26, 10, 1][c as usize];
                if m as u32 == lm {
                    34
                } else {
                    m
                }
            };
        }
        Ok(())
    }

    /// Parses a prediction unit; returns merge_flag.
    #[allow(clippy::too_many_arguments)]
    fn prediction_unit<T: BinIo>(
        &mut self,
        io: &mut T,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        part_idx: u32,
        skip: bool,
    ) -> R<bool> {
        let cs = self.ctx_slice;
        let left =
            self.nb(x0 as i32 - 1, y0 as i32 + h as i32 - 1).map_or(7, |b| (b.flags & (F_SKIP | F_MERGE)) as u32);
        let above =
            self.nb(x0 as i32 + w as i32 - 1, y0 as i32 - 1).map_or(7, |b| (b.flags & (F_SKIP | F_MERGE)) as u32);
        let shape = (31 - w.leading_zeros()) | (31 - h.leading_zeros()) << 3;
        let hpu = part_idx | (self.cu.part as u32) << 2 | shape << 5;
        let merge = skip || self.dec(io, t::MERGE_FLAG, kind::MERGE_FLAG, left | above << 3, hpu, cs, 0)? == 1;
        if merge {
            if self.sh.max_num_merge_cand > 1 {
                let cmax = self.sh.max_num_merge_cand - 1;
                let hb = left | above << 3 | (skip as u32) << 6;
                if self.dec(io, t::MERGE_IDX, kind::MERGE_IDX, 0, hb, hpu, cs)? == 1 {
                    let mut idx = 1;
                    while idx < cmax && self.byp(io, kind::MERGE_IDX, idx, hb, hpu, cmax)? == 1 {
                        idx += 1;
                    }
                }
            }
            if !skip {
                self.set_blocks(x0, y0, w, h, |b| b.flags |= F_MERGE);
            }
            return Ok(true);
        }
        let mut dir = 1u32; // bit 0: L0, bit 1: L1
        if self.sh.slice_type == B {
            let depth = self.cu.depth as usize;
            dir = if w + h != 12 {
                if self.dec(io, t::INTER_DIR + depth, kind::INTER_DIR, 0, hpu, cs, 0)? == 1 {
                    3
                } else {
                    1 + self.dec(io, t::INTER_DIR + 4, kind::INTER_DIR, 1, hpu, cs, 0)?
                }
            } else {
                1 + self.dec(io, t::INTER_DIR + 4, kind::INTER_DIR, 2, hpu, cs, 0)?
            };
        }
        let mut mvd_sum = 0u32;
        for list in 0..2u32 {
            if dir & (1 << list) == 0 {
                continue;
            }
            let nref = if list == 0 { self.sh.num_ref_l0 } else { self.sh.num_ref_l1 };
            if nref > 1 {
                let cmax = nref - 1;
                let mut r = 0;
                while r < cmax {
                    let hb = list | dir << 1 | q(nref, 15) << 3;
                    let b = if r < 2 {
                        self.dec(io, t::REF_IDX + r as usize, kind::REF_IDX, r, hb, cs, 0)?
                    } else {
                        self.byp(io, kind::REF_IDX, r, hb, 0, 0)?
                    };
                    if b == 0 {
                        break;
                    }
                    r += 1;
                }
            }
            let mut mv = [0u32; 2];
            if !(list == 1 && self.sh.mvd_l1_zero && dir == 3) {
                mv = self.mvd_coding(io, x0, y0, w, h, list, dir)?;
            }
            mvd_sum += mv[0] + mv[1];
            let hm = mvd_class(mv[0]) | mvd_class(mv[1]) << 4 | list << 8 | dir << 9;
            self.dec(io, t::MVP_FLAG, kind::MVP_FLAG, hm, hpu, cs, 0)?;
        }
        self.cu.mvd += mvd_sum;
        let mc = mvd_class(mvd_sum) as u8;
        self.set_blocks(x0, y0, w, h, |b| b.mvd = mc);
        Ok(false)
    }

    /// Parses mvd_coding; returns |MvdLX| per component.
    #[allow(clippy::too_many_arguments)]
    fn mvd_coding<T: BinIo>(
        &mut self,
        io: &mut T,
        x0: u32,
        y0: u32,
        w: u32,
        h: u32,
        list: u32,
        dir: u32,
    ) -> R<[u32; 2]> {
        let cs = self.ctx_slice;
        let nl = self.nb(x0 as i32 - 1, y0 as i32 + h as i32 - 1).map_or(0, |b| b.mvd as u32);
        let na = self.nb(x0 as i32 + w as i32 - 1, y0 as i32 - 1).map_or(0, |b| b.mvd as u32);
        let co = self.colo(x0 + w / 2, y0 + h / 2).mvd as u32;
        let hb = nl | na << 4 | co << 8;
        let hl = list | dir << 1 | (self.cu.part as u32) << 3;
        let g0x = self.dec(io, t::MVD_GT0, kind::MVD, hl, hb, cs, 0)?;
        let g0y = self.dec(io, t::MVD_GT0, kind::MVD, hl | 1 << 6 | g0x << 7, hb, cs, 0)?;
        let g1x = if g0x == 1 { self.dec(io, t::MVD_GT1, kind::MVD, hl | 2 << 6 | g0y << 8, hb, cs, 0)? } else { 0 };
        let g1y = if g0y == 1 {
            self.dec(io, t::MVD_GT1, kind::MVD, hl | 3 << 6 | g0x << 8 | g1x << 9, hb, cs, 0)?
        } else {
            0
        };
        let mut out = [0u32; 2];
        for (comp, (g0, g1)) in [(g0x, g1x), (g0y, g1y)].into_iter().enumerate() {
            if g0 == 0 {
                continue;
            }
            let mut v = 1;
            if g1 == 1 {
                let hc = comp as u32 | list << 1 | (q(out[0], 255)) << 2;
                v = 2 + self.eg_k(io, 1, kind::MVD_SUF, hb, hc)?;
            }
            self.byp(io, kind::MVD_SIGN, comp as u32 | list << 1, q(v, 15), 0, 0)?;
            out[comp] = v;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_tree<T: BinIo>(
        &mut self,
        io: &mut T,
        x0: u32,
        y0: u32,
        xb: u32,
        yb: u32,
        log2: u32,
        depth: u32,
        blk: u32,
        pcb: bool,
        pcr: bool,
    ) -> R<()> {
        let cs = self.ctx_slice;
        let c = self.cu;
        let nzl = self.nb(x0 as i32 - 1, y0 as i32).map_or(16, |b| b.nz as u32);
        let nza = self.nb(x0 as i32, y0 as i32 - 1).map_or(16, |b| b.nz as u32);
        let co = self.colo(x0 + (1 << log2) / 2, y0 + (1 << log2) / 2).nz as u32;
        let hn = nzl | nza << 5;
        let split = if log2 <= self.sps.log2_max_tb
            && log2 > self.sps.log2_min_tb
            && depth < c.max_trafo_depth
            && !(c.intra_split && depth == 0)
        {
            let ha = log2 | depth << 3 | (c.intra as u32) << 6 | (c.part as u32) << 7;
            self.dec(io, t::SPLIT_TU + (5 - log2) as usize, kind::SPLIT_TU, ha, hn, q(co, 31), cs)? == 1
        } else {
            let inter_split = self.sps.max_th_depth_inter == 0 && !c.intra && c.part != P2NX2N && depth == 0;
            log2 > self.sps.log2_max_tb || (c.intra_split && depth == 0) || inter_split
        };
        let (mut cb, mut cr) = (false, false);
        if self.chroma != 0 && log2 > 2 {
            let ha = depth | log2 << 3 | (c.intra as u32) << 6;
            let nzc = self.nb(x0 as i32 - 1, y0 as i32).map_or(16, |b| b.nzc as u32)
                | self.nb(x0 as i32, y0 as i32 - 1).map_or(16, |b| b.nzc as u32) << 5;
            if depth == 0 || pcb {
                cb = self.dec(io, t::CBF_CHROMA + depth as usize, kind::CBF_CHROMA, ha, nzc, cs, 2)? == 1;
            }
            if depth == 0 || pcr {
                cr = self.dec(io, t::CBF_CHROMA + depth as usize, kind::CBF_CHROMA, ha | 1 << 7, nzc, cs, cb as u32)?
                    == 1;
            }
        }
        if split {
            let h = 1u32 << (log2 - 1);
            for k in 0..4u32 {
                self.transform_tree(io, x0 + (k & 1) * h, y0 + (k >> 1) * h, x0, y0, log2 - 1, depth + 1, k, cb, cr)?;
            }
            return Ok(());
        }
        let mut cbf_luma = true;
        if c.intra || depth != 0 || cb || cr {
            let ha = depth | log2 << 3 | (c.intra as u32) << 6 | (cb as u32 | (cr as u32) << 1) << 7;
            cbf_luma = self.dec(io, t::CBF_LUMA + (depth == 0) as usize, kind::CBF_LUMA, ha, hn, q(co, 31), cs)? == 1;
        }
        self.transform_unit(io, x0, y0, xb, yb, log2, blk, cbf_luma, cb, cr, pcb, pcr)
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_unit<T: BinIo>(
        &mut self,
        io: &mut T,
        x0: u32,
        y0: u32,
        xb: u32,
        yb: u32,
        log2: u32,
        blk: u32,
        cbf_luma: bool,
        cb: bool,
        cr: bool,
        pcb: bool,
        pcr: bool,
    ) -> R<()> {
        let chroma4 = self.chroma != 0 && log2 == 2;
        let cbf_chroma = if chroma4 { pcb || pcr } else { cb || cr };
        let size = 1u32 << log2;
        if cbf_luma || cbf_chroma {
            if self.pps.cu_qp_delta && !self.is_dqp_coded {
                self.cu_qp_delta(io)?;
            }
            let (mut nz, mut nzc) = (0u32, 0u32);
            self.tu_nz = [0, 0];
            if cbf_luma {
                nz = self.residual(io, x0, y0, log2, 0)?;
                self.tu_nz[0] = nz;
            }
            if self.chroma != 0 {
                if log2 > 2 {
                    if cb {
                        nzc += self.residual(io, x0, y0, log2 - 1, 1)?;
                        self.tu_nz[1] = nzc;
                    }
                    if cr {
                        nzc += self.residual(io, x0, y0, log2 - 1, 2)?;
                    }
                } else if blk == 3 {
                    if pcb {
                        nzc += self.residual(io, xb, yb, 2, 1)?;
                        self.tu_nz[1] = nzc;
                    }
                    if pcr {
                        nzc += self.residual(io, xb, yb, 2, 2)?;
                    }
                }
            }
            self.cu.nz += nz;
            let area4 = (size * size / 16).max(1);
            let d = nz.div_ceil(area4).min(15) as u8;
            let dc = nzc.div_ceil(area4).min(15) as u8;
            let coded = if cbf_luma { F_CODED } else { 0 };
            self.set_blocks(x0, y0, size, size, |b| {
                b.nz = d;
                b.nzc = dc;
                b.flags |= coded;
            });
        } else {
            self.set_blocks(x0, y0, size, size, |b| {
                b.nz = 0;
                b.nzc = 0;
            });
        }
        Ok(())
    }

    fn cu_qp_delta<T: BinIo>(&mut self, io: &mut T) -> R<()> {
        let cs = self.ctx_slice;
        let hc = self.cu.intra as u32 | self.cu.log2 << 1;
        let mut v = 0u32;
        while v < 5 && self.dec(io, t::DQP + (v > 0) as usize, kind::DQP, v, hc, cs, 0)? == 1 {
            v += 1;
        }
        if v == 5 {
            v += self.eg_k(io, 0, kind::DQP, 1 << 8, hc)?;
        }
        let neg = v > 0 && self.byp(io, kind::DQP, 1 << 9, q(v, 15), hc, 0)? == 1;
        let off = 6 * (self.sps.bit_depth_luma as i32 - 8);
        let d = if neg { -(v as i32) } else { v as i32 };
        if d < -(26 + off / 2) || d > 25 + off / 2 {
            return Err(Bad);
        }
        self.is_dqp_coded = true;
        self.dqp = d;
        self.qp = ((self.qp + d + 52 + 2 * off) % (52 + off)) - off;
        Ok(())
    }

    /// Parses residual_coding (7.3.8.11) for one transform block; returns the number of
    /// non-zero coefficients.
    fn residual<T: BinIo>(&mut self, io: &mut T, x0: u32, y0: u32, log2: u32, c_idx: usize) -> R<u32> {
        let cs = self.ctx_slice;
        let chroma = c_idx > 0;
        let ch = chroma as u32;
        let size = 1usize << log2;
        let c = self.cu;
        // Neighbouring transform blocks and the co-located block, as model context.
        let (nl, na) = if chroma {
            (
                self.nb(x0 as i32 - 1, y0 as i32).map_or(16, |b| b.nzc as u32),
                self.nb(x0 as i32, y0 as i32 - 1).map_or(16, |b| b.nzc as u32),
            )
        } else {
            (
                self.nb(x0 as i32 - 1, y0 as i32).map_or(16, |b| b.nz as u32),
                self.nb(x0 as i32, y0 as i32 - 1).map_or(16, |b| b.nz as u32),
            )
        };
        let luma_size = if chroma { 2u32 << log2 } else { 1u32 << log2 };
        let co = q(self.colo(x0 + luma_size / 2, y0 + luma_size / 2).nz as u32, 15);
        let hnb = q(nl, 16) | q(na, 16) << 5 | co << 10;
        let hnbq = ql(nl) | ql(na) << 3 | ql(co) << 6;
        // Coefficients already decoded in this transform unit (luma, then Cb) for chroma blocks.
        let hsame = if chroma { 1 + ql(self.tu_nz[0]) + 8 * ql(self.tu_nz[1]) * (c_idx == 2) as u32 } else { 0 };
        let mut tskip = false;
        if self.pps.transform_skip && !c.tq_bypass && log2 <= 2 {
            tskip = self.dec(io, t::TSKIP + ch as usize, kind::TSKIP, ch | (c.intra as u32) << 1, hnb, cs, 0)? == 1;
        }
        let hk = ch | (log2 - 2) << 1 | (c.intra as u32) << 3 | (tskip as u32) << 4;
        // last_sig_coeff_{x,y}_prefix and suffix
        let cmax = (log2 << 1) - 1;
        let (off, shift) = if chroma { (15, log2 - 2) } else { (3 * (log2 - 2) + ((log2 - 1) >> 2), (log2 + 1) >> 2) };
        // The last position of the previous block of this size and component is the best
        // single predictor of this one.
        let prev = self.last_prev[ch as usize][(log2 - 2) as usize];
        let lc = hnbq | hsame << 9;
        let mut px = 0;
        while px < cmax
            && self.dec(
                io,
                t::LAST_X + (off + (px >> shift)) as usize,
                kind::LAST,
                px | hk << 4,
                q(prev.0 as u32, 32),
                lc,
                cs,
            )? == 1
        {
            px += 1;
        }
        let lby = q(prev.1 as u32, 32) | px << 6;
        let mut py = 0;
        while py < cmax
            && self.dec(
                io,
                t::LAST_Y + (off + (py >> shift)) as usize,
                kind::LAST,
                py | 1 << 3 | hk << 4,
                lby,
                lc,
                cs | (px | 16) << 16,
            )? == 1
        {
            py += 1;
        }
        let suffix = |p: &mut Self, io: &mut T, prefix: u32, axis: u32| -> R<u32> {
            if prefix <= 3 {
                return Ok(prefix);
            }
            let n = (prefix >> 1) - 1;
            let mut s = 0;
            for i in (0..n).rev() {
                s |= p.byp(io, kind::LAST, i | axis << 3 | hk << 4, prefix, 0, 0)? << i;
            }
            Ok((1 << n) * (2 + (prefix & 1)) + s)
        };
        let mut lx = suffix(self, io, px, 0)?;
        let mut ly = suffix(self, io, py, 1)?;
        self.last_prev[ch as usize][(log2 - 2) as usize] = (lx as u8, ly as u8);
        // scanIdx (7.4.9.11)
        let mut scan = 0usize;
        if c.intra && (log2 == 2 || (log2 == 3 && !chroma)) {
            let m = if chroma { c.ipm_c } else { self.ipm_at(x0, y0) };
            if (6..=14).contains(&m) {
                scan = 2;
            } else if (22..=30).contains(&m) {
                scan = 1;
            }
        }
        if scan == 2 {
            std::mem::swap(&mut lx, &mut ly);
        }
        let sb_log2 = (log2 - 2) as usize;
        let sb_n = 1usize << sb_log2;
        let last_sb = SCAN_POS[sb_log2][scan][(ly as usize >> 2) * sb_n + (lx as usize >> 2)] as usize;
        let last_pos = SCAN_POS[2][scan][((ly & 3) as usize) * 4 + (lx & 3) as usize] as usize;
        for v in self.tu_abs[..size * size].iter_mut() {
            *v = 0;
        }
        for v in self.tu_sgn[..size * size].iter_mut() {
            *v = 0;
        }
        let mut csbf = [0u8; 64];
        let sign_hiding = self.pps.sign_data_hiding && !c.tq_bypass;
        let sg_left = self.nb(x0 as i32 - 1, y0 as i32).map(|b| b.sg);
        let sg_above = self.nb(x0 as i32, y0 as i32 - 1).map(|b| b.sg);
        let mut c1 = 1u32;
        let mut total = 0u32;
        let mut big = 0u32;
        let mut coded_sbs = 0u32;
        let scan_sb = &SCAN[sb_log2][scan];
        let scan4 = &SCAN[2][scan];
        for i in (0..=last_sb).rev() {
            let (xs, ys) = (scan_sb[i].0 as usize, scan_sb[i].1 as usize);
            let right = xs + 1 < sb_n && csbf[ys * 8 + xs + 1] != 0;
            let below = ys + 1 < sb_n && csbf[(ys + 1) * 8 + xs] != 0;
            let mut infer_dc = false;
            if i < last_sb && i > 0 {
                let inc = (right || below) as usize + 2 * chroma as usize;
                // Levels already decoded in the sub-blocks to the right and below.
                let mut nbsum = 0u32;
                if xs + 1 < sb_n {
                    nbsum += self.sb_sum(size, xs + 1, ys);
                }
                if ys + 1 < sb_n {
                    nbsum += self.sb_sum(size, xs, ys + 1);
                }
                let ha = right as u32 | (below as u32) << 1 | hk << 2;
                let hb = (xs | ys << 3) as u32 | q(coded_sbs, 15) << 6;
                csbf[ys * 8 + xs] = self.dec(io, t::CSBF + inc, kind::CSBF, ha, hb, q(nbsum, 31) | hnb << 5, cs)? as u8;
                infer_dc = true;
            } else {
                csbf[ys * 8 + xs] = 1;
            }
            if csbf[ys * 8 + xs] == 0 {
                continue;
            }
            coded_sbs += 1;
            // sig_coeff_flag
            let prev_csbf = right as u32 | (below as u32) << 1;
            let mut sig = [false; 16];
            let mut nsig = 0u32;
            let start = if i == last_sb {
                sig[last_pos] = true;
                nsig = 1;
                let (xp, yp) = scan4[last_pos];
                self.tu_abs[(ys * 4 + yp as usize) * size + xs * 4 + xp as usize] = 1;
                last_pos as i32 - 1
            } else {
                15
            };
            let mut n = start;
            while n >= 0 {
                let (xp, yp) = (scan4[n as usize].0 as usize, scan4[n as usize].1 as usize);
                let (xc, yc) = (xs * 4 + xp, ys * 4 + yp);
                if n > 0 || !infer_dc {
                    let sig_ctx = if log2 == 2 {
                        CTX_IDX_MAP[(yc << 2) + xc] as usize
                    } else if xc + yc == 0 {
                        0
                    } else {
                        let mut s = match prev_csbf {
                            0 => {
                                if xp + yp == 0 {
                                    2
                                } else if xp + yp < 3 {
                                    1
                                } else {
                                    0
                                }
                            }
                            1 => {
                                if yp == 0 {
                                    2
                                } else if yp == 1 {
                                    1
                                } else {
                                    0
                                }
                            }
                            2 => {
                                if xp == 0 {
                                    2
                                } else if xp == 1 {
                                    1
                                } else {
                                    0
                                }
                            }
                            _ => 2,
                        };
                        if !chroma {
                            if xs + ys > 0 {
                                s += 3;
                            }
                            s += if log2 == 3 {
                                if scan == 0 {
                                    9
                                } else {
                                    15
                                }
                            } else {
                                21
                            };
                        } else {
                            s += if log2 == 3 { 9 } else { 12 };
                        }
                        s
                    };
                    let ctx = t::SIG + if chroma { 27 } else { 0 } + sig_ctx;
                    let (tc, ts, _) = self.template(size, xc, yc);
                    let ha = tc | q(ts, 15) << 3 | q(nsig, 3) << 7 | ch << 9;
                    let hb = (xc | yc << 5) as u32 | (log2 - 2) << 10 | ch << 12 | (scan as u32) << 13;
                    let hc = hnbq | q(total, 15) << 9 | hsame << 13;
                    let s = self.dec(io, ctx, kind::SIG, ha, hb, hc, cs | hk << 11)? == 1;
                    if s {
                        sig[n as usize] = true;
                        nsig += 1;
                        infer_dc = false;
                        self.tu_abs[yc * size + xc] = 1;
                    }
                } else {
                    // Inferred DC of a coded sub-block with no other significant coefficient.
                    sig[0] = true;
                    nsig += 1;
                    self.tu_abs[yc * size + xc] = 1;
                }
                n -= 1;
            }
            // coeff_abs_level_greater1_flag / greater2_flag
            let ctx_set = (if i == 0 || chroma { 0 } else { 2 }) + (c1 == 0) as u32;
            c1 = 1;
            let mut gt1 = [false; 16];
            let mut gt2 = [false; 16];
            let mut num_gt1 = 0;
            let mut first_gt1: i32 = -1;
            let mut first_sig = 16i32;
            let mut last_sig = -1i32;
            let mut seen = 0u32;
            for n in (0..16).rev() {
                if !sig[n] {
                    continue;
                }
                let (xp, yp) = (scan4[n].0 as usize, scan4[n].1 as usize);
                let (xc, yc) = (xs * 4 + xp, ys * 4 + yp);
                if num_gt1 < 8 {
                    let inc = (ctx_set * 4 + c1.min(3)) as usize + if chroma { 16 } else { 0 };
                    let (tc, ts, _) = self.template(size, xc, yc);
                    let ha = c1 | q(ts, 15) << 2 | tc << 6 | ch << 9 | dclass((xc + yc) as u32).min(3) << 10;
                    let hb = (xc | yc << 5) as u32 | (log2 - 2) << 10 | ch << 12;
                    let hc = q(nsig, 16) | q(seen, 15) << 5 | ctx_set << 9 | q(big, 7) << 11;
                    let g = self.dec(io, t::GT1 + inc, kind::GT1, ha, hb, hc, cs | hk << 11 | hnbq << 16)? == 1;
                    gt1[n] = g;
                    num_gt1 += 1;
                    if g {
                        c1 = 0;
                        big += 1;
                        if first_gt1 < 0 {
                            first_gt1 = n as i32;
                        }
                        self.tu_abs[yc * size + xc] = 2;
                    } else if c1 > 0 && c1 < 3 {
                        c1 += 1;
                    }
                }
                seen += 1;
                if last_sig < 0 {
                    last_sig = n as i32;
                }
                first_sig = n as i32;
            }
            if first_gt1 >= 0 {
                let n = first_gt1 as usize;
                let (xp, yp) = (scan4[n].0 as usize, scan4[n].1 as usize);
                let (xc, yc) = (xs * 4 + xp, ys * 4 + yp);
                let (tc, ts, _) = self.template(size, xc, yc);
                let ha = q(ts, 15) | tc << 4 | ch << 7;
                let hb = (xc | yc << 5) as u32 | (log2 - 2) << 10 | ch << 12;
                let inc = ctx_set as usize + if chroma { 4 } else { 0 };
                let g = self.dec(io, t::GT2 + inc, kind::GT2, ha, hb, q(nsig, 16) | hnb << 5, cs | hk << 11)? == 1;
                gt2[n] = g;
                if g {
                    self.tu_abs[yc * size + xc] = 3;
                }
            }
            // coeff_sign_flag
            let hidden = sign_hiding && last_sig - first_sig > 3;
            let nb_sign = |v: Option<u32>, pos: u32| -> u32 {
                match v {
                    None => 3,
                    Some(g) if chroma => (g >> (12 + 2 * pos)) & 3,
                    Some(g) => (g >> (2 * pos)) & 3,
                }
            };
            let mut sign = [0u32; 16];
            for n in (0..16).rev() {
                if !sig[n] || (hidden && n as i32 == first_sig) {
                    continue;
                }
                let (xp, yp) = (scan4[n].0 as usize, scan4[n].1 as usize);
                let (xc, yc) = (xs * 4 + xp, ys * 4 + yp);
                let sr = if xc + 1 < size { self.tu_sgn[yc * size + xc + 1] } else { 3 } as u32;
                let sb = if yc + 1 < size { self.tu_sgn[(yc + 1) * size + xc] } else { 3 } as u32;
                let lvl = q(self.tu_abs[yc * size + xc] as u32, 3);
                let ha = ch | ((xc + yc == 0) as u32) << 1 | lvl << 2;
                let hb = sr | sb << 2 | (q(xc as u32, 3) | q(yc as u32, 3) << 2) << 4;
                // The same low-frequency coefficient's sign in the left and upper blocks.
                let lf = xc + yc <= if chroma { 1 } else { 2 };
                let hd = if lf {
                    let pos = LF_POS[yc * 3 + xc] as u32;
                    let other = if c_idx == 2 { (self.tu_cb_sg >> (2 * pos)) & 3 } else { 3 };
                    (1 + pos) | nb_sign(sg_left, pos) << 3 | nb_sign(sg_above, pos) << 5 | other << 7
                } else {
                    0
                };
                sign[n] = self.byp(io, kind::SIGN, ha, hb, c.intra as u32, hd)?;
                self.tu_sgn[yc * size + xc] = 1 + sign[n] as u8;
            }
            // coeff_abs_level_remaining
            let mut num_sig = 0;
            let mut rice = 0u32;
            for n in (0..16).rev() {
                if !sig[n] {
                    continue;
                }
                let (xp, yp) = (scan4[n].0 as usize, scan4[n].1 as usize);
                let (xc, yc) = (xs * 4 + xp, ys * 4 + yp);
                let base = 1 + gt1[n] as u32 + gt2[n] as u32;
                let thresh = if num_sig < 8 {
                    if n as i32 == first_gt1 {
                        3
                    } else {
                        2
                    }
                } else {
                    1
                };
                let mut abs = base;
                if base == thresh {
                    let (tc, _, tf) = self.template(size, xc, yc);
                    let hb = qsum(tf) | tc << 4 | base << 7 | ch << 9;
                    let hc = (q(xc as u32, 7) + q(yc as u32, 7)) | (log2 - 2) << 4 | (c.intra as u32) << 6;
                    let rem = self.coeff_remaining(io, rice, hb, hc)?;
                    abs = base + rem;
                    if abs > 3 * (1 << rice) {
                        rice = (rice + 1).min(4);
                    }
                    if base == 1 {
                        big += (abs >= 2) as u32;
                    }
                }
                self.tu_abs[yc * size + xc] = abs.min(65535) as u16;
                num_sig += 1;
            }
            total += nsig;
        }
        // Low-frequency signs, as context for the neighbouring blocks.
        let sg = |x: usize, y: usize| if x < size && y < size { self.tu_sgn[y * size + x] as u32 } else { 0 };
        let luma_size = if chroma { 2u32 << log2 } else { 1u32 << log2 };
        if chroma {
            if c_idx == 1 {
                let v = sg(0, 0) | sg(1, 0) << 2 | sg(0, 1) << 4;
                self.tu_cb_sg = v;
                self.set_blocks(x0, y0, luma_size, luma_size, |b| b.sg = (b.sg & 0xFFF) | v << 12);
            }
        } else {
            let v = sg(0, 0) | sg(1, 0) << 2 | sg(0, 1) << 4 | sg(2, 0) << 6 | sg(1, 1) << 8 | sg(0, 2) << 10;
            self.set_blocks(x0, y0, luma_size, luma_size, |b| b.sg = (b.sg & !0xFFF) | v);
        }
        Ok(total)
    }

    /// coeff_abs_level_remaining (9.3.3.11) with Rice parameter `k`.
    fn coeff_remaining<T: BinIo>(&mut self, io: &mut T, k: u32, hb: u32, hc: u32) -> R<u32> {
        let mut prefix = 0u32;
        while self.byp(io, kind::REM, q(prefix, 15) | k << 4, hb, hc, 0)? == 1 {
            prefix += 1;
            if prefix > 32 {
                return Err(Bad);
            }
        }
        let (n, base) = if prefix < 4 {
            (k, prefix << k)
        } else {
            let u = prefix - 4;
            if k + 1 + u > 30 {
                return Err(Bad);
            }
            (k + 1 + u, (4 << k) + (1 << (k + 1 + u)) - (1 << (k + 1)))
        };
        let mut s = 0u32;
        let mut acc = 1u32;
        for i in (0..n).rev() {
            let bit = self.byp(io, kind::REM, 16 | (i.min(15)) | k << 5 | q(prefix, 7) << 8, hb, hc, acc.min(255))?;
            acc = (acc << 1) | bit;
            s |= bit << i;
        }
        Ok(base + s)
    }

    /// IntraPredModeY at a luma position of the current CU.
    fn ipm_at(&self, x: u32, y: u32) -> u8 {
        let c = &self.cu;
        if c.part == PNXN {
            let h = 1u32 << (c.log2 - 1);
            let i = ((x - c.x0) >= h) as usize + 2 * ((y - c.y0) >= h) as usize;
            c.ipm[i]
        } else {
            c.ipm[0]
        }
    }

    /// Significant neighbours (count, capped level sum) of a coefficient among those already
    /// decoded: right, two right, below, two below, and below-right.
    #[inline]
    fn template(&self, size: usize, x: usize, y: usize) -> (u32, u32, u32) {
        let mut cnt = 0u32;
        let mut sum = 0u32;
        let mut full = 0u32;
        let mut add = |v: u16| {
            if v != 0 {
                cnt += 1;
                sum += (v as u32).min(4);
                full += (v as u32).min(64);
            }
        };
        if x + 1 < size {
            add(self.tu_abs[y * size + x + 1]);
            if x + 2 < size {
                add(self.tu_abs[y * size + x + 2]);
            }
            if y + 1 < size {
                add(self.tu_abs[(y + 1) * size + x + 1]);
            }
        }
        if y + 1 < size {
            add(self.tu_abs[(y + 1) * size + x]);
            if y + 2 < size {
                add(self.tu_abs[(y + 2) * size + x]);
            }
        }
        (cnt, sum, full)
    }

    /// Capped sum of the levels of a decoded 4x4 sub-block.
    fn sb_sum(&self, size: usize, xs: usize, ys: usize) -> u32 {
        let mut s = 0u32;
        for y in ys * 4..ys * 4 + 4 {
            for x in xs * 4..xs * 4 + 4 {
                s += (self.tu_abs[y * size + x] as u32).min(4);
            }
        }
        s
    }
}
