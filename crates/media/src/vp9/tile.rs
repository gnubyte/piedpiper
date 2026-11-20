//! VP9 tile parser: partitions, mode info, motion-vector prediction and coefficient tokens,
//! bit-exact with libvpx (`vp9_decodeframe.c`, `vp9_decodemv.c`, `vp9_detokenize.c`). Like the
//! H.264 parser it never reads bits itself: every boolean comes from a [`BoolIo`] together
//! with the probability the bitstream codes it with and a [`Hint`] (context) for PiedPiper's
//! model.

use super::tables::*;
use super::{
    FrameCtx, FrameHeader, MvRef, ALTREF_FRAME, COMPOUND_REFERENCE, GOLDEN_FRAME, LAST_FRAME, REFERENCE_MODE_SELECT,
    SWITCHABLE, TX_MODE_SELECT,
};
use crate::bits::{Bad, R};
use crate::h264::mb::Hint;

/// Source/sink of boolean decisions with libvpx probabilities (of a 0, in 1/256).
pub trait BoolIo {
    fn read(&mut self, prob: u8, h: &Hint) -> R<u32>;
}

/// A [`BoolIo`] that is told where each tile starts and ends.
pub trait TileIo: BoolIo {
    fn begin(&mut self, tile: usize) -> R<()>;
    fn end(&mut self, tile: usize) -> R<()>;
}

/// Model symbol classes for VP9 (40..=63; H.264 and the generic symbols use lower values).
pub mod kind {
    pub const MARKER: u8 = 40;
    pub const PARTITION: u8 = 41;
    pub const SKIP: u8 = 42;
    pub const TX: u8 = 43;
    pub const IS_INTER: u8 = 44;
    pub const YMODE: u8 = 45;
    pub const UVMODE: u8 = 46;
    pub const REF: u8 = 47;
    pub const INTER_MODE: u8 = 48;
    pub const INTERP: u8 = 49;
    pub const MV_JOINT: u8 = 50;
    pub const MV_SIGN: u8 = 51;
    pub const MV_CLASS: u8 = 52;
    pub const MV_BITS: u8 = 53;
    pub const MV_FR: u8 = 54;
    pub const MV_HP: u8 = 55;
    pub const TOKEN: u8 = 56;
    pub const EXTRA: u8 = 57;
    pub const SIGN: u8 = 58;
    /// Frame header bytes, raw frames and tail fix-ups (stream.rs).
    pub const HDR: u8 = 59;
    pub const RAW: u8 = 60;
    pub const FIX: u8 = 61;
    pub const COUNT: usize = 62;
}

/// Mixer contexts (`Hint::ctx`) of the syntax elements. The model selects its first mixer
/// by this value when a probability is supplied, so all of them stay below 460.
mod cx {
    pub const MARKER: usize = 0;
    /// + partition context + 16 * tree node
    pub const PARTITION: usize = 1;
    pub const SKIP: usize = 49;
    pub const TX: usize = 52;
    pub const IS_INTER: usize = 70;
    pub const KF_Y: usize = 74;
    pub const KF_UV: usize = 83;
    pub const Y: usize = 92;
    pub const UV: usize = 101;
    pub const COMP_INTER: usize = 110;
    pub const COMP_REF: usize = 115;
    pub const SINGLE_REF1: usize = 120;
    pub const SINGLE_REF2: usize = 125;
    pub const INTER_MODE: usize = 130;
    pub const INTERP: usize = 151;
    pub const MV_JOINT: usize = 159;
    /// + 30 * component
    pub const MV: usize = 162;
    pub const TOKEN: usize = 222;
    pub const PARETO: usize = 294;
    pub const SIGN: usize = 310;
    pub const EXTRA: usize = 318;
}

/// libvpx `vp9_cat6_prob_high12`: 18 extra-bit probabilities for 12-bit video; 10-bit video
/// uses the last 16.
const CAT6_PROB_HIGH12: [u8; 18] = [255, 255, 255, 255, 254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129];

const BLOCK_8X8: u8 = 3;
const BLOCK_64X64: u8 = 12;

#[derive(Clone, Copy, Default)]
pub struct Mi {
    pub sb_type: u8,
    pub mode: u8,
    pub uv_mode: u8,
    pub ref_frame: [i8; 2],
    pub mv: [[i16; 2]; 2],
    pub bmi_mode: [u8; 4],
    pub bmi_mv: [[[i16; 2]; 2]; 4],
    pub skip: bool,
    pub tx_size: u8,
    pub interp: u8,
    /// Non-zero coefficients of the block (all planes, saturating; model context only).
    pub nzc: u16,
}

impl Mi {
    fn is_inter(&self) -> bool {
        self.ref_frame[0] > 0
    }
    fn has_second(&self) -> bool {
        self.ref_frame[1] > 0
    }
}

/// Frame-wide state shared by the tiles of one frame.
pub struct FrameParse {
    pub mi_rows: usize,
    pub mi_cols: usize,
    pub grid: Vec<Mi>,
    pub frame_mvs: Vec<MvRef>,
    above_ctx: [Vec<u8>; 3],
    /// Per 4x4 column: end of block (capped at 63) | DC sign << 6, for the model only.
    above_eob: [Vec<u8>; 3],
    above_seg: Vec<u8>,
    /// Per 4x4 of each plane: end of block (capped at 63) of the transform covering it.
    pub eob_map: [Vec<u8>; 3],
}

impl FrameParse {
    pub fn new(h: &FrameHeader) -> Self {
        let (mi_rows, mi_cols) = (h.mi_rows(), h.mi_cols());
        let aligned = (mi_cols + 7) & !7;
        let ctx = || [vec![0; aligned * 2 + 16], vec![0; aligned + 16], vec![0; aligned + 16]];
        FrameParse {
            mi_rows,
            mi_cols,
            grid: vec![Mi::default(); mi_rows * mi_cols],
            frame_mvs: vec![MvRef { ref_frame: [0, -1], mv: [[0; 2]; 2] }; mi_rows * mi_cols],
            above_ctx: ctx(),
            above_eob: ctx(),
            above_seg: vec![0; aligned + 8],
            eob_map: [vec![0; mi_rows * mi_cols * 4], vec![0; mi_rows * mi_cols], vec![0; mi_rows * mi_cols]],
        }
    }

    /// Byte ranges of the tiles in the frame data after the compressed header (each tile but
    /// the last has a 4-byte big-endian size), in tile-row then tile-column order.
    pub fn tile_ranges(h: &FrameHeader, data: &[u8], start: usize) -> R<Vec<(usize, usize)>> {
        let n = (1usize << h.log2_tile_cols) * (1usize << h.log2_tile_rows);
        let mut out = Vec::with_capacity(n);
        let mut p = start;
        for i in 0..n {
            if i + 1 == n {
                if p > data.len() {
                    return Err(Bad);
                }
                out.push((p, data.len() - p));
            } else {
                let b = data.get(p..p + 4).ok_or(Bad)?;
                let size = u32::from_be_bytes(b.try_into().unwrap()) as usize;
                p += 4;
                if size > data.len() - p {
                    return Err(Bad);
                }
                out.push((p, size));
                p += size;
            }
        }
        Ok(out)
    }
}

fn tile_offset(idx: usize, mis: usize, log2: u32) -> usize {
    let sb = (mis + 7) >> 3;
    (((idx * sb) >> log2) << 3).min(mis)
}

/// Parses one tile.
pub struct TileParser<'a> {
    h: &'a FrameHeader,
    fc: &'a FrameCtx,
    f: &'a mut FrameParse,
    prev_mvs: Option<&'a [MvRef]>,
    /// The last shown frame's mode info and ends of block (same size), for the model.
    colo: Option<Colo<'a>>,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    left_ctx: [[u8; 16]; 3],
    left_eob: [[u8; 16]; 3],
    left_seg: [u8; 8],
    bmode_wl: u32,
    bmode_hl: u32,
    /// Frame class for the model: quantizer / 16 | shown << 4 | intra << 5.
    fq: u32,
    /// Number of non-zero coefficients of each plane of the current block.
    blk_eob: [u32; 3],
    /// Token classes by raster position of the transform block being read. Never cleared:
    /// every position read (scan neighbours) was written earlier in the same block.
    cache: Vec<u8>,
}

/// The last shown frame's mode info and per-4x4 ends of block.
pub type Colo<'a> = (&'a [Mi], &'a [Vec<u8>; 3]);

#[inline]
fn hint(ctx: usize, kind: u8, a: u32, b: u32, c: u32, d: u32) -> Hint {
    Hint { ctx: ctx.min(459) as u16, kind, a, b, c, d }
}

/// Reads a libvpx tree; node `n` uses mixer context `h.ctx + n * stride` and has `n` in
/// the top bits of `h.a`.
#[inline]
fn tree<T: BoolIo>(io: &mut T, t: &[i8], probs: &[u8], h: Hint, stride: usize) -> R<u32> {
    let mut i: i32 = 0;
    loop {
        let node = (i >> 1) as usize;
        let mut hh = h;
        hh.ctx = (h.ctx as usize + node * stride).min(459) as u16;
        hh.a |= (node as u32) << 28;
        let bit = io.read(*probs.get(node).ok_or(Bad)?, &hh)?;
        i = t[(i + bit as i32) as usize] as i32;
        if i <= 0 {
            return Ok((-i) as u32);
        }
    }
}

/// For every scan table and scan position: up to five raster neighbours (left, above,
/// above-left, two left, two above) that come earlier in the scan, so their tokens are known
/// when the position is coded. `u16::MAX` marks a missing neighbour.
struct ScanNeighbours {
    tables: Vec<Vec<[u16; 5]>>,
}

fn scan_neighbours() -> &'static ScanNeighbours {
    static CELL: std::sync::OnceLock<ScanNeighbours> = std::sync::OnceLock::new();
    CELL.get_or_init(|| {
        let scans: [(&[i16], usize); 10] = [
            (&DEFAULT_SCAN_4X4, 4),
            (&COL_SCAN_4X4, 4),
            (&ROW_SCAN_4X4, 4),
            (&DEFAULT_SCAN_8X8, 8),
            (&COL_SCAN_8X8, 8),
            (&ROW_SCAN_8X8, 8),
            (&DEFAULT_SCAN_16X16, 16),
            (&COL_SCAN_16X16, 16),
            (&ROW_SCAN_16X16, 16),
            (&DEFAULT_SCAN_32X32, 32),
        ];
        let tables = scans
            .iter()
            .map(|&(scan, w)| {
                let mut inv = vec![usize::MAX; w * w];
                for (i, &pos) in scan.iter().enumerate() {
                    inv[pos as usize] = i;
                }
                scan.iter()
                    .enumerate()
                    .map(|(i, &pos)| {
                        let (r, c) = (pos as i32 / w as i32, pos as i32 % w as i32);
                        let mut out = [u16::MAX; 5];
                        for (k, (dr, dc)) in [(0, -1), (-1, 0), (-1, -1), (0, -2), (-2, 0)].iter().enumerate() {
                            let (rr, cc) = (r + dr, c + dc);
                            if rr >= 0 && cc >= 0 {
                                let n = (rr * w as i32 + cc) as usize;
                                if inv[n] < i {
                                    out[k] = n as u16;
                                }
                            }
                        }
                        out
                    })
                    .collect()
            })
            .collect();
        ScanNeighbours { tables }
    })
}

/// Bucket of a count: 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33+.
#[inline]
fn eob_bucket(e: u32) -> u32 {
    if e < 3 {
        e
    } else {
        (32 - (e - 1).leading_zeros()).min(7)
    }
}

impl<'a> TileParser<'a> {
    pub fn new(
        h: &'a FrameHeader,
        fc: &'a FrameCtx,
        f: &'a mut FrameParse,
        prev_mvs: Option<&'a [MvRef]>,
        colo: Option<Colo<'a>>,
        tile_row: usize,
        tile_col: usize,
    ) -> Self {
        let (mi_rows, mi_cols) = (f.mi_rows, f.mi_cols);
        TileParser {
            h,
            fc,
            prev_mvs,
            colo,
            row_start: tile_offset(tile_row, mi_rows, h.log2_tile_rows),
            row_end: tile_offset(tile_row + 1, mi_rows, h.log2_tile_rows),
            col_start: tile_offset(tile_col, mi_cols, h.log2_tile_cols),
            col_end: tile_offset(tile_col + 1, mi_cols, h.log2_tile_cols),
            f,
            left_ctx: [[0; 16]; 3],
            left_eob: [[0; 16]; 3],
            left_seg: [0; 8],
            bmode_wl: 1,
            bmode_hl: 1,
            fq: (h.base_q >> 4).min(15) | (h.show_frame as u32) << 4 | (h.intra() as u32) << 5,
            blk_eob: [0; 3],
            cache: vec![0; 1024],
        }
    }

    pub fn run<T: BoolIo>(&mut self, io: &mut T) -> R<()> {
        if io.read(128, &hint(cx::MARKER, kind::MARKER, 0, 0, 0, 0))? != 0 {
            return Err(Bad);
        }
        let mut mi_row = self.row_start;
        while mi_row < self.row_end {
            self.left_ctx = [[0; 16]; 3];
            self.left_eob = [[0; 16]; 3];
            self.left_seg = [0; 8];
            let mut mi_col = self.col_start;
            while mi_col < self.col_end {
                self.partition(io, mi_row, mi_col, BLOCK_64X64, 4)?;
                mi_col += 8;
            }
            mi_row += 8;
        }
        Ok(())
    }

    fn partition<T: BoolIo>(&mut self, io: &mut T, mi_row: usize, mi_col: usize, bsize: u8, n4x4_l2: u32) -> R<()> {
        let n8x8_l2 = n4x4_l2 - 1;
        let num_8x8 = 1usize << n8x8_l2;
        let hbs = num_8x8 >> 1;
        if mi_row >= self.f.mi_rows || mi_col >= self.f.mi_cols {
            return Ok(());
        }
        let has_rows = mi_row + hbs < self.f.mi_rows;
        let has_cols = mi_col + hbs < self.f.mi_cols;
        let bsl = n8x8_l2;
        let above_s = self.f.above_seg[mi_col] as u32;
        let left_s = self.left_seg[mi_row & 7] as u32;
        let ctx = (((left_s >> bsl) & 1) * 2 + ((above_s >> bsl) & 1)) as usize + bsl as usize * 4;
        let fc = self.fc;
        let probs: &[u8] = if self.h.intra() { &KF_PARTITION_PROBS[ctx * 3..ctx * 3 + 3] } else { &fc.partition[ctx][..] };
        let pc = self.colo_mi(mi_row, mi_col).map_or(31, |m| m.sb_type as u32 | (m.skip as u32) << 4);
        let hh = hint(
            cx::PARTITION + ctx,
            kind::PARTITION,
            ctx as u32 | (has_rows as u32) << 4 | (has_cols as u32) << 5 | (self.h.intra() as u32) << 6,
            above_s | left_s << 8 | bsl << 16,
            self.fq | pc << 8,
            pc | bsl << 8,
        );
        let edge = |node: usize| {
            let mut h1 = hh;
            h1.ctx += 16 * node as u16;
            h1.a |= (node as u32) << 28;
            h1
        };
        let p = if has_rows && has_cols {
            tree(io, &PARTITION_TREE, probs, hh, 16)?
        } else if !has_rows && has_cols {
            if io.read(probs[1], &edge(1))? == 1 {
                3
            } else {
                1
            }
        } else if has_rows && !has_cols {
            if io.read(probs[2], &edge(2))? == 1 {
                3
            } else {
                2
            }
        } else {
            3
        };
        let subsize = SUBSIZE_LOOKUP[p as usize * 13 + bsize as usize];
        if hbs == 0 {
            self.bmode_wl = if p & 2 != 0 { 0 } else { 1 };
            self.bmode_hl = if p & 1 != 0 { 0 } else { 1 };
            self.block(io, mi_row, mi_col, subsize, 1, 1)?;
        } else {
            match p {
                0 => self.block(io, mi_row, mi_col, subsize, n4x4_l2, n4x4_l2)?,
                1 => {
                    self.block(io, mi_row, mi_col, subsize, n4x4_l2, n8x8_l2)?;
                    if has_rows {
                        self.block(io, mi_row + hbs, mi_col, subsize, n4x4_l2, n8x8_l2)?;
                    }
                }
                2 => {
                    self.block(io, mi_row, mi_col, subsize, n8x8_l2, n4x4_l2)?;
                    if has_cols {
                        self.block(io, mi_row, mi_col + hbs, subsize, n8x8_l2, n4x4_l2)?;
                    }
                }
                _ => {
                    self.partition(io, mi_row, mi_col, subsize, n8x8_l2)?;
                    self.partition(io, mi_row, mi_col + hbs, subsize, n8x8_l2)?;
                    self.partition(io, mi_row + hbs, mi_col, subsize, n8x8_l2)?;
                    self.partition(io, mi_row + hbs, mi_col + hbs, subsize, n8x8_l2)?;
                }
            }
        }
        if bsize >= BLOCK_8X8 && (bsize == BLOCK_8X8 || p != 3) {
            let a = PARTITION_CONTEXT_LOOKUP[subsize as usize * 2];
            let l = PARTITION_CONTEXT_LOOKUP[subsize as usize * 2 + 1];
            for k in 0..num_8x8 {
                self.f.above_seg[mi_col + k] = a;
                self.left_seg[((mi_row & 7) + k).min(7)] = l;
            }
        }
        Ok(())
    }

    fn mi_at(&self, row: usize, col: usize) -> &Mi {
        &self.f.grid[row * self.f.mi_cols + col]
    }

    /// The co-located block of the last shown frame.
    fn colo_mi(&self, row: usize, col: usize) -> Option<Mi> {
        self.colo.map(|(g, _)| g[row * self.f.mi_cols + col])
    }

    fn block<T: BoolIo>(&mut self, io: &mut T, mi_row: usize, mi_col: usize, bsize: u8, bwl: u32, bhl: u32) -> R<()> {
        if bsize > BLOCK_64X64 {
            return Err(Bad);
        }
        let bw = 1usize << (bwl - 1);
        let bh = 1usize << (bhl - 1);
        let (mi_rows, mi_cols) = (self.f.mi_rows, self.f.mi_cols);
        let x_mis = bw.min(mi_cols - mi_col);
        let y_mis = bh.min(mi_rows - mi_row);
        let above = (mi_row > 0).then(|| *self.mi_at(mi_row - 1, mi_col));
        let left = (mi_col > self.col_start).then(|| *self.mi_at(mi_row, mi_col - 1));
        let edges = Edges {
            top: -((mi_row as i32) * 64),
            bottom: (mi_rows as i32 - bh as i32 - mi_row as i32) * 64,
            left: -((mi_col as i32) * 64),
            right: (mi_cols as i32 - bw as i32 - mi_col as i32) * 64,
        };
        let mut mi = Mi { sb_type: bsize, ..Default::default() };
        let colo = self.colo_mi(mi_row, mi_col);
        if self.h.intra() {
            self.intra_frame_mode_info(io, &mut mi, above.as_ref(), left.as_ref(), colo.as_ref())?;
        } else {
            self.inter_frame_mode_info(io, &mut mi, above.as_ref(), left.as_ref(), colo.as_ref(), mi_row, mi_col, &edges)?;
        }
        for y in 0..y_mis {
            for x in 0..x_mis {
                let i = (mi_row + y) * mi_cols + mi_col + x;
                self.f.grid[i] = mi;
                if !self.h.intra() {
                    self.f.frame_mvs[i] = MvRef { ref_frame: mi.ref_frame, mv: mi.mv };
                }
            }
        }
        let ss = [(0usize, 0usize), (1, 1), (1, 1)];
        if mi.skip {
            for (plane, &(sx, sy)) in ss.iter().enumerate() {
                let n4w = (bw * 2) >> sx;
                let n4h = (bh * 2) >> sy;
                let a0 = (mi_col * 2) >> sx;
                let l0 = ((mi_row * 2) & 15) >> sy;
                self.f.above_ctx[plane][a0..a0 + n4w].fill(0);
                self.f.above_eob[plane][a0..a0 + n4w].fill(0);
                for k in 0..n4h {
                    self.left_ctx[plane][(l0 + k).min(15)] = 0;
                    self.left_eob[plane][(l0 + k).min(15)] = 0;
                }
            }
        }
        self.blk_eob = [0; 3];
        let inter = mi.is_inter();
        if !inter || !mi.skip {
            let mut eobtotal = 0;
            for (plane, &(sx, sy)) in ss.iter().enumerate() {
                let tx = if plane == 0 {
                    mi.tx_size
                } else if bsize < BLOCK_8X8 {
                    0
                } else {
                    UV_TXSIZE_LOOKUP[bsize as usize * 16 + mi.tx_size as usize * 4 + sx * 2 + sy]
                };
                let n4w = ((bw * 2) >> sx) as i32;
                let n4h = ((bh * 2) >> sy) as i32;
                let step = 1i32 << tx;
                let max_w = n4w + if edges.right >= 0 { 0 } else { edges.right >> (5 + sx) };
                let max_h = n4h + if edges.bottom >= 0 { 0 } else { edges.bottom >> (5 + sy) };
                let lim_w = if edges.right >= 0 { 0 } else { max_w };
                let lim_h = if edges.bottom >= 0 { 0 } else { max_h };
                let mut row = 0;
                while row < max_h {
                    let mut col = 0;
                    while col < max_w {
                        let pos = Pos { plane, mi_row, mi_col, x: col, y: row, tx, lim_w, lim_h };
                        if !inter {
                            if !mi.skip {
                                let mut mode = if plane == 0 { mi.mode } else { mi.uv_mode };
                                if bsize < BLOCK_8X8 && plane == 0 {
                                    mode = mi.bmi_mode[((row << 1) + col) as usize];
                                }
                                let tx_type = if plane > 0 || self.h.lossless { 0 } else { INTRA_MODE_TO_TX_TYPE[mode as usize] };
                                self.tokens(io, &mi, &pos, tx_type, mode)?;
                            }
                        } else {
                            eobtotal += self.tokens(io, &mi, &pos, 0, mi.mode)?;
                        }
                        col += step;
                    }
                    row += step;
                }
            }
            let promote = inter && bsize >= BLOCK_8X8 && eobtotal == 0;
            let nzc = self.blk_eob.iter().sum::<u32>().min(65535) as u16;
            for y in 0..y_mis {
                for x in 0..x_mis {
                    let m = &mut self.f.grid[(mi_row + y) * mi_cols + mi_col + x];
                    m.skip |= promote;
                    m.nzc = nzc;
                }
            }
        }
        Ok(())
    }

    fn tx_size<T: BoolIo>(&mut self, io: &mut T, mi: &Mi, above: Option<&Mi>, left: Option<&Mi>, allow_select: bool, info: u32) -> R<u8> {
        let max_tx = MAX_TXSIZE_LOOKUP[mi.sb_type as usize];
        if allow_select && self.h.tx_mode == TX_MODE_SELECT && mi.sb_type >= BLOCK_8X8 {
            let mut a = above.map_or(max_tx, |m| if m.skip { max_tx } else { m.tx_size });
            let mut l = left.map_or(max_tx, |m| if m.skip { max_tx } else { m.tx_size });
            if left.is_none() {
                l = a;
            }
            if above.is_none() {
                a = l;
            }
            let ctx = ((a + l) > max_tx) as usize;
            let fc = self.fc;
            let probs: &[u8] = match max_tx {
                1 => &fc.tx8[ctx],
                2 => &fc.tx16[ctx],
                _ => &fc.tx32[ctx],
            };
            let fq = self.fq;
            let hh = |n: usize| {
                hint(
                    cx::TX + (max_tx as usize - 1) * 6 + ctx * 3 + n,
                    kind::TX,
                    max_tx as u32 | (n as u32) << 2 | (a as u32) << 4 | (l as u32) << 6,
                    mi.sb_type as u32 | info << 4,
                    fq,
                    0,
                )
            };
            let mut tx = io.read(probs[0], &hh(0))? as u8;
            if tx != 0 && max_tx >= 2 {
                tx += io.read(probs[1], &hh(1))? as u8;
                if tx != 1 && max_tx >= 3 {
                    tx += io.read(probs[2], &hh(2))? as u8;
                }
            }
            Ok(tx)
        } else {
            Ok(max_tx.min(TX_MODE_TO_BIGGEST_TX_SIZE[self.h.tx_mode as usize]))
        }
    }

    fn skip<T: BoolIo>(&mut self, io: &mut T, above: Option<&Mi>, left: Option<&Mi>, colo: Option<&Mi>, bsize: u8) -> R<bool> {
        let ctx = above.map_or(0, |m| m.skip as usize) + left.map_or(0, |m| m.skip as usize);
        let nb = |m: Option<&Mi>| m.map_or(15, |m| m.is_inter() as u32 | (m.tx_size as u32) << 1);
        let cd = colo.map_or(0, |m| 1 | (m.skip as u32) << 1 | (m.is_inter() as u32) << 2 | eob_bucket(m.nzc as u32) << 3 | (m.sb_type as u32) << 6);
        let hh = hint(cx::SKIP + ctx, kind::SKIP, ctx as u32 | (bsize as u32) << 2, nb(above) | nb(left) << 4, self.fq | (bsize as u32) << 6, cd);
        Ok(io.read(self.fc.skip[ctx], &hh)? == 1)
    }

    fn intra_frame_mode_info<T: BoolIo>(&mut self, io: &mut T, mi: &mut Mi, above: Option<&Mi>, left: Option<&Mi>, colo: Option<&Mi>) -> R<()> {
        mi.skip = self.skip(io, above, left, colo, mi.sb_type)?;
        mi.tx_size = self.tx_size(io, mi, above, left, true, mi.skip as u32)?;
        mi.ref_frame = [0, -1];
        let fq = self.fq;
        let y_mode = |m: &Mi, b: usize| if m.sb_type < BLOCK_8X8 { m.bmi_mode[b] } else { m.mode };
        let read_b = |io: &mut T, mi: &mut Mi, b: usize| -> R<u8> {
            let a = if b < 2 { above.map_or(0, |m| y_mode(m, b + 2)) } else { mi.bmi_mode[b - 2] };
            let l = if b == 0 || b == 2 { left.map_or(0, |m| y_mode(m, b + 1)) } else { mi.bmi_mode[b - 1] };
            let o = (a as usize * 10 + l as usize) * 9;
            let hh = hint(cx::KF_Y, kind::YMODE, a as u32 | (l as u32) << 4, mi.sb_type as u32 | (b as u32) << 4, fq, 1);
            Ok(tree(io, &INTRA_MODE_TREE, &KF_Y_MODE_PROB[o..o + 9], hh, 1)? as u8)
        };
        match mi.sb_type {
            0 => {
                for b in 0..4 {
                    let m = read_b(io, mi, b)?;
                    mi.bmi_mode[b] = m;
                }
                mi.mode = mi.bmi_mode[3];
            }
            1 => {
                let m0 = read_b(io, mi, 0)?;
                mi.bmi_mode[0] = m0;
                mi.bmi_mode[2] = m0;
                let m1 = read_b(io, mi, 1)?;
                mi.bmi_mode[1] = m1;
                mi.bmi_mode[3] = m1;
                mi.mode = m1;
            }
            2 => {
                let m0 = read_b(io, mi, 0)?;
                mi.bmi_mode[0] = m0;
                mi.bmi_mode[1] = m0;
                let m2 = read_b(io, mi, 2)?;
                mi.bmi_mode[2] = m2;
                mi.bmi_mode[3] = m2;
                mi.mode = m2;
            }
            _ => {
                mi.mode = read_b(io, mi, 0)?;
            }
        }
        let o = mi.mode as usize * 9;
        let hh = hint(cx::KF_UV, kind::UVMODE, mi.mode as u32, mi.sb_type as u32, fq, 1);
        mi.uv_mode = tree(io, &INTRA_MODE_TREE, &KF_UV_MODE_PROB[o..o + 9], hh, 1)? as u8;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn inter_frame_mode_info<T: BoolIo>(
        &mut self,
        io: &mut T,
        mi: &mut Mi,
        above: Option<&Mi>,
        left: Option<&Mi>,
        colo: Option<&Mi>,
        mi_row: usize,
        mi_col: usize,
        edges: &Edges,
    ) -> R<()> {
        mi.skip = self.skip(io, above, left, colo, mi.sb_type)?;
        let ctx = match (above, left) {
            (Some(a), Some(l)) => {
                let (ai, li) = (!a.is_inter(), !l.is_inter());
                if ai && li {
                    3
                } else {
                    (ai || li) as usize
                }
            }
            (Some(e), None) | (None, Some(e)) => 2 * (!e.is_inter()) as usize,
            (None, None) => 0,
        };
        let refs = |m: Option<&Mi>| m.map_or(15, |m| (m.ref_frame[0] + 1) as u32 | ((m.ref_frame[1] + 1) as u32) << 2);
        let hh = hint(
            cx::IS_INTER + ctx,
            kind::IS_INTER,
            ctx as u32 | (mi.sb_type as u32) << 2 | (mi.skip as u32) << 6,
            refs(above) | refs(left) << 4,
            self.fq,
            colo.map_or(0, |m| 1 | (m.is_inter() as u32) << 1 | ((m.ref_frame[0] + 1) as u32) << 2 | (m.skip as u32) << 5),
        );
        let inter = io.read(self.fc.intra_inter[ctx], &hh)? == 1;
        mi.tx_size = self.tx_size(io, mi, above, left, !mi.skip || !inter, mi.skip as u32 | (inter as u32) << 1 | 4)?;
        if inter {
            self.inter_block(io, mi, above, left, colo, mi_row, mi_col, edges)
        } else {
            self.intra_block(io, mi)
        }
    }

    fn intra_block<T: BoolIo>(&mut self, io: &mut T, mi: &mut Mi) -> R<()> {
        let fc = self.fc;
        let fq = self.fq;
        let bs = mi.sb_type;
        let read_y = |io: &mut T, group: usize, b: u32| -> R<u8> {
            let hh = hint(cx::Y, kind::YMODE, group as u32 | b << 4, bs as u32, fq, 0);
            Ok(tree(io, &INTRA_MODE_TREE, &fc.y_mode[group], hh, 1)? as u8)
        };
        match bs {
            0 => {
                for b in 0..4 {
                    mi.bmi_mode[b] = read_y(io, 0, b as u32)?;
                }
                mi.mode = mi.bmi_mode[3];
            }
            1 => {
                let m = read_y(io, 0, 0)?;
                mi.bmi_mode[0] = m;
                mi.bmi_mode[2] = m;
                let m = read_y(io, 0, 1)?;
                mi.bmi_mode[1] = m;
                mi.bmi_mode[3] = m;
                mi.mode = m;
            }
            2 => {
                let m = read_y(io, 0, 0)?;
                mi.bmi_mode[0] = m;
                mi.bmi_mode[1] = m;
                let m = read_y(io, 0, 2)?;
                mi.bmi_mode[2] = m;
                mi.bmi_mode[3] = m;
                mi.mode = m;
            }
            s => mi.mode = read_y(io, SIZE_GROUP[s as usize] as usize, 0)?,
        }
        let hh = hint(cx::UV, kind::UVMODE, mi.mode as u32, bs as u32, fq, 0);
        mi.uv_mode = tree(io, &INTRA_MODE_TREE, &fc.uv_mode[mi.mode as usize], hh, 1)? as u8;
        mi.interp = 3;
        mi.ref_frame = [0, -1];
        Ok(())
    }

    fn ref_frames<T: BoolIo>(&mut self, io: &mut T, mi: &mut Mi, above: Option<&Mi>, left: Option<&Mi>, colo: Option<&Mi>) -> R<()> {
        let h = self.h;
        let fq = self.fq;
        let refs = |m: Option<&Mi>| m.map_or(15, |m| (m.ref_frame[0] + 1) as u32 | ((m.ref_frame[1] + 1) as u32) << 2);
        let nb = refs(above) | refs(left) << 4;
        let bs = mi.sb_type as u32 | refs(colo) << 4 | (colo.is_some_and(|m| m.skip) as u32) << 8;
        let mode = if h.reference_mode == REFERENCE_MODE_SELECT {
            let fixed = h.comp_fixed_ref;
            let ctx = match (above, left) {
                (Some(a), Some(l)) => {
                    if !a.has_second() && !l.has_second() {
                        ((a.ref_frame[0] == fixed) ^ (l.ref_frame[0] == fixed)) as usize
                    } else if !a.has_second() {
                        2 + (a.ref_frame[0] == fixed || !a.is_inter()) as usize
                    } else if !l.has_second() {
                        2 + (l.ref_frame[0] == fixed || !l.is_inter()) as usize
                    } else {
                        4
                    }
                }
                (Some(e), None) | (None, Some(e)) => {
                    if !e.has_second() {
                        (e.ref_frame[0] == fixed) as usize
                    } else {
                        3
                    }
                }
                (None, None) => 1,
            };
            io.read(self.fc.comp_inter[ctx], &hint(cx::COMP_INTER + ctx, kind::REF, ctx as u32, nb, fq, bs))? as u8
        } else {
            h.reference_mode
        };
        if mode == COMPOUND_REFERENCE {
            let idx = h.sign_bias[h.comp_fixed_ref as usize] as usize;
            let ctx = self.comp_ref_ctx(above, left);
            let bit = io.read(self.fc.comp_ref[ctx], &hint(cx::COMP_REF + ctx, kind::REF, ctx as u32 | 8, nb, fq, bs))?;
            mi.ref_frame[idx] = h.comp_fixed_ref;
            mi.ref_frame[1 - idx] = h.comp_var_ref[bit as usize];
        } else {
            let ctx0 = single_ref_p1(above, left);
            let bit0 = io.read(self.fc.single_ref[ctx0][0], &hint(cx::SINGLE_REF1 + ctx0, kind::REF, ctx0 as u32 | 16, nb, fq, bs))?;
            mi.ref_frame[0] = if bit0 == 1 {
                let ctx1 = single_ref_p2(above, left);
                let bit1 = io.read(self.fc.single_ref[ctx1][1], &hint(cx::SINGLE_REF2 + ctx1, kind::REF, ctx1 as u32 | 24, nb, fq, bs))?;
                if bit1 == 1 {
                    ALTREF_FRAME
                } else {
                    GOLDEN_FRAME
                }
            } else {
                LAST_FRAME
            };
            mi.ref_frame[1] = -1;
        }
        Ok(())
    }

    fn comp_ref_ctx(&self, above: Option<&Mi>, left: Option<&Mi>) -> usize {
        let h = self.h;
        let fix_ref_idx = h.sign_bias[h.comp_fixed_ref as usize] as usize;
        let var_ref_idx = 1 - fix_ref_idx;
        let v1 = h.comp_var_ref[1];
        match (above, left) {
            (Some(a), Some(l)) => {
                let (ai, li) = (!a.is_inter(), !l.is_inter());
                if ai && li {
                    2
                } else if ai || li {
                    let e = if ai { l } else { a };
                    if !e.has_second() {
                        1 + 2 * (e.ref_frame[0] != v1) as usize
                    } else {
                        1 + 2 * (e.ref_frame[var_ref_idx] != v1) as usize
                    }
                } else {
                    let l_sg = !l.has_second();
                    let a_sg = !a.has_second();
                    let vrfa = if a_sg { a.ref_frame[0] } else { a.ref_frame[var_ref_idx] };
                    let vrfl = if l_sg { l.ref_frame[0] } else { l.ref_frame[var_ref_idx] };
                    if vrfa == vrfl && v1 == vrfa {
                        0
                    } else if l_sg && a_sg {
                        if (vrfa == h.comp_fixed_ref && vrfl == h.comp_var_ref[0])
                            || (vrfl == h.comp_fixed_ref && vrfa == h.comp_var_ref[0])
                        {
                            4
                        } else if vrfa == vrfl {
                            3
                        } else {
                            1
                        }
                    } else if l_sg || a_sg {
                        let vrfc = if l_sg { vrfa } else { vrfl };
                        let rfs = if a_sg { vrfa } else { vrfl };
                        if vrfc == v1 && rfs != v1 {
                            1
                        } else if rfs == v1 && vrfc != v1 {
                            2
                        } else {
                            4
                        }
                    } else if vrfa == vrfl {
                        4
                    } else {
                        2
                    }
                }
            }
            (Some(e), None) | (None, Some(e)) => {
                if !e.is_inter() {
                    2
                } else if e.has_second() {
                    4 * (e.ref_frame[var_ref_idx] != v1) as usize
                } else {
                    3 * (e.ref_frame[0] != v1) as usize
                }
            }
            (None, None) => 2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn inter_block<T: BoolIo>(
        &mut self,
        io: &mut T,
        mi: &mut Mi,
        above: Option<&Mi>,
        left: Option<&Mi>,
        colo: Option<&Mi>,
        mi_row: usize,
        mi_col: usize,
        edges: &Edges,
    ) -> R<()> {
        self.ref_frames(io, mi, above, left, colo)?;
        let compound = mi.has_second();
        let bs = mi.sb_type;
        let search = &MV_REF_BLOCKS[bs as usize * 16..bs as usize * 16 + 16];
        // mode context from the two nearest neighbours
        let mut counter = 0usize;
        let mut nb_modes = 0u32;
        for i in 0..2 {
            let (r, c) = (search[2 * i] as i32, search[2 * i + 1] as i32);
            match self.candidate(mi_row, mi_col, r, c) {
                Some(m) => {
                    counter += MODE_2_COUNTER[m.mode as usize];
                    nb_modes |= (m.mode as u32) << (4 * i);
                }
                None => nb_modes |= 15 << (4 * i),
            }
        }
        let mode_ctx = COUNTER_TO_CONTEXT[counter] as usize;
        if mode_ctx > 6 {
            return Err(Bad);
        }
        let fq = self.fq;
        let rf0 = mi.ref_frame[0] as u32;
        let cm = colo.map_or(255, |m| m.mode as u32 | ((m.ref_frame[0] + 1) as u32) << 4 | (m.skip as u32) << 7);
        let read_mode = |io: &mut T, fc: &FrameCtx, b: u32| -> R<u8> {
            let hh = hint(
                cx::INTER_MODE + mode_ctx * 3,
                kind::INTER_MODE,
                mode_ctx as u32 | rf0 << 4 | (compound as u32) << 6,
                bs as u32 | b << 4,
                nb_modes,
                fq | cm << 6,
            );
            Ok(10 + tree(io, &INTER_MODE_TREE, &fc.inter_mode[mode_ctx], hh, 1)? as u8)
        };
        if bs >= BLOCK_8X8 {
            mi.mode = read_mode(io, self.fc, 0)?;
        }
        mi.interp = if self.h.interp_filter == SWITCHABLE {
            let lt = left.map_or(3, |m| m.interp);
            let at = above.map_or(3, |m| m.interp);
            let ctx = if lt == at {
                lt
            } else if lt == 3 {
                at
            } else if at == 3 {
                lt
            } else {
                3
            } as usize;
            let hh = hint(cx::INTERP + ctx * 2, kind::INTERP, ctx as u32 | (lt as u32) << 2 | (at as u32) << 4, bs as u32 | (mi.mode as u32) << 4, fq, 0);
            tree(io, &SWITCHABLE_INTERP_TREE, &self.fc.interp[ctx], hh, 1)? as u8
        } else {
            self.h.interp_filter
        };
        let allow_hp = self.h.allow_hp;
        let mut tmv = [None; 2];
        if let Some(m) = colo.filter(|m| m.is_inter()) {
            for (r, t) in tmv.iter_mut().enumerate().take(1 + compound as usize) {
                let rf = mi.ref_frame[r];
                *t = if m.ref_frame[0] == rf {
                    Some(m.mv[0])
                } else if m.ref_frame[1] == rf {
                    Some(m.mv[1])
                } else {
                    None
                };
            }
        }
        if bs < BLOCK_8X8 {
            let n4w = 1usize << self.bmode_wl;
            let n4h = 1usize << self.bmode_hl;
            let mut best_ref = [[0i16; 2]; 2];
            let mut got_new = false;
            let mut b_mode = 10u8;
            let mut idy = 0;
            while idy < 2 {
                let mut idx = 0;
                while idx < 2 {
                    let j = idy * 2 + idx;
                    b_mode = read_mode(io, self.fc, 1 + j as u32)?;
                    let mut best_sub = [[0i16; 2]; 2];
                    if b_mode == 10 || b_mode == 11 {
                        for (r, sub) in best_sub.iter_mut().enumerate().take(1 + compound as usize) {
                            *sub = self.sub8x8_mv(mi, search, b_mode, j, r, mi_row, mi_col, edges);
                        }
                    } else if b_mode == 13 && !got_new {
                        for (r, best) in best_ref.iter_mut().enumerate().take(1 + compound as usize) {
                            let (list, _) = self.find_mv_refs(13, mi.ref_frame[r], search, mi_row, mi_col, -1, edges);
                            *best = lower_precision(list[0], allow_hp);
                        }
                        got_new = true;
                    }
                    let mv = self.assign_mv(io, b_mode, &best_ref, &best_sub, compound, bs, j as u32, &tmv)?;
                    mi.bmi_mv[j] = mv;
                    mi.bmi_mode[j] = b_mode;
                    if n4h == 2 {
                        mi.bmi_mv[j + 2] = mv;
                        mi.bmi_mode[j + 2] = b_mode;
                    }
                    if n4w == 2 {
                        mi.bmi_mv[j + 1] = mv;
                        mi.bmi_mode[j + 1] = b_mode;
                    }
                    idx += n4w;
                }
                idy += n4h;
            }
            mi.mode = b_mode;
            mi.mv = mi.bmi_mv[3];
        } else {
            let mut best = [[0i16; 2]; 2];
            if mi.mode != 12 {
                for (r, b) in best.iter_mut().enumerate().take(1 + compound as usize) {
                    let (list, count) = self.find_mv_refs(mi.mode, mi.ref_frame[r], search, mi_row, mi_col, -1, edges);
                    *b = lower_precision(list[count - 1], allow_hp);
                }
            }
            mi.mv = self.assign_mv(io, mi.mode, &best, &best, compound, bs, 0, &tmv)?;
        }
        Ok(())
    }

    fn candidate(&self, mi_row: usize, mi_col: usize, dr: i32, dc: i32) -> Option<&Mi> {
        let r = mi_row as i32 + dr;
        let c = mi_col as i32 + dc;
        if r < 0 || c < self.col_start as i32 || r >= self.f.mi_rows as i32 || c >= self.col_end as i32 {
            return None;
        }
        Some(self.mi_at(r as usize, c as usize))
    }

    /// libvpx `dec_find_mv_refs`: returns the candidate list and its length.
    #[allow(clippy::too_many_arguments)]
    fn find_mv_refs(&self, mode: u8, ref_frame: i8, search: &[i8], mi_row: usize, mi_col: usize, block: i32, edges: &Edges) -> ([[i16; 2]; 2], usize) {
        let sb = &self.h.sign_bias;
        let mut list = [[0i16; 2]; 2];
        let mut count = 0usize;
        let early_break = mode != 11;
        let mut different_ref_found = false;
        let prev = self.prev_mvs.and_then(|p| p.get(mi_row * self.f.mi_cols + mi_col));
        // returns true when the search is done
        let add = |list: &mut [[i16; 2]; 2], count: &mut usize, mv: [i16; 2]| -> bool {
            if *count > 0 {
                if mv != list[0] {
                    list[*count] = mv;
                    *count += 1;
                    return true;
                }
                false
            } else {
                list[0] = mv;
                *count = 1;
                early_break
            }
        };
        let done = 'search: {
            let mut i = 0;
            if block >= 0 {
                while i < 2 {
                    let (r, c) = (search[2 * i] as i32, search[2 * i + 1] as i32);
                    if let Some(cand) = self.candidate(mi_row, mi_col, r, c) {
                        different_ref_found = true;
                        let sub = |which: usize| -> [i16; 2] {
                            if cand.sb_type < BLOCK_8X8 {
                                cand.bmi_mv[IDX_N_COLUMN_TO_SUBBLOCK[block as usize][(c == 0) as usize]][which]
                            } else {
                                cand.mv[which]
                            }
                        };
                        if cand.ref_frame[0] == ref_frame {
                            if add(&mut list, &mut count, sub(0)) {
                                break 'search true;
                            }
                        } else if cand.ref_frame[1] == ref_frame && add(&mut list, &mut count, sub(1)) {
                            break 'search true;
                        }
                    }
                    i += 1;
                }
            }
            while i < 8 {
                let (r, c) = (search[2 * i] as i32, search[2 * i + 1] as i32);
                if let Some(cand) = self.candidate(mi_row, mi_col, r, c) {
                    different_ref_found = true;
                    if cand.ref_frame[0] == ref_frame {
                        if add(&mut list, &mut count, cand.mv[0]) {
                            break 'search true;
                        }
                    } else if cand.ref_frame[1] == ref_frame && add(&mut list, &mut count, cand.mv[1]) {
                        break 'search true;
                    }
                }
                i += 1;
            }
            if let Some(p) = prev {
                if p.ref_frame[0] == ref_frame {
                    if add(&mut list, &mut count, p.mv[0]) {
                        break 'search true;
                    }
                } else if p.ref_frame[1] == ref_frame && add(&mut list, &mut count, p.mv[1]) {
                    break 'search true;
                }
            }
            if different_ref_found {
                for i in 0..8 {
                    let (r, c) = (search[2 * i] as i32, search[2 * i + 1] as i32);
                    if let Some(cand) = self.candidate(mi_row, mi_col, r, c) {
                        if cand.is_inter() {
                            if cand.ref_frame[0] != ref_frame
                                && add(&mut list, &mut count, scale(cand.mv[0], cand.ref_frame[0], ref_frame, sb))
                            {
                                break 'search true;
                            }
                            if cand.has_second()
                                && cand.ref_frame[1] != ref_frame
                                && cand.mv[1] != cand.mv[0]
                                && add(&mut list, &mut count, scale(cand.mv[1], cand.ref_frame[1], ref_frame, sb))
                            {
                                break 'search true;
                            }
                        }
                    }
                }
            }
            if let Some(p) = prev {
                if p.ref_frame[0] != ref_frame
                    && p.ref_frame[0] > 0
                    && add(&mut list, &mut count, scale(p.mv[0], p.ref_frame[0], ref_frame, sb))
                {
                    break 'search true;
                }
                if p.ref_frame[1] > 0
                    && p.ref_frame[1] != ref_frame
                    && p.mv[1] != p.mv[0]
                    && add(&mut list, &mut count, scale(p.mv[1], p.ref_frame[1], ref_frame, sb))
                {
                    break 'search true;
                }
            }
            false
        };
        if !done {
            count = if mode == 11 { 2 } else { 1 };
        }
        for mv in list.iter_mut().take(count) {
            mv[1] = (mv[1] as i32).clamp(edges.left - 128, edges.right + 128) as i16;
            mv[0] = (mv[0] as i32).clamp(edges.top - 128, edges.bottom + 128) as i16;
        }
        (list, count)
    }

    #[allow(clippy::too_many_arguments)]
    fn sub8x8_mv(&self, mi: &Mi, search: &[i8], b_mode: u8, block: usize, r: usize, mi_row: usize, mi_col: usize, edges: &Edges) -> [i16; 2] {
        let rf = mi.ref_frame[r];
        match block {
            0 => {
                let (list, count) = self.find_mv_refs(b_mode, rf, search, mi_row, mi_col, 0, edges);
                list[count - 1]
            }
            1 | 2 => {
                if b_mode == 10 {
                    mi.bmi_mv[0][r]
                } else {
                    let (list, _) = self.find_mv_refs(b_mode, rf, search, mi_row, mi_col, block as i32, edges);
                    list.iter().copied().find(|&mv| mv != mi.bmi_mv[0][r]).unwrap_or([0, 0])
                }
            }
            _ => {
                if b_mode == 10 {
                    mi.bmi_mv[2][r]
                } else if mi.bmi_mv[2][r] != mi.bmi_mv[1][r] {
                    mi.bmi_mv[1][r]
                } else if mi.bmi_mv[2][r] != mi.bmi_mv[0][r] {
                    mi.bmi_mv[0][r]
                } else {
                    let (list, _) = self.find_mv_refs(b_mode, rf, search, mi_row, mi_col, 3, edges);
                    list.iter().copied().find(|&mv| mv != mi.bmi_mv[2][r]).unwrap_or([0, 0])
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_mv<T: BoolIo>(
        &mut self,
        io: &mut T,
        mode: u8,
        best_ref: &[[i16; 2]; 2],
        near: &[[i16; 2]; 2],
        compound: bool,
        bsize: u8,
        sub: u32,
        tmv: &[Option<[i16; 2]>; 2],
    ) -> R<[[i16; 2]; 2]> {
        match mode {
            13 => {
                let mut out = [[0i16; 2]; 2];
                for (r, mv) in out.iter_mut().enumerate().take(1 + compound as usize) {
                    *mv = self.read_mv(io, best_ref[r], bsize | (r as u8) << 4, sub, tmv[r])?;
                }
                Ok(out)
            }
            10 | 11 => Ok(*near),
            _ => Ok([[0, 0]; 2]),
        }
    }

    fn read_mv<T: BoolIo>(&mut self, io: &mut T, rf: [i16; 2], info: u8, sub: u32, tmv: Option<[i16; 2]>) -> R<[i16; 2]> {
        let fc = self.fc;
        let use_hp = self.h.allow_hp && rf[0].unsigned_abs() < 64 && rf[1].unsigned_abs() < 64;
        let mag = |v: i16| eob_bucket(v.unsigned_abs() as u32 >> 2);
        let pred = mag(rf[0]) | mag(rf[1]) << 3;
        // the co-located vector of the last frame, as a difference from the predicted one
        let pd = |k: usize| tmv.map(|m| m[k] as i32 - rf[k] as i32);
        let (p0, p1) = (pd(0), pd(1));
        let t = |d: Option<i32>| d.map_or(0, |d| 1 | ((d > 0) as u32) << 1 | ((d < 0) as u32) << 2 | eob_bucket(d.unsigned_abs() >> 1) << 3);
        let hh = hint(cx::MV_JOINT, kind::MV_JOINT, info as u32 | (use_hp as u32) << 8 | sub << 9, pred, self.fq, t(p0) | t(p1) << 6);
        let joint = tree(io, &MV_JOINT_TREE, &fc.mv_joints, hh, 1)?;
        let mut diff = [0i32; 2];
        if joint == 2 || joint == 3 {
            diff[0] = self.mv_component(io, 0, use_hp, joint, pred, p0)?;
        }
        if joint == 1 || joint == 3 {
            let row = (diff[0] != 0) as u32 | ((diff[0] < 0) as u32) << 1;
            diff[1] = self.mv_component(io, 1, use_hp, joint, pred | row << 6, p1)?;
        }
        let mv = [rf[0] as i32 + diff[0], rf[1] as i32 + diff[1]];
        // libvpx flags vectors outside +-2^14 as corrupt
        if mv.iter().any(|v| v.abs() >= 1 << 14) {
            return Err(Bad);
        }
        Ok([mv[0] as i16, mv[1] as i16])
    }

    /// Reads one vector component. `pd` is the difference the last frame's co-located vector
    /// predicts; the hints say, field by field, what that prediction would code.
    fn mv_component<T: BoolIo>(&mut self, io: &mut T, comp: usize, use_hp: bool, joint: u32, pred: u32, pd: Option<i32>) -> R<i32> {
        let c = &self.fc.mv[comp];
        let base = cx::MV + comp * 30;
        let fq = self.fq;
        let id = comp as u32 | joint << 1;
        // the predicted difference split like the bitstream codes it
        let (ps, pclass, pdd, pfr, php) = match pd {
            Some(v) if v != 0 => {
                let z = v.unsigned_abs() as i32 - 1;
                let cl = if z >= 8192 { 10 } else { (31 - ((z >> 3) | 1).leading_zeros()) as i32 };
                let off = z - if cl == 0 { 0 } else { 2 << (cl + 2) };
                (1 + (v < 0) as u32, cl as u32, off >> 3, (off >> 1) & 3, off & 1)
            }
            Some(_) => (3, 15, 0, 0, 0),
            None => (0, 15, 0, 0, 0),
        };
        let t = ps | pclass << 2;
        let sign = io.read(c.sign, &hint(base, kind::MV_SIGN, id, pred, fq, t))?;
        let ts = t | sign << 8 | ((ps == 1 + sign) as u32) << 9;
        let class = tree(io, &MV_CLASS_TREE, &c.classes, hint(base + 1, kind::MV_CLASS, id | sign << 3, pred, fq, ts), 1)? as i32;
        let same = (class as u32 == pclass) as u32;
        let (d, mag) = if class == 0 {
            let hh = hint(base + 11, kind::MV_BITS, id, pred, fq, ts | same << 10 | ((pdd & 1) as u32) << 11);
            (io.read(c.class0, &hh)? as i32, 0)
        } else {
            let mut d = 0;
            for i in 0..class as usize {
                // does the prediction agree with the bits read so far, and what is its next bit?
                let agree = same != 0 && (pdd & ((1 << i) - 1)) == d;
                let pb = agree as u32 | (((pdd >> i) & 1) as u32) << 1;
                let hh = hint(base + 12 + i, kind::MV_BITS, id | (i as u32) << 3 | (class as u32) << 7, d as u32 & 0xFF, pred, ts | pb << 10);
                d |= (io.read(c.bits[i], &hh)? as i32) << i;
            }
            (d, 2 << (class + 2))
        };
        let agree = same != 0 && d == pdd;
        let pf = agree as u32 | (pfr as u32) << 1;
        let (fp, fbase): (&[u8], usize) = if class == 0 { (&c.class0_fp[d as usize & 1], base + 22) } else { (&c.fp, base + 25) };
        let fr = tree(io, &MV_FP_TREE, fp, hint(fbase, kind::MV_FR, id | (class as u32) << 3, d as u32 & 1 | pf << 1, pred, ts | pf << 10), 1)? as i32;
        let hp = if use_hp {
            let ph = (agree && fr == pfr) as u32 | (php as u32) << 1;
            let hh = hint(base + 28 + (class != 0) as usize, kind::MV_HP, id | (class as u32) << 3, fr as u32 | ph << 2, pred, ts | ph << 10);
            io.read(if class == 0 { c.class0_hp } else { c.hp }, &hh)? as i32
        } else {
            1
        };
        let mag = mag + ((d << 3) | (fr << 1) | hp) + 1;
        Ok(if sign == 1 { -mag } else { mag })
    }

    /// libvpx `vp9_decode_block_tokens` + `decode_coefs`; returns the end of block.
    fn tokens<T: BoolIo>(&mut self, io: &mut T, mi: &Mi, pos: &Pos, tx_type: u8, mode: u8) -> R<usize> {
        let plane = pos.plane;
        let tx = pos.tx;
        let (sx, sy) = if plane == 0 { (0, 0) } else { (1, 1) };
        let a0 = ((pos.mi_col * 2) >> sx) + pos.x as usize;
        let l0 = (((pos.mi_row * 2) & 15) >> sy) + pos.y as usize;
        let n = 1usize << tx;
        let mut above_nz = false;
        let mut left_nz = false;
        let mut above_e = 0u32;
        let mut left_e = 0u32;
        let mut dc_ctx = 0u32;
        for k in 0..n {
            if let Some(&v) = self.f.above_ctx[plane].get(a0 + k) {
                above_nz |= v != 0;
            }
            if let Some(&v) = self.left_ctx[plane].get(l0 + k) {
                left_nz |= v != 0;
            }
            if let Some(&v) = self.f.above_eob[plane].get(a0 + k) {
                above_e = above_e.max(v as u32 & 63);
                if k == 0 {
                    dc_ctx = v as u32 >> 6;
                }
            }
            if let Some(&v) = self.left_eob[plane].get(l0 + k) {
                left_e = left_e.max(v as u32 & 63);
                if k == 0 {
                    dc_ctx |= (v as u32 >> 6) << 2;
                }
            }
        }
        let mut ctx = above_nz as usize + left_nz as usize;
        // end of block at the same place in the last shown frame
        let (pw, ph) = ((self.f.mi_cols * 2) >> sx, (self.f.mi_rows * 2) >> sy);
        let (col4, row4) = (a0, ((pos.mi_row * 2) >> sy) + pos.y as usize);
        let pe = match self.colo {
            Some((_, e)) if col4 < pw && row4 < ph => eob_bucket(e[plane][row4 * pw + col4] as u32),
            _ => 8,
        };
        let (scan, nb, si): (&[i16], &[i16], usize) = match (tx, if tx == 3 { 0 } else { tx_type }) {
            (0, 1) => (&ROW_SCAN_4X4, &ROW_SCAN_4X4_NEIGHBORS, 2),
            (0, 2) => (&COL_SCAN_4X4, &COL_SCAN_4X4_NEIGHBORS, 1),
            (0, _) => (&DEFAULT_SCAN_4X4, &DEFAULT_SCAN_4X4_NEIGHBORS, 0),
            (1, 1) => (&ROW_SCAN_8X8, &ROW_SCAN_8X8_NEIGHBORS, 5),
            (1, 2) => (&COL_SCAN_8X8, &COL_SCAN_8X8_NEIGHBORS, 4),
            (1, _) => (&DEFAULT_SCAN_8X8, &DEFAULT_SCAN_8X8_NEIGHBORS, 3),
            (2, 1) => (&ROW_SCAN_16X16, &ROW_SCAN_16X16_NEIGHBORS, 8),
            (2, 2) => (&COL_SCAN_16X16, &COL_SCAN_16X16_NEIGHBORS, 7),
            (2, _) => (&DEFAULT_SCAN_16X16, &DEFAULT_SCAN_16X16_NEIGHBORS, 6),
            _ => (&DEFAULT_SCAN_32X32, &DEFAULT_SCAN_32X32_NEIGHBORS, 9),
        };
        let nbr5 = &scan_neighbours().tables[si];
        let max_eob = 16usize << (tx << 1);
        let ptype = (plane > 0) as usize;
        let rf = mi.is_inter() as usize;
        let fc = self.fc;
        let probs = &fc.coef[tx as usize][ptype][rf];
        let band_tr: &[u8] = if tx == 0 { &COEFBAND_TRANS_4X4 } else { &COEFBAND_TRANS_8X8PLUS };
        let mut cache = std::mem::take(&mut self.cache);
        if cache.len() < 1024 {
            cache = vec![0; 1024];
        }
        let mut c = 0usize;
        let mut nz = 0u32;
        let mut prev_tok = 0u32;
        let mut dc_sign = 0u8;
        let tag = tx as u32 | (ptype as u32) << 2 | (rf as u32) << 3;
        let nbe = eob_bucket(above_e) | eob_bucket(left_e) << 3;
        let luma = if plane == 0 { 0 } else { eob_bucket(self.blk_eob[0]) | eob_bucket(self.blk_eob[1]) << 3 };
        let blk = (mi.sb_type as u32) | (mode as u32) << 4 | (tx_type as u32) << 8;
        let fq = self.fq;
        // Position-dependent parts of the token hints, computed once per position.
        let pos_hint = |c: usize, ctx: usize, nz: u32, prev: u32, cache: &[u8]| -> [u32; 4] {
            let (n0, n1) = if c > 0 && c < max_eob {
                (cache[nb[2 * c] as usize] as u32, cache[nb[2 * c + 1] as usize] as u32)
            } else {
                (0, 0)
            };
            let mut mag = 0u32;
            for &n in &nbr5[c.min(max_eob - 1)] {
                if n != u16::MAX {
                    mag += (cache[n as usize] as u32).min(3);
                }
            }
            let dc = if c == 0 { dc_ctx } else { 0 };
            [
                tag << 4 | (ctx as u32) << 8 | (c.min(255) as u32) << 11,
                tag << 4 | mag.min(15) << 8 | (ctx as u32) << 12 | nz.min(15) << 15 | nbe << 19,
                tag << 4 | n0 << 8 | n1 << 11 | prev.min(7) << 14 | (c.min(15) as u32) << 17 | (tx_type as u32) << 21 | pe << 23,
                tag << 4 | fq << 8 | blk << 14 | dc << 24 | luma << 28,
            ]
        };
        let mk = |node: usize, band: usize, ph: &[u32; 4]| -> Hint {
            let mctx = if node < 3 {
                let grp = if band == 0 {
                    0
                } else if band < 3 {
                    1
                } else {
                    2
                };
                cx::TOKEN + ((node * 4 + tx as usize) * 2 + ptype) * 3 + grp
            } else {
                cx::PARETO + (node - 3) * 2 + ptype
            };
            let nd = node as u32;
            hint(mctx, kind::TOKEN, nd | ph[0], nd | ph[1], nd | ph[2], nd | ph[3])
        };
        while c < max_eob {
            let mut band = band_tr[c] as usize;
            let mut p = &probs[band][ctx];
            let mut ph = pos_hint(c, ctx, nz, prev_tok, &cache);
            if io.read(p[0], &mk(0, band, &ph))? == 0 {
                break;
            }
            while io.read(p[1], &mk(1, band, &ph))? == 0 {
                cache[scan[c] as usize] = 0;
                prev_tok = 0;
                c += 1;
                if c >= max_eob {
                    self.cache = cache;
                    return self.finish_tokens(pos, a0, l0, n, c, nz, dc_sign);
                }
                ctx = ((1 + cache[nb[2 * c] as usize] as u32 + cache[nb[2 * c + 1] as usize] as u32) >> 1) as usize;
                band = band_tr[c] as usize;
                p = &probs[band][ctx];
                ph = pos_hint(c, ctx, nz, prev_tok, &cache);
            }
            let tok;
            if io.read(p[2], &mk(2, band, &ph))? == 0 {
                tok = 1u8;
            } else {
                let pi = (p[2] as usize).max(1) * 8;
                let pp = &PARETO8_FULL[pi - 8..pi];
                let node = |k: usize, io: &mut T| io.read(pp[k], &mk(3 + k, band, &ph));
                if node(0, io)? == 1 {
                    if node(3, io)? == 1 {
                        tok = 5;
                        let cat = if node(5, io)? == 1 {
                            if node(7, io)? == 1 {
                                6
                            } else {
                                5
                            }
                        } else if node(6, io)? == 1 {
                            4
                        } else {
                            3
                        };
                        self.extra(io, cat, tag)?;
                    } else {
                        tok = 4;
                        let cat = if node(4, io)? == 1 { 2 } else { 1 };
                        self.extra(io, cat, tag)?;
                    }
                } else if node(1, io)? == 1 {
                    tok = 3;
                    node(2, io)?;
                } else {
                    tok = 2;
                }
            }
            cache[scan[c] as usize] = tok;
            prev_tok = tok as u32;
            nz += 1;
            let sh = hint(
                cx::SIGN + tx as usize * 2 + ptype,
                kind::SIGN,
                tok as u32 | (c.min(15) as u32) << 3 | tag << 7,
                if c == 0 { dc_ctx | 16 } else { 0 },
                0,
                0,
            );
            let s = io.read(128, &sh)?;
            if c == 0 {
                dc_sign = 1 + s as u8;
            }
            c += 1;
            if c < max_eob {
                ctx = ((1 + cache[nb[2 * c] as usize] as u32 + cache[nb[2 * c + 1] as usize] as u32) >> 1) as usize;
            }
        }
        self.cache = cache;
        self.finish_tokens(pos, a0, l0, n, c, nz, dc_sign)
    }

    fn extra<T: BoolIo>(&mut self, io: &mut T, cat: usize, tag: u32) -> R<()> {
        const OFF: [usize; 7] = [0, 0, 1, 3, 6, 10, 15];
        let probs: &[u8] = match cat {
            1 => &CAT1_PROB,
            2 => &CAT2_PROB,
            3 => &CAT3_PROB,
            4 => &CAT4_PROB,
            5 => &CAT5_PROB,
            _ => match self.h.bit_depth {
                12 => &CAT6_PROB_HIGH12,
                10 => &CAT6_PROB_HIGH12[2..],
                _ => &CAT6_PROB,
            },
        };
        let mut acc = 1u32;
        for (i, &p) in probs.iter().enumerate() {
            let hh = hint(cx::EXTRA + OFF[cat] + i, kind::EXTRA, cat as u32 | (i as u32) << 3 | acc.min(255) << 7, tag, 0, 0);
            let b = io.read(p, &hh)?;
            acc = (acc << 1) | b;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_tokens(&mut self, pos: &Pos, a0: usize, l0: usize, n: usize, eob: usize, nz: u32, dc_sign: u8) -> R<usize> {
        let v = (eob > 0) as u8;
        let e = eob.min(63) as u8 | dc_sign << 6;
        let mut keep_a = n;
        let mut keep_l = n;
        if n > 1 {
            if pos.lim_w != 0 && n as i32 + pos.x > pos.lim_w {
                keep_a = (pos.lim_w - pos.x).max(0) as usize;
            }
            if pos.lim_h != 0 && n as i32 + pos.y > pos.lim_h {
                keep_l = (pos.lim_h - pos.y).max(0) as usize;
            }
        }
        let plane = pos.plane;
        for k in 0..n {
            if let Some(a) = self.f.above_ctx[plane].get_mut(a0 + k) {
                *a = if k < keep_a { v } else { 0 };
            }
            if let Some(a) = self.f.above_eob[plane].get_mut(a0 + k) {
                *a = if k < keep_a { e } else { 0 };
            }
            if l0 + k < 16 {
                self.left_ctx[plane][l0 + k] = if k < keep_l { v } else { 0 };
                self.left_eob[plane][l0 + k] = if k < keep_l { e } else { 0 };
            }
        }
        let sh = (plane > 0) as usize;
        let (pw, ph) = ((self.f.mi_cols * 2) >> sh, (self.f.mi_rows * 2) >> sh);
        let row4 = ((pos.mi_row * 2) >> sh) + pos.y as usize;
        let col4 = ((pos.mi_col * 2) >> sh) + pos.x as usize;
        let em = eob.min(63) as u8;
        for r in row4..(row4 + n).min(ph) {
            let line = &mut self.f.eob_map[plane][r * pw..r * pw + pw];
            for v in &mut line[col4.min(pw)..(col4 + n).min(pw)] {
                *v = em;
            }
        }
        self.blk_eob[plane] += nz;
        Ok(eob)
    }
}

/// Position of a transform block: plane, block origin, offset in 4x4 units, size and the
/// frame-edge limits of `vp9_decode_block_tokens`.
struct Pos {
    plane: usize,
    mi_row: usize,
    mi_col: usize,
    x: i32,
    y: i32,
    tx: u8,
    lim_w: i32,
    lim_h: i32,
}

struct Edges {
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
}

const MODE_2_COUNTER: [usize; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
const COUNTER_TO_CONTEXT: [u8; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];
const IDX_N_COLUMN_TO_SUBBLOCK: [[usize; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

fn scale(mv: [i16; 2], cand_ref: i8, this_ref: i8, sb: &[u8; 4]) -> [i16; 2] {
    if sb[cand_ref as usize] != sb[this_ref as usize] {
        [mv[0].wrapping_neg(), mv[1].wrapping_neg()]
    } else {
        mv
    }
}

fn lower_precision(mut mv: [i16; 2], allow_hp: bool) -> [i16; 2] {
    let use_hp = allow_hp && mv[0].unsigned_abs() < 64 && mv[1].unsigned_abs() < 64;
    if !use_hp {
        for v in mv.iter_mut() {
            if *v & 1 != 0 {
                *v += if *v > 0 { -1 } else { 1 };
            }
        }
    }
    mv
}

fn single_ref_p1(above: Option<&Mi>, left: Option<&Mi>) -> usize {
    const L: i8 = LAST_FRAME;
    match (above, left) {
        (Some(a), Some(l)) => {
            let (ai, li) = (!a.is_inter(), !l.is_inter());
            if ai && li {
                2
            } else if ai || li {
                let e = if ai { l } else { a };
                if !e.has_second() {
                    4 * (e.ref_frame[0] == L) as usize
                } else {
                    1 + (e.ref_frame[0] == L || e.ref_frame[1] == L) as usize
                }
            } else {
                let (a2, l2) = (a.has_second(), l.has_second());
                let (a0, a1, l0, l1) = (a.ref_frame[0], a.ref_frame[1], l.ref_frame[0], l.ref_frame[1]);
                if a2 && l2 {
                    1 + (a0 == L || a1 == L || l0 == L || l1 == L) as usize
                } else if a2 || l2 {
                    let rfs = if !a2 { a0 } else { l0 };
                    let crf1 = if a2 { a0 } else { l0 };
                    let crf2 = if a2 { a1 } else { l1 };
                    if rfs == L {
                        3 + (crf1 == L || crf2 == L) as usize
                    } else {
                        (crf1 == L || crf2 == L) as usize
                    }
                } else {
                    2 * (a0 == L) as usize + 2 * (l0 == L) as usize
                }
            }
        }
        (Some(e), None) | (None, Some(e)) => {
            if !e.is_inter() {
                2
            } else if !e.has_second() {
                4 * (e.ref_frame[0] == L) as usize
            } else {
                1 + (e.ref_frame[0] == L || e.ref_frame[1] == L) as usize
            }
        }
        (None, None) => 2,
    }
}

fn single_ref_p2(above: Option<&Mi>, left: Option<&Mi>) -> usize {
    const L: i8 = LAST_FRAME;
    const G: i8 = GOLDEN_FRAME;
    const A: i8 = ALTREF_FRAME;
    match (above, left) {
        (Some(a), Some(l)) => {
            let (ai, li) = (!a.is_inter(), !l.is_inter());
            if ai && li {
                2
            } else if ai || li {
                let e = if ai { l } else { a };
                if !e.has_second() {
                    if e.ref_frame[0] == L {
                        3
                    } else {
                        4 * (e.ref_frame[0] == G) as usize
                    }
                } else {
                    1 + 2 * (e.ref_frame[0] == G || e.ref_frame[1] == G) as usize
                }
            } else {
                let (a2, l2) = (a.has_second(), l.has_second());
                let (a0, a1, l0, l1) = (a.ref_frame[0], a.ref_frame[1], l.ref_frame[0], l.ref_frame[1]);
                if a2 && l2 {
                    if a0 == l0 && a1 == l1 {
                        3 * (a0 == G || a1 == G || l0 == G || l1 == G) as usize
                    } else {
                        2
                    }
                } else if a2 || l2 {
                    let rfs = if !a2 { a0 } else { l0 };
                    let crf1 = if a2 { a0 } else { l0 };
                    let crf2 = if a2 { a1 } else { l1 };
                    if rfs == G {
                        3 + (crf1 == G || crf2 == G) as usize
                    } else if rfs == A {
                        (crf1 == G || crf2 == G) as usize
                    } else {
                        1 + 2 * (crf1 == G || crf2 == G) as usize
                    }
                } else if a0 == L && l0 == L {
                    3
                } else if a0 == L || l0 == L {
                    let edge0 = if a0 == L { l0 } else { a0 };
                    4 * (edge0 == G) as usize
                } else {
                    2 * (a0 == G) as usize + 2 * (l0 == G) as usize
                }
            }
        }
        (Some(e), None) | (None, Some(e)) => {
            if !e.is_inter() || (e.ref_frame[0] == L && !e.has_second()) {
                2
            } else if !e.has_second() {
                4 * (e.ref_frame[0] == G) as usize
            } else {
                3 * (e.ref_frame[0] == G || e.ref_frame[1] == G) as usize
            }
        }
        (None, None) => 2,
    }
}
