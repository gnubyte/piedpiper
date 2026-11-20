//! VP8 frame parser: the first partition (frame header, probability updates, macroblock
//! modes and motion vectors) and the token partitions, bit-exact with libvpx's decoder. Every
//! boolean goes through a [`PartIo`] together with its partition, libvpx's probability (of a
//! 0, in 1/256) and a [`Hint`] (context) for PiedPiper's model.

use super::tables::*;
use crate::bits::{Bad, R};
use crate::h264::mb::Hint;

/// Source/sink of boolean decisions. Partition 0 is the first partition (header and modes),
/// 1.. the token partitions.
pub trait PartIo {
    fn read(&mut self, part: usize, prob: u8, h: &Hint) -> R<u32>;
    /// Called once the header has said how many token partitions follow (1, 2, 4 or 8).
    fn token_partitions(&mut self, n: usize) -> R<()>;
}

/// Model symbol classes (40..=63, as for VP9; a file is one or the other).
pub mod kind {
    pub const HEADER: u8 = 40;
    pub const COEF_UPDATE: u8 = 41;
    pub const SEGMENT: u8 = 42;
    pub const SKIP: u8 = 43;
    pub const IS_INTER: u8 = 44;
    pub const REF: u8 = 45;
    pub const YMODE: u8 = 46;
    pub const BMODE: u8 = 47;
    pub const UVMODE: u8 = 48;
    pub const MV_MODE: u8 = 49;
    pub const SPLIT: u8 = 50;
    pub const SUB_MV: u8 = 51;
    pub const MV_SHORT: u8 = 52;
    pub const MV_LONG: u8 = 53;
    pub const MV_SIGN: u8 = 54;
    pub const TOKEN: u8 = 56;
    pub const EXTRA: u8 = 57;
    pub const SIGN: u8 = 58;
    /// Raw frame bytes, header bytes and tail fix-ups (stream.rs).
    pub const HDR: u8 = 59;
    pub const RAW: u8 = 60;
    pub const FIX: u8 = 61;
}

/// Mixer contexts (`Hint::ctx`); all below 460.
mod cx {
    pub const HEADER: usize = 0;
    pub const COEF_UPDATE: usize = 1;
    /// + (node * 4 + type) * 3 + band group
    pub const TOKEN: usize = 2;
    pub const SIGN: usize = 134;
    pub const EXTRA: usize = 138;
    pub const SKIP: usize = 164;
    pub const IS_INTER: usize = 165;
    pub const REF: usize = 166;
    pub const KF_Y: usize = 168;
    pub const Y: usize = 172;
    pub const KF_B: usize = 176;
    pub const B: usize = 185;
    pub const KF_UV: usize = 194;
    pub const UV: usize = 197;
    /// + node * 6 + count
    pub const MV_MODE: usize = 200;
    pub const SPLIT: usize = 224;
    /// + context * 3 + node
    pub const SUB_MV: usize = 227;
    /// + 19 * component + probability index
    pub const MV: usize = 242;
    pub const SEGMENT: usize = 280;
}

pub const B_PRED: u8 = 4;
pub const NEARESTMV: u8 = 5;
pub const NEARMV: u8 = 6;
pub const ZEROMV: u8 = 7;
pub const NEWMV: u8 = 8;
pub const SPLITMV: u8 = 9;

const BANDS: [u8; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];
const ZIGZAG: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const CAT3: [u8; 3] = [173, 148, 140];
const CAT4: [u8; 4] = [176, 155, 140, 135];
const CAT5: [u8; 5] = [180, 157, 141, 134, 130];
const CAT6: [u8; 11] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];
const SUB_MV_REF_PROB3: [[u8; 3]; 8] =
    [[147, 136, 18], [223, 1, 34], [106, 145, 1], [208, 1, 1], [179, 121, 1], [223, 1, 34], [179, 121, 1], [208, 1, 1]];
const MBSPLIT_FILL_COUNT: [usize; 4] = [8, 8, 4, 1];
const MBSPLIT_FILL_OFFSET: [[u8; 16]; 4] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15],
    [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
];
/// Largest frame parsed, in macroblocks (8192 x 8192 pixels).
const MAX_MBS: usize = 1 << 18;

/// Probabilities that persist between frames (libvpx `FRAME_CONTEXT`; the non-key-frame
/// block-mode probabilities never change and are left out).
#[derive(Clone)]
pub struct Probs {
    /// [type][band][context][node]
    pub coef: [u8; 1056],
    pub ymode: [u8; 4],
    pub uvmode: [u8; 3],
    /// [component][19]
    pub mvc: [u8; 38],
}

impl Probs {
    pub fn defaults() -> Self {
        Probs { coef: DEFAULT_COEF_PROBS, ymode: YMODE_PROB, uvmode: UV_MODE_PROB, mvc: DEFAULT_MV_CONTEXT }
    }
}

/// One macroblock's mode info.
#[derive(Clone, Copy, Default)]
pub struct Mb {
    pub mode: u8,
    pub uv_mode: u8,
    /// 0 intra, 1 last, 2 golden, 3 altref
    pub ref_frame: u8,
    /// (row, col) in quarter pixels
    pub mv: [i16; 2],
    pub bmodes: [u8; 16],
    pub bmvs: [[i16; 2]; 16],
    pub skip: bool,
    /// Non-zero coefficients (saturating; model context only).
    pub nzc: u16,
}

impl Mb {
    fn has_y2(&self) -> bool {
        self.mode != B_PRED && self.mode != SPLITMV
    }
}

/// State carried between frames.
#[derive(Default)]
pub struct State {
    pub probs: Option<Probs>,
    pub sign_bias: [bool; 4],
    pub width: u32,
    pub height: u32,
    /// Mode info and per-4x4 ends of block of the last frame of the same size (model
    /// context only).
    pub prev: Vec<Mb>,
    pub prev_eob: Vec<u8>,
}

/// The raw bytes in front of the first partition.
#[derive(Clone, Copy, Debug)]
pub struct Tag {
    pub key: bool,
    pub version: u8,
    pub show: bool,
    pub first_size: usize,
    /// 3, or 10 for key frames (start code and dimensions).
    pub len: usize,
    pub width: u32,
    pub height: u32,
}

pub fn read_tag(f: &[u8]) -> R<Tag> {
    let b = f.get(..3).ok_or(Bad)?;
    let raw = b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16;
    let key = raw & 1 == 0;
    let mut t = Tag { key, version: ((raw >> 1) & 7) as u8, show: (raw >> 4) & 1 == 1, first_size: (raw >> 5) as usize, len: 3, width: 0, height: 0 };
    if key {
        let k = f.get(3..10).ok_or(Bad)?;
        if k[..3] != [0x9d, 0x01, 0x2a] {
            return Err(Bad);
        }
        t.width = (k[3] as u32 | (k[4] as u32) << 8) & 0x3fff;
        t.height = (k[5] as u32 | (k[6] as u32) << 8) & 0x3fff;
        t.len = 10;
        if t.width == 0 || t.height == 0 {
            return Err(Bad);
        }
    }
    if t.first_size == 0 {
        return Err(Bad);
    }
    Ok(t)
}

/// What the parse found out about a frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameInfo {
    pub parts: usize,
    pub q: u32,
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

fn bmode_of(mode: u8) -> u8 {
    match mode {
        1 => 2, // V_PRED -> B_VE_PRED
        2 => 3, // H_PRED -> B_HE_PRED
        3 => 1, // TM_PRED -> B_TM_PRED
        _ => 0,
    }
}

struct Parser<'a, T: PartIo> {
    io: &'a mut T,
    key: bool,
    mb_rows: usize,
    mb_cols: usize,
    /// (mb_rows + 1) x (mb_cols + 1) with a zeroed border row and column, as libvpx keeps it.
    grid: Vec<Mb>,
    colo: Option<&'a [Mb]>,
    colo_eob: Option<&'a [u8]>,
    probs: Probs,
    sign_bias: [bool; 4],
    q: u32,
    fq: u32,
    seg_map: bool,
    seg_probs: [u8; 3],
    skip_enabled: bool,
    prob_skip: u8,
    prob_intra: u8,
    prob_last: u8,
    prob_gf: u8,
    parts: usize,
    /// per macroblock column: y0-3, u0-1, v0-1, y2 nonzero flags, and the same as ends of block
    above_ctx: Vec<[u8; 9]>,
    above_eob: Vec<[u8; 9]>,
    /// coefficient levels (raster order, capped) of the blocks above and to the left
    above_lv: Vec<[[u8; 16]; 9]>,
    left_lv: [[u8; 16]; 9],
    left_ctx: [u8; 9],
    left_eob: [u8; 9],
    eob_map: Vec<u8>,
}

impl<T: PartIo> Parser<'_, T> {
    #[inline]
    fn hdr_bit(&mut self, id: u32, prob: u8) -> R<u32> {
        self.io.read(0, prob, &hint(cx::HEADER, kind::HEADER, id, self.key as u32, 0, 0))
    }

    fn literal(&mut self, id: u32, n: u32) -> R<u32> {
        let mut v = 0;
        for i in 0..n {
            v = (v << 1) | self.io.read(0, 128, &hint(cx::HEADER, kind::HEADER, id | i << 8, self.key as u32, v & 0xFF, 0))?;
        }
        Ok(v)
    }

    fn tree(&mut self, part: usize, t: &[i8], probs: &[u8], h: Hint, stride: usize) -> R<u32> {
        let mut i: i32 = 0;
        loop {
            let node = (i >> 1) as usize;
            let mut hh = h;
            hh.ctx = (h.ctx as usize + node * stride).min(459) as u16;
            hh.a |= (node as u32) << 28;
            let bit = self.io.read(part, *probs.get(node).ok_or(Bad)?, &hh)?;
            i = t[(i + bit as i32) as usize] as i32;
            if i <= 0 {
                return Ok((-i) as u32);
            }
        }
    }

    fn header(&mut self, st: &mut State) -> R<()> {
        if self.key {
            self.hdr_bit(1, 128)?; // colour space
            self.hdr_bit(2, 128)?; // clamping type
        }
        if self.hdr_bit(3, 128)? == 1 {
            // segmentation
            self.seg_map = self.hdr_bit(4, 128)? == 1;
            let update_data = self.hdr_bit(5, 128)? == 1;
            if update_data {
                self.hdr_bit(6, 128)?; // absolute or delta
                for (i, bits) in [7u32, 6].iter().enumerate() {
                    for j in 0..4u32 {
                        if self.hdr_bit(7 | (i as u32) << 4 | j << 5, 128)? == 1 {
                            self.literal(8 | (i as u32) << 4, *bits)?;
                            self.hdr_bit(9, 128)?;
                        }
                    }
                }
            }
            if self.seg_map {
                self.seg_probs = [255; 3];
                for i in 0..3 {
                    if self.hdr_bit(10 | (i as u32) << 4, 128)? == 1 {
                        self.seg_probs[i] = self.literal(11, 8)? as u8;
                    }
                }
            }
        }
        self.hdr_bit(12, 128)?; // filter type
        self.literal(13, 6)?; // filter level
        self.literal(14, 3)?; // sharpness
        if self.hdr_bit(15, 128)? == 1 && self.hdr_bit(16, 128)? == 1 {
            for i in 0..8u32 {
                if self.hdr_bit(17 | i << 5, 128)? == 1 {
                    self.literal(18, 6)?;
                    self.hdr_bit(19, 128)?;
                }
            }
        }
        self.parts = 1 << self.literal(20, 2)?;
        self.io.token_partitions(self.parts)?;
        self.q = self.literal(21, 7)?;
        self.fq = (self.q >> 3) | (self.key as u32) << 4;
        for i in 0..5u32 {
            if self.hdr_bit(22 | i << 5, 128)? == 1 {
                self.literal(23, 4)?;
                self.hdr_bit(24, 128)?;
            }
        }
        if !self.key {
            let golden = self.hdr_bit(25, 128)? == 1;
            let alt = self.hdr_bit(26, 128)? == 1;
            if !golden {
                self.literal(27, 2)?;
            }
            if !alt {
                self.literal(28, 2)?;
            }
            self.sign_bias[2] = self.hdr_bit(29, 128)? == 1;
            self.sign_bias[3] = self.hdr_bit(30, 128)? == 1;
        }
        let refresh_entropy = self.hdr_bit(31, 128)? == 1;
        if !refresh_entropy {
            // libvpx saves the context here and restores it after the frame
            st.probs = Some(self.probs.clone());
        }
        if !self.key {
            self.hdr_bit(32, 128)?; // refresh last
        }
        for i in 0..1056 {
            let up = COEF_UPDATE_PROBS[i];
            let hh = hint(cx::COEF_UPDATE, kind::COEF_UPDATE, (i % 11) as u32 | ((i / 33) as u32 & 31) << 4, up as u32, 0, 0);
            if self.io.read(0, up, &hh)? == 1 {
                self.probs.coef[i] = self.literal(33, 8)? as u8;
            }
        }
        self.skip_enabled = self.hdr_bit(34, 128)? == 1;
        if self.skip_enabled {
            self.prob_skip = self.literal(35, 8)? as u8;
        }
        if !self.key {
            self.prob_intra = self.literal(36, 8)? as u8;
            self.prob_last = self.literal(37, 8)? as u8;
            self.prob_gf = self.literal(38, 8)? as u8;
            if self.hdr_bit(39, 128)? == 1 {
                for i in 0..4 {
                    self.probs.ymode[i] = self.literal(40, 8)? as u8;
                }
            }
            if self.hdr_bit(41, 128)? == 1 {
                for i in 0..3 {
                    self.probs.uvmode[i] = self.literal(42, 8)? as u8;
                }
            }
            for i in 0..38 {
                let up = MV_UPDATE_PROBS[i];
                let hh = hint(cx::HEADER, kind::HEADER, 43 | (i as u32) << 8, up as u32, 0, 0);
                if self.io.read(0, up, &hh)? == 1 {
                    let x = self.literal(44, 7)? as u8;
                    self.probs.mvc[i] = if x != 0 { x << 1 } else { 1 };
                }
            }
        }
        Ok(())
    }

    fn at(&self, r: isize, c: isize) -> &Mb {
        &self.grid[((r + 1) as usize) * (self.mb_cols + 1) + (c + 1) as usize]
    }

    fn colo_mb(&self, r: usize, c: usize) -> Option<Mb> {
        self.colo.map(|g| g[r * self.mb_cols + c])
    }

    fn modes(&mut self) -> R<()> {
        for r in 0..self.mb_rows {
            for c in 0..self.mb_cols {
                let mb = self.mb_modes(r, c)?;
                let i = (r + 1) * (self.mb_cols + 1) + c + 1;
                self.grid[i] = mb;
            }
        }
        Ok(())
    }

    fn mb_modes(&mut self, r: usize, c: usize) -> R<Mb> {
        let mut mb = Mb::default();
        let above = *self.at(r as isize - 1, c as isize);
        let left = *self.at(r as isize, c as isize - 1);
        let colo = self.colo_mb(r, c);
        let fq = self.fq;
        let nb = |m: &Mb| m.mode as u32 | (m.ref_frame as u32) << 4 | (m.skip as u32) << 6;
        let cm = colo.map_or(255, |m| m.mode as u32 | (m.ref_frame as u32) << 4 | (m.skip as u32) << 6 | bucket(m.nzc as u32) << 7);
        if self.seg_map {
            let p = self.seg_probs;
            let h = hint(cx::SEGMENT, kind::SEGMENT, 0, 0, 0, 0);
            if self.io.read(0, p[0], &h)? == 1 {
                self.io.read(0, p[2], &hint(cx::SEGMENT + 2, kind::SEGMENT, 2, 0, 0, 0))?;
            } else {
                self.io.read(0, p[1], &hint(cx::SEGMENT + 1, kind::SEGMENT, 1, 0, 0, 0))?;
            }
        }
        if self.skip_enabled {
            let h = hint(cx::SKIP, kind::SKIP, above.skip as u32 | (left.skip as u32) << 1, nb(&above) | nb(&left) << 8, fq, cm);
            mb.skip = self.io.read(0, self.prob_skip, &h)? == 1;
        }
        let ctxbits = mb.skip as u32;
        if self.key {
            let h = hint(cx::KF_Y, kind::YMODE, ctxbits | bmode_of(above.mode) as u32 | (bmode_of(left.mode) as u32) << 4, nb(&above) | nb(&left) << 8, fq, cm);
            mb.mode = match self.tree(0, &KF_YMODE_TREE, &KF_YMODE_PROB, h, 1)? {
                0 => 0,
                v => v as u8,
            };
            if mb.mode == B_PRED {
                for i in 0..16 {
                    let a = if i < 4 {
                        if above.mode == B_PRED {
                            above.bmodes[i + 12]
                        } else {
                            bmode_of(above.mode)
                        }
                    } else {
                        mb.bmodes[i - 4]
                    };
                    let l = if i & 3 == 0 {
                        if left.mode == B_PRED {
                            left.bmodes[i + 3]
                        } else {
                            bmode_of(left.mode)
                        }
                    } else {
                        mb.bmodes[i - 1]
                    };
                    let o = (a as usize * 10 + l as usize) * 9;
                    let h = hint(cx::KF_B, kind::BMODE, a as u32 | (l as u32) << 4, i as u32, fq, 1);
                    mb.bmodes[i] = self.tree(0, &BMODE_TREE, &KF_BMODE_PROB[o..o + 9], h, 1)? as u8;
                }
            }
            let h = hint(cx::KF_UV, kind::UVMODE, mb.mode as u32, 1, fq, 0);
            mb.uv_mode = self.tree(0, &UV_MODE_TREE, &KF_UV_MODE_PROB, h, 1)? as u8;
            return Ok(mb);
        }
        let h = hint(cx::IS_INTER, kind::IS_INTER, (above.ref_frame > 0) as u32 | ((left.ref_frame > 0) as u32) << 1 | ctxbits << 2, nb(&above) | nb(&left) << 8, fq, cm);
        mb.ref_frame = self.io.read(0, self.prob_intra, &h)? as u8;
        if mb.ref_frame == 0 {
            let probs = self.probs.ymode;
            let h = hint(cx::Y, kind::YMODE, ctxbits, nb(&above) | nb(&left) << 8, fq, cm);
            mb.mode = self.tree(0, &YMODE_TREE, &probs, h, 1)? as u8;
            if mb.mode == B_PRED {
                for i in 0..16 {
                    // the above and left sub-block modes (libvpx uses them only in key frames)
                    let a = if i < 4 {
                        if above.mode == B_PRED {
                            above.bmodes[i + 12]
                        } else {
                            bmode_of(above.mode)
                        }
                    } else {
                        mb.bmodes[i - 4]
                    };
                    let l = if i & 3 == 0 {
                        if left.mode == B_PRED {
                            left.bmodes[i + 3]
                        } else {
                            bmode_of(left.mode)
                        }
                    } else {
                        mb.bmodes[i - 1]
                    };
                    let h = hint(cx::B, kind::BMODE, a as u32 | (l as u32) << 4, i as u32, fq, (above.ref_frame > 0) as u32 | ((left.ref_frame > 0) as u32) << 1);
                    mb.bmodes[i] = self.tree(0, &BMODE_TREE, &BMODE_PROB, h, 1)? as u8;
                }
            }
            let probs = self.probs.uvmode;
            let h = hint(cx::UV, kind::UVMODE, mb.mode as u32, 0, fq, 0);
            mb.uv_mode = self.tree(0, &UV_MODE_TREE, &probs, h, 1)? as u8;
            return Ok(mb);
        }
        let rh = |k: u32| hint(cx::REF + k as usize, kind::REF, k | (above.ref_frame as u32) << 2 | (left.ref_frame as u32) << 4, fq, cm, 0);
        if self.io.read(0, self.prob_last, &rh(0))? == 1 {
            mb.ref_frame = 2 + self.io.read(0, self.prob_gf, &rh(1))? as u8;
        }
        self.inter_modes(&mut mb, r, c, &above, &left, colo.as_ref())?;
        Ok(mb)
    }

    #[allow(clippy::too_many_arguments)]
    fn inter_modes(&mut self, mb: &mut Mb, r: usize, c: usize, above: &Mb, left: &Mb, colo: Option<&Mb>) -> R<()> {
        let aboveleft = *self.at(r as isize - 1, c as isize - 1);
        let sb = self.sign_bias;
        let bias = |m: &Mb, mut mv: [i16; 2]| {
            if sb[m.ref_frame as usize] != sb[mb.ref_frame as usize] {
                mv = [mv[0].wrapping_neg(), mv[1].wrapping_neg()];
            }
            mv
        };
        let mut near = [[0i16; 2]; 4];
        let mut cnt = [0u32; 4];
        let mut n = 0usize; // index of the last filled slot
        if above.ref_frame != 0 {
            if above.mv != [0, 0] {
                n += 1;
                near[n] = bias(above, above.mv);
            }
            cnt[n] += 2;
        }
        if left.ref_frame != 0 {
            if left.mv != [0, 0] {
                let mv = bias(left, left.mv);
                if mv != near[n] {
                    n += 1;
                    near[n] = mv;
                }
                cnt[n] += 2;
            } else {
                cnt[0] += 2;
            }
        }
        if aboveleft.ref_frame != 0 {
            if aboveleft.mv != [0, 0] {
                let mv = bias(&aboveleft, aboveleft.mv);
                if mv != near[n] {
                    n += 1;
                    near[n] = mv;
                }
                cnt[n] += 1;
            } else {
                cnt[0] += 1;
            }
        }
        let mi = |row: usize, col: usize, rows: usize, cols: usize| {
            (
                -((col as i32 * 16) << 3),
                ((cols as i32 - 1 - col as i32) * 16) << 3,
                -((row as i32 * 16) << 3),
                ((rows as i32 - 1 - row as i32) * 16) << 3,
            )
        };
        let (le, re, te, be) = mi(r, c, self.mb_rows, self.mb_cols);
        let clamp = |mv: [i16; 2]| -> [i16; 2] {
            let col = (mv[1] as i32).clamp(le - 128, re + 128);
            let row = (mv[0] as i32).clamp(te - 128, be + 128);
            [row as i16, col as i16]
        };
        let fq = self.fq;
        let cm = colo.map_or(255, |m| m.mode as u32 | (m.ref_frame as u32) << 4 | (m.skip as u32) << 6);
        let nbm = above.mode as u32 | (left.mode as u32) << 4 | (aboveleft.mode as u32) << 8;
        let mh = |node: usize, count: u32| {
            hint(cx::MV_MODE + node * 6 + count.min(5) as usize, kind::MV_MODE, node as u32 | count << 4 | (mb.ref_frame as u32) << 8, nbm, cm, fq | (n as u32) << 6)
        };
        if self.io.read(0, MODE_CONTEXTS[cnt[0].min(5) as usize * 4], &mh(0, cnt[0]))? == 0 {
            mb.mode = ZEROMV;
            mb.mv = [0, 0];
            return Ok(());
        }
        cnt[1] += ((cnt[3] > 0) & (near[n] == near[1])) as u32;
        if cnt[2] > cnt[1] {
            cnt.swap(1, 2);
            near.swap(1, 2);
        }
        if self.io.read(0, MODE_CONTEXTS[cnt[1].min(5) as usize * 4 + 1], &mh(1, cnt[1]))? == 0 {
            mb.mode = NEARESTMV;
            mb.mv = clamp(near[1]);
            return Ok(());
        }
        if self.io.read(0, MODE_CONTEXTS[cnt[2].min(5) as usize * 4 + 2], &mh(2, cnt[2]))? == 0 {
            mb.mode = NEARMV;
            mb.mv = clamp(near[2]);
            return Ok(());
        }
        let near_index = (cnt[1] >= cnt[0]) as usize;
        let best = clamp(near[near_index]);
        let split_cnt = ((above.mode == SPLITMV) as u32 + (left.mode == SPLITMV) as u32) * 2 + (aboveleft.mode == SPLITMV) as u32;
        if self.io.read(0, MODE_CONTEXTS[split_cnt.min(5) as usize * 4 + 3], &mh(3, split_cnt))? == 1 {
            self.split_mv(mb, above, left, best, colo)?;
            mb.mv = mb.bmvs[15];
            mb.mode = SPLITMV;
        } else {
            let t = colo.filter(|m| m.ref_frame == mb.ref_frame).map(|m| [m.mv[0] as i32 - best[0] as i32, m.mv[1] as i32 - best[1] as i32]);
            let dr = self.mv_component(0, t.map(|t| t[0]))? * 2;
            let dc = self.mv_component(1, t.map(|t| t[1]))? * 2;
            mb.mv = [(best[0] as i32 + dr) as i16, (best[1] as i32 + dc) as i16];
            mb.mode = NEWMV;
        }
        Ok(())
    }

    fn split_mv(&mut self, mb: &mut Mb, above: &Mb, left: &Mb, best: [i16; 2], colo: Option<&Mb>) -> R<()> {
        let fq = self.fq;
        let sh = |node: usize| hint(cx::SPLIT + node, kind::SPLIT, node as u32 | ((above.mode == SPLITMV) as u32) << 4 | ((left.mode == SPLITMV) as u32) << 5, fq, 0, 0);
        let (s, num) = if self.io.read(0, 110, &sh(0))? == 1 {
            if self.io.read(0, 111, &sh(1))? == 1 {
                (self.io.read(0, 150, &sh(2))? as usize, 2)
            } else {
                (2, 4)
            }
        } else {
            (3, 16)
        };
        let t = colo.filter(|m| m.ref_frame == mb.ref_frame).map(|m| [m.mv[0] as i32 - best[0] as i32, m.mv[1] as i32 - best[1] as i32]);
        for j in 0..num {
            let k = MBSPLIT_OFFSET[s * 16 + j] as usize;
            let leftmv = if k & 3 == 0 {
                if left.mode != SPLITMV {
                    left.mv
                } else {
                    left.bmvs[k + 3]
                }
            } else {
                mb.bmvs[k - 1]
            };
            let abovemv = if k >> 2 == 0 {
                if above.mode != SPLITMV {
                    above.mv
                } else {
                    above.bmvs[k + 12]
                }
            } else {
                mb.bmvs[k - 4]
            };
            let lez = leftmv == [0, 0];
            let aez = abovemv == [0, 0];
            let lea = leftmv == abovemv;
            let ctx = (aez as usize) << 2 | (lez as usize) << 1 | lea as usize;
            let prob = SUB_MV_REF_PROB3[ctx];
            let h = |node: usize| hint(cx::SUB_MV + ctx.min(4) * 3 + node, kind::SUB_MV, node as u32 | (ctx as u32) << 4 | (s as u32) << 8, j as u32, fq, 0);
            let blockmv = if self.io.read(0, prob[0], &h(0))? == 1 {
                if self.io.read(0, prob[1], &h(1))? == 1 {
                    if self.io.read(0, prob[2], &h(2))? == 1 {
                        let dr = self.mv_component(0, t.map(|t| t[0]))? * 2;
                        let dc = self.mv_component(1, t.map(|t| t[1]))? * 2;
                        [(best[0] as i32 + dr) as i16, (best[1] as i32 + dc) as i16]
                    } else {
                        [0, 0]
                    }
                } else {
                    abovemv
                }
            } else {
                leftmv
            };
            let fc = MBSPLIT_FILL_COUNT[s];
            for &o in &MBSPLIT_FILL_OFFSET[s][j * fc..j * fc + fc] {
                mb.bmvs[o as usize] = blockmv;
            }
        }
        Ok(())
    }

    /// libvpx `read_mvcomponent`; `pd` is what the co-located vector predicts.
    fn mv_component(&mut self, comp: usize, pd: Option<i32>) -> R<i32> {
        let p: [u8; 19] = self.probs.mvc[comp * 19..comp * 19 + 19].try_into().unwrap();
        let base = cx::MV + comp * 19;
        let fq = self.fq;
        // predicted magnitude (in the units read here) and sign
        let (pm, ps) = pd.map_or((1023u32, 0u32), |d| ((d / 2).unsigned_abs().min(1023), 1 + (d < 0) as u32));
        let pc = if pm < 8 { pm } else { 8 + bucket(pm >> 3) };
        let mut x: i32;
        if self.io.read(0, p[0], &hint(base, kind::MV_LONG, comp as u32, pc | ps << 5, fq, 0))? == 1 {
            x = 0;
            for i in 0..3 {
                let h = hint(base + 9 + i, kind::MV_LONG, comp as u32 | (i as u32) << 1, x as u32, pc, (pm >> i) & 1);
                x += (self.io.read(0, p[9 + i], &h)? as i32) << i;
            }
            let mut i = 9;
            while i > 3 {
                let h = hint(base + 9 + i, kind::MV_LONG, comp as u32 | (i as u32) << 1, (x >> 4) as u32, pc, (pm >> i) & 1);
                x += (self.io.read(0, p[9 + i], &h)? as i32) << i;
                i -= 1;
            }
            if x & 0xFFF0 == 0 || self.io.read(0, p[12], &hint(base + 12, kind::MV_LONG, comp as u32 | 3 << 1, (x >> 4) as u32, pc, (pm >> 3) & 1))? == 1 {
                x += 8;
            }
        } else {
            let h = hint(base + 2, kind::MV_SHORT, comp as u32, pc | ps << 5, fq, 0);
            x = self.tree(0, &SMALL_MVTREE, &p[2..9], h, 1)? as i32;
        }
        if x != 0 && self.io.read(0, p[1], &hint(base + 1, kind::MV_SIGN, comp as u32, ps | (x.min(15) as u32) << 2, pc, fq))? == 1 {
            x = -x;
        }
        Ok(x)
    }

    fn tokens(&mut self) -> R<()> {
        for r in 0..self.mb_rows {
            let part = 1 + r % self.parts;
            self.left_ctx = [0; 9];
            self.left_eob = [0; 9];
            self.left_lv = [[0; 16]; 9];
            for c in 0..self.mb_cols {
                let mut mb = *self.at(r as isize, c as isize);
                let colo = self.colo_mb(r, c);
                if mb.skip {
                    let n = if mb.has_y2() { 9 } else { 8 };
                    self.above_ctx[c][..n].fill(0);
                    self.above_eob[c][..n].fill(0);
                    self.above_lv[c][..n].fill([0; 16]);
                    self.left_lv[..n].fill([0; 16]);
                    self.left_ctx[..n].fill(0);
                    self.left_eob[..n].fill(0);
                    mb.nzc = 0;
                } else {
                    mb.nzc = self.mb_tokens(part, r, c, &mb, colo.as_ref())?.min(65535) as u16;
                }
                let i = (r + 1) * (self.mb_cols + 1) + c + 1;
                self.grid[i].nzc = mb.nzc;
            }
        }
        Ok(())
    }

    fn mb_tokens(&mut self, part: usize, r: usize, c: usize, mb: &Mb, colo: Option<&Mb>) -> R<u32> {
        let mut total = 0u32;
        let blk = mb.mode as u32 | (mb.ref_frame as u32) << 4;
        let cn = colo.map_or(8, |m| bucket(m.nzc as u32));
        let first = if mb.has_y2() {
            let ctx = self.above_ctx[c][8] + self.left_ctx[8];
            let ne = bucket(self.above_eob[c][8] as u32) | bucket(self.left_eob[8] as u32) << 3;
            let (alv, llv) = (self.above_lv[c][8], self.left_lv[8]);
            let (eob, nz, lv) = self.coeffs(part, 1, ctx as usize, 0, blk, ne, cn, 24, &alv, &llv)?;
            self.above_lv[c][8] = lv;
            self.left_lv[8] = lv;
            self.above_ctx[c][8] = (eob > 0) as u8;
            self.left_ctx[8] = (eob > 0) as u8;
            self.above_eob[c][8] = eob as u8;
            self.left_eob[8] = eob as u8;
            total += nz;
            1
        } else {
            0
        };
        let ytype = if first == 1 { 0 } else { 3 };
        let pw = self.mb_cols * 4;
        for i in 0..16 {
            let (ax, ly) = (i & 3, i >> 2);
            let ctx = self.above_ctx[c][ax] + self.left_ctx[ly];
            let ne = bucket(self.above_eob[c][ax] as u32) | bucket(self.left_eob[ly] as u32) << 3;
            let pos4 = (r * 4 + ly) * pw + c * 4 + ax;
            let pe = self.colo_eob.map_or(8, |e| bucket(e[pos4] as u32));
            let (alv, llv) = (self.above_lv[c][ax], self.left_lv[ly]);
            let (eob, nz, lv) = self.coeffs(part, ytype, ctx as usize, first, blk, ne, pe, i as u32, &alv, &llv)?;
            self.above_lv[c][ax] = lv;
            self.left_lv[ly] = lv;
            let nzflag = (eob > 0) as u8;
            self.above_ctx[c][ax] = nzflag;
            self.left_ctx[ly] = nzflag;
            self.above_eob[c][ax] = eob as u8;
            self.left_eob[ly] = eob as u8;
            self.eob_map[pos4] = eob as u8;
            total += nz;
        }
        for i in 16..24 {
            let v = (i > 19) as usize;
            let ax = 4 + (v << 1) + (i & 1);
            let ly = 4 + (v << 1) + ((i & 3) > 1) as usize;
            let ctx = self.above_ctx[c][ax] + self.left_ctx[ly];
            let ne = bucket(self.above_eob[c][ax] as u32) | bucket(self.left_eob[ly] as u32) << 3;
            let (alv, llv) = (self.above_lv[c][ax], self.left_lv[ly]);
            let (eob, nz, lv) = self.coeffs(part, 2, ctx as usize, 0, blk, ne, cn, i as u32, &alv, &llv)?;
            self.above_lv[c][ax] = lv;
            self.left_lv[ly] = lv;
            let nzflag = (eob > 0) as u8;
            self.above_ctx[c][ax] = nzflag;
            self.left_ctx[ly] = nzflag;
            self.above_eob[c][ax] = eob as u8;
            self.left_eob[ly] = eob as u8;
            total += nz;
        }
        Ok(total)
    }

    /// libvpx `GetCoeffs`; returns (end of block, non-zero coefficients).
    #[allow(clippy::too_many_arguments)]
    fn coeffs(
        &mut self,
        part: usize,
        typ: usize,
        ctx0: usize,
        first: usize,
        blk: u32,
        ne: u32,
        pe: u32,
        bi: u32,
        alv: &[u8; 16],
        llv: &[u8; 16],
    ) -> R<(usize, u32, [u8; 16])> {
        let base = typ * 8 * 3 * 11;
        let probs_at = |band: usize, ctx: usize| base + (band * 3 + ctx) * 11;
        let mut levels = [0u8; 16];
        let fq = self.fq;
        let mut nz = 0u32;
        let mut prev = ctx0 as u32;
        let mk = |node: usize, pos: usize, ctx: usize, nz: u32, prev: u32, levels: &[u8; 16]| -> Hint {
            let band = BANDS[pos] as usize;
            let grp = if band == 0 {
                0
            } else if band < 4 {
                1
            } else {
                2
            };
            // already decoded neighbours of this coefficient inside the block (raster order)
            let rp = ZIGZAG[pos.min(15)] as usize;
            let up = if rp >= 4 { levels[rp - 4] as u32 & 127 } else { 0 };
            let lf = if rp & 3 != 0 { levels[rp - 1] as u32 & 127 } else { 0 };
            // the same coefficient in the blocks above and to the left
            let (al, ll) = (alv[rp] as u32 & 127, llv[rp] as u32 & 127);
            let nd = node as u32;
            let t = typ as u32;
            hint(
                cx::TOKEN + (node * 4 + typ) * 3 + grp,
                kind::TOKEN,
                nd | t << 4 | (ctx as u32) << 6 | (pos as u32) << 8,
                nd | t << 4 | nz.min(15) << 6 | ne << 10 | (pos.min(3) as u32) << 16 | al.min(3) << 18 | ll.min(3) << 20,
                nd | t << 4 | up.min(3) << 6 | lf.min(3) << 8 | prev.min(3) << 10 | (pos as u32) << 12,
                nd | t << 4 | fq << 6 | blk << 12 | pe << 18 | bi.min(24) << 22,
            )
        };
        let mut n = first;
        let mut p = probs_at(BANDS[n] as usize, ctx0);
        let mut ctx = ctx0;
        if self.io.read(part, self.probs.coef[p], &mk(0, n, ctx, nz, prev, &levels))? == 0 {
            return Ok((0, 0, levels));
        }
        loop {
            n += 1;
            let pos = n - 1;
            if self.io.read(part, self.probs.coef[p + 1], &mk(1, pos, ctx, nz, prev, &levels))? == 0 {
                p = probs_at(BANDS[n] as usize, 0);
                ctx = 0;
                prev = 0;
            } else {
                let coef = self.probs.coef;
                let rd = |io: &mut T, node: usize, levels: &[u8; 16]| io.read(part, coef[p + node], &mk(node, pos, ctx, nz, prev, levels));
                let v: u32;
                if rd(self.io, 2, &levels)? == 0 {
                    v = 1;
                } else if rd(self.io, 3, &levels)? == 0 {
                    v = if rd(self.io, 4, &levels)? == 0 { 2 } else { 3 + rd(self.io, 5, &levels)? };
                } else if rd(self.io, 6, &levels)? == 0 {
                    if rd(self.io, 7, &levels)? == 0 {
                        v = 5 + self.extra(part, 1, &[159], typ)?;
                    } else {
                        v = 7 + self.extra(part, 2, &[165, 145], typ)?;
                    }
                } else {
                    let bit1 = rd(self.io, 8, &levels)?;
                    let bit0 = rd(self.io, 9 + bit1 as usize, &levels)?;
                    let cat = 2 * bit1 + bit0;
                    let tab: &[u8] = match cat {
                        0 => &CAT3,
                        1 => &CAT4,
                        2 => &CAT5,
                        _ => &CAT6,
                    };
                    v = 3 + (8 << cat) + self.extra(part, 3 + cat as usize, tab, typ)?;
                }
                let class = if v == 1 { 1 } else { 2 };
                nz += 1;
                // signs of the same coefficient in the neighbouring blocks: 0 none, 1 +, 2 -
                let rp = ZIGZAG[pos] as usize;
                let sg = |x: u8| if x & 127 == 0 { 0 } else { 1 + (x >> 7) as u32 };
                let sh = hint(
                    cx::SIGN + typ,
                    kind::SIGN,
                    (pos as u32) | (v.min(15)) << 4 | (typ as u32) << 8,
                    sg(alv[rp]) | sg(llv[rp]) << 2 | (pos as u32) << 4,
                    0,
                    0,
                );
                let neg = self.io.read(part, 128, &sh)?;
                levels[rp] = v.min(127) as u8 | (neg as u8) << 7;
                p = probs_at(BANDS[n] as usize, class);
                ctx = class;
                prev = v.min(3);
                if n == 16 || self.io.read(part, self.probs.coef[p], &mk(0, n.min(15), ctx, nz, prev, &levels))? == 0 {
                    return Ok((n, nz, levels));
                }
            }
            if n == 16 {
                return Ok((16, nz, levels));
            }
        }
    }

    fn extra(&mut self, part: usize, cat: usize, probs: &[u8], typ: usize) -> R<u32> {
        const OFF: [usize; 7] = [0, 0, 1, 3, 6, 10, 15];
        let mut v = 0u32;
        for (i, &p) in probs.iter().enumerate() {
            let h = hint(cx::EXTRA + OFF[cat] + i, kind::EXTRA, cat as u32 | (i as u32) << 3 | (typ as u32) << 7, v.min(255), 0, 0);
            v = (v << 1) | self.io.read(part, p, &h)?;
        }
        Ok(v)
    }
}

/// Parses one frame (after its tag) and updates the state carried to later frames.
pub fn parse_frame<T: PartIo>(st: &mut State, tag: &Tag, io: &mut T) -> R<FrameInfo> {
    if tag.key {
        st.probs = Some(Probs::defaults());
        st.sign_bias = [false; 4];
        st.width = tag.width;
        st.height = tag.height;
    }
    let mut probs = st.probs.clone().ok_or(Bad)?;
    if !tag.key && st.width == 0 {
        return Err(Bad);
    }
    let mb_cols = (st.width as usize).div_ceil(16);
    let mb_rows = (st.height as usize).div_ceil(16);
    if mb_cols * mb_rows > MAX_MBS {
        return Err(Bad);
    }
    std::mem::swap(&mut probs, st.probs.as_mut().ok_or(Bad)?);
    let prev = std::mem::take(&mut st.prev);
    let prev_eob = std::mem::take(&mut st.prev_eob);
    let n = mb_rows * mb_cols;
    let colo = (prev.len() == n && prev_eob.len() == n * 16).then_some(());
    let saved;
    let result = {
        let mut p = Parser {
            io,
            key: tag.key,
            mb_rows,
            mb_cols,
            grid: vec![Mb::default(); (mb_rows + 1) * (mb_cols + 1)],
            colo: colo.map(|_| &prev[..]),
            colo_eob: colo.map(|_| &prev_eob[..]),
            probs: st.probs.clone().ok_or(Bad)?,
            sign_bias: st.sign_bias,
            q: 0,
            fq: 0,
            seg_map: false,
            seg_probs: [255; 3],
            skip_enabled: false,
            prob_skip: 0,
            prob_intra: 0,
            prob_last: 0,
            prob_gf: 0,
            parts: 1,
            above_ctx: vec![[0; 9]; mb_cols],
            above_eob: vec![[0; 9]; mb_cols],
            above_lv: vec![[[0; 16]; 9]; mb_cols],
            left_lv: [[0; 16]; 9],
            left_ctx: [0; 9],
            left_eob: [0; 9],
            eob_map: vec![0; n * 16],
        };
        // `header` stores the context to restore (refresh_entropy_probs == 0) in st.probs;
        // keep the pre-frame copy apart so a failed parse leaves a deterministic state.
        let mut tmp = State { probs: None, ..Default::default() };
        let r = p.header(&mut tmp).and_then(|_| p.modes()).and_then(|_| p.tokens());
        saved = tmp.probs;
        r.map(|_| {
            let mut grid = Vec::with_capacity(n);
            for row in 0..mb_rows {
                grid.extend_from_slice(&p.grid[(row + 1) * (mb_cols + 1) + 1..(row + 2) * (mb_cols + 1)]);
            }
            (FrameInfo { parts: p.parts, q: p.q }, p.probs, p.sign_bias, grid, p.eob_map)
        })
    };
    let (info, new_probs, sign_bias, grid, eob_map) = result?;
    st.probs = Some(saved.unwrap_or(new_probs));
    st.sign_bias = sign_bias;
    st.prev = grid;
    st.prev_eob = eob_map;
    Ok(info)
}
