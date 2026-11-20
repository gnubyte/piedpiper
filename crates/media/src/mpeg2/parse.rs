//! MPEG-2 video headers (ISO/IEC 13818-2 clause 6) and a slice parser that reads every bit
//! through a [`BitIo`], together with a [`Hint`] saying where the bit sits: which syntax
//! element, which node of its code tree, and the surrounding state the model can use.
//! Code tables are annex B of Rec. ITU-T H.262 (tables B.1 to B.4 and B.9 to B.15, checked code by code).

use crate::bits::{Bad, BitReader, R};
use crate::h264::mb::Hint;

/// Source/sink of slice bits.
pub trait BitIo {
    fn bit(&mut self, h: &Hint) -> R<u32>;
    /// Whether another macroblock follows. The bitstream says so by the next 23 bits not
    /// being zero; the recoder codes it as a decision of its own.
    fn more(&mut self, h: &Hint) -> R<bool>;
}

/// Model symbol classes (40..=63).
pub mod kind {
    pub const SLICE_HDR: u8 = 40;
    pub const MBA: u8 = 41;
    pub const MB_TYPE: u8 = 42;
    pub const MODES: u8 = 43;
    pub const QUANT: u8 = 44;
    pub const MOTION_CODE: u8 = 45;
    pub const MOTION_SIGN: u8 = 46;
    pub const RESIDUAL: u8 = 47;
    pub const DMV: u8 = 48;
    pub const CBP: u8 = 49;
    pub const DC_SIZE: u8 = 50;
    pub const DC_DIFF: u8 = 51;
    pub const COEF: u8 = 52;
    pub const COEF_SIGN: u8 = 53;
    pub const ESCAPE: u8 = 54;
    pub const MORE: u8 = 55;
    /// Stream-level symbols (stream.rs).
    pub const TAIL: u8 = 58;
    pub const HDR: u8 = 59;
    pub const RAW: u8 = 60;
    pub const START: u8 = 61;
}

/// Mixer contexts (`Hint::ctx`, all below 460; the model is given a neutral probability so
/// that it selects mixers by these).
mod cx {
    pub const SLICE_HDR: usize = 0;
    pub const MBA: usize = 16;
    /// + picture type * 8 + depth
    pub const MB_TYPE: usize = 30;
    pub const MODES: usize = 62;
    pub const QUANT: usize = 70;
    pub const MOTION_CODE: usize = 76;
    pub const MOTION_SIGN: usize = 90;
    pub const RESIDUAL: usize = 92;
    pub const DMV: usize = 102;
    pub const CBP: usize = 106;
    pub const DC_SIZE: usize = 120;
    pub const DC_DIFF: usize = 144;
    /// + table * 24 + depth
    pub const COEF: usize = 170;
    pub const COEF_SIGN: usize = 270;
    pub const ESCAPE: usize = 280;
    pub const MORE: usize = 300;
}

const NONE: i32 = i32::MIN;
const ESC: i32 = 111;
const EOB: i32 = 112;
const MBA_ESC: i32 = 34;
const MBA_STUFF: i32 = 35;

/// A prefix code as a binary trie: children are node indices (> 0), leaves are -(value + 1).
pub struct Vlc {
    nodes: Vec<[i32; 2]>,
}

impl Vlc {
    fn new(codes: &[(u32, u8, i32)]) -> Self {
        let mut nodes = vec![[NONE, NONE]];
        for &(code, len, val) in codes {
            let mut n = 0usize;
            for i in (0..len).rev() {
                let b = ((code >> i) & 1) as usize;
                if i == 0 {
                    nodes[n][b] = -(val + 1);
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

/// macroblock_address_increment codes for increments 1 to 33 and macroblock_escape (table B.1),
/// then macroblock_stuffing, which exists only in MPEG-1 (ISO/IEC 11172-2): the note under H.262
/// table B.1 says MPEG-2 does not have it, and the parser rejects a slice that uses it.
const MBA_TABLE: [(u8, u8); 35] = [
    (0x1, 1), (0x3, 3), (0x2, 3), (0x3, 4), (0x2, 4), (0x3, 5), (0x2, 5), (0x7, 7), (0x6, 7), (0xb, 8), (0xa, 8), (0x9, 8),
    (0x8, 8), (0x7, 8), (0x6, 8), (0x17, 10), (0x16, 10), (0x15, 10), (0x14, 10), (0x13, 10), (0x12, 10), (0x23, 11),
    (0x22, 11), (0x21, 11), (0x20, 11), (0x1f, 11), (0x1e, 11), (0x1d, 11), (0x1c, 11), (0x1b, 11), (0x1a, 11), (0x19, 11),
    (0x18, 11), (0x8, 11), (0xf, 11),
];
const PAT_TABLE: [(u8, u8); 64] = [
    (0x1, 9), (0xb, 5), (0x9, 5), (0xd, 6), (0xd, 4), (0x17, 7), (0x13, 7), (0x1f, 8), (0xc, 4), (0x16, 7), (0x12, 7), (0x1e, 8),
    (0x13, 5), (0x1b, 8), (0x17, 8), (0x13, 8), (0xb, 4), (0x15, 7), (0x11, 7), (0x1d, 8), (0x11, 5), (0x19, 8), (0x15, 8),
    (0x11, 8), (0xf, 6), (0xf, 8), (0xd, 8), (0x3, 9), (0xf, 5), (0xb, 8), (0x7, 8), (0x7, 9), (0xa, 4), (0x14, 7), (0x10, 7),
    (0x1c, 8), (0xe, 6), (0xe, 8), (0xc, 8), (0x2, 9), (0x10, 5), (0x18, 8), (0x14, 8), (0x10, 8), (0xe, 5), (0xa, 8), (0x6, 8),
    (0x6, 9), (0x12, 5), (0x1a, 8), (0x16, 8), (0x12, 8), (0xd, 5), (0x9, 8), (0x5, 8), (0x5, 9), (0xc, 5), (0x8, 8), (0x4, 8),
    (0x4, 9), (0x7, 3), (0xa, 5), (0x8, 5), (0xc, 6),
];
const MV_TABLE: [(u8, u8); 17] = [
    (0x1, 1), (0x1, 2), (0x1, 3), (0x1, 4), (0x3, 6), (0x5, 7), (0x4, 7), (0x3, 7), (0xb, 9), (0xa, 9), (0x9, 9), (0x11, 10),
    (0x10, 10), (0xf, 10), (0xe, 10), (0xd, 10), (0xc, 10),
];
const DC_LUM: [(u16, u8); 12] =
    [(0x4, 3), (0x0, 2), (0x1, 2), (0x5, 3), (0x6, 3), (0xe, 4), (0x1e, 5), (0x3e, 6), (0x7e, 7), (0xfe, 8), (0x1fe, 9), (0x1ff, 9)];
const DC_CHROMA: [(u16, u8); 12] = [
    (0x0, 2), (0x1, 2), (0x2, 2), (0x6, 3), (0xe, 4), (0x1e, 5), (0x3e, 6), (0x7e, 7), (0xfe, 8), (0x1fe, 9), (0x3fe, 10), (0x3ff, 10),
];
/// Table B.14 (with the escape and end-of-block codes last).
const B14: [(u16, u8); 113] = [
    (0x3, 2), (0x4, 4), (0x5, 5), (0x6, 7), (0x26, 8), (0x21, 8), (0xa, 10), (0x1d, 12), (0x18, 12), (0x13, 12), (0x10, 12),
    (0x1a, 13), (0x19, 13), (0x18, 13), (0x17, 13), (0x1f, 14), (0x1e, 14), (0x1d, 14), (0x1c, 14), (0x1b, 14), (0x1a, 14),
    (0x19, 14), (0x18, 14), (0x17, 14), (0x16, 14), (0x15, 14), (0x14, 14), (0x13, 14), (0x12, 14), (0x11, 14), (0x10, 14),
    (0x18, 15), (0x17, 15), (0x16, 15), (0x15, 15), (0x14, 15), (0x13, 15), (0x12, 15), (0x11, 15), (0x10, 15), (0x3, 3),
    (0x6, 6), (0x25, 8), (0xc, 10), (0x1b, 12), (0x16, 13), (0x15, 13), (0x1f, 15), (0x1e, 15), (0x1d, 15), (0x1c, 15),
    (0x1b, 15), (0x1a, 15), (0x19, 15), (0x13, 16), (0x12, 16), (0x11, 16), (0x10, 16), (0x5, 4), (0x4, 7), (0xb, 10),
    (0x14, 12), (0x14, 13), (0x7, 5), (0x24, 8), (0x1c, 12), (0x13, 13), (0x6, 5), (0xf, 10), (0x12, 12), (0x7, 6), (0x9, 10),
    (0x12, 13), (0x5, 6), (0x1e, 12), (0x14, 16), (0x4, 6), (0x15, 12), (0x7, 7), (0x11, 12), (0x5, 7), (0x11, 13), (0x27, 8),
    (0x10, 13), (0x23, 8), (0x1a, 16), (0x22, 8), (0x19, 16), (0x20, 8), (0x18, 16), (0xe, 10), (0x17, 16), (0xd, 10),
    (0x16, 16), (0x8, 10), (0x15, 16), (0x1f, 12), (0x1a, 12), (0x19, 12), (0x17, 12), (0x16, 12), (0x1f, 13), (0x1e, 13),
    (0x1d, 13), (0x1c, 13), (0x1b, 13), (0x1f, 16), (0x1e, 16), (0x1d, 16), (0x1c, 16), (0x1b, 16), (0x1, 6), (0x2, 2),
];
/// Table B.15 (intra blocks with intra_vlc_format = 1).
const B15: [(u16, u8); 113] = [
    (0x02, 2), (0x06, 3), (0x07, 4), (0x1c, 5), (0x1d, 5), (0x05, 6), (0x04, 6), (0x7b, 7), (0x7c, 7), (0x23, 8), (0x22, 8),
    (0xfa, 8), (0xfb, 8), (0xfe, 8), (0xff, 8), (0x1f, 14), (0x1e, 14), (0x1d, 14), (0x1c, 14), (0x1b, 14), (0x1a, 14),
    (0x19, 14), (0x18, 14), (0x17, 14), (0x16, 14), (0x15, 14), (0x14, 14), (0x13, 14), (0x12, 14), (0x11, 14), (0x10, 14),
    (0x18, 15), (0x17, 15), (0x16, 15), (0x15, 15), (0x14, 15), (0x13, 15), (0x12, 15), (0x11, 15), (0x10, 15), (0x02, 3),
    (0x06, 5), (0x79, 7), (0x27, 8), (0x20, 8), (0x16, 13), (0x15, 13), (0x1f, 15), (0x1e, 15), (0x1d, 15), (0x1c, 15),
    (0x1b, 15), (0x1a, 15), (0x19, 15), (0x13, 16), (0x12, 16), (0x11, 16), (0x10, 16), (0x05, 5), (0x07, 7), (0xfc, 8),
    (0x0c, 10), (0x14, 13), (0x07, 5), (0x26, 8), (0x1c, 12), (0x13, 13), (0x06, 6), (0xfd, 8), (0x12, 12), (0x07, 6),
    (0x04, 9), (0x12, 13), (0x06, 7), (0x1e, 12), (0x14, 16), (0x04, 7), (0x15, 12), (0x05, 7), (0x11, 12), (0x78, 7),
    (0x11, 13), (0x7a, 7), (0x10, 13), (0x21, 8), (0x1a, 16), (0x25, 8), (0x19, 16), (0x24, 8), (0x18, 16), (0x05, 9),
    (0x17, 16), (0x07, 9), (0x16, 16), (0x0d, 10), (0x15, 16), (0x1f, 12), (0x1a, 12), (0x19, 12), (0x17, 12), (0x16, 12),
    (0x1f, 13), (0x1e, 13), (0x1d, 13), (0x1c, 13), (0x1b, 13), (0x1f, 16), (0x1e, 16), (0x1d, 16), (0x1c, 16), (0x1b, 16),
    (0x01, 6), (0x06, 4),
];
const LEVEL: [u8; 111] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1,
    2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1,
];
const RUN: [u8; 111] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9,
    10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

pub const INTRA: u8 = 0x01;
pub const PAT: u8 = 0x02;
pub const BACK: u8 = 0x04;
pub const FOR: u8 = 0x08;
pub const QUANT: u8 = 0x10;

struct Tables {
    mba: Vlc,
    mb_type: [Vlc; 3],
    cbp: Vlc,
    mv: Vlc,
    dmv: Vlc,
    dc: [Vlc; 2],
    /// B.14 subsequent, B.14 first coefficient of a non-intra block, B.15
    coef: [Vlc; 3],
}

fn tables() -> &'static Tables {
    static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mba: Vec<(u32, u8, i32)> = MBA_TABLE.iter().enumerate().map(|(i, &(c, l))| (c as u32, l, i as i32 + 1)).collect();
        let t = |e: &[(u8, u8, u8)]| Vlc::new(&e.iter().map(|&(c, l, v)| (c as u32, l, v as i32)).collect::<Vec<_>>());
        let coef = |tab: &[(u16, u8); 113], first: bool| {
            let mut e: Vec<(u32, u8, i32)> = tab.iter().enumerate().map(|(i, &(c, l))| (c as u32, l, i as i32)).collect();
            if first {
                e.pop(); // no end of block before the first coefficient
                e[0] = (1, 1, 0); // '1s' is run 0, level 1
            }
            Vlc::new(&e)
        };
        Tables {
            mba: Vlc::new(&mba),
            mb_type: [
                t(&[(1, 1, INTRA), (1, 2, QUANT | INTRA)]),
                t(&[(3, 5, INTRA), (1, 2, PAT), (1, 3, FOR), (1, 1, FOR | PAT), (1, 6, QUANT | INTRA), (1, 5, QUANT | PAT), (2, 5, QUANT | FOR | PAT)]),
                t(&[
                    (3, 5, INTRA),
                    (2, 3, BACK),
                    (3, 3, BACK | PAT),
                    (2, 4, FOR),
                    (3, 4, FOR | PAT),
                    (2, 2, FOR | BACK),
                    (3, 2, FOR | BACK | PAT),
                    (1, 6, QUANT | INTRA),
                    (2, 6, QUANT | BACK | PAT),
                    (3, 6, QUANT | FOR | PAT),
                    (2, 5, QUANT | FOR | BACK | PAT),
                ]),
            ],
            cbp: Vlc::new(&PAT_TABLE.iter().enumerate().map(|(i, &(c, l))| (c as u32, l, i as i32)).collect::<Vec<_>>()),
            mv: Vlc::new(&MV_TABLE.iter().enumerate().map(|(i, &(c, l))| (c as u32, l, i as i32)).collect::<Vec<_>>()),
            dmv: Vlc::new(&[(0, 1, 0), (2, 2, 1), (3, 2, 2)]),
            dc: [
                Vlc::new(&DC_LUM.iter().enumerate().map(|(i, &(c, l))| (c as u32, l, i as i32)).collect::<Vec<_>>()),
                Vlc::new(&DC_CHROMA.iter().enumerate().map(|(i, &(c, l))| (c as u32, l, i as i32)).collect::<Vec<_>>()),
            ],
            coef: [coef(&B14, false), coef(&B14, true), coef(&B15, false)],
        }
    })
}

/// Sequence-level parameters slice parsing depends on.
#[derive(Clone, Copy, Debug, Default)]
pub struct Seq {
    pub width: u32,
    pub height: u32,
    /// A sequence extension was seen (MPEG-2 rather than MPEG-1).
    pub mpeg2: bool,
    /// 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4
    pub chroma: u8,
    pub scalable: bool,
    pub progressive: bool,
}

/// Picture-level parameters slice parsing depends on.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pic {
    /// 1 I, 2 P, 3 B, 4 D
    pub ptype: u8,
    pub f_code: [[u8; 2]; 2],
    /// 1 top field, 2 bottom field, 3 frame
    pub structure: u8,
    pub frame_pred_frame_dct: bool,
    pub concealment: bool,
    pub intra_vlc: bool,
    pub alternate_scan: bool,
    pub q_scale_type: bool,
    /// A picture coding extension followed the picture header.
    pub ext: bool,
}

/// Reads a sequence header (payload after the `B3` start code).
pub fn sequence_header(p: &[u8], seq: &mut Seq) -> R<()> {
    let mut r = BitReader::new(p);
    seq.width = r.u(12)?;
    seq.height = r.u(12)?;
    seq.mpeg2 = false;
    seq.chroma = 1;
    seq.scalable = false;
    Ok(())
}

/// Reads an extension (payload after the `B5` start code).
pub fn extension(p: &[u8], seq: &mut Seq, pic: &mut Pic) -> R<()> {
    let mut r = BitReader::new(p);
    match r.u(4)? {
        1 => {
            r.u(8)?; // profile and level
            seq.progressive = r.flag()?;
            seq.chroma = r.u(2)? as u8;
            seq.width |= r.u(2)? << 12;
            seq.height |= r.u(2)? << 12;
            seq.mpeg2 = true;
        }
        5 => seq.scalable = true,
        8 => {
            for s in 0..2 {
                for t in 0..2 {
                    pic.f_code[s][t] = r.u(4)? as u8;
                }
            }
            r.u(2)?; // intra_dc_precision
            pic.structure = r.u(2)? as u8;
            r.bit()?; // top_field_first
            pic.frame_pred_frame_dct = r.flag()?;
            pic.concealment = r.flag()?;
            pic.q_scale_type = r.flag()?;
            pic.intra_vlc = r.flag()?;
            pic.alternate_scan = r.flag()?;
            pic.ext = true;
        }
        _ => {}
    }
    Ok(())
}

/// Reads a picture header (payload after the `00` start code).
pub fn picture_header(p: &[u8], pic: &mut Pic) -> R<()> {
    let mut r = BitReader::new(p);
    r.u(10)?; // temporal reference
    *pic = Pic { ptype: r.u(3)? as u8, ..Default::default() };
    Ok(())
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

/// Model context shared by the slices of a picture: per macroblock the number of
/// coefficients of each block, for the row above and for the previous picture.
#[derive(Default)]
pub struct PicCtx {
    cols: usize,
    above: Vec<[u8; 12]>,
    cur: Vec<[u8; 12]>,
    colo: Vec<[u8; 12]>,
}

impl PicCtx {
    /// Starts a picture of `width` x `height` pixels.
    pub fn picture(&mut self, width: u32, height: u32) {
        let cols = (width as usize).div_ceil(16);
        let n = cols * (height as usize).div_ceil(16);
        if n > 1 << 16 || cols == 0 {
            *self = PicCtx::default();
            return;
        }
        if self.cur.len() != n {
            self.cur = vec![[0; 12]; n];
            self.colo = vec![[0; 12]; n];
        }
        std::mem::swap(&mut self.cur, &mut self.colo);
        self.cur.fill([0; 12]);
        self.above = vec![[0; 12]; cols];
        self.cols = cols;
    }

    fn get(&self, addr: i64) -> ([u8; 12], [u8; 12]) {
        if self.cols == 0 || addr < 0 {
            return ([0; 12], [0; 12]);
        }
        let a = addr as usize;
        (self.above[a % self.cols], self.colo.get(a).copied().unwrap_or([0; 12]))
    }

    fn set(&mut self, addr: i64, blk: &[u32; 12]) {
        if self.cols == 0 || addr < 0 {
            return;
        }
        let a = addr as usize;
        let mut v = [0u8; 12];
        for (d, &s) in v.iter_mut().zip(blk) {
            *d = s.min(255) as u8;
        }
        let c = a % self.cols;
        self.above[c] = v;
        if let Some(x) = self.cur.get_mut(a) {
            *x = v;
        }
    }
}

/// What a slice parse found (for statistics and the model's cross-slice context).
#[derive(Clone, Copy, Debug, Default)]
pub struct SliceInfo {
    pub mbs: u32,
    pub coefs: u32,
}

/// Parses one slice (payload after its start code) through `io`.
pub struct SliceParser<'a> {
    seq: &'a Seq,
    pic: &'a Pic,
    ctx: &'a mut PicCtx,
    /// address of the last macroblock (skipped ones included)
    addr: i64,
    /// coefficient counts of the macroblock above and of the co-located one of the previous picture
    above: [u8; 12],
    colo: [u8; 12],
    t: &'static Tables,
    q: u32,
    row: u32,
    prev_type: u8,
    prev_cbp: u32,
    /// motion codes of the previous macroblock, per [r][s][t] (model context)
    prev_mc: [[[i32; 2]; 2]; 2],
    /// coefficients of each block of the previous macroblock (model context)
    prev_blk: [u32; 12],
    info: SliceInfo,
}

impl<'a> SliceParser<'a> {
    pub fn new(seq: &'a Seq, pic: &'a Pic, ctx: &'a mut PicCtx, slice_code: u8) -> R<Self> {
        if !seq.mpeg2 || seq.scalable || !pic.ext || pic.structure == 0 || !(1..=3).contains(&pic.ptype) || !(1..=3).contains(&seq.chroma) {
            return Err(Bad);
        }
        let cols = ctx.cols as i64;
        Ok(SliceParser {
            seq,
            pic,
            addr: (slice_code as i64 - 1) * cols - 1,
            ctx,
            above: [0; 12],
            colo: [0; 12],
            t: tables(),
            q: 0,
            row: slice_code as u32,
            prev_type: 0,
            prev_cbp: 0,
            prev_mc: [[[0; 2]; 2]; 2],
            prev_blk: [0; 12],
            info: SliceInfo::default(),
        })
    }

    fn vlc<T: BitIo>(&self, io: &mut T, v: &Vlc, h: Hint, ctx_base: usize, max_depth: usize) -> R<i32> {
        let mut n = 0usize;
        let mut depth = 0usize;
        loop {
            let mut hh = h;
            hh.ctx = (ctx_base + depth.min(max_depth)).min(459) as u16;
            hh.a |= (n as u32) << 20;
            hh.c ^= (n as u32) << 17;
            let b = io.bit(&hh)? as usize;
            let c = v.nodes[n][b];
            if c == NONE {
                return Err(Bad);
            }
            if c < 0 {
                return Ok(-c - 1);
            }
            n = c as usize;
            depth += 1;
        }
    }

    fn bits<T: BitIo>(&self, io: &mut T, n: u32, h: Hint, ctx: usize) -> R<u32> {
        let mut v = 0u32;
        for i in 0..n {
            let mut hh = h;
            hh.ctx = (ctx + (i as usize).min(11)).min(459) as u16;
            hh.b ^= (i << 24) | (v & 0xFFF) << 12;
            v = (v << 1) | io.bit(&hh)?;
        }
        Ok(v)
    }

    pub fn run<T: BitIo>(mut self, io: &mut T) -> R<SliceInfo> {
        let pt = self.pic.ptype as u32;
        if self.seq.height > 2800 {
            self.bits(io, 3, hint(cx::SLICE_HDR, kind::SLICE_HDR, 1, 0, 0, 0), cx::SLICE_HDR)?;
        }
        self.q = self.bits(io, 5, hint(cx::SLICE_HDR + 1, kind::SLICE_HDR, 2, pt, 0, 0), cx::SLICE_HDR + 1)?;
        if self.bits(io, 1, hint(cx::SLICE_HDR + 7, kind::SLICE_HDR, 3, pt, 0, 0), cx::SLICE_HDR + 7)? == 1 {
            self.bits(io, 1, hint(cx::SLICE_HDR + 8, kind::SLICE_HDR, 4, pt, 0, 0), cx::SLICE_HDR + 8)?;
            self.bits(io, 7, hint(cx::SLICE_HDR + 9, kind::SLICE_HDR, 5, 0, 0, 0), cx::SLICE_HDR + 9)?;
            while self.bits(io, 1, hint(cx::SLICE_HDR + 10, kind::SLICE_HDR, 6, 0, 0, 0), cx::SLICE_HDR + 10)? == 1 {
                self.bits(io, 8, hint(cx::SLICE_HDR + 11, kind::SLICE_HDR, 7, 0, 0, 0), cx::SLICE_HDR + 11)?;
            }
        }
        loop {
            self.macroblock(io)?;
            self.info.mbs += 1;
            if self.info.mbs > 1 << 16 {
                return Err(Bad);
            }
            let h = hint(cx::MORE, kind::MORE, self.info.mbs.min(255), self.row & 0xFF, pt, 0);
            if !io.more(&h)? {
                break;
            }
        }
        Ok(self.info)
    }

    fn macroblock<T: BitIo>(&mut self, io: &mut T) -> R<()> {
        let pt = self.pic.ptype as u32;
        let t = self.t;
        // address increment, with escapes
        let mut inc = 0u32;
        loop {
            let h = hint(cx::MBA, kind::MBA, (inc.min(66) / 33) << 8, pt | (self.prev_type as u32) << 4, self.info.mbs.min(63), 0);
            match self.vlc(io, &t.mba, h, cx::MBA, 11)? {
                MBA_ESC => inc += 33,
                MBA_STUFF => return Err(Bad),
                v => {
                    inc += v as u32;
                    break;
                }
            }
        }
        let skipped = inc - 1;
        for k in 1..inc as i64 {
            self.ctx.set(self.addr + k, &[0; 12]);
        }
        self.addr += inc as i64;
        (self.above, self.colo) = self.ctx.get(self.addr);
        let tot = |b: &[u8; 12]| b.iter().map(|&x| x as u32).sum::<u32>();
        let nbt = bucket(tot(&self.above)) | bucket(tot(&self.colo)) << 3;
        let pat = |b: &[u8; 12]| b[..6].iter().fold(0u32, |a, &x| a << 1 | (x > 0) as u32);
        let nbp = pat(&self.above) | pat(&self.colo) << 6;
        let h = hint(cx::MB_TYPE + (pt as usize - 1) * 8, kind::MB_TYPE, pt << 8, self.prev_type as u32 | bucket(skipped) << 8, self.prev_cbp | nbt << 6, self.q >> 2 | nbt << 6);
        let mbt = self.vlc(io, &t.mb_type[pt as usize - 1], h, cx::MB_TYPE + (pt as usize - 1) * 8, 7)? as u8;
        let intra = mbt & INTRA != 0;
        let motion = mbt & (FOR | BACK) != 0;
        let frame = self.pic.structure == 3;
        let mut motion_type = 2u32; // frame-based (frame pictures) / field-based (field pictures)
        if motion {
            if frame {
                if !self.pic.frame_pred_frame_dct {
                    motion_type = self.bits(io, 2, hint(cx::MODES, kind::MODES, 1, mbt as u32, pt, 0), cx::MODES)?;
                }
            } else {
                motion_type = self.bits(io, 2, hint(cx::MODES + 2, kind::MODES, 2, mbt as u32, pt, 0), cx::MODES + 2)?;
            }
        } else if !frame {
            motion_type = 1;
        }
        if frame && !self.pic.frame_pred_frame_dct && (intra || mbt & PAT != 0) {
            self.bits(io, 1, hint(cx::MODES + 4, kind::MODES, 3, mbt as u32, motion_type, 0), cx::MODES + 4)?;
        }
        if mbt & QUANT != 0 {
            self.q = self.bits(io, 5, hint(cx::QUANT, kind::QUANT, self.q, mbt as u32, pt, 0), cx::QUANT)?;
        }
        // motion vector count, format and dual prime (tables 6-17 and 6-18)
        let (count, field_fmt, dmv) = if frame {
            match motion_type {
                1 => (2, true, false),
                3 => (1, true, true),
                _ => (1, false, false),
            }
        } else {
            match motion_type {
                2 => (2, true, false),
                3 => (1, true, true),
                _ => (1, true, false),
            }
        };
        if intra {
            self.prev_mc = [[[0; 2]; 2]; 2];
        }
        if mbt & FOR != 0 || (intra && self.pic.concealment) {
            let (c, f, d) = if intra { (1, !frame, false) } else { (count, field_fmt, dmv) };
            self.motion_vectors(io, 0, c, f, d, mbt)?;
        }
        if mbt & BACK != 0 {
            self.motion_vectors(io, 1, count, field_fmt, dmv, mbt)?;
        }
        if intra && self.pic.concealment {
            self.bits(io, 1, hint(cx::MODES + 6, kind::MODES, 4, 0, 0, 0), cx::MODES + 6)?;
        }
        let nblocks = match self.seq.chroma {
            1 => 6,
            2 => 8,
            _ => 12,
        };
        let mut pattern = if intra { (1u32 << nblocks) - 1 } else { 0 };
        if mbt & PAT != 0 {
            let h = hint(cx::CBP, kind::CBP, 0, self.prev_cbp & 63, mbt as u32 | pt << 5, motion_type | nbp << 2);
            let cbp = self.vlc(io, &t.cbp, h, cx::CBP, 9)? as u32;
            pattern = cbp << (nblocks - 6);
            if self.seq.chroma == 2 {
                pattern |= self.bits(io, 2, hint(cx::CBP + 10, kind::CBP, 1, cbp, 0, 0), cx::CBP + 10)?;
            } else if self.seq.chroma == 3 {
                pattern |= self.bits(io, 6, hint(cx::CBP + 12, kind::CBP, 2, cbp, 0, 0), cx::CBP + 12)?;
            }
            self.prev_cbp = cbp;
        } else {
            self.prev_cbp = if intra { 63 } else { 0 };
        }
        let mut blk = [0u32; 12];
        for (i, b) in blk.iter_mut().enumerate().take(nblocks) {
            if pattern & (1 << (nblocks - 1 - i)) != 0 {
                *b = self.block(io, i, intra)?;
            }
        }
        self.ctx.set(self.addr, &blk);
        self.prev_blk = blk;
        self.prev_type = mbt;
        Ok(())
    }

    fn motion_vectors<T: BitIo>(&mut self, io: &mut T, s: usize, count: usize, field_fmt: bool, dmv: bool, mbt: u8) -> R<()> {
        let sel = |io: &mut T, me: &Self, r: usize| me.bits(io, 1, hint(cx::MODES + 5, kind::MODES, 5 | (r as u32) << 4 | (s as u32) << 5, mbt as u32, 0, 0), cx::MODES + 5);
        if count == 1 {
            if field_fmt && !dmv {
                sel(io, self, 0)?;
            }
            self.motion_vector(io, 0, s, dmv, mbt)?;
        } else {
            sel(io, self, 0)?;
            self.motion_vector(io, 0, s, dmv, mbt)?;
            sel(io, self, 1)?;
            self.motion_vector(io, 1, s, dmv, mbt)?;
        }
        Ok(())
    }

    fn motion_vector<T: BitIo>(&mut self, io: &mut T, r: usize, s: usize, dmv: bool, mbt: u8) -> R<()> {
        let t = self.t;
        let mut other = 31u32;
        for tt in 0..2 {
            let prev = self.prev_mc[r][s][tt];
            let pm = prev.unsigned_abs().min(16) | ((prev < 0) as u32) << 5;
            let id = (r as u32) | (s as u32) << 1 | (tt as u32) << 2;
            let fc = self.pic.f_code[s][tt] as u32;
            let h = hint(cx::MOTION_CODE + tt * 7, kind::MOTION_CODE, id << 8, pm | fc << 6, mbt as u32, other);
            let m = self.vlc(io, &t.mv, h, cx::MOTION_CODE + tt * 7, 6)?;
            let mut code = m;
            if m != 0 {
                let neg = self.bits(io, 1, hint(cx::MOTION_SIGN + tt, kind::MOTION_SIGN, id, m as u32, pm, other), cx::MOTION_SIGN + tt)?;
                if neg == 1 {
                    code = -m;
                }
                if fc > 1 && fc < 15 {
                    self.bits(io, fc - 1, hint(cx::RESIDUAL + tt * 4, kind::RESIDUAL, id | fc << 3, m as u32, pm, 0), cx::RESIDUAL + tt * 4)?;
                }
            }
            if dmv {
                self.vlc(io, &t.dmv, hint(cx::DMV + tt * 2, kind::DMV, id, 0, 0, 0), cx::DMV + tt * 2, 1)?;
            }
            self.prev_mc[r][s][tt] = code;
            other = code.unsigned_abs().min(15) | ((code < 0) as u32) << 4;
        }
        Ok(())
    }

    /// Reads one block's coefficients; returns how many there were.
    fn block<T: BitIo>(&mut self, io: &mut T, i: usize, intra: bool) -> R<u32> {
        let t = self.t;
        let lum = i < 4;
        let pt = self.pic.ptype as u32;
        // (the same block above or in the previous picture made coefficient contexts too
        // sparse and cost more than it gave; macroblock types and patterns use it)
        let near = bucket(self.prev_blk[i]) | (if i > 0 { bucket(self.prev_blk[i - 1]) } else { 0 }) << 3;
        let mut n = 0u32;
        let mut pos: u32;
        if intra {
            let c = !lum as usize;
            let h = hint(cx::DC_SIZE + c * 12, kind::DC_SIZE, (c as u32) << 8, near, pt, self.q >> 2);
            let size = self.vlc(io, &t.dc[c], h, cx::DC_SIZE + c * 12, 10)? as u32;
            if size > 0 {
                self.bits(io, size, hint(cx::DC_DIFF + c * 12, kind::DC_DIFF, (c as u32) | size << 1, near, 0, 0), cx::DC_DIFF + c * 12)?;
            }
            pos = 1;
        } else {
            pos = 0;
        }
        let tab = if intra && self.pic.intra_vlc { 2 } else { 0 };
        let mut first = !intra;
        let mut prev_level = 0u32;
        let mut prev_run = 0u32;
        let mut prev_sign = 2u32;
        loop {
            let ti = if first { 1 } else { tab };
            let set = if intra { 2 + (ti == 2) as usize } else { ti };
            let ctx = cx::COEF + set * 24;
            let h = hint(
                ctx,
                kind::COEF,
                (ti as u32) << 16 | (intra as u32) << 18 | (lum as u32) << 19 | bucket(pos).min(7) << 28,
                pos.min(63) | n.min(15) << 6 | (intra as u32) << 16,
                bucket(prev_level) | prev_run.min(15) << 3 | (i as u32) << 7 | pos.min(63) << 11,
                self.q >> 1 | pt << 5 | (self.pic.alternate_scan as u32) << 12 | near << 13,
            );
            let v = self.vlc(io, &t.coef[ti], h, ctx + (pos.min(7) * 3) as usize, 2)?;
            if v == EOB {
                break;
            }
            let (run, level) = if v == ESC {
                let run = self.bits(io, 6, hint(cx::ESCAPE, kind::ESCAPE, 1, pos.min(63), 0, 0), cx::ESCAPE)?;
                let lv = self.bits(io, 12, hint(cx::ESCAPE + 8, kind::ESCAPE, 2, run, 0, 0), cx::ESCAPE + 8)?;
                let level = if lv & 0x800 != 0 { 4096 - lv } else { lv };
                (run, level)
            } else {
                let (run, level) = (RUN[v as usize] as u32, LEVEL[v as usize] as u32);
                let h = hint(
                    cx::COEF_SIGN + lum as usize * 5 + (pos + run).min(4) as usize,
                    kind::COEF_SIGN,
                    (lum as u32) | (intra as u32) << 1,
                    (pos + run).min(63),
                    level.min(15),
                    prev_sign | (i as u32) << 2 | (pos + run).min(63) << 6,
                );
                prev_sign = io.bit(&h)?;
                (run, level)
            };
            pos += run;
            if pos > 63 {
                return Err(Bad);
            }
            pos += 1;
            n += 1;
            prev_level = level;
            prev_run = run;
            first = false;
        }
        self.info.coefs += n;
        Ok(n)
    }
}
