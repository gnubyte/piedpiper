//! AC-3 (Dolby Digital, ATSC A/52) audio frames, coded bit by bit.
//!
//! DVD rips carry several AC-3 tracks, about a fifth of the file. AC-3 has no entropy coder
//! to undo: exponents are differential values packed three to a 7-bit code, and mantissas
//! are fixed-length or grouped codes whose sizes come from a psychoacoustic bit allocation.
//! As with MPEG-2 video, nothing is re-encoded. [`parse_frame`] reads every bit of a frame
//! through a [`BitIo`] with a [`Hint`] (syntax element, place in its code, context such as the
//! previous frame's value of the same field or the previous exponents of the channel), runs
//! the A/52 bit allocation to know how long each mantissa is, and the decoder runs the same
//! parser on the bits the model returns, so a frame is rebuilt bit for bit by construction.
//! The sync word and both CRCs are not coded: only frames whose CRCs check out are recoded,
//! and the decoder recomputes them.

use crate::bits::{Bad, BitWriter, R};
use crate::h264::mb::Hint;
use crate::model::{code_bin, code_num, ArithDec, ArithEnc, Model};
use crate::vp9::stream::Rec;

// --- A/52 tables -------------------------------------------------------------------------------
// From ETSI TS 102 366 V1.4.1 (2017-09), the ETSI edition of the AC-3 standard (ATSC A/52),
// checked value by value: BAND_START is bndtab (table 6.12) with the end of the last band
// appended, LOG_ADD is latab (6.14), HTH is hth (6.15) as [48, 44.1, 32 kHz] per band,
// FRAME_WORDS is table 4.13, BAP_TAB is baptab (6.16), SLOW_DECAY to FAST_GAIN are tables 6.6
// to 6.11 and MANT_BITS is table 6.17. bin_to_band() gives masktab (6.13) and frame58() the 5/8
// sizes of table 6.34.

const BAND_START: [u8; 51] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 34, 37, 40, 43, 46, 49, 55, 61, 67, 73, 79, 85, 97, 109, 121, 133, 157, 181, 205, 229, 253];
const LOG_ADD: [u8; 260] = [64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 52, 51, 50, 49, 48, 47, 47, 46, 45, 44, 44, 43, 42, 41, 41, 40, 39, 38, 38, 37, 36, 36, 35, 35, 34, 33, 33, 32, 32, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 21, 20, 20, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const HTH: [[u16; 3]; 50] = [[1232, 1264, 1408], [1232, 1264, 1408], [1088, 1120, 1200], [1024, 1040, 1104], [992, 992, 1056], [960, 976, 1008], [944, 960, 992], [944, 944, 976], [928, 944, 960], [928, 928, 944], [928, 928, 944], [928, 928, 944], [928, 928, 928], [912, 928, 928], [912, 912, 928], [912, 912, 928], [896, 912, 928], [896, 896, 928], [880, 896, 928], [880, 896, 928], [864, 880, 912], [864, 880, 912], [848, 864, 912], [848, 864, 912], [832, 848, 896], [832, 848, 896], [816, 832, 896], [800, 832, 880], [784, 800, 864], [768, 784, 848], [752, 768, 832], [752, 752, 816], [752, 752, 800], [752, 752, 784], [768, 752, 768], [784, 768, 752], [832, 800, 752], [912, 848, 752], [992, 912, 768], [1056, 992, 784], [1120, 1056, 816], [1168, 1104, 848], [1184, 1184, 960], [1120, 1168, 1040], [1088, 1120, 1136], [1088, 1088, 1184], [1312, 1152, 1120], [2048, 1584, 1088], [2112, 2112, 1104], [2112, 2112, 1248]];
/// Frame sizes in 16-bit words by frmsizecod and fscod.
const FRAME_WORDS: [[u16; 3]; 38] = [[64, 69, 96], [64, 70, 96], [80, 87, 120], [80, 88, 120], [96, 104, 144], [96, 105, 144], [112, 121, 168], [112, 122, 168], [128, 139, 192], [128, 140, 192], [160, 174, 240], [160, 175, 240], [192, 208, 288], [192, 209, 288], [224, 243, 336], [224, 244, 336], [256, 278, 384], [256, 279, 384], [320, 348, 480], [320, 349, 480], [384, 417, 576], [384, 418, 576], [448, 487, 672], [448, 488, 672], [512, 557, 768], [512, 558, 768], [640, 696, 960], [640, 697, 960], [768, 835, 1152], [768, 836, 1152], [896, 975, 1344], [896, 976, 1344], [1024, 1114, 1536], [1024, 1115, 1536], [1152, 1253, 1728], [1152, 1254, 1728], [1280, 1393, 1920], [1280, 1394, 1920]];
const BAP_TAB: [u8; 64] = [0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15];
const SLOW_DECAY: [u8; 4] = [15, 17, 19, 21];
const FAST_DECAY: [u8; 4] = [63, 83, 103, 123];
const SLOW_GAIN: [u16; 4] = [1344, 1240, 1144, 1040];
const DB_PER_BIT: [u16; 4] = [0, 1792, 2304, 2816];
const FLOOR: [i16; 8] = [752, 688, 624, 560, 496, 368, 240, -2048];
const FAST_GAIN: [u16; 8] = [128, 256, 384, 512, 640, 768, 896, 1024];
/// Bits of a directly coded mantissa by bap (0 for no bits or grouped codes).
const MANT_BITS: [u32; 16] = [0, 0, 0, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16];

fn bin_to_band(bin: usize) -> usize {
    // BAND_START is short: search it (bins below 28 map to themselves)
    if bin < 28 {
        return bin;
    }
    BAND_START.iter().rposition(|&s| (s as usize) <= bin).unwrap_or(0).min(49)
}

/// logadd() of ETSI TS 102 366 clause 6.2.2.3.
fn logadd(a: i32, b: i32) -> i32 {
    let c = a - b;
    let address = (c.abs() >> 1).min(255) as usize;
    if c >= 0 {
        a + LOG_ADD[address] as i32
    } else {
        b + LOG_ADD[address] as i32
    }
}

/// calc_lowcomp() of ETSI TS 102 366 clause 6.2.2.4.
fn calc_lowcomp(a: i32, b0: i32, b1: i32, bin: usize) -> i32 {
    if bin < 7 {
        if b0 + 256 == b1 {
            384
        } else if b0 > b1 {
            (a - 64).max(0)
        } else {
            a
        }
    } else if bin < 20 {
        if b0 + 256 == b1 {
            320
        } else if b0 > b1 {
            (a - 64).max(0)
        } else {
            a
        }
    } else {
        (a - 128).max(0)
    }
}

// --- CRC ---------------------------------------------------------------------------------------

const fn crc_table() -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u16) << 8;
        let mut k = 0;
        while k < 8 {
            c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

static CRC: [u16; 256] = crc_table();

/// CRC-16 (x^16 + x^15 + x^2 + 1, MSB first, no inversion) as A/52 uses it.
pub fn crc16(data: &[u8]) -> u16 {
    data.iter().fold(0u16, |c, &b| (c << 8) ^ CRC[((c >> 8) as u8 ^ b) as usize])
}

/// Bytes of the first CRC's region (5/8 of a frame of `size` bytes).
fn frame58(size: usize) -> usize {
    ((size >> 2) + (size >> 4)) << 1
}

/// The `crc1` value that makes the CRC of `frame[2..frame58]` zero (the field itself is at
/// bytes 2..4 and is ignored).
fn crc1_for(frame: &[u8]) -> u16 {
    let end = frame58(frame.len());
    let l = end - 4;
    // the CRC of the region is linear in the field: A * field + CRC(region with a zero field)
    let mut cols = [0u16; 16];
    for (i, col) in cols.iter_mut().enumerate() {
        let mut c = crc16(&(1u16 << i).to_be_bytes());
        for _ in 0..l {
            c = (c << 8) ^ CRC[(c >> 8) as usize];
        }
        *col = c;
    }
    let mut zeroed = frame[2..end].to_vec();
    zeroed[0] = 0;
    zeroed[1] = 0;
    let t = crc16(&zeroed);
    // solve A x = t by Gaussian elimination over GF(2)
    let mut rows = [0u32; 16];
    for (r, row) in rows.iter_mut().enumerate() {
        let mut a = 0u32;
        for (i, &col) in cols.iter().enumerate() {
            a |= (((col >> r) & 1) as u32) << i;
        }
        *row = a | (((t >> r) & 1) as u32) << 16;
    }
    for i in 0..16 {
        let Some(p) = (i..16).find(|&r| (rows[r] >> i) & 1 == 1) else { return 0 };
        rows.swap(i, p);
        for r in 0..16 {
            if r != i && (rows[r] >> i) & 1 == 1 {
                rows[r] ^= rows[i];
            }
        }
    }
    (0..16).fold(0u16, |x, i| x | (((rows[i] >> 16) & 1) as u16) << i)
}

/// Writes both CRCs of a frame whose other bytes are final.
pub fn fill_crcs(frame: &mut [u8]) {
    let c1 = crc1_for(frame);
    frame[2..4].copy_from_slice(&c1.to_be_bytes());
    let n = frame.len();
    let c2 = crc16(&frame[2..n - 2]);
    frame[n - 2..].copy_from_slice(&c2.to_be_bytes());
}

/// The size of a recodable AC-3 frame at the start of `d` (sync word, a header this parser
/// handles, both CRCs right), or `None`.
pub fn frame_at(d: &[u8]) -> Option<usize> {
    if d.len() < 8 || d[0] != 0x0B || d[1] != 0x77 {
        return None;
    }
    let fscod = (d[4] >> 6) as usize;
    let frm = (d[4] & 63) as usize;
    let bsid = d[5] >> 3;
    if fscod == 3 || frm > 37 || bsid > 8 {
        return None;
    }
    let size = FRAME_WORDS[frm][fscod] as usize * 2;
    if size > d.len() || crc16(&d[2..size]) != 0 || crc16(&d[2..frame58(size)]) != 0 {
        return None;
    }
    Some(size)
}

// --- contexts ------------------------------------------------------------------------------------

/// Where frame bits come from and go to.
pub trait BitIo {
    fn bit(&mut self, h: &Hint) -> R<u32>;
}

/// Hint kinds (the AC-3 streams have models of their own).
mod kind {
    pub const SIDE: u8 = 4;
    pub const EXP: u8 = 21;
    pub const MANT: u8 = 22;
    pub const MGRP: u8 = 23;
    pub const AUX: u8 = 24;
}

/// Mixer contexts.
mod cx {
    pub const SIDE: usize = 0;
    pub const EXP: usize = 64;
    pub const MANT: usize = 96;
    pub const MGRP: usize = 232;
    pub const AUX: usize = 260;
}

#[inline]
fn hint(ctx: usize, kind: u8, a: u32, b: u32, c: u32, d: u32) -> Hint {
    Hint { ctx: ctx.min(459) as u16, kind, a, b, c, d }
}

const CPL: usize = 0;
const LFE: usize = 6;
/// Side-information slots per stream: (block 0..=6, element 0..63, index 0..127).
const SIDE_KEYS: usize = 7 * 64 * 128;

/// What the model remembers of an AC-3 stream (frames with the same rate and size).
struct Stream {
    sig: u16,
    side: Vec<u16>,
    exps: [[u8; 256]; 7],
    mags: [[u8; 256]; 7],
}

impl Stream {
    fn new(sig: u16) -> Self {
        Stream { sig, side: vec![u16::MAX; SIDE_KEYS], exps: [[0; 256]; 7], mags: [[8; 256]; 7] }
    }
}

/// Model context kept across the frames of a chunk.
#[derive(Default)]
pub struct Ctx {
    streams: Vec<Stream>,
    last_sig: u16,
}

impl Ctx {
    fn stream(&mut self, sig: u16) -> usize {
        if let Some(i) = self.streams.iter().position(|s| s.sig == sig) {
            return i;
        }
        if self.streams.len() >= 16 {
            self.streams.remove(0);
        }
        self.streams.push(Stream::new(sig));
        self.streams.len() - 1
    }
}

// --- parsing -------------------------------------------------------------------------------------

/// Per-frame decoding state (A/52 syntax state; resets with every frame).
struct Frame {
    acmod: u32,
    nfch: usize,
    lfeon: bool,
    fscod: usize,
    cplinu: bool,
    chincpl: [bool; 6],
    blksw: [bool; 7],
    phsflginu: bool,
    ncplbnd: usize,
    cplstrt: usize,
    cplend: usize,
    endmant: [usize; 7],
    expstr: [u8; 7],
    dexps: [[u8; 256]; 7],
    bap: [[u8; 256]; 7],
    sdecay: i32,
    fdecay: i32,
    sgain: i32,
    dbknee: i32,
    floor: i32,
    snroffset: [i32; 7],
    fgain: [i32; 7],
    cplfleak: i32,
    cplsleak: i32,
    dba_mode: [u8; 7],
    dba: [[(u8, u8, u8); 8]; 7],
    dba_nseg: [usize; 7],
}

impl Default for Frame {
    fn default() -> Self {
        Frame {
            acmod: 0,
            nfch: 0,
            lfeon: false,
            fscod: 0,
            cplinu: false,
            chincpl: [false; 6],
            blksw: [false; 7],
            phsflginu: false,
            ncplbnd: 0,
            cplstrt: 0,
            cplend: 0,
            endmant: [0; 7],
            expstr: [0; 7],
            dexps: [[0; 256]; 7],
            bap: [[0; 256]; 7],
            sdecay: 0,
            fdecay: 0,
            sgain: 0,
            dbknee: 0,
            floor: 0,
            snroffset: [0; 7],
            fgain: [0; 7],
            cplfleak: 0,
            cplsleak: 0,
            dba_mode: [2; 7],
            dba: [[(0, 0, 0); 8]; 7],
            dba_nseg: [0; 7],
        }
    }
}

struct Parser<'a, T: BitIo> {
    io: &'a mut T,
    ctx: &'a mut Ctx,
    st: usize,
    /// Bits read so far (the sync word and crc1 included) and the limit (before crc2).
    pos: usize,
    limit: usize,
    blk: usize,
    f: Box<Frame>,
}

impl<T: BitIo> Parser<'_, T> {
    #[inline]
    fn bit(&mut self, h: &Hint) -> R<u32> {
        if self.pos >= self.limit {
            return Err(Bad);
        }
        self.pos += 1;
        self.io.bit(h)
    }

    /// A side-information field of `n` bits, coded with the previous frame's value of the
    /// same field (same block, element and index) as context.
    fn side(&mut self, n: u32, elem: u32, idx: usize) -> R<u32> {
        let key = (self.blk.min(6) * 64 + elem as usize) * 128 + idx.min(127);
        let prev = self.ctx.streams[self.st].side[key] as u32;
        let mut v = 0u32;
        for j in 0..n {
            let node = if j < 11 { 1 << j | v } else { 0x800 | j };
            let pb = if prev == u16::MAX as u32 { 2 } else { (prev >> (n - 1 - j)) & 1 };
            let h = hint(
                cx::SIDE + elem as usize,
                kind::SIDE,
                elem << 24 | (idx.min(127) as u32) << 12 | node,
                prev | j << 16,
                elem << 24 | (self.blk as u32) << 20 | node,
                pb | (j.min(15)) << 2 | n << 8,
            );
            v = v << 1 | self.bit(&h)?;
        }
        if n <= 16 {
            self.ctx.streams[self.st].side[key] = v as u16;
        }
        Ok(v)
    }

    /// Raw bytes (additional BSI, skip fields).
    fn bytes(&mut self, count: usize, elem: u32) -> R<()> {
        for i in 0..count {
            self.side(8, elem, i)?;
        }
        Ok(())
    }

    fn header(&mut self) -> R<()> {
        self.blk = 6;
        let bsid = self.side(5, 2, 0)?;
        if bsid > 8 {
            return Err(Bad);
        }
        self.side(3, 3, 0)?; // bsmod
        let acmod = self.side(3, 4, 0)?;
        if acmod & 1 == 1 && acmod != 1 {
            self.side(2, 5, 0)?; // cmixlev
        }
        if acmod & 4 != 0 {
            self.side(2, 6, 0)?; // surmixlev
        }
        if acmod == 2 {
            self.side(2, 7, 0)?; // dsurmod
        }
        let lfeon = self.side(1, 8, 0)? == 1;
        for i in 0..if acmod == 0 { 2 } else { 1 } {
            self.side(5, 9, i)?; // dialnorm
            if self.side(1, 10, i)? == 1 {
                self.side(8, 11, i)?; // compr
            }
            if self.side(1, 12, i)? == 1 {
                self.side(8, 13, i)?; // langcod
            }
            if self.side(1, 14, i)? == 1 {
                self.side(7, 15, i)?; // mixlevel, roomtyp
            }
        }
        self.side(2, 16, 0)?; // copyrightb, origbs
        if bsid == 6 {
            if self.side(1, 17, 0)? == 1 {
                self.side(14, 18, 0)?; // dmixmod, lt/rt and lo/ro mix levels
            }
            if self.side(1, 19, 0)? == 1 {
                self.side(14, 20, 0)?; // dsurexmod, dheadphonmod, adconvtyp, xbsi2, encinfo
            }
        } else {
            for i in 0..2 {
                if self.side(1, 21, i)? == 1 {
                    self.side(14, 22, i)?; // timecode
                }
            }
        }
        if self.side(1, 23, 0)? == 1 {
            let l = self.side(6, 24, 0)? as usize;
            self.bytes(l + 1, 25)?;
        }
        let f = &mut self.f;
        f.acmod = acmod;
        f.nfch = [2, 1, 2, 3, 3, 4, 4, 5][acmod as usize];
        f.lfeon = lfeon;
        Ok(())
    }

    /// The slots in bitstream order: coupling (if in use), full-bandwidth channels, LFE.
    fn slots(&self) -> Vec<usize> {
        let mut v = Vec::with_capacity(7);
        if self.f.cplinu {
            v.push(CPL);
        }
        v.extend(1..=self.f.nfch);
        if self.f.lfeon {
            v.push(LFE);
        }
        v
    }

    fn block(&mut self, blk: usize) -> R<()> {
        self.blk = blk;
        let nfch = self.f.nfch;
        let acmod = self.f.acmod;
        for ch in 1..=nfch {
            self.f.blksw[ch] = self.side(1, 26, ch)? == 1;
        }
        for ch in 1..=nfch {
            self.side(1, 27, ch)?; // dithflag
        }
        for i in 0..if acmod == 0 { 2 } else { 1 } {
            if self.side(1, 28, i)? == 1 {
                self.side(8, 29, i)?; // dynrng
            }
        }
        // coupling strategy
        if self.side(1, 30, 0)? == 1 {
            self.f.cplinu = self.side(1, 31, 0)? == 1;
            if self.f.cplinu {
                if acmod < 2 {
                    return Err(Bad);
                }
                for ch in 1..=nfch {
                    self.f.chincpl[ch] = self.side(1, 32, ch)? == 1;
                }
                if acmod == 2 {
                    self.f.phsflginu = self.side(1, 33, 0)? == 1;
                }
                let beg = self.side(4, 34, 0)? as usize;
                let end = self.side(4, 35, 0)? as usize + 3;
                if beg >= end {
                    return Err(Bad);
                }
                self.f.cplstrt = beg * 12 + 37;
                self.f.cplend = end * 12 + 37;
                let nsub = end - beg;
                let mut nbnd = nsub;
                for b in 1..nsub {
                    if self.side(1, 36, b)? == 1 {
                        nbnd -= 1;
                    }
                }
                self.f.ncplbnd = nbnd;
            } else {
                self.f.chincpl = [false; 6];
                self.f.phsflginu = false;
            }
        } else if blk == 0 {
            return Err(Bad);
        }
        // coupling coordinates
        if self.f.cplinu {
            let mut any = false;
            for ch in 1..=nfch {
                if !self.f.chincpl[ch] {
                    continue;
                }
                if self.side(1, 37, ch)? == 1 {
                    any = true;
                    self.side(2, 38, ch)?; // mstrcplco
                    for b in 0..self.f.ncplbnd {
                        self.side(8, 39, ch * 20 + b)?; // cplcoexp, cplcomant
                    }
                } else if blk == 0 {
                    return Err(Bad);
                }
            }
            if acmod == 2 && any && self.f.phsflginu {
                for b in 0..self.f.ncplbnd {
                    self.side(1, 40, b)?; // phsflg
                }
            }
        }
        // rematrixing
        if acmod == 2 && self.side(1, 41, 0)? == 1 {
            // number of rematrixing bands (table 4.10)
            let cplbegf = (self.f.cplstrt.max(37) - 37) / 12;
            let n = if !self.f.cplinu || cplbegf > 2 {
                4
            } else if cplbegf > 0 {
                3
            } else {
                2
            };
            for b in 0..n {
                self.side(1, 42, b)?;
            }
        }
        // exponent strategies
        if self.f.cplinu {
            self.f.expstr[CPL] = self.side(2, 43, CPL)? as u8;
        }
        for ch in 1..=nfch {
            self.f.expstr[ch] = self.side(2, 43, ch)? as u8;
        }
        if self.f.lfeon {
            self.f.expstr[LFE] = self.side(1, 43, LFE)? as u8;
        }
        if blk == 0 && self.slots().iter().any(|&s| self.f.expstr[s] == 0) {
            return Err(Bad);
        }
        // channel bandwidth
        for ch in 1..=nfch {
            if self.f.expstr[ch] != 0 {
                if self.f.chincpl[ch] {
                    self.f.endmant[ch] = self.f.cplstrt;
                } else {
                    let bw = self.side(6, 44, ch)? as usize;
                    if bw > 60 {
                        return Err(Bad);
                    }
                    self.f.endmant[ch] = bw * 3 + 73;
                }
            }
        }
        self.f.endmant[LFE] = 7;
        self.f.endmant[CPL] = self.f.cplend;
        // exponents
        for s in self.slots() {
            let strat = self.f.expstr[s] as usize;
            if strat == 0 {
                continue;
            }
            let gs = 3usize << (strat - 1);
            let (ngrps, first) = match s {
                CPL => ((self.f.cplend - self.f.cplstrt) / gs, self.f.cplstrt),
                LFE => (2, 1),
                _ => ((self.f.endmant[s] + gs - 4) / gs, 1),
            };
            let abs = self.side(4, 45, s)?;
            let absexp = if s == CPL { abs << 1 } else { abs };
            if s != CPL {
                self.f.dexps[s][0] = absexp as u8;
            }
            self.exponents(s, strat, ngrps, absexp, first)?;
            if s != CPL && s != LFE {
                self.side(2, 46, s)?; // gainrng
            }
        }
        // bit allocation parameters
        if self.side(1, 47, 0)? == 1 {
            let v = self.side(11, 48, 0)? as usize;
            self.f.sdecay = SLOW_DECAY[v >> 9] as i32;
            self.f.fdecay = FAST_DECAY[(v >> 7) & 3] as i32;
            self.f.sgain = SLOW_GAIN[(v >> 5) & 3] as i32;
            self.f.dbknee = DB_PER_BIT[(v >> 3) & 3] as i32;
            self.f.floor = FLOOR[v & 7] as i32;
        } else if blk == 0 {
            return Err(Bad);
        }
        if self.side(1, 49, 0)? == 1 {
            let csnr = (self.side(6, 50, 0)? as i32 - 15) << 4;
            for s in self.slots() {
                let fs = self.side(4, 51, s)? as i32;
                self.f.snroffset[s] = (csnr + fs) << 2;
                self.f.fgain[s] = FAST_GAIN[self.side(3, 52, s)? as usize] as i32;
            }
        } else if blk == 0 {
            return Err(Bad);
        }
        if self.f.cplinu {
            if self.side(1, 53, 0)? == 1 {
                let v = self.side(6, 54, 0)? as i32;
                self.f.cplfleak = v >> 3;
                self.f.cplsleak = v & 7;
            } else if blk == 0 {
                return Err(Bad);
            }
        }
        // delta bit allocation
        let dslots: Vec<usize> = self.slots().into_iter().filter(|&s| s != LFE).collect();
        if self.side(1, 55, 0)? == 1 {
            for &s in &dslots {
                let m = self.side(2, 56, s)? as u8;
                if m == 3 {
                    return Err(Bad);
                }
                self.f.dba_mode[s] = m;
            }
            for &s in &dslots {
                if self.f.dba_mode[s] == 1 {
                    let n = self.side(3, 57, s)? as usize + 1;
                    self.f.dba_nseg[s] = n;
                    for seg in 0..n {
                        let v = self.side(12, 58, s * 8 + seg)?;
                        self.f.dba[s][seg] = ((v >> 7) as u8, ((v >> 3) & 15) as u8, (v & 7) as u8);
                    }
                }
            }
        } else if blk == 0 {
            self.f.dba_mode = [2; 7];
        }
        // skip field
        if self.side(1, 59, 0)? == 1 {
            let l = self.side(9, 60, 0)? as usize;
            self.bytes(l, 61)?;
        }
        // bit allocation
        for s in self.slots() {
            let (start, end) = match s {
                CPL => (self.f.cplstrt, self.f.cplend),
                _ => (0, self.f.endmant[s]),
            };
            self.bit_alloc(s, start, end)?;
        }
        self.mantissas()
    }

    fn exponents(&mut self, s: usize, strat: usize, ngrps: usize, absexp: u32, first: usize) -> R<()> {
        let rep = [1usize, 1, 2, 4][strat];
        let class = match s {
            CPL => 0,
            LFE => 2,
            _ => 1,
        };
        let mut prev = absexp as i32;
        let mut bin = first;
        let st = self.st;
        for g in 0..ngrps {
            // the delta pattern the previous exponents of this channel had at these bins
            let old = &self.ctx.streams[st].exps[s];
            let mut expect = 0u32;
            let mut base = prev;
            for k in 0..3 {
                let b = (bin + k * rep).min(255);
                let d = (old[b] as i32 - base).clamp(-2, 2);
                expect = expect * 5 + (d + 2) as u32;
                base += d;
            }
            let o0 = (old[bin.min(255)] as i32 - prev).clamp(-8, 8) + 8;
            // the same pattern in the channel coded before this one, in this block
            let xs = if (2..LFE).contains(&s) { s - 1 } else { 0 };
            let xexpect = if xs >= 1 && bin + 2 * rep < self.f.endmant[xs] {
                let x = &self.f.dexps[xs];
                let mut e = 0u32;
                let mut base = prev;
                for k in 0..3 {
                    let b = bin + k * rep;
                    let d = (x[b] as i32 - base).clamp(-2, 2);
                    e = e * 5 + (d + 2) as u32;
                    base += d;
                }
                e
            } else {
                127
            };
            let mut v = 0u32;
            for j in 0..7 {
                let node = 1 << j | v;
                let h = hint(
                    cx::EXP + strat * 8 + j,
                    kind::EXP,
                    node | (strat as u32) << 8 | class << 10 | (g.min(15) as u32) << 12,
                    expect | (prev as u32) << 7 | node << 12,
                    node | (o0 as u32) << 8 | (strat as u32) << 13 | class << 15,
                    xexpect | class << 7 | (self.blk as u32) << 9,
                );
                v = v << 1 | self.bit(&h)?;
            }
            if v >= 125 {
                return Err(Bad);
            }
            for d in [v / 25, (v / 5) % 5, v % 5] {
                prev += d as i32 - 2;
                if !(0..=24).contains(&prev) {
                    return Err(Bad);
                }
                for _ in 0..rep {
                    if bin < 256 {
                        self.f.dexps[s][bin] = prev as u8;
                    }
                    bin += 1;
                }
            }
        }
        let e = self.f.dexps[s];
        self.ctx.streams[st].exps[s] = e;
        Ok(())
    }

    /// The parametric bit allocation of ETSI TS 102 366 clause 6.2.2 for channel slot `s`
    /// (mantissas `start..end`), step by step as in its pseudo-code and with its variable names:
    /// exponents to psd (6.2.2.2), psd integration into bands (6.2.2.3), the excitation function
    /// (6.2.2.4), the masking curve (6.2.2.5), delta bit allocation (6.2.2.6) and the bap of each
    /// mantissa (6.2.2.7).
    fn bit_alloc(&mut self, s: usize, start: usize, end: usize) -> R<()> {
        if end <= start || end > 253 {
            return Err(Bad);
        }
        let f = &mut self.f;
        // 6.2.2.2
        let mut psd = [0i32; 256];
        for bin in start..end {
            psd[bin] = 3072 - ((f.dexps[s][bin] as i32) << 7);
        }
        // 6.2.2.3 (bndtab[k] + bndsz[k] is BAND_START[k + 1]; masktab[bin] is bin_to_band(bin))
        let mut bndpsd = [0i32; 50];
        let mut j = start;
        let mut k = bin_to_band(start);
        loop {
            let lastbin = (BAND_START[k + 1] as usize).min(end);
            bndpsd[k] = psd[j];
            j += 1;
            while j < lastbin {
                bndpsd[k] = logadd(bndpsd[k], psd[j]);
                j += 1;
            }
            k += 1;
            if end <= lastbin {
                break;
            }
        }
        // 6.2.2.4
        let (fgain, sgain, fdecay, sdecay) = (f.fgain[s], f.sgain, f.fdecay, f.sdecay);
        let bndstrt = bin_to_band(start);
        let bndend = bin_to_band(end - 1) + 1;
        let mut excite = [0i32; 50];
        let (mut fastleak, mut slowleak) = (0i32, 0i32);
        let mut begin;
        if bndstrt == 0 {
            // full-bandwidth and lfe channels; calc_lowcomp() is skipped for bin 6 of the lfe
            // channel (the only one with bndend 7)
            let skip = |bin: usize| bndend == 7 && bin == 6;
            let mut lowcomp = calc_lowcomp(0, bndpsd[0], bndpsd[1], 0);
            excite[0] = bndpsd[0] - fgain - lowcomp;
            lowcomp = calc_lowcomp(lowcomp, bndpsd[1], bndpsd[2], 1);
            excite[1] = bndpsd[1] - fgain - lowcomp;
            begin = 7;
            for bin in 2..7 {
                if !skip(bin) {
                    lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);
                }
                fastleak = bndpsd[bin] - fgain;
                slowleak = bndpsd[bin] - sgain;
                excite[bin] = fastleak - lowcomp;
                if !skip(bin) && bndpsd[bin] <= bndpsd[bin + 1] {
                    begin = bin + 1;
                    break;
                }
            }
            for bin in begin..bndend.min(22) {
                if !skip(bin) {
                    lowcomp = calc_lowcomp(lowcomp, bndpsd[bin], bndpsd[bin + 1], bin);
                }
                fastleak = (fastleak - fdecay).max(bndpsd[bin] - fgain);
                slowleak = (slowleak - sdecay).max(bndpsd[bin] - sgain);
                excite[bin] = (fastleak - lowcomp).max(slowleak);
            }
            begin = 22;
        } else {
            // coupling channel, with the leak values of 6.2.2.1
            fastleak = (f.cplfleak << 8) + 768;
            slowleak = (f.cplsleak << 8) + 768;
            begin = bndstrt;
        }
        for bin in begin..bndend {
            fastleak = (fastleak - fdecay).max(bndpsd[bin] - fgain);
            slowleak = (slowleak - sdecay).max(bndpsd[bin] - sgain);
            excite[bin] = fastleak.max(slowleak);
        }
        // 6.2.2.5
        let mut mask = [0i32; 50];
        for bin in bndstrt..bndend {
            if bndpsd[bin] < f.dbknee {
                excite[bin] += (f.dbknee - bndpsd[bin]) >> 2;
            }
            mask[bin] = excite[bin].max(HTH[bin][f.fscod] as i32);
        }
        // 6.2.2.6, for deltbae 0 (reuse) and 1 (new). The first offset counts from the channel's
        // first band (for the coupling channel, its first band: 4.4.3.51).
        if f.dba_mode[s] <= 1 {
            let mut band = bndstrt;
            for seg in 0..f.dba_nseg[s].min(8) {
                let (deltoffst, deltlen, deltba) = f.dba[s][seg];
                band += deltoffst as usize;
                if band >= 50 || deltlen as usize > 50 - band {
                    return Err(Bad);
                }
                let delta = if deltba >= 4 { (deltba as i32 - 3) << 7 } else { (deltba as i32 - 4) << 7 };
                for _ in 0..deltlen {
                    mask[band] += delta;
                    band += 1;
                }
            }
        }
        // 6.2.2.7. 6.2.2.1 gives no mantissas at all when every snr offset of the block is zero;
        // this parser applies that per channel (snroffset -960 is csnroffst = fsnroffst = 0).
        let snroffset = f.snroffset[s];
        if snroffset == -960 {
            f.bap[s] = [0; 256];
            return Ok(());
        }
        let floor = f.floor;
        let mut i = start;
        let mut j = bin_to_band(start);
        loop {
            let lastbin = (BAND_START[j + 1] as usize).min(end);
            let m = ((mask[j] - snroffset - floor).max(0) & 0x1fe0) + floor;
            while i < lastbin {
                let address = ((psd[i] - m) >> 5).clamp(0, 63) as usize;
                f.bap[s][i] = BAP_TAB[address];
                i += 1;
            }
            j += 1;
            if end <= lastbin {
                break;
            }
        }
        Ok(())
    }

    /// Mantissas of one block: full-bandwidth channels in order, the coupling channel after
    /// the first coupled one, then the LFE. Grouped codes are shared across channels.
    fn mantissas(&mut self) -> R<()> {
        let mut order = Vec::with_capacity(7);
        let mut got_cpl = false;
        for ch in 1..=self.f.nfch {
            order.push(ch);
            if self.f.chincpl[ch] && self.f.cplinu && !got_cpl {
                order.push(CPL);
                got_cpl = true;
            }
        }
        if self.f.lfeon {
            order.push(LFE);
        }
        // pending grouped levels (magnitudes for the context) and how many are left
        let mut g1 = (0usize, [0u8; 3]);
        let mut g2 = (0usize, [0u8; 3]);
        let mut g4 = (0usize, [0u8; 2]);
        let st = self.st;
        for s in order {
            let (start, end) = match s {
                CPL => (self.f.cplstrt, self.f.cplend),
                _ => (0, self.f.endmant[s]),
            };
            let class = match s {
                CPL => 0u32,
                LFE => 2,
                _ => 1,
            };
            let rep = [1usize, 1, 2, 4][self.f.expstr[s].min(3) as usize];
            let mut prev_mag = 8u32;
            for bin in start..end {
                let bap = self.f.bap[s][bin] as usize;
                if bap == 0 {
                    self.ctx.streams[st].mags[s][bin] = 0;
                    prev_mag = 0;
                    continue;
                }
                let exp = self.f.dexps[s][bin] as u32;
                let gpos = ((bin - start) % rep) as u32;
                let tmag = self.ctx.streams[st].mags[s][bin] as u32;
                let sw = ((1..LFE).contains(&s) && self.f.blksw[s]) as u32;
                let base_b = exp | gpos << 5 | prev_mag << 7;
                // the same bin of the channel coded before this one in the block (channels of a
                // mix are correlated)
                let xs = match s {
                    CPL => 1,
                    LFE => 0,
                    _ => s - 1,
                };
                let xmag = if xs >= 1 && bin < self.f.endmant[xs] { self.ctx.streams[st].mags[xs][bin] as u32 } else { 9 };
                let base_d = (bin as u32 >> 3).min(31) | xmag << 5 | (s as u32) << 9;
                let mag = match bap {
                    1 | 2 | 4 => {
                        let (left, n, levels, gi): (usize, u32, u32, usize) = match bap {
                            1 => (g1.0, 5, 3, 0),
                            2 => (g2.0, 7, 5, 1),
                            _ => (g4.0, 7, 11, 2),
                        };
                        let level = if left > 0 {
                            match bap {
                                1 => {
                                    g1.0 -= 1;
                                    g1.1[2 - g1.0]
                                }
                                2 => {
                                    g2.0 -= 1;
                                    g2.1[2 - g2.0]
                                }
                                _ => {
                                    g4.0 -= 1;
                                    g4.1[1]
                                }
                            }
                        } else {
                            let mut v = 0u32;
                            for j in 0..n {
                                let node = 1 << j | v;
                                let h = hint(
                                    cx::MGRP + gi * 8 + j as usize,
                                    kind::MGRP,
                                    node | (bap as u32) << 9 | class << 13 | xmag << 15,
                                    base_b,
                                    node | (bap as u32) << 9 | tmag << 13 | sw << 17,
                                    base_d,
                                );
                                v = v << 1 | self.bit(&h)?;
                            }
                            match bap {
                                1 => {
                                    let l = [(v / 9).min(2), ((v / 3) % 3), v % 3];
                                    g1 = (2, [l[0] as u8, l[1] as u8, l[2] as u8]);
                                    l[0] as u8
                                }
                                2 => {
                                    let l = [(v / 25).min(4), (v / 5) % 5, v % 5];
                                    g2 = (2, [l[0] as u8, l[1] as u8, l[2] as u8]);
                                    l[0] as u8
                                }
                                _ => {
                                    let l = [(v / 11).min(10), v % 11];
                                    g4 = (1, [l[0] as u8, l[1] as u8]);
                                    l[0] as u8
                                }
                            }
                        };
                        let center = levels / 2;
                        ((level as u32).abs_diff(center) * 8 / (center + 1)).min(7)
                    }
                    _ => {
                        let n = MANT_BITS[bap];
                        let mut v = 0u32;
                        for j in 0..n {
                            let node = if j < 8 { 1 << j | v } else { 0x100 | j };
                            let h = hint(
                                cx::MANT + bap * 8 + (j as usize).min(7),
                                kind::MANT,
                                node | (bap as u32) << 9 | class << 13 | xmag << 15,
                                base_b,
                                node | (bap as u32) << 9 | tmag << 13 | sw << 17,
                                base_d,
                            );
                            v = v << 1 | self.bit(&h)?;
                        }
                        match bap {
                            3 => (v.abs_diff(3) * 8 / 4).min(7),
                            5 => (v.abs_diff(7) * 8 / 8).min(7),
                            _ => {
                                // two's complement: magnitude's top three bits
                                let neg = v >> (n - 1) == 1;
                                let m = if neg { (!v) & ((1 << (n - 1)) - 1) } else { v };
                                (m >> (n - 4)).min(7)
                            }
                        }
                    }
                };
                self.ctx.streams[st].mags[s][bin] = mag as u8;
                prev_mag = mag;
            }
        }
        Ok(())
    }
}

/// Parses one frame after its sync word and crc1 (`io` delivers the bits from fscod on).
/// Returns the frame size in bytes and the bits used (sync word and crc1 included).
pub fn parse_frame<T: BitIo>(io: &mut T, ctx: &mut Ctx) -> R<(usize, usize)> {
    // fscod and frmsizecod: coded with the previous frame's values as context
    let last = ctx.last_sig as u32;
    let mut v = 0u32;
    for j in 0..8u32 {
        let node = 1 << j | v;
        let h = hint(cx::SIDE + 62, kind::SIDE, 62 << 24 | node, last | j << 16, 62 << 24 | node, 8 << 8);
        v = v << 1 | io.bit(&h)?;
    }
    let (fscod, frm) = ((v >> 6) as usize, (v & 63) as usize);
    if fscod == 3 || frm > 37 {
        return Err(Bad);
    }
    let size = FRAME_WORDS[frm][fscod] as usize * 2;
    let sig = v as u16;
    ctx.last_sig = sig;
    let st = ctx.stream(sig);
    let mut p = Parser { io, ctx, st, pos: 40, limit: size * 8 - 16, blk: 6, f: Box::default() };
    p.f.fscod = fscod;
    p.header()?;
    for blk in 0..6 {
        p.block(blk)?;
    }
    Ok((size, p.pos))
}

// --- frame coding ----------------------------------------------------------------------------------

/// Reads a frame's bits (from fscod on) and records them for the model.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    recs: &'a mut Vec<Rec>,
}

impl BitIo for Reader<'_> {
    #[inline]
    fn bit(&mut self, h: &Hint) -> R<u32> {
        let b = *self.data.get(self.pos >> 3).ok_or(Bad)? >> (7 - (self.pos & 7)) & 1;
        self.pos += 1;
        self.recs.push(Rec { h: *h, prob: 128, bit: b });
        Ok(b as u32)
    }
}

/// Decodes a frame's bits from the model and writes them.
struct Writer<'a, 'd> {
    m: &'a mut Model,
    c: &'a mut ArithDec<'d>,
    w: BitWriter,
}

impl BitIo for Writer<'_, '_> {
    #[inline]
    fn bit(&mut self, h: &Hint) -> R<u32> {
        if self.c.overrun() {
            return Err(Bad);
        }
        let b = code_bin(self.m, self.c, h, Some(0), 0);
        self.w.put(b);
        Ok(b)
    }
}

fn aux_hint(k: usize, left: usize, prev: u32) -> Hint {
    hint(cx::AUX + left.min(17), kind::AUX, (left.min(4095) as u32) << 2 | prev, k.min(4095) as u32, 0, 0)
}

/// Whether the parser handles a frame that [`frame_at`] accepted (parsing depends only on
/// the frame's bits, never on the model context).
/// `probe` holds scratch buffers (reused; their contents do not matter).
pub fn parses(frame: &[u8], probe: &mut Probe) -> bool {
    probe.recs.clear();
    let mut r = Reader { data: frame, pos: 32, recs: &mut probe.recs };
    matches!(parse_frame(&mut r, &mut probe.ctx), Ok((size, _)) if size == frame.len())
}

/// Scratch buffers for [`parses`].
#[derive(Default)]
pub struct Probe {
    recs: Vec<Rec>,
    ctx: Ctx,
}

/// Codes one frame that [`parses`] accepted.
fn encode_frame(frame: &[u8], ctx: &mut Ctx, m: &mut Model, enc: &mut ArithEnc, recs: &mut Vec<Rec>) -> R<()> {
    recs.clear();
    let mut r = Reader { data: frame, pos: 32, recs };
    let (size, used) = parse_frame(&mut r, ctx)?;
    if size != frame.len() {
        return Err(Bad);
    }
    let n = size * 8 - 16;
    let mut prev = 2;
    for k in used..n {
        let b = (frame[k >> 3] >> (7 - (k & 7))) & 1;
        r.recs.push(Rec { h: aux_hint(k - used, n - k, prev), prob: 128, bit: b });
        prev = b as u32;
    }
    for r in recs.iter() {
        code_bin(m, enc, &r.h, Some(0), r.bit as u32);
    }
    Ok(())
}

/// Decodes one frame coded by [`encode_frame`] (sync word and CRCs filled in).
pub fn decode_frame(ctx: &mut Ctx, m: &mut Model, c: &mut ArithDec) -> R<Vec<u8>> {
    let mut w = BitWriter::new();
    w.put_bits(0x0B77, 16);
    w.put_bits(0, 16);
    let mut io = Writer { m, c, w };
    let (size, used) = parse_frame(&mut io, ctx)?;
    let n = size * 8 - 16;
    let mut prev = 2;
    for k in used..n {
        if io.c.overrun() {
            return Err(Bad);
        }
        let b = code_bin(io.m, io.c, &aux_hint(k - used, n - k, prev), Some(0), 0);
        io.w.put(b);
        prev = b;
    }
    io.w.put_bits(0, 16);
    let mut frame = io.w.finish();
    if frame.len() != size {
        return Err(Bad);
    }
    fill_crcs(&mut frame);
    Ok(frame)
}

/// Numbers in an AC-3 stream.
pub const N_COUNT: u32 = 70;
pub const N_GAP: u32 = 71;

/// Codes a run of frames (`(offset, size)` in `input`, in file order, each accepted by
/// [`parses`]) with a fresh model of 2^`bits` slots.
pub fn encode_run(input: &[u8], frames: &[(usize, usize)], bits: u32) -> R<Vec<u8>> {
    let mut ctx = Ctx::default();
    let mut recs = Vec::new();
    let mut m = Model::new(bits);
    let mut enc = ArithEnc::new();
    code_num(&mut m, &mut enc, N_COUNT, frames.len() as u64);
    let mut prev_end = frames.first().map_or(0, |&(o, _)| o);
    code_num(&mut m, &mut enc, N_GAP, prev_end as u64);
    for (i, &(o, s)) in frames.iter().enumerate() {
        if i > 0 {
            code_num(&mut m, &mut enc, N_GAP, (o.checked_sub(prev_end).ok_or(Bad)?) as u64);
        }
        encode_frame(&input[o..o + s], &mut ctx, &mut m, &mut enc, &mut recs)?;
        prev_end = o + s;
    }
    Ok(enc.finish())
}

/// Development aid: the model cost of the frames' bits by syntax group, as (group, bits in
/// the frames, bits under the model), coding the run as [`encode_run`] would.
pub fn cost_breakdown(input: &[u8], frames: &[(usize, usize)], bits: u32) -> Vec<(String, u64, f64)> {
    let mut ctx = Ctx::default();
    let mut m = Model::new(bits);
    let mut recs = Vec::new();
    let mut tally: std::collections::BTreeMap<String, (u64, f64)> = std::collections::BTreeMap::new();
    for &(o, s) in frames {
        let frame = &input[o..o + s];
        recs.clear();
        let mut r = Reader { data: frame, pos: 32, recs: &mut recs };
        let Ok((size, used)) = parse_frame(&mut r, &mut ctx) else { continue };
        let n = size * 8 - 16;
        let mut prev = 2;
        for k in used..n {
            let b = (frame[k >> 3] >> (7 - (k & 7))) & 1;
            recs.push(Rec { h: aux_hint(k - used, n - k, prev), prob: 128, bit: b });
            prev = b as u32;
        }
        for r in &recs {
            let p = m.predict(&r.h, Some(0)) as f64 / 65536.0;
            let cost = -(if r.bit == 1 { p } else { 1.0 - p }).max(1e-9).log2();
            m.update(r.bit as u32);
            let name = match r.h.kind {
                kind::SIDE => format!("side {:02}", r.h.a >> 24),
                kind::EXP => format!("exp D{}", [0, 15, 25, 45][((r.h.a >> 8) & 3) as usize]),
                kind::MANT => format!("mant bap {:02}", (r.h.a >> 9) & 15),
                kind::MGRP => format!("mgrp bap {}", (r.h.a >> 9) & 15),
                _ => "aux".to_string(),
            };
            let e = tally.entry(name).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += cost;
        }
    }
    tally.into_iter().map(|(k, (n, c))| (k, n, c)).collect()
}

/// Decodes a run coded by [`encode_run`]: `(offset, frame bytes)` in order.
pub fn decode_run(data: &[u8], bits: u32, file_len: usize) -> R<Vec<(usize, Vec<u8>)>> {
    let mut ctx = Ctx::default();
    let mut m = Model::new(bits);
    let mut c = ArithDec::new(data);
    let n = code_num(&mut m, &mut c, N_COUNT, 0);
    if n > file_len as u64 / 128 + 1 {
        return Err(Bad);
    }
    let mut out = Vec::with_capacity(n as usize);
    let mut pos = code_num(&mut m, &mut c, N_GAP, 0) as usize;
    for i in 0..n {
        if i > 0 {
            pos = pos.checked_add(code_num(&mut m, &mut c, N_GAP, 0) as usize).ok_or(Bad)?;
        }
        let f = decode_frame(&mut ctx, &mut m, &mut c)?;
        if pos.checked_add(f.len()).is_none_or(|e| e > file_len) {
            return Err(Bad);
        }
        let len = f.len();
        out.push((pos, f));
        pos += len;
        if c.overrun() {
            return Err(Bad);
        }
    }
    Ok(out)
}

/// Finds the recodable AC-3 frames in `data[from..to]` (scanning for sync words; a frame
/// counts only if both CRCs check out).
pub fn scan(data: &[u8], from: usize, to: usize, out: &mut Vec<(usize, usize)>) {
    let mut i = from;
    while i + 8 <= to {
        if data[i] == 0x0B && data[i + 1] == 0x77 {
            if let Some(s) = frame_at(&data[i..to]) {
                out.push((i, s));
                i += s;
                continue;
            }
        }
        i += 1;
    }
}

/// Synthetic AC-3 frames for tests: the frame parser driven with random but valid field
/// values (the generator reads each field's element, width and context from its hint).
#[doc(hidden)]
pub mod synth {
    use super::{fill_crcs, kind, parse_frame, BitIo, Ctx};
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

    struct Gen<'a> {
        rng: &'a mut Rng,
        w: BitWriter,
        queue: VecDeque<u32>,
        acmod: u32,
        cplbeg: u32,
        frmsize: u32,
    }

    impl Gen<'_> {
        fn push(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                self.queue.push_back((v >> i) & 1);
            }
        }

        /// A value for a side field at its first bit.
        fn side(&mut self, elem: u32, idx: u32, blk: u32, n: u32) -> u32 {
            let r = &mut *self.rng;
            let any = r.step() & ((1u32 << n.min(31)) - 1);
            match elem {
                62 => self.frmsize & 63, // fscod 0
                2 => [6, 8][r.below(2) as usize],
                4 => {
                    self.acmod = r.below(8);
                    self.acmod
                }
                23 | 59 => (r.below(8) == 0) as u32,
                24 | 60 => r.below(3),
                30 | 37 | 47 | 49 | 53 => (blk == 0 || r.below(4) == 0) as u32,
                31 => (self.acmod >= 2 && r.below(3) != 0) as u32,
                34 => {
                    self.cplbeg = r.below(8);
                    self.cplbeg
                }
                35 => self.cplbeg.saturating_sub(2) + r.below(16 - self.cplbeg.saturating_sub(2)),
                43 => {
                    if idx == 6 {
                        (blk == 0 || r.below(2) == 0) as u32
                    } else if blk == 0 {
                        1 + r.below(3)
                    } else {
                        r.below(4)
                    }
                }
                44 => r.below(61),
                45 => {
                    if idx == 0 {
                        r.below(13)
                    } else {
                        r.below(16)
                    }
                }
                50 => 8 + r.below(30),
                55 => (r.below(4) == 0) as u32,
                56 => r.below(3),
                57 => r.below(3),
                58 => r.below(3) << 7 | r.below(5) << 3 | r.below(8),
                _ => any,
            }
        }
    }

    impl BitIo for Gen<'_> {
        fn bit(&mut self, h: &Hint) -> R<u32> {
            if self.queue.is_empty() {
                match h.kind {
                    kind::SIDE => {
                        let n = (h.d >> 8) & 31;
                        let v = self.side(h.a >> 24, (h.a >> 12) & 127, (h.c >> 20) & 7, n);
                        self.push(v, n);
                    }
                    kind::EXP => {
                        // three deltas that keep the exponent within 0..=24
                        let mut prev = ((h.b >> 7) & 31) as i32;
                        let mut v = 0;
                        for _ in 0..3 {
                            let lo = (prev - 2).max(0);
                            let hi = (prev + 2).min(24);
                            let e = lo + self.rng.below((hi - lo + 1) as u32) as i32;
                            v = v * 5 + (e - prev + 2) as u32;
                            prev = e;
                        }
                        self.push(v, 7);
                    }
                    _ => {
                        let b = self.rng.step() & 1;
                        self.queue.push_back(b);
                    }
                }
            }
            let b = self.queue.pop_front().unwrap_or(0);
            self.w.put(b);
            Ok(b)
        }
    }

    /// A valid frame of size code `frmsize` (48 kHz), or `None` when the random choices did
    /// not fit (try another seed).
    pub fn frame(rng: &mut Rng, frmsize: u32) -> Option<Vec<u8>> {
        let mut w = BitWriter::new();
        w.put_bits(0x0B77, 16);
        w.put_bits(0, 16);
        let mut g = Gen { rng, w, queue: VecDeque::new(), acmod: 0, cplbeg: 0, frmsize };
        let (size, used) = parse_frame(&mut g, &mut Ctx::default()).ok()?;
        let mut w = g.w;
        for _ in used..size * 8 {
            w.put(0);
        }
        let mut f = w.finish();
        fill_crcs(&mut f);
        Some(f)
    }

    /// `n` frames from a few streams (size codes), each found by trying seeds.
    pub fn frames(seed: u64, n: usize) -> Vec<Vec<u8>> {
        let mut rng = Rng(seed | 1);
        let mut out = Vec::new();
        while out.len() < n {
            let frm = [28, 30, 32, 34][out.len() % 4];
            for _ in 0..200 {
                if let Some(f) = frame(&mut rng, frm) {
                    out.push(f);
                    break;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_frames_roundtrip() {
        let frames = synth::frames(7, 24);
        let mut data = Vec::new();
        let mut pos = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            data.extend((0..(i * 13 % 29)).map(|k| (k * 7 + i) as u8)); // container bytes between frames
            assert_eq!(frame_at(f), Some(f.len()), "frame {i}");
            pos.push((data.len(), f.len()));
            data.extend_from_slice(f);
        }
        let mut found = Vec::new();
        scan(&data, 0, data.len(), &mut found);
        assert_eq!(found, pos);
        let mut probe = Probe::default();
        assert!(found.iter().all(|&(o, s)| parses(&data[o..o + s], &mut probe)));
        let coded = encode_run(&data, &found, 18).unwrap();
        let back = decode_run(&coded, 18, data.len()).unwrap();
        assert_eq!(back.len(), frames.len());
        for ((o, f), &(o2, s2)) in back.iter().zip(&found) {
            assert_eq!(*o, o2);
            assert_eq!(&f[..], &data[o2..o2 + s2]);
        }
    }

    #[test]
    fn crc1_is_recomputed() {
        // a fake "frame": random bytes, then both CRCs filled; recomputing them from a copy
        // with the CRC fields cleared must give the same bytes
        let size = FRAME_WORDS[20][0] as usize * 2;
        let mut f: Vec<u8> = (0..size).map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(7) as u8).collect();
        f[0] = 0x0B;
        f[1] = 0x77;
        fill_crcs(&mut f);
        assert_eq!(crc16(&f[2..]), 0);
        assert_eq!(crc16(&f[2..frame58(size)]), 0);
        let mut g = f.clone();
        g[2..4].fill(0);
        let n = g.len();
        g[n - 2..].fill(0);
        fill_crcs(&mut g);
        assert_eq!(f, g);
    }

    #[test]
    fn band_table_matches() {
        for bin in 0..253 {
            let b = bin_to_band(bin);
            assert!(BAND_START[b] as usize <= bin && bin < BAND_START[b + 1] as usize, "bin {bin}");
        }
    }
}
