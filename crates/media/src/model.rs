//! Context-mixing bit model and binary arithmetic coder.
//!
//! Integer arithmetic only, so every platform computes identical probabilities (the same
//! rule PiedPiper's neural codec follows).
//!
//! For an H.264 bin the model mixes (in the logistic domain) CABAC's own probability for the
//! bin's context with six adaptive estimates indexed by hashes of the context and the
//! [`Hint`] the parser supplies, then refines the result with an adaptive probability map.
//! Because CABAC's estimate is one of the inputs, the mixer learns how far to trust it and the
//! result is rarely worse than the original coding.
//!
//! Speed: the hashed tables are far larger than the CPU caches, so a bin's table positions
//! ([`Slots`]) depend only on its hint and can be computed and prefetched early: the encoder
//! prefetches several recorded bins ahead, and the decoder prefetches the possible next bins
//! the parser announces. Macroblock-level syntax elements, which have few contexts, use a
//! small table that stays in the caches. Each update is applied lazily at the next prediction,
//! after that prediction's memory loads have started.

use crate::h264::mb::{kind, Hint};
use crate::h264::tables::RANGE_LPS;

// --- logistic helpers (12-bit probabilities, stretch domain scaled by 256) -----------------

const SQ_T: [i32; 33] = [
    1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975,
    4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
];

#[inline]
const fn squash(d: i32) -> i32 {
    if d > 2047 {
        return 4095;
    }
    if d < -2047 {
        return 1;
    }
    let w = d & 127;
    let i = ((d >> 7) + 16) as usize;
    (SQ_T[i] * (128 - w) + SQ_T[i + 1] * w + 64) >> 7
}

const fn build_stretch() -> [i16; 4096] {
    let mut t = [0i16; 4096];
    let mut pi = 0i32;
    let mut x = -2047i32;
    while x <= 2047 {
        let v = squash(x);
        let mut j = pi;
        while j <= v {
            t[j as usize] = x as i16;
            j += 1;
        }
        pi = v + 1;
        x += 1;
    }
    let mut j = pi;
    while j < 4096 {
        t[j as usize] = 2047;
        j += 1;
    }
    t
}

static STRETCH: [i16; 4096] = build_stretch();

#[inline]
fn stretch(p12: u32) -> i32 {
    STRETCH[p12.min(4095) as usize] as i32
}

/// Probability (16-bit) that a CABAC context in state `s` codes a 1.
const fn build_cabac_p() -> [u16; 128] {
    let mut t = [0u16; 128];
    let mut s = 0;
    while s < 128 {
        let p = s >> 1;
        // pLPS from the 4th column of rangeTabLPS (range ~480).
        let mut lps = (RANGE_LPS[p][3] as u32 * 65536 + 240) / 480;
        if lps < 64 {
            lps = 64;
        }
        if lps > 32768 {
            lps = 32768;
        }
        t[s] = if s & 1 == 1 { (65536 - lps) as u16 } else { lps as u16 };
        if t[s] == 0 {
            t[s] = 1;
        }
        s += 1;
    }
    t
}

static CABAC_P: [u16; 128] = build_cabac_p();

/// CABAC's probability (16-bit) of a 1 in context state `s` (for statistics).
pub fn cabac_p1(s: u8) -> u32 {
    CABAC_P[s as usize & 127] as u32
}

const fn build_recip() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        t[n] = (65536 * 2) / (2 * n as u32 + 3);
        n += 1;
    }
    t
}

static RECIP: [u32; 256] = build_recip();

// --- binary arithmetic coder ---------------------------------------------------------------

/// Carry-less binary arithmetic encoder (32-bit, 16-bit probabilities).
pub struct ArithEnc {
    low: u32,
    high: u32,
    pub out: Vec<u8>,
}

impl Default for ArithEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithEnc {
    pub fn new() -> Self {
        ArithEnc { low: 0, high: 0xFFFF_FFFF, out: Vec::new() }
    }

    /// Codes `bit` with `p1` = P(bit = 1) in 1..=65535.
    #[inline]
    pub fn encode(&mut self, bit: u32, p1: u32) {
        let range = self.high - self.low;
        let mid = self.low + ((range as u64 * p1 as u64) >> 16) as u32;
        if bit != 0 {
            self.high = mid;
        } else {
            self.low = mid + 1;
        }
        while (self.low ^ self.high) & 0xFF00_0000 == 0 {
            self.out.push((self.high >> 24) as u8);
            self.low <<= 8;
            self.high = (self.high << 8) | 0xFF;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        let v = self.low;
        self.out.extend_from_slice(&v.to_be_bytes());
        self.out
    }
}

/// Decoder matching [`ArithEnc`].
pub struct ArithDec<'a> {
    low: u32,
    high: u32,
    x: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> ArithDec<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = ArithDec { low: 0, high: 0xFFFF_FFFF, x: 0, data, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next() as u32;
        }
        d
    }

    #[inline]
    fn next(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    #[inline]
    pub fn decode(&mut self, p1: u32) -> u32 {
        let range = self.high - self.low;
        let mid = self.low + ((range as u64 * p1 as u64) >> 16) as u32;
        let bit = if self.x <= mid {
            self.high = mid;
            1
        } else {
            self.low = mid + 1;
            0
        };
        while (self.low ^ self.high) & 0xFF00_0000 == 0 {
            self.low <<= 8;
            self.high = (self.high << 8) | 0xFF;
            self.x = (self.x << 8) | self.next() as u32;
        }
        bit
    }

    /// True if the decoder consumed more input than exists (corrupt stream).
    pub fn overrun(&self) -> bool {
        self.pos > self.data.len() + 8
    }
}

// --- the model -----------------------------------------------------------------------------

/// Non-H.264 symbol classes (kinds above the H.264 syntax elements).
pub mod sym {
    pub const RAW_OTHER: u8 = 32;
    pub const RAW_NAL: u8 = 33;
    pub const NUM: u8 = 34;
    pub const HDR: u8 = 35;
    pub const FLAG: u8 = 36;
    pub const NALHDR: u8 = 37;
    pub const FIX: u8 = 38;
}

const N_SLOT: usize = 6;
const N_IN: usize = N_SLOT + 2;
const N_MIX: usize = 460 + 64;
const N_MIX2: usize = 64 * 256;
const APM_N: usize = 33;

/// Adaptation limit of the context slots (higher = slower, steadier statistics).
const LIMIT: usize = 30;
/// Mixer learning rates.
const LR1: i64 = 8;
const LR2: i64 = 8;
/// Valid hash-table sizes (log2 of the number of slots); stored in every stream.
pub const MIN_TABLE_BITS: u32 = 16;
pub const MAX_TABLE_BITS: u32 = 24;

/// Where the context slots of a bin live in the table. It depends only on the bin's
/// [`Hint`], so it can be computed, and the memory prefetched, before the bin is coded.
#[derive(Clone, Copy, Default)]
pub struct Slots {
    idx: [u32; N_SLOT],
    chk: [u32; N_SLOT],
    mix2: u32,
    /// In the small table (macroblock-level syntax elements) instead of the large one.
    small: bool,
}

/// Bin kinds whose contexts live in the small table: macroblock-level syntax elements with
/// few distinct contexts. A table that small stays in the CPU caches, so these bins (whose
/// contexts the decoder cannot know early enough to prefetch) rarely wait for memory, while
/// coefficient bins (most of the data, with many rare contexts) and motion vectors use the
/// large table.
const SMALL_KINDS: u64 = (1 << kind::SKIP)
    | (1 << kind::MB_TYPE)
    | (1 << kind::SUB_TYPE)
    | (1 << kind::REF)
    | (1 << kind::DQP)
    | (1 << kind::IPRED)
    | (1 << kind::CPRED)
    | (1 << kind::CBP_L)
    | (1 << kind::CBP_C)
    | (1 << kind::T8)
    | (1 << kind::CBF)
    | (1 << kind::MVD_SIGN);

/// log2 of the small table's size for a model whose large table has `2^bits` slots.
fn small_bits(bits: u32) -> u32 {
    (bits - 6).clamp(12, 18)
}

/// Distinguishes the slot types, so different context combinations of one bin never share
/// a table position systematically.
const SALT: [u64; N_SLOT] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
    0x4528_21E6_38D0_1377,
    0xBE54_66CF_34E9_0C6C,
];

/// True for bins coded with a single adaptive counter instead of the mix (terminating bins:
/// end_of_slice_flag and the I_PCM flag). They have no [`Slots`].
#[inline]
pub fn is_simple(h: &Hint) -> bool {
    h.kind == kind::END || h.kind == kind::PCM
}

#[inline(always)]
fn prefetch_line<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetching is only a hint to the CPU; it never faults, whatever the address.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(p as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: as above.
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

/// The adaptive model. One instance codes a whole file.
///
/// The update after a bin is deferred until the next prediction has computed its table
/// positions and started loading them, so the slow table reads overlap the update work. The
/// arithmetic, and therefore every coded byte, is the same as updating immediately.
pub struct Model {
    slots: Vec<u32>,
    small: Vec<u32>,
    weights: Vec<[i32; N_IN]>,
    weights2: Vec<[i32; N_IN]>,
    apm: Vec<u16>,
    bits: u32,
    sbits: u32,
    // state of the bin being coded
    cur: Slots,
    st: [i32; N_IN],
    mix: usize,
    apm_i: usize,
    apm_w: i32,
    p1: i32,
    p2: i32,
    p_fin: i32,
    /// Terminating bins (end_of_slice, I_PCM) use one adaptive counter per hint value
    /// instead of the full mix: they are almost always 0 and are 10-15% of all bins.
    simple: [u16; 8],
    simple_i: Option<usize>,
    /// The bin whose update is still to be applied (2 = none).
    pending: u32,
}

impl Model {
    /// A fresh model with `2^table_bits` context slots (4 bytes each), plus a small table
    /// (1/64 of that, at most 1 MB) for macroblock-level syntax elements.
    pub fn new(table_bits: u32) -> Self {
        let bits = table_bits.clamp(MIN_TABLE_BITS, MAX_TABLE_BITS);
        let mut w0 = [0i32; N_IN];
        w0[0] = 30000;
        for w in w0.iter_mut().take(N_IN - 1).skip(1) {
            *w = 8000;
        }
        let mut apm = vec![0u16; N_MIX * APM_N];
        for c in 0..N_MIX {
            for j in 0..APM_N {
                apm[c * APM_N + j] = (squash((j as i32 - 16) * 128) * 16) as u16;
            }
        }
        let sbits = small_bits(bits);
        Model {
            slots: vec![0x8000_0000; 1 << bits],
            small: vec![0x8000_0000; 1 << sbits],
            weights: vec![w0; N_MIX],
            weights2: vec![w0; N_MIX2],
            apm,
            bits,
            sbits,
            cur: Slots::default(),
            st: [0; N_IN],
            mix: 0,
            apm_i: 0,
            apm_w: 0,
            p1: 2048,
            p2: 2048,
            p_fin: 2048,
            simple: [2048; 8],
            simple_i: None,
            pending: 2,
        }
    }

    /// Table positions of the context slots of a bin with hint `h`.
    #[inline]
    pub fn slots(&self, h: &Hint) -> Slots {
        let k = h.kind as u64;
        let c = h.ctx as u64;
        let (a, b, hc, hd) = (h.a as u64, h.b as u64, h.c as u64, h.d as u64);
        let base = (k << 48 | c << 32).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let ha = (a ^ 0x5851_F42D_4C95_7F2D).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let hb = (b ^ 0x1405_7B7E_F767_814F).wrapping_mul(0x94D0_49BB_1331_11EB);
        let hcc = (hc ^ 0x2545_F491_4F6C_DD1D).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let hdd = (hd ^ 0x9FB2_1C65_1E98_DF25).wrapping_mul(0xA076_1D64_78BD_642F);
        let hk0 = (k << 48).wrapping_mul(0xE703_7ED1_A0B4_28DB);
        // Context combinations, one per slot (the products above are well mixed in their
        // high bits, which give the position; the next 8 bits are the slot's check byte).
        let keys: [u64; N_SLOT] = [
            base ^ ha.rotate_left(5) ^ hdd.rotate_left(29), // kind, ctx, a, d
            base ^ ha, // kind, ctx, a
            base ^ hcc.rotate_left(31), // kind, ctx, c
            base ^ ha ^ hb.rotate_left(7), // kind, ctx, a, b
            base ^ ha.rotate_left(47) ^ hcc.rotate_left(13) ^ hdd, // kind, ctx, a, c, d
            hk0 ^ ha.rotate_left(3) ^ hb.rotate_left(29) ^ hcc.rotate_left(41) ^ hdd.rotate_left(53), // kind, a, b, c, d (no ctx)
        ];
        let small = h.kind < 64 && (SMALL_KINDS >> h.kind) & 1 == 1;
        let shift = 64 - if small { self.sbits } else { self.bits };
        let mut s = Slots { mix2: (((h.kind as usize & 63) << 8) | (ha >> 56) as usize) as u32, small, ..Default::default() };
        for (i, key) in keys.iter().enumerate() {
            let x = key ^ SALT[i];
            s.idx[i] = (x >> shift) as u32;
            s.chk[i] = ((x >> (shift - 8)) & 0xFF) as u32;
        }
        s
    }

    /// Starts loading the table memory of `s` (a hint to the CPU; never changes results).
    #[inline]
    pub fn prefetch(&self, s: &Slots) {
        let base = if s.small { self.small.as_ptr() } else { self.slots.as_ptr() };
        for &i in &s.idx {
            prefetch_line(base.wrapping_add(i as usize));
        }
        prefetch_line(self.weights2.as_ptr().wrapping_add(s.mix2 as usize));
    }

    /// Probability (16-bit) that the next bin is 1. `state` is the CABAC context state, or
    /// `None` for bins CABAC does not model (bypass, terminate, non-H.264 data).
    #[inline]
    pub fn predict(&mut self, h: &Hint, state: Option<u8>) -> u32 {
        if is_simple(h) {
            return self.predict_simple(h);
        }
        let s = self.slots(h);
        self.predict_at(&s, h, state)
    }

    #[inline]
    fn predict_simple(&mut self, h: &Hint) -> u32 {
        self.flush();
        let i = ((h.kind == kind::PCM) as usize) << 2 | (h.a as usize & 3);
        self.simple_i = Some(i);
        (self.simple[i] as u32).clamp(1, 4095) << 4
    }

    /// [`Model::predict`] for a bin whose slots `s` = `self.slots(h)` were computed earlier.
    #[inline]
    pub fn predict_at(&mut self, s: &Slots, h: &Hint, state: Option<u8>) -> u32 {
        if is_simple(h) {
            return self.predict_simple(h);
        }
        self.prefetch(s);
        self.flush();
        self.simple_i = None;
        self.cur = *s;
        self.st[0] = match state {
            Some(s) => stretch((CABAC_P[s as usize] >> 4) as u32),
            None => 0,
        };
        let tab = if s.small { &self.small } else { &self.slots };
        for i in 0..N_SLOT {
            let v = tab[s.idx[i] as usize];
            // A slot owned by another context counts as empty.
            let p16 = if (v >> 8) & 0xFF == s.chk[i] { v >> 16 } else { 0x8000 };
            self.st[i + 1] = stretch(p16 >> 4);
        }
        self.st[N_IN - 1] = 256;
        self.mix = if state.is_some() { (h.ctx as usize).min(459) } else { 460 + (h.kind as usize & 63) };
        let dot = |w: &[i32; N_IN], st: &[i32; N_IN]| -> i32 {
            let mut s: i64 = 0;
            for i in 0..N_IN {
                s += st[i] as i64 * w[i] as i64;
            }
            (s >> 16).clamp(-2047, 2047) as i32
        };
        let d1 = dot(&self.weights[self.mix], &self.st);
        let d2 = dot(&self.weights2[s.mix2 as usize], &self.st);
        self.p1 = squash(d1);
        self.p2 = squash(d2);
        let d = (d1 + d2) >> 1;
        let p_mix = squash(d);
        // APM indexed by the CABAC context, interpolated on the stretched mixer output.
        let sv = d + 2048;
        let lo = (sv >> 7) as usize;
        self.apm_w = sv & 127;
        self.apm_i = self.mix * APM_N + lo.min(APM_N - 2);
        let pa = (self.apm[self.apm_i] as i32 * (128 - self.apm_w) + self.apm[self.apm_i + 1] as i32 * self.apm_w) >> 11;
        self.p_fin = ((p_mix + 3 * pa) >> 2).clamp(1, 4095);
        (self.p_fin as u32) << 4
    }

    /// Learns from the bin just predicted (applied before the next prediction).
    #[inline]
    pub fn update(&mut self, bit: u32) {
        self.pending = bit;
    }

    #[inline]
    fn flush(&mut self) {
        if self.pending < 2 {
            let bit = self.pending;
            self.pending = 2;
            self.apply(bit);
        }
    }

    #[inline]
    fn apply(&mut self, bit: u32) {
        if let Some(i) = self.simple_i {
            let p = self.simple[i] as i32;
            let t = if bit != 0 { 4095 } else { 0 };
            self.simple[i] = (p + ((t - p) >> 5)) as u16;
            return;
        }
        let limit = LIMIT;
        let target = if bit != 0 { 65535 } else { 0 };
        let tab = if self.cur.small { &mut self.small } else { &mut self.slots };
        for i in 0..N_SLOT {
            let chk = self.cur.chk[i];
            let s = &mut tab[self.cur.idx[i] as usize];
            let (p, n) = if (*s >> 8) & 0xFF == chk { ((*s >> 16) as i32, (*s & 0xFF) as usize) } else { (0x8000, 0) };
            let np = p + (((target - p) as i64 * RECIP[n] as i64) >> 16) as i32;
            let nn = if n < limit { n + 1 } else { n };
            *s = (np.clamp(16, 65519) as u32) << 16 | chk << 8 | nn as u32;
        }
        let err1 = ((bit as i32) << 12) - self.p1;
        let err2 = ((bit as i32) << 12) - self.p2;
        let (lr, lr2) = (LR1, LR2);
        let w = &mut self.weights[self.mix];
        for i in 0..N_IN {
            w[i] += ((self.st[i] as i64 * err1 as i64 * lr) >> 14) as i32;
        }
        let w = &mut self.weights2[self.cur.mix2 as usize];
        for i in 0..N_IN {
            w[i] += ((self.st[i] as i64 * err2 as i64 * lr2) >> 14) as i32;
        }
        let i0 = self.apm_i;
        let wt = self.apm_w;
        let a0 = self.apm[i0] as i32;
        self.apm[i0] = (a0 + (((target - a0) * (128 - wt)) >> 13)) as u16;
        let a1 = self.apm[i0 + 1] as i32;
        self.apm[i0 + 1] = (a1 + (((target - a1) * wt) >> 13)) as u16;
    }
}

/// Either side of the arithmetic coder, so the same code paths serve compression and
/// decompression.
pub trait Coder {
    /// Codes (encoder) or decodes (decoder) one bit with probability `p1` of a 1.
    fn code(&mut self, bit: u32, p1: u32) -> u32;
}

impl Coder for ArithEnc {
    #[inline]
    fn code(&mut self, bit: u32, p1: u32) -> u32 {
        self.encode(bit, p1);
        bit
    }
}

impl Coder for ArithDec<'_> {
    #[inline]
    fn code(&mut self, _bit: u32, p1: u32) -> u32 {
        self.decode(p1)
    }
}

/// Codes one bin through the model.
#[inline]
pub fn code_bin<C: Coder>(m: &mut Model, c: &mut C, h: &Hint, state: Option<u8>, bit: u32) -> u32 {
    let p = m.predict(h, state);
    let b = c.code(bit, p);
    m.update(b);
    b
}

/// Codes a flag with its own small context.
pub fn code_flag<C: Coder>(m: &mut Model, c: &mut C, id: u32, ctx: u32, bit: u32) -> u32 {
    code_bin(m, c, &Hint { ctx: 0, kind: sym::FLAG, a: id, b: ctx, c: 0, d: 0 }, None, bit)
}

/// Codes an unsigned number (Elias-gamma binarisation with adaptive bits).
pub fn code_num<C: Coder>(m: &mut Model, c: &mut C, id: u32, v: u64) -> u64 {
    let x = v.wrapping_add(1);
    let nbits = 64 - x.leading_zeros(); // 1..=64 when encoding
    // unary length
    let mut n = 1u32;
    loop {
        let more = (n < nbits) as u32;
        let b = code_bin(m, c, &Hint { ctx: 0, kind: sym::NUM, a: id, b: n.min(40), c: 0, d: 0 }, None, more);
        if b == 0 || n >= 64 {
            break;
        }
        n += 1;
    }
    let mut val = 1u64;
    for i in (0..n - 1).rev() {
        let bit = ((x >> i) & 1) as u32;
        let ctxv = (n << 8 | (n - 1 - i).min(7) << 4 | (val & 0xF) as u32).min(0xFFFF);
        let b = code_bin(m, c, &Hint { ctx: 0, kind: sym::NUM, a: id | 0x100, b: ctxv, c: 0, d: 0 }, None, bit);
        val = (val << 1) | b as u64;
    }
    val.wrapping_sub(1)
}

/// Order-0/1/2 context for coding raw bytes.
#[derive(Default, Clone, Copy)]
pub struct ByteCtx {
    c1: u32,
    c2: u32,
}

/// Codes one byte MSB first.
pub fn code_byte<C: Coder>(m: &mut Model, c: &mut C, kind_id: u8, bc: &mut ByteCtx, byte: u8) -> u8 {
    let mut c0 = 1u32;
    for i in (0..8).rev() {
        let bit = ((byte >> i) & 1) as u32;
        let h = Hint { ctx: c0 as u16, kind: kind_id, a: bc.c1 | c0 << 8, b: bc.c2 << 8 | bc.c1 | c0 << 16, c: 0, d: 0 };
        let b = code_bin(m, c, &h, None, bit);
        c0 = (c0 << 1) | b;
    }
    let v = (c0 & 0xFF) as u8;
    bc.c2 = bc.c1;
    bc.c1 = v as u32;
    v
}

/// Kinds used by H.264 bins are all below this value.
pub const H264_KINDS: usize = kind::COUNT;
