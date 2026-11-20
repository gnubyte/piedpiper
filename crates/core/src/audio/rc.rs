//! The residual coder of audio version 2: a small context-mixing model driving a binary
//! arithmetic coder. Integers only, so it is bit-exact everywhere.
//!
//! A residual `r` is binarised as its bit length `e` (0 for zero), the bits of `|r|` below
//! the leading one, and a sign. `e` is coded relative to `e0`, the bit length of a running
//! average of recent magnitudes: first "is `e ≥ e0`?", then unary steps up or down. Those
//! decisions, which carry most of the information, and the first bit under the leading one
//! each mix three adaptive models, conditioned on fast, slow and very fast averages of recent
//! magnitudes, through an online-learned logistic mixer (as in the PAQ family). The next bit
//! is modelled per (average, length); the rest per (length, position); the sign on the
//! previous residual's sign.
//!
//! Each block of frames starts with a header per channel: an adaptive flag, and for a block
//! predictor ([`super::lpc`]) its order, shift and coefficients as plain bits.

use super::lpc::{self, Lpc};

/// Probability counters adapt at rate 1/(n + 1.5) after n observations, down to 1/256.5.
const LIMIT: usize = 255;
/// Running averages of |r| (×16) move by 1/2^shift of the difference per residual.
const FAST: u32 = 4;
const SLOW: u32 = 7;
const VFAST: u32 = 1;
/// Mixer learning rate.
const LR: i32 = 16;
/// Context sizes: quarter-octave buckets of the fast average; positions of the slow and very
/// fast averages relative to `e0`; decisions about `e`; bit lengths.
const NQ: usize = 96;
const NR: usize = 32;
const NSLOT: usize = 17;
const NE: usize = 24;
/// Largest bit length the decoder accepts (residuals stay below 2^21).
const EMAX: u32 = 23;
/// Spread of mixer inputs ("stretch") is ±2047, about ±8 in natural log-odds.
const BIAS: i32 = 256;

#[derive(Clone, Copy)]
struct Bit {
    /// P(1) in 1/65536.
    p: u32,
    n: u32,
}

const NEW: Bit = Bit { p: 32768, n: 0 };

/// round(65536 / (n + 1.5)) for n = 0..=LIMIT.
fn rates() -> Vec<u32> {
    (0..=LIMIT as u32).map(|n| (131072 + (2 * n + 3) / 2) / (2 * n + 3)).collect()
}

#[inline]
fn learn(b: &mut Bit, bit: u32, rates: &[u32]) {
    let target = if bit != 0 { 65535i64 } else { 0 };
    let p = b.p as i64;
    b.p = (p + (((target - p) * rates[b.n as usize] as i64) >> 16)).clamp(32, 65504) as u32;
    if (b.n as usize) < LIMIT {
        b.n += 1;
    }
}

/// Binary arithmetic coding (the carry-less 32-bit coder of the PAQ family).
trait Coder {
    /// Codes `bit` (ignored when decoding) with P(1) = `p16` / 65536 and returns it.
    fn code_p(&mut self, p16: u32, bit: u32) -> u32;

    #[inline]
    fn code(&mut self, m: &mut Bit, bit: u32, rates: &[u32]) -> u32 {
        let b = self.code_p(m.p, bit);
        learn(m, b, rates);
        b
    }
}

struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Coder for Encoder {
    #[inline]
    fn code_p(&mut self, p16: u32, bit: u32) -> u32 {
        let mid = self.x1 + (((self.x2 - self.x1) as u64 * p16 as u64) >> 16) as u32;
        if bit != 0 {
            self.x2 = mid;
        } else {
            self.x1 = mid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
        bit
    }
}

struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    buf: &'a [u8],
    pos: usize,
}

impl Decoder<'_> {
    /// Past the end the stream reads as 0xFF bytes (the encoder's flush relies on it).
    #[inline]
    fn byte(&mut self) -> u32 {
        let b = self.buf.get(self.pos).copied().unwrap_or(0xFF);
        self.pos += 1;
        b as u32
    }
}

impl Coder for Decoder<'_> {
    #[inline]
    fn code_p(&mut self, p16: u32, _bit: u32) -> u32 {
        let mid = self.x1 + (((self.x2 - self.x1) as u64 * p16 as u64) >> 16) as u32;
        let bit = (self.x <= mid) as u32;
        if bit != 0 {
            self.x2 = mid;
        } else {
            self.x1 = mid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.byte();
        }
        bit
    }
}

/// 4096 / (1 + e^(-d/256)), from an interpolated table (integers only).
fn squash(d: i32) -> i32 {
    const T: [i32; 33] = [
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068,
        4079, 4085, 4089, 4092, 4093, 4094,
    ];
    if d > 2047 {
        return 4095;
    }
    if d < -2047 {
        return 1;
    }
    let w = d & 127;
    let i = ((d >> 7) + 16) as usize;
    ((T[i] * (128 - w) + T[i + 1] * w + 64) >> 7).clamp(1, 4095)
}

/// Inverse of [`squash`] on 12-bit probabilities.
fn stretch_table() -> Vec<i16> {
    let mut t = vec![2047i16; 4096];
    let mut next = 0usize;
    for d in -2047..=2047 {
        let v = squash(d) as usize;
        for slot in &mut t[next..=v] {
            *slot = d as i16;
        }
        next = v + 1;
    }
    t
}

/// Three models of one family of binary decisions, one per context.
struct Mixed {
    a: Vec<Bit>,
    b: Vec<Bit>,
    c: Vec<Bit>,
}

impl Mixed {
    fn new(slots: usize) -> Self {
        Mixed { a: vec![NEW; NQ * slots], b: vec![NEW; NR * slots], c: vec![NEW; NR * slots] }
    }
}

struct Channel {
    /// Running averages of |r| × 16: fast, slow, very fast.
    fast: i64,
    slow: i64,
    vfast: i64,
    /// Sign of the previous residual: 0 negative, 1 zero, 2 positive.
    sign: usize,
    e: Mixed,
    m1: Mixed,
    m2: Vec<Bit>,
    rest: Vec<Bit>,
    s: Vec<Bit>,
    /// Mixer weights (16.16) per selector: 4 inputs (three models and a bias).
    w: Vec<i32>,
    /// "This block uses a block predictor."
    flag: Bit,
}

impl Channel {
    fn new() -> Self {
        Channel {
            fast: 256,
            slow: 256,
            vfast: 256,
            sign: 1,
            e: Mixed::new(NSLOT),
            m1: Mixed::new(NE),
            m2: vec![NEW; NQ * NE * 2],
            rest: vec![NEW; NE * NE],
            s: vec![NEW; 3 * 8],
            w: vec![19660; (NSLOT + NE) * 4],
            flag: NEW,
        }
    }
}

#[inline]
fn bit_len(u: u64) -> u32 {
    64 - u.leading_zeros()
}

/// Quarter-octave bucket of an average (×16).
#[inline]
fn bucket(a: i64) -> usize {
    let v = a.max(0) as u64 + 16;
    let l = 63 - v.leading_zeros();
    let frac = (v >> (l - 2)) & 3;
    ((((l - 4) * 4) as u64) + frac).min(NQ as u64 - 1) as usize
}

/// The residual model of all channels.
struct Model {
    ch: Vec<Channel>,
    rates: Vec<u32>,
    stretch: Vec<i16>,
}

impl Model {
    fn new(channels: usize) -> Self {
        Model { ch: (0..channels).map(|_| Channel::new()).collect(), rates: rates(), stretch: stretch_table() }
    }

    /// Codes one decision with the mixed models at indices `ia`, `ib`, `ic` and mixer
    /// selector `sel`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn mixed<C: Coder>(coder: &mut C, m: &mut Mixed, w: &mut [i32], stretch: &[i16], rates: &[u32], i: [usize; 3], sel: usize, bit: u32) -> u32 {
        let st = [
            stretch[(m.a[i[0]].p >> 4) as usize] as i32,
            stretch[(m.b[i[1]].p >> 4) as usize] as i32,
            stretch[(m.c[i[2]].p >> 4) as usize] as i32,
            BIAS,
        ];
        let w = &mut w[sel * 4..sel * 4 + 4];
        let mut dot = 0i64;
        for k in 0..4 {
            dot += w[k] as i64 * st[k] as i64;
        }
        let p12 = squash((dot >> 16).clamp(-2047, 2047) as i32);
        let b = coder.code_p((p12 as u32) << 4, bit);
        let err = (((b as i32) << 12) - p12) * LR;
        for k in 0..4 {
            w[k] += (st[k] * err) >> 15;
        }
        learn(&mut m.a[i[0]], b, rates);
        learn(&mut m.b[i[1]], b, rates);
        learn(&mut m.c[i[2]], b, rates);
        b
    }

    /// Codes channel `c`'s block header (`lpc` is ignored when decoding) and returns it, or
    /// `None` if a decoded order is out of range.
    fn header<C: Coder>(&mut self, coder: &mut C, c: usize, lpc: Option<&Lpc>) -> Option<Option<Lpc>> {
        if coder.code(&mut self.ch[c].flag, lpc.is_some() as u32, &self.rates) == 0 {
            return Some(None);
        }
        let mut raw = |v: u64, bits: u32| -> u64 {
            let mut out = 0;
            for k in (0..bits).rev() {
                out |= (coder.code_p(32768, ((v >> k) & 1) as u32) as u64) << k;
            }
            out
        };
        let order = raw(lpc.map_or(0, |l| l.q.len() as u64), 6) as usize;
        if order > lpc::MAX_ORDER {
            return None;
        }
        let shift = raw(lpc.map_or(0, |l| l.shift as u64), 4) as u32;
        let width = 64 - lpc::COEF_BITS;
        let q = (0..order)
            .map(|j| {
                let v = raw(lpc.map_or(0, |l| l.q[j] as u64 & ((1 << lpc::COEF_BITS) - 1)), lpc::COEF_BITS);
                ((v << width) as i64) >> width
            })
            .collect();
        Some(Some(Lpc { q, shift }))
    }

    /// Codes residual `r` of channel `c` (the value is ignored when decoding) and returns it.
    #[inline]
    fn code<C: Coder>(&mut self, coder: &mut C, c: usize, r: i64) -> i64 {
        let Model { ch, rates, stretch } = self;
        let m = &mut ch[c];
        let u = r.unsigned_abs();
        let e = bit_len(u);
        let q = bucket(m.fast);
        let e0 = bit_len(((m.fast + 8) >> 4) as u64).min(EMAX - 1);
        let rel = |a: i64| (bucket(a) as i64 - 4 * e0 as i64 + 16).clamp(0, NR as i64 - 1) as usize;
        let (rb, rc) = (rel(m.slow), rel(m.vfast));
        let ctx = |s: usize, slots: usize| [q * slots + s, rb * slots + s, rc * slots + s];

        let up = if e0 == 0 { 1 } else { Self::mixed(coder, &mut m.e, &mut m.w, stretch, rates, ctx(0, NSLOT), 0, (e >= e0) as u32) };
        let mut len;
        if up != 0 {
            len = e0;
            while len < EMAX {
                let s = 1 + (len - e0).min(7) as usize;
                if Self::mixed(coder, &mut m.e, &mut m.w, stretch, rates, ctx(s, NSLOT), s, (e > len) as u32) == 0 {
                    break;
                }
                len += 1;
            }
        } else {
            len = e0 - 1;
            while len > 0 {
                let s = 9 + (e0 - 1 - len).min(7) as usize;
                if Self::mixed(coder, &mut m.e, &mut m.w, stretch, rates, ctx(s, NSLOT), s, (e < len) as u32) == 0 {
                    break;
                }
                len -= 1;
            }
        }

        let mut v: u64 = if len == 0 { 0 } else { 1 << (len - 1) };
        if len >= 2 {
            let k = len as usize;
            let b1 = Self::mixed(coder, &mut m.m1, &mut m.w, stretch, rates, ctx(k, NE), NSLOT + k, ((u >> (len - 2)) & 1) as u32);
            v |= (b1 as u64) << (len - 2);
            if len >= 3 {
                let b2 = coder.code(&mut m.m2[(q * NE + k) * 2 + b1 as usize], ((u >> (len - 3)) & 1) as u32, rates);
                v |= (b2 as u64) << (len - 3);
                for pos in (0..len - 3).rev() {
                    let b = coder.code(&mut m.rest[k * NE + pos as usize], ((u >> pos) & 1) as u32, rates);
                    v |= (b as u64) << pos;
                }
            }
        }
        let mut out = v as i64;
        if v != 0 && coder.code(&mut m.s[m.sign * 8 + len.min(7) as usize], (r < 0) as u32, rates) != 0 {
            out = -out;
        }

        let a = (v as i64) << 4;
        m.fast += (a - m.fast) >> FAST;
        m.slow += (a - m.slow) >> SLOW;
        m.vfast += (a - m.vfast) >> VFAST;
        m.sign = (1 + out.signum()) as usize;
        out
    }
}

/// Residual stream of `res` (frame-major: channel 0, then channel 1, of each frame), with
/// the block headers `plans` (one per block and channel, block-major). `progress(i)` is
/// called every so often with the number of residuals coded; returning `false` stops early.
pub(super) fn encode(res: &[i32], channels: usize, plans: &[Option<Lpc>], progress: &mut dyn FnMut(usize) -> bool) -> Option<Vec<u8>> {
    let mut model = Model::new(channels);
    let mut enc = Encoder { x1: 0, x2: u32::MAX, out: Vec::with_capacity(res.len() / 2) };
    let block = lpc::BLOCK * channels;
    for (i, &r) in res.iter().enumerate() {
        if i % (1 << 18) == 0 && !progress(i) {
            return None;
        }
        if i % block == 0 {
            for c in 0..channels {
                model.header(&mut enc, c, plans[i / block * channels + c].as_ref());
            }
        }
        model.code(&mut enc, i % channels, r as i64);
    }
    enc.out.push((enc.x1 >> 24) as u8);
    Some(enc.out)
}

/// Decodes residuals one at a time, in the order they were coded.
pub(super) struct ResidualDecoder<'a> {
    dec: Decoder<'a>,
    model: Model,
}

impl<'a> ResidualDecoder<'a> {
    pub(super) fn new(buf: &'a [u8], channels: usize) -> Self {
        let mut dec = Decoder { x1: 0, x2: u32::MAX, x: 0, buf, pos: 0 };
        for _ in 0..4 {
            dec.x = (dec.x << 8) | dec.byte();
        }
        ResidualDecoder { dec, model: Model::new(channels) }
    }

    /// The next residual, which belongs to channel `c`.
    #[inline]
    pub(super) fn next(&mut self, c: usize) -> i64 {
        self.model.code(&mut self.dec, c, 0)
    }

    /// Channel `c`'s next block header (`None` if corrupt).
    pub(super) fn header(&mut self, c: usize) -> Option<Option<Lpc>> {
        self.model.header(&mut self.dec, c, None)
    }

    /// Bytes read so far, including the read-ahead.
    pub(super) fn consumed(&self) -> usize {
        self.dec.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn roundtrip_with(res: &[i32], channels: usize, plans: &[Option<Lpc>]) -> usize {
        let bytes = encode(res, channels, plans, &mut |_| true).unwrap();
        let mut d = ResidualDecoder::new(&bytes, channels);
        let block = lpc::BLOCK * channels;
        for (i, &r) in res.iter().enumerate() {
            if i % block == 0 {
                for c in 0..channels {
                    assert_eq!(d.header(c), Some(plans[i / block * channels + c].clone()), "header at {i}");
                }
            }
            assert_eq!(d.next(i % channels), r as i64, "residual {i}");
        }
        bytes.len()
    }

    fn roundtrip(res: &[i32], channels: usize) -> usize {
        let blocks = res.len().div_ceil(lpc::BLOCK * channels);
        roundtrip_with(res, channels, &vec![None; blocks * channels])
    }

    #[test]
    fn block_headers_roundtrip() {
        let mut rng = Rng::new(3);
        let res: Vec<i32> = (0..3 * lpc::BLOCK * 2 + 100).map(|_| (rng.next_u64() % 41) as i32 - 20).collect();
        let full = Lpc { q: (0..lpc::MAX_ORDER as i64).map(|j| if j % 2 == 0 { 16383 - j } else { -16384 + j }).collect(), shift: 15 };
        let plans = vec![Some(full), None, None, Some(Lpc { q: vec![], shift: 0 }), Some(Lpc { q: vec![2, -1], shift: 0 }), None, None, Some(Lpc { q: vec![-7], shift: 3 })];
        roundtrip_with(&res, 2, &plans);
    }

    #[test]
    fn extremes_and_noise_roundtrip() {
        let mut rng = Rng::new(7);
        let mut res: Vec<i32> = vec![0; 1000];
        res.extend((0..5000).map(|_| (rng.next_u64() % 2001) as i32 - 1000));
        res.extend([(1 << 21) - 1, -(1 << 21) + 1, 1, -1, 0, 65535, -65536, 3, -2]);
        res.extend((0..3000).map(|i| ((rng.next_u64() % 64) as i32 - 32) << (i % 18)));
        res.extend(vec![0; 777]);
        for channels in [1, 2] {
            let n = res.len() / channels * channels;
            roundtrip(&res[..n], channels);
        }
        assert!(roundtrip(&[], 1) <= 1);
    }

    #[test]
    fn small_residuals_cost_little() {
        let mut rng = Rng::new(9);
        let res: Vec<i32> = (0..20000).map(|_| (rng.next_u64() % 3) as i32 - 1).collect();
        // Entropy of a uniform ternary source: log2(3) ≈ 1.585 bits per residual.
        assert!(roundtrip(&res, 1) * 8 < 20000 * 165 / 100);
        assert!(roundtrip(&vec![0; 100000], 2) < 200, "silence is almost free");
    }

    #[test]
    fn squash_and_stretch_are_monotonic_inverses() {
        let t = stretch_table();
        for d in -2047..2047 {
            assert!(squash(d) <= squash(d + 1));
        }
        for p in 1..4095usize {
            let d = t[p] as i32;
            assert!(squash(d) >= p as i32 || d == 2047, "p={p} d={d}");
            assert!(t[p] <= t[p + 1]);
        }
    }
}
