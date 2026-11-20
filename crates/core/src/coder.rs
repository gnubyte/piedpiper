//! Multi-symbol range coder (LZMA-style carry handling, 48-bit precision).
//!
//! Model probabilities become integer frequencies `max(1, floor(p * 2^24))` — the same
//! recipe as TufaZip's `make_freq_table` — so every symbol stays codable and the integer
//! table is identical in encoder and decoder. With the range kept in `[2^40, 2^48)` and
//! totals below `2^25`, the coding loss versus ideal `-log2 p` is negligible.

const RANGE_MAX: u64 = (1 << 48) - 1;
const TOP: u64 = 1 << 40;
const LOW_MASK: u64 = (1 << 48) - 1;
const FREQ_SCALE: f32 = 16_777_216.0; // 2^24

#[inline(always)]
pub fn freq(p: f32) -> u64 {
    let f = (p * FREQ_SCALE) as u32;
    (if f == 0 { 1 } else { f }) as u64
}

pub struct Encoder {
    low: u64,
    range: u64,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl Encoder {
    pub fn new(out: Vec<u8>) -> Self {
        Encoder { low: 0, range: RANGE_MAX, cache: 0, cache_size: 1, out }
    }

    #[inline]
    fn shift_low(&mut self) {
        if (self.low & LOW_MASK) < (0xFF << 40) || (self.low >> 48) != 0 {
            let carry = (self.low >> 48) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 40) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & ((1 << 40) - 1)) << 8;
    }

    #[inline]
    pub fn encode(&mut self, start: u64, size: u64, total: u64) {
        debug_assert!(size > 0 && start + size <= total);
        let r = self.range / total;
        self.low += r * start;
        self.range = r * size;
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Encodes `sym` with the distribution `probs`.
    #[inline]
    pub fn encode_symbol(&mut self, probs: &[f32], sym: usize) {
        let mut total = 0u64;
        let mut start = 0u64;
        for (s, &p) in probs.iter().enumerate() {
            if s == sym {
                start = total;
            }
            total += freq(p);
        }
        self.encode(start, freq(probs[sym]), total);
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..7 {
            self.shift_low();
        }
        self.out
    }
}

pub struct Decoder<'a> {
    code: u64,
    range: u64,
    r: u64,
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        // Byte 0 is the encoder's initial (always zero) cache byte.
        let mut d = Decoder { code: 0, range: RANGE_MAX, r: 1, data, pos: 1 };
        for _ in 0..6 {
            d.code = (d.code << 8) | d.next();
        }
        d
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b as u64
    }

    #[inline]
    pub fn decode_freq(&mut self, total: u64) -> u64 {
        self.r = self.range / total;
        let v = self.code / self.r;
        if v < total {
            v
        } else {
            total - 1
        }
    }

    #[inline]
    pub fn decode_update(&mut self, start: u64, size: u64) {
        self.code = self.code.wrapping_sub(self.r * start);
        self.range = self.r * size;
        while self.range < TOP {
            self.code = ((self.code << 8) | self.next()) & ((1 << 56) - 1);
            self.range <<= 8;
        }
    }

    /// Decodes one symbol with the distribution `probs`.
    #[inline]
    pub fn decode_symbol(&mut self, probs: &[f32]) -> usize {
        let mut total = 0u64;
        for &p in probs {
            total += freq(p);
        }
        let target = self.decode_freq(total);
        let mut cum = 0u64;
        for (s, &p) in probs.iter().enumerate() {
            let f = freq(p);
            if cum + f > target {
                self.decode_update(cum, f);
                return s;
            }
            cum += f;
        }
        unreachable!("target < total")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn roundtrip_random_distributions() {
        let mut rng = Rng::new(42);
        for &v in &[2usize, 3, 17, 256, 4097] {
            let n = 3000;
            let mut dists = Vec::new();
            let mut syms = Vec::new();
            for i in 0..n {
                // Mix of peaked, flat and degenerate distributions.
                let mut p: Vec<f32> = (0..v).map(|_| (rng.uniform() + 1.0).powi(8)).collect();
                if i % 7 == 0 {
                    p.iter_mut().for_each(|x| *x = 1e-12);
                    p[i % v] = 1.0;
                }
                let s: f32 = p.iter().sum();
                p.iter_mut().for_each(|x| *x /= s);
                let sym = if i % 11 == 0 { (rng.next_u64() % v as u64) as usize } else {
                    // sample roughly by probability
                    let u = (rng.uniform() + 1.0) / 2.0;
                    let mut c = 0.0;
                    let mut k = v - 1;
                    for (j, &pj) in p.iter().enumerate() { c += pj; if u < c { k = j; break; } }
                    k
                };
                dists.push(p);
                syms.push(sym);
            }
            let mut enc = Encoder::new(Vec::new());
            for (p, &s) in dists.iter().zip(&syms) {
                enc.encode_symbol(p, s);
            }
            let bytes = enc.finish();
            let mut dec = Decoder::new(&bytes);
            for (i, (p, &s)) in dists.iter().zip(&syms).enumerate() {
                assert_eq!(dec.decode_symbol(p), s, "vocab {v} symbol {i}");
            }
        }
    }

    #[test]
    fn near_ideal_cost() {
        // 10k fair bits must cost ~1250 bytes.
        let mut enc = Encoder::new(Vec::new());
        let p = [0.5f32, 0.5];
        let mut rng = Rng::new(1);
        for _ in 0..10_000 {
            enc.encode_symbol(&p, (rng.next_u64() & 1) as usize);
        }
        let n = enc.finish().len();
        assert!((1250..1262).contains(&n), "{n}");
    }
}
