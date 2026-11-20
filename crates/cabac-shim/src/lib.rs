//! Adaptive binary arithmetic coding for preflate-rs, written for PiedPiper (MIT).
//!
//! preflate-rs (Apache-2.0) depends on the `cabac` crate, which is LGPL-3.0. PiedPiper links
//! statically into WebAssembly, Android, Python and C binaries, so the workspace replaces that
//! crate (`[patch.crates-io]` in the root `Cargo.toml`) with this one. It implements only the
//! interface preflate-rs calls — the [`CabacWriter`] / [`CabacReader`] traits and the
//! [`vp8`] writer/reader/context types — on top of an LZMA-style binary range coder (the
//! public-domain design from Igor Pavlov's LZMA SDK) with count-based adaptive
//! probabilities. Its output is not compatible with the original crate's; it only has to
//! agree with itself, and preflate's correction data never leaves a PiedPiper stream.
//!
//! Everything is integer arithmetic, so encoder and decoder agree on every platform.

use std::io::{self, Read, Write};

/// Longest unary value a reader accepts (preflate writes at most 32).
const MAX_UNARY: usize = 1 << 16;

fn ctx_index<const N: usize>(i: usize) -> usize {
    i.min(N - 1)
}

/// Writes context-modelled bits.
pub trait CabacWriter<CTX> {
    /// Codes `bit` with the adaptive probability in `context` (and updates it).
    fn put(&mut self, bit: bool, context: &mut CTX) -> io::Result<()>;

    /// Codes `bit` with probability one half.
    fn put_bypass(&mut self, bit: bool) -> io::Result<()>;

    /// Codes `value` in unary (`value` ones, then a zero); bit `i` uses context
    /// `min(i, N - 1)`.
    fn put_unary_encoded<const N: usize>(&mut self, value: usize, context: &mut [CTX; N]) -> io::Result<()> {
        if N == 0 {
            for _ in 0..value {
                self.put_bypass(true)?;
            }
            return self.put_bypass(false);
        }
        for i in 0..value {
            self.put(true, &mut context[ctx_index::<N>(i)])?;
        }
        self.put(false, &mut context[ctx_index::<N>(value)])
    }

    /// Codes the low `num_bits` bits of `bits`, most significant first; bit `i` (counting
    /// from the least significant) uses context `min(i, N - 1)`.
    fn put_n_bits<const N: usize>(&mut self, bits: u64, num_bits: usize, context: &mut [CTX; N]) -> io::Result<()> {
        if num_bits > 64 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "more than 64 bits"));
        }
        for i in (0..num_bits).rev() {
            let bit = (bits >> i) & 1 == 1;
            if N == 0 {
                self.put_bypass(bit)?;
            } else {
                self.put(bit, &mut context[ctx_index::<N>(i)])?;
            }
        }
        Ok(())
    }

    /// Flushes the coder. Nothing may be written afterwards.
    fn finish(&mut self) -> io::Result<()>;
}

/// Reads what a [`CabacWriter`] wrote, given the same sequence of contexts.
pub trait CabacReader<CTX> {
    fn get(&mut self, context: &mut CTX) -> io::Result<bool>;

    fn get_bypass(&mut self) -> io::Result<bool>;

    fn get_unary_encoded<const N: usize>(&mut self, context: &mut [CTX; N]) -> io::Result<usize> {
        let mut value = 0;
        loop {
            let bit = if N == 0 { self.get_bypass()? } else { self.get(&mut context[ctx_index::<N>(value)])? };
            if !bit {
                return Ok(value);
            }
            value += 1;
            if value > MAX_UNARY {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "unary value too large"));
            }
        }
    }

    fn get_n_bits<const N: usize>(&mut self, num_bits: usize, context: &mut [CTX; N]) -> io::Result<u64> {
        if num_bits > 64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "more than 64 bits"));
        }
        let mut v = 0u64;
        for i in (0..num_bits).rev() {
            let bit = if N == 0 { self.get_bypass()? } else { self.get(&mut context[ctx_index::<N>(i)])? };
            if bit {
                v |= 1 << i;
            }
        }
        Ok(v)
    }
}

/// The coder preflate-rs uses (named after the module it expects; the algorithm is an
/// LZMA-style range coder, not VP8's boolean coder).
pub mod vp8 {
    use super::*;

    const PROB_BITS: u32 = 12;
    const TOP: u32 = 1 << 24;
    /// Adaptation slows down until a context has seen this many bits.
    const RATE_LIMIT: u32 = 62;

    /// An adaptive estimate of P(bit = 0).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VP8Context {
        /// P(0) in units of 2⁻¹⁶.
        p0: u32,
        /// Bits seen so far (saturating at [`RATE_LIMIT`]).
        n: u32,
    }

    impl Default for VP8Context {
        fn default() -> Self {
            VP8Context { p0: 1 << 15, n: 0 }
        }
    }

    impl VP8Context {
        /// P(0) for the coder, in units of 2⁻¹², kept away from 0 and 1.
        #[inline]
        fn prob(&self) -> u32 {
            (self.p0 >> (16 - PROB_BITS)).clamp(1, (1 << PROB_BITS) - 1)
        }

        /// Moves the estimate towards the coded bit, by 1/(n + 2): the running frequency
        /// at first, then exponential forgetting once `n` saturates.
        #[inline]
        fn update(&mut self, bit: bool) {
            let d = self.n + 2;
            if bit {
                self.p0 -= self.p0 / d;
            } else {
                self.p0 += ((1 << 16) - self.p0) / d;
            }
            if self.n < RATE_LIMIT {
                self.n += 1;
            }
        }
    }

    pub struct VP8Writer<W: Write> {
        out: W,
        low: u64,
        range: u32,
        cache: u8,
        cache_size: u64,
        finished: bool,
    }

    impl<W: Write> VP8Writer<W> {
        pub fn new(out: W) -> io::Result<Self> {
            Ok(VP8Writer { out, low: 0, range: u32::MAX, cache: 0, cache_size: 1, finished: false })
        }

        #[inline]
        fn shift_low(&mut self) -> io::Result<()> {
            if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
                let carry = (self.low >> 32) as u8;
                let mut b = self.cache;
                loop {
                    self.out.write_all(&[b.wrapping_add(carry)])?;
                    b = 0xFF;
                    self.cache_size -= 1;
                    if self.cache_size == 0 {
                        break;
                    }
                }
                self.cache = (self.low >> 24) as u8;
            }
            self.cache_size += 1;
            self.low = (self.low & 0x00FF_FFFF) << 8;
            Ok(())
        }

        #[inline]
        fn encode(&mut self, bit: bool, p0: u32) -> io::Result<()> {
            debug_assert!(!self.finished);
            let bound = (self.range >> PROB_BITS) * p0;
            if bit {
                self.low += bound as u64;
                self.range -= bound;
            } else {
                self.range = bound;
            }
            while self.range < TOP {
                self.range <<= 8;
                self.shift_low()?;
            }
            Ok(())
        }
    }

    impl<W: Write> CabacWriter<VP8Context> for VP8Writer<W> {
        #[inline]
        fn put(&mut self, bit: bool, context: &mut VP8Context) -> io::Result<()> {
            self.encode(bit, context.prob())?;
            context.update(bit);
            Ok(())
        }

        fn put_bypass(&mut self, bit: bool) -> io::Result<()> {
            self.encode(bit, 1 << (PROB_BITS - 1))
        }

        fn finish(&mut self) -> io::Result<()> {
            if !self.finished {
                for _ in 0..5 {
                    self.shift_low()?;
                }
                self.finished = true;
            }
            self.out.flush()
        }
    }

    pub struct VP8Reader<R: Read> {
        input: R,
        code: u32,
        range: u32,
    }

    impl<R: Read> VP8Reader<R> {
        pub fn new(input: R) -> io::Result<Self> {
            let mut r = VP8Reader { input, code: 0, range: u32::MAX };
            // The first byte is the writer's initial (always zero) cache byte.
            for _ in 0..5 {
                r.code = (r.code << 8) | r.next_byte()? as u32;
            }
            Ok(r)
        }

        /// Next input byte; zeros past the end (like the writer's flush padding).
        #[inline]
        fn next_byte(&mut self) -> io::Result<u8> {
            let mut b = [0u8; 1];
            loop {
                match self.input.read(&mut b) {
                    Ok(0) => return Ok(0),
                    Ok(_) => return Ok(b[0]),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            }
        }

        #[inline]
        fn decode(&mut self, p0: u32) -> io::Result<bool> {
            let bound = (self.range >> PROB_BITS) * p0;
            let bit = if self.code < bound {
                self.range = bound;
                false
            } else {
                self.code -= bound;
                self.range -= bound;
                true
            };
            while self.range < TOP {
                self.range <<= 8;
                self.code = (self.code << 8) | self.next_byte()? as u32;
            }
            Ok(bit)
        }
    }

    impl<R: Read> CabacReader<VP8Context> for VP8Reader<R> {
        #[inline]
        fn get(&mut self, context: &mut VP8Context) -> io::Result<bool> {
            let bit = self.decode(context.prob())?;
            context.update(bit);
            Ok(bit)
        }

        fn get_bypass(&mut self) -> io::Result<bool> {
            self.decode(1 << (PROB_BITS - 1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::vp8::{VP8Context, VP8Reader, VP8Writer};
    use super::*;
    use std::io::Cursor;

    /// SplitMix64, for reproducible test data.
    fn rng(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn roundtrip_mixed_operations() {
        let mut seed = 7;
        let mut ops = Vec::new();
        for _ in 0..20_000 {
            let r = rng(&mut seed);
            ops.push(match r % 4 {
                0 => (0, (r >> 8) % 100 < 90, 0usize),  // skewed context bit
                1 => (1, (r >> 8) & 1 == 1, 0),         // bypass
                2 => (2, false, ((r >> 8) % 40) as usize), // unary
                _ => (3, false, ((r >> 8) % 33) as usize), // n bits
            });
        }
        let vals: Vec<u64> = (0..ops.len()).map(|_| rng(&mut seed)).collect();

        let mut buf = Vec::new();
        {
            let mut w = VP8Writer::new(&mut buf).unwrap();
            let mut c = VP8Context::default();
            let mut cu = [VP8Context::default(); 32];
            let mut cb = [VP8Context::default(); 32];
            for (k, &(op, bit, n)) in ops.iter().enumerate() {
                match op {
                    0 => w.put(bit, &mut c).unwrap(),
                    1 => w.put_bypass(bit).unwrap(),
                    2 => w.put_unary_encoded(n, &mut cu).unwrap(),
                    _ => w.put_n_bits(vals[k], n, &mut cb).unwrap(),
                }
            }
            w.finish().unwrap();
        }
        let mut r = VP8Reader::new(Cursor::new(&buf)).unwrap();
        let mut c = VP8Context::default();
        let mut cu = [VP8Context::default(); 32];
        let mut cb = [VP8Context::default(); 32];
        for (k, &(op, bit, n)) in ops.iter().enumerate() {
            match op {
                0 => assert_eq!(r.get(&mut c).unwrap(), bit),
                1 => assert_eq!(r.get_bypass().unwrap(), bit),
                2 => assert_eq!(r.get_unary_encoded(&mut cu).unwrap(), n),
                _ => assert_eq!(r.get_n_bits(n, &mut cb).unwrap(), vals[k] & ((1u128 << n) - 1) as u64),
            }
        }
    }

    #[test]
    fn skewed_bits_compress() {
        let mut buf = Vec::new();
        let mut w = VP8Writer::new(&mut buf).unwrap();
        let mut c = VP8Context::default();
        for i in 0..100_000 {
            w.put(i % 50 == 0, &mut c).unwrap();
        }
        w.finish().unwrap();
        // H(0.02) is about 0.14 bits per bit: 1767 bytes.
        assert!(buf.len() < 1950, "{}", buf.len());
        let mut r = VP8Reader::new(Cursor::new(&buf)).unwrap();
        let mut c = VP8Context::default();
        for i in 0..100_000 {
            assert_eq!(r.get(&mut c).unwrap(), i % 50 == 0);
        }
    }

    #[test]
    fn carries_propagate() {
        // Long runs of near-certain bits followed by surprises exercise the carry path.
        let mut seed = 3;
        let bits: Vec<bool> = (0..200_000).map(|i| if i % 5000 < 4990 { false } else { rng(&mut seed) & 1 == 1 }).collect();
        let mut buf = Vec::new();
        let mut w = VP8Writer::new(&mut buf).unwrap();
        let mut c = [VP8Context::default(); 2];
        for (i, &b) in bits.iter().enumerate() {
            w.put(b, &mut c[i & 1]).unwrap();
        }
        w.finish().unwrap();
        let mut r = VP8Reader::new(Cursor::new(&buf)).unwrap();
        let mut c = [VP8Context::default(); 2];
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(r.get(&mut c[i & 1]).unwrap(), b, "bit {i}");
        }
    }
}
