//! The H.264 CABAC arithmetic decoding and encoding engines (9.3.1.2, 9.3.3.2, 9.3.4).
//!
//! Context states are kept by the caller as `pStateIdx << 1 | valMPS` so that the same
//! state array can drive a decoder, an encoder or a probability model.

use super::tables::{CTX_INIT, RANGE_LPS, TRANS_LPS, TRANS_MPS};
use crate::bits::BitWriter;

/// Contexts used by 4:2:0 / 4:2:2 frame and field coding (ctxIdx 0..459).
pub const NUM_CTX: usize = 460;

/// Initialises all context states for a slice (9.3.1.1). `model` is 0 for I slices and
/// `cabac_init_idc + 1` for P and B slices.
pub fn init_states(states: &mut [u8; NUM_CTX], slice_qp: i32, model: usize) {
    let qp = slice_qp.clamp(0, 51);
    for (i, s) in states.iter_mut().enumerate() {
        let [m, n] = CTX_INIT[i][model];
        let pre = (((m as i32 * qp) >> 4) + n as i32).clamp(1, 126);
        *s = if pre <= 63 { ((63 - pre) << 1) as u8 } else { (((pre - 64) << 1) | 1) as u8 };
    }
}

/// State after coding `bin` in a context whose state was `s`.
#[inline]
pub fn next_state(s: u8, bin: u32) -> u8 {
    let p = (s >> 1) as usize;
    let mps = (s & 1) as u32;
    if bin == mps {
        (TRANS_MPS[p] << 1) | mps as u8
    } else {
        let mps = if p == 0 { 1 - mps } else { mps };
        (TRANS_LPS[p] << 1) | mps as u8
    }
}

/// Reads an existing CABAC payload.
pub struct Decoder<'a> {
    data: &'a [u8],
    bitpos: usize,
    range: u32,
    offset: u32,
}

impl<'a> Decoder<'a> {
    /// Starts decoding at byte `start` of `data` (the first byte after cabac_alignment).
    pub fn new(data: &'a [u8], start: usize) -> Self {
        let mut d = Decoder { data, bitpos: start * 8, range: 510, offset: 0 };
        for _ in 0..9 {
            d.offset = (d.offset << 1) | d.bit();
        }
        d
    }

    #[inline]
    fn bit(&mut self) -> u32 {
        let i = self.bitpos >> 3;
        let b = if i < self.data.len() { (self.data[i] >> (7 - (self.bitpos & 7))) & 1 } else { 0 };
        self.bitpos += 1;
        b as u32
    }

    /// Starts a new CABAC substream (HEVC tiles and wavefront rows) after a terminating bin
    /// equal to 1: skips the alignment bits to the next byte and re-initialises the engine.
    pub fn restart(&mut self) {
        self.bitpos = (self.bitpos + 7) & !7;
        self.range = 510;
        self.offset = 0;
        for _ in 0..9 {
            self.offset = (self.offset << 1) | self.bit();
        }
    }

    /// True once the decoder has read well past the end of the payload, which only
    /// happens when the parse has gone wrong.
    pub fn overrun(&self) -> bool {
        self.bitpos > self.data.len() * 8 + 64
    }

    #[inline]
    pub fn decision(&mut self, s: u8) -> u32 {
        let p = (s >> 1) as usize;
        let mps = (s & 1) as u32;
        let lps = RANGE_LPS[p][((self.range >> 6) & 3) as usize] as u32;
        self.range -= lps;
        let bin = if self.offset >= self.range {
            self.offset -= self.range;
            self.range = lps;
            1 - mps
        } else {
            mps
        };
        while self.range < 256 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | self.bit();
        }
        bin
    }

    #[inline]
    pub fn bypass(&mut self) -> u32 {
        self.offset = (self.offset << 1) | self.bit();
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }

    #[inline]
    pub fn terminate(&mut self) -> u32 {
        self.range -= 2;
        if self.offset >= self.range {
            1
        } else {
            while self.range < 256 {
                self.range <<= 1;
                self.offset = (self.offset << 1) | self.bit();
            }
            0
        }
    }
}

/// Writes a CABAC payload exactly as the standard's encoder does.
pub struct Encoder {
    low: u32,
    range: u32,
    first: bool,
    outstanding: u32,
    pub w: BitWriter,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Encoder { low: 0, range: 510, first: true, outstanding: 0, w: BitWriter::new() }
    }

    #[inline]
    fn put(&mut self, b: u32) {
        if self.first {
            self.first = false;
        } else {
            self.w.put(b);
        }
        while self.outstanding > 0 {
            self.w.put(1 - b);
            self.outstanding -= 1;
        }
    }

    #[inline]
    fn renorm(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put(1);
            } else {
                self.low -= 256;
                self.outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    #[inline]
    pub fn decision(&mut self, s: u8, bin: u32) {
        let p = (s >> 1) as usize;
        let mps = (s & 1) as u32;
        let lps = RANGE_LPS[p][((self.range >> 6) & 3) as usize] as u32;
        self.range -= lps;
        if bin != mps {
            self.low += self.range;
            self.range = lps;
        }
        self.renorm();
    }

    #[inline]
    pub fn bypass(&mut self, bin: u32) {
        self.low <<= 1;
        if bin != 0 {
            self.low += self.range;
        }
        if self.low >= 1024 {
            self.put(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put(0);
        } else {
            self.low -= 512;
            self.outstanding += 1;
        }
    }

    #[inline]
    pub fn terminate(&mut self, bin: u32) {
        self.range -= 2;
        if bin != 0 {
            self.low += self.range;
            self.flush();
        } else {
            self.renorm();
        }
    }

    /// Starts a new CABAC substream after a terminating bin equal to 1 (which flushed the
    /// engine): pads with zero bits to a byte boundary (byte_alignment()) and re-initialises.
    pub fn restart(&mut self) {
        while self.w.bit_len() & 7 != 0 {
            self.w.put(0);
        }
        self.low = 0;
        self.range = 510;
        self.first = true;
        self.outstanding = 0;
    }

    fn flush(&mut self) {
        self.range = 2;
        self.renorm();
        let b = (self.low >> 9) & 1;
        self.put(b);
        self.w.put_bits(((self.low >> 7) & 3) | 1, 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_decoder_agree() {
        // Pseudo-random bins through a few contexts, then a terminating bin.
        let mut st = [0u8; NUM_CTX];
        init_states(&mut st, 26, 0);
        let mut seed = 12345u32;
        let mut bins = Vec::new();
        for i in 0..20000 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let kind = i % 3;
            let bin = if (seed >> 16) % 7 < 2 { 1 } else { 0 };
            bins.push((kind, (i % 40) as usize, bin));
        }
        let mut enc = Encoder::new();
        let mut s1 = st;
        for &(kind, ctx, bin) in &bins {
            match kind {
                0 | 1 => {
                    enc.decision(s1[ctx], bin);
                    s1[ctx] = next_state(s1[ctx], bin);
                }
                _ => enc.bypass(bin),
            }
            enc.terminate(0);
        }
        enc.terminate(1);
        let data = enc.w.finish();
        let mut dec = Decoder::new(&data, 0);
        let mut s2 = st;
        for &(kind, ctx, bin) in &bins {
            let got = match kind {
                0 | 1 => {
                    let b = dec.decision(s2[ctx]);
                    s2[ctx] = next_state(s2[ctx], b);
                    b
                }
                _ => dec.bypass(),
            };
            assert_eq!(got, bin);
            assert_eq!(dec.terminate(), 0);
        }
        assert_eq!(dec.terminate(), 1);
    }
}
