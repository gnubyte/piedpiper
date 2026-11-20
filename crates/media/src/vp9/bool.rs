//! VP8/VP9 boolean arithmetic coder, bit-exact with libvpx (`vpx_dsp/bitreader.h`,
//! `vpx_dsp/bitwriter.{h,c}`). Probabilities are of a 0 bit, in 1/256.

/// Reads a boolean-coded partition. Bytes past the end read as zero, as in libvpx.
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    value: u32,
    bits: u32,
    range: u32,
}

impl<'a> BoolDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let b0 = data.first().copied().unwrap_or(0) as u32;
        let b1 = data.get(1).copied().unwrap_or(0) as u32;
        BoolDecoder { data, pos: 2, value: (b0 << 8) | b1, bits: 0, range: 255 }
    }

    #[inline]
    pub fn read(&mut self, prob: u8) -> u32 {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let big = split << 8;
        let bit = if self.value >= big {
            self.range -= split;
            self.value -= big;
            1
        } else {
            self.range = split;
            0
        };
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bits += 1;
            if self.bits == 8 {
                self.bits = 0;
                self.value |= self.data.get(self.pos).copied().unwrap_or(0) as u32;
                self.pos += 1;
            }
        }
        bit
    }

    pub fn literal(&mut self, n: u32) -> u32 {
        (0..n).fold(0, |v, _| (v << 1) | self.read(128))
    }

    /// True once the decoder has consumed well past the end (a broken parse).
    pub fn overrun(&self) -> bool {
        self.pos > self.data.len() + 8
    }
}

/// Writes a boolean-coded partition exactly like libvpx's `vpx_writer`.
pub struct BoolEncoder {
    low: u32,
    range: u32,
    count: i32,
    pub buf: Vec<u8>,
}

static NORM: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 1;
    while i < 256 {
        let mut n = 0;
        while (i << n) < 128 {
            n += 1;
        }
        t[i] = n as u8;
        i += 1;
    }
    t
};

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    /// Starts a partition (writes the marker bit, as `vpx_start_encode` does).
    pub fn new() -> Self {
        let mut e = BoolEncoder { low: 0, range: 255, count: -24, buf: Vec::new() };
        e.write(0, 128);
        e
    }

    /// Like [`BoolEncoder::new`] but without the marker bit, for streams whose first bit is
    /// replayed from a recording that already contains it.
    pub fn new_raw() -> Self {
        BoolEncoder { low: 0, range: 255, count: -24, buf: Vec::new() }
    }

    #[inline]
    pub fn write(&mut self, bit: u32, prob: u8) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let mut range = split;
        let mut low = self.low;
        if bit != 0 {
            low = low.wrapping_add(split);
            range = self.range - split;
        }
        let mut shift = NORM[range as usize] as i32;
        range <<= shift;
        let mut count = self.count + shift;
        if count >= 0 {
            let offset = shift - count;
            if (low << (offset - 1)) & 0x8000_0000 != 0 {
                let mut x = self.buf.len();
                while x > 0 && self.buf[x - 1] == 0xFF {
                    self.buf[x - 1] = 0;
                    x -= 1;
                }
                if x > 0 {
                    self.buf[x - 1] = self.buf[x - 1].wrapping_add(1);
                }
            }
            self.buf.push(((low >> (24 - offset)) & 0xFF) as u8);
            low <<= offset;
            shift = count;
            low &= 0xFF_FFFF;
            count -= 8;
        }
        low <<= shift;
        self.count = count;
        self.low = low;
        self.range = range;
    }

    /// Ends a VP8 partition like libvpx's `vp8_stop_encode` (32 zero bits, no padding byte).
    pub fn finish_vp8(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write(0, 128);
        }
        self.buf
    }

    /// Ends the partition like `vpx_stop_encode` and returns the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write(0, 128);
        }
        if self.buf.last().is_some_and(|&b| b & 0xE0 == 0xC0) {
            self.buf.push(0);
        }
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut s = 1u32;
        let bits: Vec<(u32, u8)> = (0..50_000)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let p = ((s >> 8) % 255 + 1) as u8;
                let b = ((s >> 20) % 256 >= p as u32) as u32;
                (b, p)
            })
            .collect();
        let mut e = BoolEncoder::new();
        for &(b, p) in &bits {
            e.write(b, p);
        }
        let data = e.finish();
        let mut d = BoolDecoder::new(&data);
        assert_eq!(d.read(128), 0, "marker");
        for &(b, p) in &bits {
            assert_eq!(d.read(p), b);
        }
    }
}
