//! Bit-level helpers for H.264 NAL units.

/// Any parse failure. Callers fall back to storing the data unchanged, so the reason does
/// not matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bad;

pub type R<T> = Result<T, Bad>;

/// Removes emulation-prevention bytes (`00 00 03` -> `00 00`), turning a NAL payload into
/// its RBSP.
pub fn unescape(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0;
    for &b in ebsp {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// Inserts emulation-prevention bytes the way the standard requires (7.4.1), including the
/// final `03` after an RBSP that ends in `00` (a trailing cabac_zero_word).
pub fn escape(rbsp: &[u8], out: &mut Vec<u8>) {
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    if rbsp.last() == Some(&0) {
        out.push(3);
    }
}

/// MSB-first reader over an RBSP.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn bit(&mut self) -> R<u32> {
        if self.pos >= self.data.len() * 8 {
            return Err(Bad);
        }
        let b = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(b as u32)
    }

    pub fn flag(&mut self) -> R<bool> {
        Ok(self.bit()? == 1)
    }

    pub fn u(&mut self, n: u32) -> R<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    pub fn ue(&mut self) -> R<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(Bad);
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let v = self.u(zeros)? as u64;
        u32::try_from((1u64 << zeros) - 1 + v).map_err(|_| Bad)
    }

    pub fn se(&mut self) -> R<i32> {
        let k = self.ue()? as i64;
        Ok(if k & 1 == 1 { ((k + 1) / 2) as i32 } else { -((k / 2) as i32) })
    }

    /// True while there is data before the rbsp_trailing_bits (the final `1` bit).
    pub fn more_rbsp_data(&self) -> bool {
        let mut i = self.data.len();
        while i > 0 && self.data[i - 1] == 0 {
            i -= 1;
        }
        if i == 0 {
            return false;
        }
        let tz = self.data[i - 1].trailing_zeros() as usize;
        let last_one = (i - 1) * 8 + (7 - tz);
        self.pos < last_one
    }
}

/// MSB-first writer.
#[derive(Default)]
pub struct BitWriter {
    pub buf: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn put(&mut self, b: u32) {
        self.acc = (self.acc << 1) | (b & 1);
        self.n += 1;
        if self.n == 8 {
            self.buf.push(self.acc as u8);
            self.acc = 0;
            self.n = 0;
        }
    }

    pub fn put_bits(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            self.put((v >> i) & 1);
        }
    }

    pub fn bit_len(&self) -> usize {
        self.buf.len() * 8 + self.n as usize
    }

    /// Pads with zero bits to a byte boundary and returns the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        while self.n != 0 {
            self.put(0);
        }
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_roundtrip() {
        let cases: [&[u8]; 5] = [
            &[0, 0, 0, 1, 2, 0, 0, 3],
            &[1, 2, 3],
            &[0, 0, 0, 0],
            &[5, 0, 0],
            &[0, 0, 4, 0, 0, 2, 0, 0],
        ];
        for rbsp in cases {
            let mut e = Vec::new();
            escape(rbsp, &mut e);
            assert_eq!(unescape(&e), rbsp, "{e:?}");
        }
        let mut e = Vec::new();
        escape(&[0, 0, 0, 0], &mut e);
        assert_eq!(e, [0, 0, 3, 0, 0, 3]);
    }

    #[test]
    fn exp_golomb() {
        let mut w = BitWriter::new();
        // ue: 0 -> 1, 1 -> 010, 4 -> 00101; se(-2) = ue(4)
        w.put_bits(0b1, 1);
        w.put_bits(0b010, 3);
        w.put_bits(0b00101, 5);
        w.put_bits(0b00101, 5);
        let b = w.finish();
        let mut r = BitReader::new(&b);
        assert_eq!(r.ue().unwrap(), 0);
        assert_eq!(r.ue().unwrap(), 1);
        assert_eq!(r.ue().unwrap(), 4);
        assert_eq!(r.se().unwrap(), -2);
    }
}
