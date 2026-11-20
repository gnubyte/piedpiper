//! Reversible 2-D pixel transform for uncompressed images (and PNG scanlines).
//!
//! The Transformer reads a file as a 1-D byte stream with a short window (128 tokens at
//! level 2, about 42 RGB pixels), so it never sees the pixel directly above the one it
//! predicts. This transform rewrites the pixel bytes before tokenizing:
//!
//! 1. **Colour decorrelation** ("subtract green", as in WebP lossless): with three or more
//!    channels, channels 0 and 2 become their difference to channel 1 (+128). Channel 1 is
//!    green in RGB and BGR order alike; a fourth (alpha) channel is left alone.
//! 2. **Spatial prediction** (photographic images only): each sample becomes its residual
//!    against the LOCO-I / JPEG-LS median edge predictor (MED) of its left, upper and
//!    upper-left neighbours in the same channel. Flat images (screenshots, diagrams, where
//!    most pixels repeat their left neighbour) skip this step: the model does better on
//!    their colour values than on residuals.
//!
//! Bytes outside the pixel rows (file headers, row padding, PNG filter-type bytes, trailing
//! data) pass through unchanged, so the transformed data has the same length. For PNG, the
//! scanlines are unfiltered to pixels first and re-filtered with their original filter types
//! on decode, which reproduces the inflated IDAT bytes exactly. Integer arithmetic only, so
//! the transform is bit-exact on every platform.

use crate::error::{Error, Result};
use crate::format::{get_varint, put_varint};

/// Spatial predictor applied after the colour decorrelation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predictor {
    /// Colour decorrelation only.
    Color = 0,
    /// Colour decorrelation, then MED residuals.
    ColorMed = 1,
}

/// Where the pixels are in a file and how to transform them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageLayout {
    /// Bytes before the first row (kept verbatim).
    pub header_len: u64,
    pub width: u32,
    pub height: u32,
    /// Samples per pixel, 1..=4, eight bits each.
    pub channels: u8,
    /// Bytes per row in the file, including padding and (for PNG) the filter-type byte.
    pub stride: u64,
    pub predictor: Predictor,
    /// Rows are PNG scanlines: a filter-type byte (0..=4) followed by filtered samples.
    pub png: bool,
}

/// Images smaller than this (in pixel bytes) are not worth transforming.
const MIN_PIXEL_BYTES: u64 = 4096;
/// Images where at least this share of pixels (in 1/256) repeats its left neighbour are
/// treated as flat (colour decorrelation only).
const FLAT_SHARE_256: u64 = 128;

const FLAG_MED: u8 = 1;
const FLAG_PNG: u8 = 2;

impl ImageLayout {
    fn row_bytes(&self) -> usize {
        self.width as usize * self.channels as usize
    }

    fn lead(&self) -> usize {
        usize::from(self.png)
    }

    /// Checks the layout against a buffer of `len` bytes.
    fn check(&self, len: usize) -> Result<()> {
        let bad = |m: &str| Err(Error::Format(format!("image layout: {m}")));
        if !(1..=4).contains(&self.channels) || self.width == 0 || self.height == 0 {
            return bad("bad dimensions");
        }
        let need_row = self.row_bytes() as u64 + self.lead() as u64;
        if self.stride < need_row {
            return bad("row stride too small");
        }
        let end = (self.height as u64).checked_mul(self.stride).and_then(|b| b.checked_add(self.header_len));
        match end {
            Some(e) if e <= len as u64 => Ok(()),
            _ => bad("pixel rows extend past the end of the data"),
        }
    }

    /// Serialises the layout (stored in the stream header).
    pub fn write(&self, out: &mut Vec<u8>) {
        put_varint(out, self.header_len);
        put_varint(out, self.width as u64);
        put_varint(out, self.height as u64);
        out.push(self.channels);
        put_varint(out, self.stride);
        let mut flags = 0u8;
        if self.predictor == Predictor::ColorMed {
            flags |= FLAG_MED;
        }
        if self.png {
            flags |= FLAG_PNG;
        }
        out.push(flags);
    }

    /// Reads a layout written by [`ImageLayout::write`].
    pub fn read(buf: &[u8], pos: &mut usize) -> Result<ImageLayout> {
        let small = |v: u64| u32::try_from(v).map_err(|_| Error::Format("image layout: value out of range".into()));
        let header_len = get_varint(buf, pos)?;
        let width = small(get_varint(buf, pos)?)?;
        let height = small(get_varint(buf, pos)?)?;
        let channels = *buf.get(*pos).ok_or_else(|| Error::Format("truncated header".into()))?;
        *pos += 1;
        let stride = get_varint(buf, pos)?;
        let flags = *buf.get(*pos).ok_or_else(|| Error::Format("truncated header".into()))?;
        *pos += 1;
        if flags & !(FLAG_MED | FLAG_PNG) != 0 {
            return Err(Error::Format("image layout: unknown flags".into()));
        }
        let l = ImageLayout {
            header_len,
            width,
            height,
            channels,
            stride,
            predictor: if flags & FLAG_MED != 0 { Predictor::ColorMed } else { Predictor::Color },
            png: flags & FLAG_PNG != 0,
        };
        if !(1..=4).contains(&channels) || width == 0 || height == 0 || stride < l.row_bytes() as u64 + l.lead() as u64 {
            return Err(Error::Format("image layout: bad dimensions".into()));
        }
        Ok(l)
    }
}

#[inline]
fn med(a: u8, b: u8, c: u8) -> u8 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if c >= hi {
        lo
    } else if c <= lo {
        hi
    } else {
        (a as i32 + b as i32 - c as i32) as u8
    }
}

#[inline]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i32 + b as i32 - c as i32;
    let (pa, pb, pc) = ((p - a as i32).abs(), (p - b as i32).abs(), (p - c as i32).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// PNG filter prediction for sample `i` of a row (`cur` holds the raw samples before `i`).
#[inline]
fn png_pred(kind: u8, cur: &[u8], prev: &[u8], i: usize, bpp: usize) -> u8 {
    let a = if i >= bpp { cur[i - bpp] } else { 0 };
    let b = prev[i];
    let c = if i >= bpp { prev[i - bpp] } else { 0 };
    match kind {
        1 => a,
        2 => b,
        3 => ((a as u16 + b as u16) / 2) as u8,
        4 => paeth(a, b, c),
        _ => 0,
    }
}

/// The raw samples of every row (PNG scanlines unfiltered), `height * row_bytes` bytes.
fn gather(data: &[u8], l: &ImageLayout) -> Result<Vec<u8>> {
    let rb = l.row_bytes();
    let mut px = vec![0u8; rb * l.height as usize];
    let zero = vec![0u8; rb];
    for y in 0..l.height as usize {
        let row = l.header_len as usize + y * l.stride as usize;
        let src = &data[row + l.lead()..row + l.lead() + rb];
        let (done, rest) = px.split_at_mut(y * rb);
        let cur = &mut rest[..rb];
        if !l.png {
            cur.copy_from_slice(src);
            continue;
        }
        let kind = data[row];
        if kind > 4 {
            return Err(Error::Format("image layout: invalid PNG filter type".into()));
        }
        let prev = if y == 0 { &zero[..] } else { &done[(y - 1) * rb..] };
        for i in 0..rb {
            let p = png_pred(kind, cur, prev, i, l.channels as usize);
            cur[i] = src[i].wrapping_add(p);
        }
    }
    Ok(px)
}

/// Writes raw samples back into the row layout (PNG rows re-filtered with their stored type).
fn scatter(px: &[u8], out: &mut [u8], l: &ImageLayout) -> Result<()> {
    let rb = l.row_bytes();
    let zero = vec![0u8; rb];
    for y in 0..l.height as usize {
        let row = l.header_len as usize + y * l.stride as usize;
        let cur = &px[y * rb..(y + 1) * rb];
        let dst_start = row + l.lead();
        if !l.png {
            out[dst_start..dst_start + rb].copy_from_slice(cur);
            continue;
        }
        let kind = out[row];
        if kind > 4 {
            return Err(Error::Format("image layout: invalid PNG filter type".into()));
        }
        let prev = if y == 0 { &zero[..] } else { &px[(y - 1) * rb..y * rb] };
        for i in 0..rb {
            let p = png_pred(kind, cur, prev, i, l.channels as usize);
            out[dst_start + i] = cur[i].wrapping_sub(p);
        }
    }
    Ok(())
}

/// Forward transform of the raw samples in place.
fn forward(px: &mut [u8], l: &ImageLayout) {
    let ch = l.channels as usize;
    if ch >= 3 {
        for p in px.chunks_exact_mut(ch) {
            p[0] = p[0].wrapping_sub(p[1]).wrapping_add(128);
            p[2] = p[2].wrapping_sub(p[1]).wrapping_add(128);
        }
    }
    if l.predictor != Predictor::ColorMed {
        return;
    }
    // Residuals, computed back to front so every prediction reads untouched neighbours.
    let rb = l.row_bytes();
    for i in (0..px.len()).rev() {
        let p = predict(px, i, rb, ch);
        px[i] = px[i].wrapping_sub(p);
    }
}

/// Inverse of [`forward`].
fn inverse(px: &mut [u8], l: &ImageLayout) {
    let ch = l.channels as usize;
    if l.predictor == Predictor::ColorMed {
        let rb = l.row_bytes();
        for i in 0..px.len() {
            let p = predict(px, i, rb, ch);
            px[i] = px[i].wrapping_add(p);
        }
    }
    if ch >= 3 {
        for p in px.chunks_exact_mut(ch) {
            p[0] = p[0].wrapping_add(p[1]).wrapping_sub(128);
            p[2] = p[2].wrapping_add(p[1]).wrapping_sub(128);
        }
    }
}

/// MED prediction of sample `i` from samples already reconstructed (left / up / up-left).
#[inline]
fn predict(px: &[u8], i: usize, rb: usize, ch: usize) -> u8 {
    let x = i % rb;
    let has_left = x >= ch;
    let has_up = i >= rb;
    match (has_left, has_up) {
        (false, false) => 128,
        (true, false) => px[i - ch],
        (false, true) => px[i - rb],
        (true, true) => med(px[i - ch], px[i - rb], px[i - rb - ch]),
    }
}

/// Transforms `data` (same length out). Only the pixel samples change.
pub fn encode(data: &[u8], l: &ImageLayout) -> Result<Vec<u8>> {
    l.check(data.len())?;
    let mut px = gather(data, l)?;
    forward(&mut px, l);
    let mut out = data.to_vec();
    let rb = l.row_bytes();
    for y in 0..l.height as usize {
        let at = l.header_len as usize + y * l.stride as usize + l.lead();
        out[at..at + rb].copy_from_slice(&px[y * rb..(y + 1) * rb]);
    }
    Ok(out)
}

/// Inverse of [`encode`].
pub fn decode(data: &[u8], l: &ImageLayout) -> Result<Vec<u8>> {
    l.check(data.len())?;
    let rb = l.row_bytes();
    let mut px = vec![0u8; rb * l.height as usize];
    for y in 0..l.height as usize {
        let at = l.header_len as usize + y * l.stride as usize + l.lead();
        px[y * rb..(y + 1) * rb].copy_from_slice(&data[at..at + rb]);
    }
    inverse(&mut px, l);
    let mut out = data.to_vec();
    scatter(&px, &mut out, l)?;
    Ok(out)
}

/// Chooses the predictor from the pixels: flat images get colour decorrelation only.
fn choose_predictor(px: &[u8], width: usize, height: usize, ch: usize) -> Predictor {
    if ch < 3 {
        return Predictor::ColorMed;
    }
    let rb = width * ch;
    let step = (height / 256).max(1);
    let (mut same, mut n) = (0u64, 0u64);
    for y in (0..height).step_by(step) {
        let row = &px[y * rb..(y + 1) * rb];
        for x in 1..width {
            n += 1;
            same += u64::from(row[x * ch..(x + 1) * ch] == row[(x - 1) * ch..x * ch]);
        }
    }
    if n > 0 && same * 256 >= n * FLAT_SHARE_256 {
        Predictor::Color
    } else {
        Predictor::ColorMed
    }
}

/// Finishes a candidate layout: size checks, then the predictor choice.
#[cfg_attr(not(feature = "deflate"), allow(dead_code))]
pub(crate) fn finish(data: &[u8], mut l: ImageLayout) -> Option<ImageLayout> {
    if l.width < 8 || l.height < 2 || (l.row_bytes() as u64) * (l.height as u64) < MIN_PIXEL_BYTES {
        return None;
    }
    l.check(data.len()).ok()?;
    let px = gather(data, &l).ok()?;
    l.predictor = choose_predictor(&px, l.width as usize, l.height as usize, l.channels as usize);
    Some(l)
}

/// Recognises uncompressed BMP, PPM/PGM (binary), PAM and TGA images with 8-bit samples.
pub fn detect(data: &[u8]) -> Option<ImageLayout> {
    let l = detect_bmp(data).or_else(|| detect_pnm(data)).or_else(|| detect_tga(data))?;
    finish(data, l)
}

/// Layout of a PNG's inflated IDAT data (8-bit, non-interlaced, non-palette images only).
#[cfg_attr(not(feature = "deflate"), allow(dead_code))]
pub fn png_layout(width: u32, height: u32, bit_depth: u8, color_type: u8, interlace: u8, inflated: &[u8]) -> Option<ImageLayout> {
    let channels = match color_type {
        0 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        _ => return None, // palette (3) indices are not colours
    };
    if bit_depth != 8 || interlace != 0 || width == 0 || height == 0 {
        return None;
    }
    let stride = width as u64 * channels + 1;
    if stride.checked_mul(height as u64)? != inflated.len() as u64 {
        return None;
    }
    let l = ImageLayout {
        header_len: 0,
        width,
        height,
        channels: channels as u8,
        stride,
        predictor: Predictor::ColorMed,
        png: true,
    };
    finish(inflated, l)
}

/// Layout for the inflated IDAT data `plain` of the PNG file `png` (from its IHDR chunk), if
/// `plain` is exactly the image's scanlines.
#[cfg_attr(not(feature = "deflate"), allow(dead_code))]
pub fn png_layout_for(png: &[u8], plain: &[u8]) -> Option<ImageLayout> {
    if png.len() < 33 || &png[..8] != b"\x89PNG\r\n\x1a\n" || &png[12..16] != b"IHDR" {
        return None;
    }
    let be = |i: usize| u32::from_be_bytes([png[i], png[i + 1], png[i + 2], png[i + 3]]);
    png_layout(be(16), be(20), png[24], png[25], png[28], plain)
}

fn u16le(d: &[u8], i: usize) -> Option<u32> {
    Some(u16::from_le_bytes(d.get(i..i + 2)?.try_into().ok()?) as u32)
}

fn u32le(d: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(i..i + 4)?.try_into().ok()?))
}

fn detect_bmp(d: &[u8]) -> Option<ImageLayout> {
    if d.len() < 54 || &d[..2] != b"BM" {
        return None;
    }
    let offset = u32le(d, 10)? as u64;
    let info = u32le(d, 14)?;
    let width = u32le(d, 18)? as i32;
    let height = u32le(d, 22)? as i32;
    let planes = u16le(d, 26)?;
    let bpp = u16le(d, 28)?;
    let compression = u32le(d, 30)?;
    if info < 40 || planes != 1 || width <= 0 || height == 0 || height == i32::MIN {
        return None;
    }
    let channels = match (bpp, compression) {
        (24, 0) => 3,
        (32, 0) | (32, 3) => 4,
        _ => return None, // palette, 16-bit and RLE images
    };
    let width = width as u32;
    Some(ImageLayout {
        header_len: offset,
        width,
        height: height.unsigned_abs(),
        channels,
        stride: (width as u64 * bpp as u64).div_ceil(32) * 4,
        predictor: Predictor::ColorMed,
        png: false,
    })
}

/// Reads the next whitespace-separated header token of a PNM file, skipping comments.
fn pnm_token(d: &[u8], pos: &mut usize) -> Option<u64> {
    loop {
        match d.get(*pos)? {
            b'#' => {
                while *d.get(*pos)? != b'\n' {
                    *pos += 1;
                }
            }
            c if c.is_ascii_whitespace() => *pos += 1,
            _ => break,
        }
    }
    let start = *pos;
    while d.get(*pos).is_some_and(|c| c.is_ascii_digit()) {
        *pos += 1;
    }
    if *pos == start || *pos - start > 9 {
        return None;
    }
    std::str::from_utf8(&d[start..*pos]).ok()?.parse().ok()
}

fn detect_pnm(d: &[u8]) -> Option<ImageLayout> {
    if d.len() < 8 || d[0] != b'P' {
        return None;
    }
    let (width, height, channels, maxval, header_len) = match d[1] {
        b'5' | b'6' => {
            let mut pos = 2;
            let w = pnm_token(d, &mut pos)?;
            let h = pnm_token(d, &mut pos)?;
            let m = pnm_token(d, &mut pos)?;
            if !d.get(pos)?.is_ascii_whitespace() {
                return None;
            }
            (w, h, if d[1] == b'5' { 1 } else { 3 }, m, pos + 1)
        }
        b'7' => {
            let end = d.windows(7).take(4096).position(|w| w == b"ENDHDR\n")?;
            let text = std::str::from_utf8(&d[2..end]).ok()?;
            let (mut w, mut h, mut c, mut m) = (0, 0, 0, 0);
            for line in text.lines() {
                let mut it = line.split_ascii_whitespace();
                let v = |it: &mut std::str::SplitAsciiWhitespace| it.next().and_then(|s| s.parse::<u64>().ok());
                match it.next() {
                    Some("WIDTH") => w = v(&mut it)?,
                    Some("HEIGHT") => h = v(&mut it)?,
                    Some("DEPTH") => c = v(&mut it)?,
                    Some("MAXVAL") => m = v(&mut it)?,
                    _ => {}
                }
            }
            (w, h, c, m, end + 7)
        }
        _ => return None,
    };
    if !(1..=4).contains(&channels) || !(1..=255).contains(&maxval) || width == 0 || height == 0 {
        return None;
    }
    Some(ImageLayout {
        header_len: header_len as u64,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        channels: channels as u8,
        stride: width * channels,
        predictor: Predictor::ColorMed,
        png: false,
    })
}

fn detect_tga(d: &[u8]) -> Option<ImageLayout> {
    if d.len() < 18 || d[1] != 0 || d[3..8].iter().any(|&b| b != 0) {
        return None; // colour-mapped images are not handled
    }
    let channels = match (d[2], d[16]) {
        (2, 24) => 3,
        (2, 32) => 4,
        (3, 8) => 1,
        _ => return None,
    };
    if d[17] & 0xC0 != 0 {
        return None; // interleaved rows
    }
    let (width, height) = (u16le(d, 12)?, u16le(d, 14)?);
    let header_len = 18 + d[0] as u64;
    let pixels = width as u64 * height as u64 * channels as u64;
    // TGA has no magic number: require the 2.0 footer or an exact size match.
    let footer = d.len() >= 26 && &d[d.len() - 18..] == b"TRUEVISION-XFILE.\0";
    if !footer && header_len + pixels != d.len() as u64 {
        return None;
    }
    Some(ImageLayout {
        header_len,
        width,
        height,
        channels,
        stride: width as u64 * channels as u64,
        predictor: Predictor::ColorMed,
        png: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut r = crate::rng::Rng::new(seed);
        (0..n).map(|_| r.next_u64() as u8).collect()
    }

    /// A smooth photo-like gradient with noise, `w*h*ch` samples.
    fn photo(w: usize, h: usize, ch: usize) -> Vec<u8> {
        let n = noise(w * h * ch, 5);
        (0..w * h * ch)
            .map(|i| {
                let (p, c) = (i / ch, i % ch);
                ((p % w) * 2 + (p / w) * 3 + c * 40 + (n[i] % 9) as usize) as u8
            })
            .collect()
    }

    fn roundtrip(data: &[u8], l: &ImageLayout) {
        let t = encode(data, l).unwrap();
        assert_eq!(t.len(), data.len());
        assert_eq!(decode(&t, l).unwrap(), data, "{l:?}");
        let mut b = Vec::new();
        l.write(&mut b);
        let mut pos = 0;
        assert_eq!(ImageLayout::read(&b, &mut pos).unwrap(), *l);
        assert_eq!(pos, b.len());
    }

    #[test]
    fn bmp_with_row_padding() {
        for (bpp, ch) in [(24u16, 3usize), (32, 4)] {
            let (w, h) = (61usize, 40usize); // odd width: 24-bit rows need padding
            let stride = (w * bpp as usize).div_ceil(32) * 4;
            let mut f = vec![0u8; 54];
            f[..2].copy_from_slice(b"BM");
            f[10..14].copy_from_slice(&54u32.to_le_bytes());
            f[14..18].copy_from_slice(&40u32.to_le_bytes());
            f[18..22].copy_from_slice(&(w as i32).to_le_bytes());
            f[22..26].copy_from_slice(&(-(h as i32)).to_le_bytes());
            f[26..28].copy_from_slice(&1u16.to_le_bytes());
            f[28..30].copy_from_slice(&bpp.to_le_bytes());
            let px = photo(w, h, ch);
            for y in 0..h {
                f.extend_from_slice(&px[y * w * ch..(y + 1) * w * ch]);
                f.extend(std::iter::repeat_n(0xEE, stride - w * ch));
            }
            f.extend_from_slice(b"trailing");
            let l = detect(&f).expect("bmp detected");
            assert_eq!((l.width, l.height, l.channels, l.stride, l.header_len), (w as u32, h as u32, ch as u8, stride as u64, 54));
            assert_eq!(l.predictor, Predictor::ColorMed);
            roundtrip(&f, &l);
            // Padding and trailing bytes are untouched.
            let t = encode(&f, &l).unwrap();
            assert_eq!(&t[t.len() - 8..], b"trailing");
            if stride > w * ch {
                assert_eq!(t[54 + w * ch], 0xEE);
            }
        }
    }

    #[test]
    fn pnm_and_pam() {
        let px = photo(64, 48, 3);
        let mut ppm = b"P6\n# made by a test\n64 48\n255\n".to_vec();
        ppm.extend_from_slice(&px);
        let l = detect(&ppm).expect("ppm");
        assert_eq!((l.channels, l.header_len as usize), (3, ppm.len() - px.len()));
        roundtrip(&ppm, &l);
        let mut pgm = b"P5 128 40 255\n".to_vec();
        pgm.extend_from_slice(&photo(128, 40, 1));
        roundtrip(&pgm, &detect(&pgm).expect("pgm"));
        let mut pam = b"P7\nWIDTH 40\nHEIGHT 30\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n".to_vec();
        pam.extend_from_slice(&photo(40, 30, 4));
        let l = detect(&pam).expect("pam");
        assert_eq!(l.channels, 4);
        roundtrip(&pam, &l);
        // 16-bit samples are left alone.
        let mut wide = b"P6 64 48 65535\n".to_vec();
        wide.extend_from_slice(&photo(64, 96, 3));
        assert!(detect(&wide).is_none());
    }

    #[test]
    fn tga_needs_footer_or_exact_size() {
        let (w, h) = (50usize, 40usize);
        let mut t = vec![0u8; 18];
        t[2] = 2;
        t[12..14].copy_from_slice(&(w as u16).to_le_bytes());
        t[14..16].copy_from_slice(&(h as u16).to_le_bytes());
        t[16] = 24;
        t.extend_from_slice(&photo(w, h, 3));
        let l = detect(&t).expect("tga");
        roundtrip(&t, &l);
        let mut extra = t.clone();
        extra.push(0);
        assert!(detect(&extra).is_none());
        extra.extend_from_slice(&[0; 8]);
        extra.extend_from_slice(b"TRUEVISION-XFILE.\0");
        roundtrip(&extra, &detect(&extra).expect("tga with footer"));
    }

    #[test]
    fn png_scanlines_all_filter_types() {
        let (w, h, ch) = (40usize, 35usize, 3usize);
        let px = photo(w, h, ch);
        // Filter each row with a different type, like an encoder would.
        let mut idat = Vec::new();
        for y in 0..h {
            let kind = (y % 5) as u8;
            idat.push(kind);
            let cur = &px[y * w * ch..(y + 1) * w * ch];
            let prev = if y == 0 { vec![0; w * ch] } else { px[(y - 1) * w * ch..y * w * ch].to_vec() };
            for i in 0..w * ch {
                idat.push(cur[i].wrapping_sub(png_pred(kind, cur, &prev, i, ch)));
            }
        }
        let l = png_layout(w as u32, h as u32, 8, 2, 0, &idat).expect("png layout");
        roundtrip(&idat, &l);
        // The transformed pixels match what the same image gives without PNG filtering.
        let plain = ImageLayout { png: false, stride: (w * ch) as u64, ..l };
        let t = encode(&idat, &l).unwrap();
        let tp = encode(&px, &plain).unwrap();
        for y in 0..h {
            assert_eq!(&t[y * (w * ch + 1) + 1..(y + 1) * (w * ch + 1)], &tp[y * w * ch..(y + 1) * w * ch]);
        }
        assert!(png_layout(w as u32, h as u32, 8, 3, 0, &idat).is_none()); // palette
        assert!(png_layout(w as u32, h as u32, 16, 2, 0, &idat).is_none());
        let mut bad = idat.clone();
        bad[0] = 9;
        assert!(encode(&bad, &l).is_err());
    }

    #[test]
    fn flat_images_skip_spatial_prediction() {
        let (w, h) = (64usize, 64usize);
        let mut px = vec![200u8; w * h * 3];
        for y in (0..h).step_by(8) {
            px[y * w * 3..y * w * 3 + 30].fill(20);
        }
        let mut ppm = b"P6 64 64 255\n".to_vec();
        ppm.extend_from_slice(&px);
        let l = detect(&ppm).unwrap();
        assert_eq!(l.predictor, Predictor::Color);
        roundtrip(&ppm, &l);
    }

    #[test]
    fn any_layout_is_reversible() {
        let data = noise(5000, 11);
        for (w, h, ch, stride, header, med, png) in
            [(10u32, 30u32, 3u8, 31u64, 7u64, true, false), (7, 50, 1, 7, 0, true, false), (12, 20, 4, 49, 3, false, true), (9, 9, 2, 19, 100, true, true)]
        {
            let mut d = data.clone();
            let l = ImageLayout { header_len: header, width: w, height: h, channels: ch, stride, predictor: if med { Predictor::ColorMed } else { Predictor::Color }, png };
            if png {
                for y in 0..h as usize {
                    d[header as usize + y * stride as usize] %= 5;
                }
            }
            roundtrip(&d, &l);
        }
        let l = ImageLayout { header_len: 0, width: 100, height: 100, channels: 3, stride: 300, predictor: Predictor::ColorMed, png: false };
        assert!(encode(&data, &l).is_err()); // rows past the end
    }
}
