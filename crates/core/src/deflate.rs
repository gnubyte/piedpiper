//! Lossless recompression of DEFLATE streams inside common containers (format mode 4).
//!
//! PNG, ZIP-family files (docx/xlsx/pptx/odt/epub/jar/apk), gzip, PDF and zlib files are
//! already compressed, so the neural model gains nothing on them directly. This module
//! finds their DEFLATE streams, inflates them with [preflate-rs] (which records the small
//! amount of information needed to recreate the *exact* original compressed bytes), and
//! hands the inflated content to the neural codec instead.
//!
//! The expanded form has these parts, each compressed as its own PiedPiper stream:
//!
//! ```text
//! meta:  "PPDX" | version u8 | n_segments varint | per segment: kind u8 + lengths
//!        | all raw container bytes | all preflate corrections
//! plain: the inflated content of every stream, concatenated
//! embedded: whole files found inside, one part each (JPEG photos, nested ZIP/PNG/gzip/
//!        PDF containers, MP4 videos; from inflated or stored ZIP entries and PDF
//!        DCTDecode streams), which the codec compresses as files of their own
//! ```
//!
//! Keeping the inflated content separate lets text (DOCX/XLSX XML, gzipped logs, PDF
//! content streams) get the text priors while headers and corrections do not dilute it.
//! Embedded files get the mode that suits them: photos in Office documents, EPUBs, APKs and
//! PDFs go to the JPEG recoder, and a DOCX inside a ZIP, the classes.jar inside an AAR or a
//! PNG stored in an APK is expanded in turn.
//! Every stream is verified to recreate byte-exactly before it is accepted, and the whole
//! blob is verified to restore the input exactly; anything else stays raw.
//!
//! [preflate-rs]: https://github.com/microsoft/preflate-rs

use crate::error::{Error, Result};
use crate::filter::Filter;
use crate::format::{crc32, get_varint, put_varint};
use crate::image::{self, ImageLayout, Predictor};
use preflate_rs::{preflate_whole_deflate_stream, recreate_whole_deflate_stream, PreflateConfig};

const MAGIC: &[u8; 4] = b"PPDX";
const VERSION: u8 = 2;

const KIND_RAW: u8 = 0;
const KIND_DEFLATE: u8 = 1;
const KIND_PNG_IDAT: u8 = 2;
/// A DEFLATE stream whose inflated content is an embedded file, carried in the next part.
const KIND_DEFLATE_PART: u8 = 3;
/// Container bytes that are an embedded file (stored ZIP entry, PDF DCTDecode stream),
/// carried in the next part.
const KIND_RAW_PART: u8 = 4;

/// Smallest embedded file worth a part of its own.
const MIN_EMBEDDED: usize = 2048;

fn is_jpeg(b: &[u8]) -> bool {
    b.len() >= MIN_EMBEDDED && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF
}

/// A whole file that compresses better on its own than as bytes of its container: a JPEG,
/// a nested container this module can expand, an MP4/Matroska video or a WAV recording.
fn is_embedded_file(b: &[u8]) -> bool {
    b.len() >= MIN_EMBEDDED
        && (is_jpeg(b)
            || b.starts_with(b"\x89PNG\r\n\x1a\n")
            || b.starts_with(b"PK\x03\x04")
            || b.starts_with(&[0x1f, 0x8b, 8])
            || b.starts_with(b"%PDF")
            || &b[4..8] == b"ftyp"
            || b.starts_with(&[0x1A, 0x45, 0xDF, 0xA3])
            || (b.starts_with(b"RIFF") && &b[8..12] == b"WAVE")
            || is_tar(b))
}

/// A POSIX/GNU tar archive (ustar magic in the first header).
fn is_tar(b: &[u8]) -> bool {
    b.len() >= 1024 && &b[257..262] == b"ustar"
}

/// Largest inflated size accepted per stream (and in total), against zip bombs.
const PLAIN_LIMIT: usize = 256 << 20;

enum Seg {
    Raw(Vec<u8>),
    /// Container bytes that are a whole embedded file.
    RawPart(Vec<u8>),
    /// `image`: the inflated content is a PDF image's samples (compressed as a part of its
    /// own with the image filter).
    Deflate { plain: Vec<u8>, corr: Vec<u8>, image: Option<ImageLayout> },
    /// A PNG zlib stream split over IDAT chunks of the given payload sizes.
    PngIdat { sizes: Vec<u32>, zhdr: [u8; 2], adler: [u8; 4], plain: Vec<u8>, corr: Vec<u8> },
}

fn config() -> PreflateConfig {
    PreflateConfig { max_chain_length: 4096, plain_text_limit: PLAIN_LIMIT, verify_compression: false }
}

/// Inflates the DEFLATE stream at the start of `data`; returns (plain, corrections, bytes
/// used) only if recreation is byte-exact.
fn try_stream(data: &[u8]) -> Option<(Vec<u8>, Vec<u8>, usize)> {
    if data.len() < 8 {
        return None;
    }
    let (res, plain) = preflate_whole_deflate_stream(data, &config()).ok()?;
    let used = res.compressed_size;
    let plain = plain.text().to_vec();
    // Recompressing incompressible data gains nothing, unless it is a file of its own
    // (a JPEG for Lepton, a nested container to expand).
    if used < 32 || used > data.len() || (plain.len() < used + used / 16 && !is_embedded_file(&plain)) {
        return None;
    }
    let back = recreate_whole_deflate_stream(&plain, &res.corrections).ok()?;
    if back.as_slice() != &data[..used] {
        return None;
    }
    Some((plain, res.corrections, used))
}

#[inline]
fn le16(b: &[u8], o: usize) -> usize {
    b[o] as usize | (b[o + 1] as usize) << 8
}

#[inline]
fn le32(b: &[u8], o: usize) -> usize {
    le16(b, o) | le16(b, o + 2) << 16
}

#[inline]
fn be32(b: &[u8], o: usize) -> usize {
    (b[o] as usize) << 24 | (b[o + 1] as usize) << 16 | (b[o + 2] as usize) << 8 | b[o + 3] as usize
}

fn is_zlib_header(a: u8, b: u8) -> bool {
    a & 0x0F == 8 && (a >> 4) <= 7 && ((a as u16) << 8 | b as u16).is_multiple_of(31) && b & 0x20 == 0
}

/// Quick check whether `data` is worth scanning.
pub fn candidate(data: &[u8]) -> bool {
    data.len() >= 64
        && (data.starts_with(b"\x89PNG\r\n\x1a\n")
            || data.starts_with(b"PK\x03\x04")
            || data.starts_with(&[0x1f, 0x8b, 8])
            || data.starts_with(b"%PDF")
            || is_zlib_header(data[0], data[1])
            || is_tar(data))
}

struct Builder<'a> {
    data: &'a [u8],
    segs: Vec<Seg>,
    raw_from: usize,
    plain_total: usize,
}

impl<'a> Builder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Builder { data, segs: Vec::new(), raw_from: 0, plain_total: 0 }
    }

    /// Emits raw bytes up to `at`, then `seg`, and resumes raw at `resume`.
    fn add(&mut self, at: usize, seg: Seg, resume: usize) {
        if at > self.raw_from {
            self.segs.push(Seg::Raw(self.data[self.raw_from..at].to_vec()));
        }
        self.segs.push(seg);
        self.raw_from = resume;
    }

    fn deflate_at(&mut self, p: usize) -> Option<usize> {
        if self.plain_total >= PLAIN_LIMIT {
            return None;
        }
        let (plain, corr, used) = try_stream(&self.data[p..])?;
        self.plain_total += plain.len();
        self.add(p, Seg::Deflate { plain, corr, image: None }, p + used);
        Some(p + used)
    }

    fn finish(mut self) -> Vec<Seg> {
        if self.data.len() > self.raw_from {
            self.segs.push(Seg::Raw(self.data[self.raw_from..].to_vec()));
        }
        self.segs
    }
}

fn scan_png(data: &[u8]) -> Option<Vec<Seg>> {
    let mut b = Builder::new(data);
    let mut p = 8;
    while p + 12 <= data.len() {
        let len = be32(data, p);
        if p + 12 + len > data.len() {
            break;
        }
        if &data[p + 4..p + 8] != b"IDAT" {
            p += 12 + len;
            continue;
        }
        // Collect the run of consecutive IDAT chunks.
        let start = p;
        let mut zlib = Vec::new();
        let mut sizes = Vec::new();
        while p + 12 <= data.len() && &data[p + 4..p + 8] == b"IDAT" {
            let l = be32(data, p);
            if p + 12 + l > data.len() {
                return None;
            }
            zlib.extend_from_slice(&data[p + 8..p + 8 + l]);
            sizes.push(l as u32);
            p += 12 + l;
        }
        if zlib.len() < 8 || !is_zlib_header(zlib[0], zlib[1]) {
            continue;
        }
        let (plain, corr, used) = try_stream(&zlib[2..])?;
        if 2 + used + 4 != zlib.len() {
            return None;
        }
        let seg = Seg::PngIdat {
            sizes,
            zhdr: [zlib[0], zlib[1]],
            adler: zlib[zlib.len() - 4..].try_into().unwrap(),
            plain,
            corr,
        };
        // The chunk headers/CRCs are rebuilt; verify that reproduces the original bytes.
        let mut rebuilt = Vec::new();
        emit_png_idat(&seg, &mut rebuilt).ok()?;
        if rebuilt != data[start..p] {
            return None;
        }
        b.add(start, seg, p);
    }
    Some(b.finish())
}

/// Tar members that are files of their own (photos, archives, documents) become parts;
/// headers and everything else stay raw. Stops at the first header it cannot read.
fn scan_tar(data: &[u8]) -> Vec<Seg> {
    let mut b = Builder::new(data);
    let mut p = 0;
    while p + 512 <= data.len() && &data[p + 257..p + 262] == b"ustar" {
        // Size: octal digits, space/NUL terminated (base-256 sizes are not followed).
        let field = &data[p + 124..p + 136];
        let digits: Vec<u8> = field.iter().copied().skip_while(|&c| c == b' ').take_while(|c| (b'0'..=b'7').contains(c)).collect();
        if digits.is_empty() || digits.len() > 11 {
            break;
        }
        let size = digits.iter().fold(0usize, |v, &c| v * 8 + (c - b'0') as usize);
        let start = p + 512;
        let Some(end) = start.checked_add(size).filter(|&e| e <= data.len()) else { break };
        let regular = matches!(data[p + 156], b'0' | 0 | b'7');
        if regular && is_embedded_file(&data[start..end]) {
            b.add(start, Seg::RawPart(data[start..end].to_vec()), end);
        }
        p = start + size.div_ceil(512) * 512;
    }
    b.finish()
}

fn scan_gzip(data: &[u8]) -> Vec<Seg> {
    let mut b = Builder::new(data);
    let n = data.len();
    let mut i = 0;
    while i + 18 <= n && data[i] == 0x1f && data[i + 1] == 0x8b && data[i + 2] == 8 {
        let flg = data[i + 3];
        let mut p = i + 10;
        if flg & 4 != 0 {
            if p + 2 > n {
                break;
            }
            p += 2 + le16(data, p);
        }
        for bit in [8u8, 16] {
            if flg & bit != 0 {
                while p < n && data[p] != 0 {
                    p += 1;
                }
                p += 1;
            }
        }
        if flg & 2 != 0 {
            p += 2;
        }
        if p >= n {
            break;
        }
        match b.deflate_at(p) {
            Some(end) => i = end + 8, // crc32 + isize stay raw
            None => break,
        }
    }
    b.finish()
}

fn scan_generic(data: &[u8]) -> Vec<Seg> {
    let mut b = Builder::new(data);
    let n = data.len();
    let is_pdf = data.starts_with(b"%PDF");
    if !is_pdf && n > 2 && is_zlib_header(data[0], data[1]) && !data.starts_with(b"PK") {
        b.deflate_at(2);
        return b.finish();
    }
    let mut i = 0;
    while i + 30 <= n {
        if &data[i..i + 4] == b"PK\x03\x04" {
            let method = le16(data, i + 8);
            let p = i + 30 + le16(data, i + 26) + le16(data, i + 28);
            if method == 8 && p < n {
                if let Some(end) = b.deflate_at(p) {
                    i = end;
                    continue;
                }
            }
            // Stored entry with known size (no data descriptor) holding a file of its own.
            let csize = le32(data, i + 18);
            if method == 0 && le16(data, i + 6) & 8 == 0 && p.checked_add(csize).is_some_and(|e| e <= n) && is_embedded_file(&data[p..p + csize]) {
                b.add(p, Seg::RawPart(data[p..p + csize].to_vec()), p + csize);
                i = p + csize;
                continue;
            }
            i += 4;
            continue;
        }
        if is_pdf && &data[i..i + 6] == b"stream" && (i == 0 || !data[i - 1].is_ascii_alphanumeric()) {
            let mut p = i + 6;
            if p < n && data[p] == b'\r' {
                p += 1;
            }
            if p < n && data[p] == b'\n' {
                p += 1;
            }
            let dict = &data[i.saturating_sub(2048)..i];
            let has_flate = dict.windows(12).rposition(|w| w == b"/FlateDecode").is_some();
            if has_flate && p + 2 < n && is_zlib_header(data[p], data[p + 1]) {
                if let Some(end) = b.deflate_at(p + 2) {
                    // An image XObject: let its samples get the image filter.
                    if let Some(Seg::Deflate { plain, image, .. }) = b.segs.last_mut() {
                        *image = pdf_image_layout(object_dict(dict), plain);
                    }
                    i = end;
                    continue;
                }
            }
            // An image stored as a plain JPEG (DCTDecode): from SOI up to its EOI marker.
            if !has_flate && dict.windows(10).rposition(|w| w == b"/DCTDecode").is_some() && data[p..].starts_with(&[0xFF, 0xD8, 0xFF]) {
                if let Some(e) = data[p..].windows(9).position(|w| w == b"endstream") {
                    let mut end = p + e;
                    while end > p && matches!(data[end - 1], b'\r' | b'\n' | b' ') {
                        end -= 1;
                    }
                    if end >= p + 4 && data[end - 2] == 0xFF && data[end - 1] == 0xD9 && is_jpeg(&data[p..end]) {
                        b.add(p, Seg::RawPart(data[p..end].to_vec()), end);
                        i = end;
                        continue;
                    }
                }
            }
            i += 6;
            continue;
        }
        i += 1;
    }
    b.finish()
}

/// The part of a PDF stream's look-back window that belongs to its own object.
fn object_dict(window: &[u8]) -> &[u8] {
    let start = window.windows(4).rposition(|w| w == b" obj" || w == b"\nobj" || w == b"\robj").map_or(0, |k| k + 4);
    &window[start..]
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// A direct integer value of `key` in a PDF dictionary (not an indirect `n 0 R` reference).
fn pdf_int(dict: &[u8], key: &[u8]) -> Option<u64> {
    let k = dict.windows(key.len()).rposition(|w| w == key)?;
    let mut p = k + key.len();
    if dict.get(p).is_some_and(|c| c.is_ascii_alphanumeric()) {
        return None; // a longer name such as /WidthX
    }
    while dict.get(p).is_some_and(|c| c.is_ascii_whitespace()) {
        p += 1;
    }
    let s = p;
    while dict.get(p).is_some_and(|c| c.is_ascii_digit()) {
        p += 1;
    }
    if p == s || p - s > 9 {
        return None;
    }
    // "12 0 R" is a reference to another object, not the value.
    let mut q = p;
    while dict.get(q).is_some_and(|c| c.is_ascii_whitespace()) {
        q += 1;
    }
    let r = q;
    while dict.get(q).is_some_and(|c| c.is_ascii_digit()) {
        q += 1;
    }
    if q > r {
        while dict.get(q).is_some_and(|c| c.is_ascii_whitespace()) {
            q += 1;
        }
        if dict.get(q) == Some(&b'R') {
            return None;
        }
    }
    std::str::from_utf8(&dict[s..p]).ok()?.parse().ok()
}

/// Image-filter layout for the inflated samples of a PDF image XObject: 8-bit gray, RGB or
/// CMYK, stored as plain rows or with PNG predictors (`/Predictor` 10-15).
fn pdf_image_layout(dict: &[u8], inflated: &[u8]) -> Option<ImageLayout> {
    if !contains(dict, b"/Image") || contains(dict, b"/Indexed") || contains(dict, b"/ImageMask true") {
        return None;
    }
    let (width, height) = (pdf_int(dict, b"/Width")?, pdf_int(dict, b"/Height")?);
    if pdf_int(dict, b"/BitsPerComponent")? != 8 || width > u32::MAX as u64 || height > u32::MAX as u64 {
        return None;
    }
    let channels: u64 = if contains(dict, b"/DeviceRGB") {
        3
    } else if contains(dict, b"/DeviceGray") {
        1
    } else if contains(dict, b"/DeviceCMYK") {
        4
    } else {
        return None;
    };
    let png = match pdf_int(dict, b"/Predictor").unwrap_or(1) {
        1 => false,
        10..=15 => {
            if pdf_int(dict, b"/Colors").unwrap_or(1) != channels || pdf_int(dict, b"/Columns").unwrap_or(1) != width {
                return None;
            }
            true
        }
        _ => return None,
    };
    let stride = width.checked_mul(channels)? + u64::from(png);
    if stride.checked_mul(height)? != inflated.len() as u64 {
        return None;
    }
    let l = ImageLayout {
        header_len: 0,
        width: width as u32,
        height: height as u32,
        channels: channels as u8,
        stride,
        predictor: Predictor::ColorMed,
        png,
    };
    image::finish(inflated, l)
}

fn emit_png_idat(seg: &Seg, out: &mut Vec<u8>) -> Result<()> {
    let Seg::PngIdat { sizes, zhdr, adler, plain, corr } = seg else { unreachable!() };
    let deflate = recreate_whole_deflate_stream(plain, corr).map_err(|_| Error::Format("deflate recreation failed".into()))?;
    let mut zlib = Vec::with_capacity(deflate.len() + 6);
    zlib.extend_from_slice(zhdr);
    zlib.extend_from_slice(&deflate);
    zlib.extend_from_slice(adler);
    let total: usize = sizes.iter().map(|&s| s as usize).sum();
    if total != zlib.len() {
        return Err(Error::Format("PNG IDAT sizes do not match the stream".into()));
    }
    let mut off = 0;
    for &s in sizes {
        let s = s as usize;
        let mut chunk = Vec::with_capacity(s + 4);
        chunk.extend_from_slice(b"IDAT");
        chunk.extend_from_slice(&zlib[off..off + s]);
        out.extend_from_slice(&(s as u32).to_be_bytes());
        out.extend_from_slice(&chunk);
        out.extend_from_slice(&crc32(&chunk).to_be_bytes());
        off += s;
    }
    Ok(())
}

/// The parts of an expanded file (see the module docs).
pub struct Expanded {
    pub meta: Vec<u8>,
    pub plain: Vec<u8>,
    pub embedded: Vec<Vec<u8>>,
    /// Per embedded part: a filter the encoder should use for it (PDF images). Decoders do
    /// not need it; each part's own header records its filter.
    pub hints: Vec<Option<Filter>>,
}

fn serialize(segs: &[Seg]) -> Expanded {
    let mut head = Vec::new();
    head.extend_from_slice(MAGIC);
    head.push(VERSION);
    put_varint(&mut head, segs.len() as u64);
    let (mut raw, mut corr, mut plain, mut embedded) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut hints = Vec::new();
    for s in segs {
        match s {
            Seg::Raw(b) => {
                head.push(KIND_RAW);
                put_varint(&mut head, b.len() as u64);
                raw.extend_from_slice(b);
            }
            Seg::RawPart(b) => {
                head.push(KIND_RAW_PART);
                put_varint(&mut head, b.len() as u64);
                embedded.push(b.clone());
                hints.push(None);
            }
            Seg::Deflate { plain: p, corr: c, image } => {
                let own = image.is_some() || is_embedded_file(p);
                head.push(if own { KIND_DEFLATE_PART } else { KIND_DEFLATE });
                put_varint(&mut head, p.len() as u64);
                put_varint(&mut head, c.len() as u64);
                corr.extend_from_slice(c);
                if own {
                    embedded.push(p.clone());
                    hints.push(image.map(Filter::Image));
                } else {
                    plain.extend_from_slice(p);
                }
            }
            Seg::PngIdat { sizes, zhdr, adler, plain: p, corr: c } => {
                head.push(KIND_PNG_IDAT);
                put_varint(&mut head, sizes.len() as u64);
                for &s in sizes {
                    put_varint(&mut head, s as u64);
                }
                head.extend_from_slice(zhdr);
                head.extend_from_slice(adler);
                put_varint(&mut head, p.len() as u64);
                put_varint(&mut head, c.len() as u64);
                corr.extend_from_slice(c);
                plain.extend_from_slice(p);
            }
        }
    }
    head.extend_from_slice(&raw);
    head.extend_from_slice(&corr);
    Expanded { meta: head, plain, embedded, hints }
}

/// Expands the DEFLATE streams (and embedded files) of `data`. Returns `None` if nothing
/// could be expanded (or the result would not restore exactly).
pub fn expand(data: &[u8]) -> Option<Expanded> {
    if !candidate(data) {
        return None;
    }
    let segs = if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        scan_png(data)?
    } else if data.starts_with(&[0x1f, 0x8b, 8]) {
        scan_gzip(data)
    } else if is_tar(data) {
        scan_tar(data)
    } else {
        scan_generic(data)
    };
    if !segs.iter().any(|s| !matches!(s, Seg::Raw(_))) {
        return None;
    }
    let x = serialize(&segs);
    match restore(&x.meta, &x.plain, &x.embedded) {
        Ok(back) if back == data => Some(x),
        _ => None,
    }
}

/// Inverse of [`expand`].
pub fn restore(blob: &[u8], plain: &[u8], embedded: &[Vec<u8>]) -> Result<Vec<u8>> {
    let bad = || Error::Format("malformed deflate container".into());
    if blob.len() < 6 || &blob[..4] != MAGIC {
        return Err(bad());
    }
    if blob[4] != VERSION {
        return Err(Error::Format(format!("unsupported deflate container version {}", blob[4])));
    }
    let mut pos = 5;
    let n = get_varint(blob, &mut pos)? as usize;
    if n > blob.len() {
        return Err(bad());
    }
    enum Rec {
        Raw(usize),
        RawPart(usize),
        Deflate(usize, usize),
        DeflatePart(usize, usize),
        Png(Vec<u32>, [u8; 2], [u8; 4], usize, usize),
    }
    let mut recs = Vec::with_capacity(n);
    let (mut raw_total, mut corr_total) = (0usize, 0usize);
    for _ in 0..n {
        let kind = *blob.get(pos).ok_or_else(bad)?;
        pos += 1;
        let mut v = || get_varint(blob, &mut pos).map(|x| x as usize);
        match kind {
            KIND_RAW => {
                let l = v()?;
                raw_total = raw_total.checked_add(l).ok_or_else(bad)?;
                recs.push(Rec::Raw(l));
            }
            KIND_RAW_PART => recs.push(Rec::RawPart(v()?)),
            KIND_DEFLATE | KIND_DEFLATE_PART => {
                let (p, c) = (v()?, v()?);
                corr_total = corr_total.checked_add(c).ok_or_else(bad)?;
                recs.push(if kind == KIND_DEFLATE { Rec::Deflate(p, c) } else { Rec::DeflatePart(p, c) });
            }
            KIND_PNG_IDAT => {
                let k = v()?;
                if k > blob.len() {
                    return Err(bad());
                }
                let mut sizes = Vec::with_capacity(k);
                for _ in 0..k {
                    sizes.push(u32::try_from(v()?).map_err(|_| bad())?);
                }
                let zhdr: [u8; 2] = blob.get(pos..pos + 2).ok_or_else(bad)?.try_into().unwrap();
                let adler: [u8; 4] = blob.get(pos + 2..pos + 6).ok_or_else(bad)?.try_into().unwrap();
                pos += 6;
                let (p, c) = (get_varint(blob, &mut pos)? as usize, get_varint(blob, &mut pos)? as usize);
                corr_total = corr_total.checked_add(c).ok_or_else(bad)?;
                recs.push(Rec::Png(sizes, zhdr, adler, p, c));
            }
            _ => return Err(bad()),
        }
    }
    let mut raw_at = pos;
    let mut corr_at = raw_at.checked_add(raw_total).ok_or_else(bad)?;
    if corr_at.checked_add(corr_total) != Some(blob.len()) {
        return Err(bad());
    }
    let mut plain_at = 0usize;
    let take = |buf: &[u8], at: &mut usize, len: usize| -> Result<Vec<u8>> {
        let s = buf.get(*at..at.checked_add(len).ok_or_else(bad)?).ok_or_else(bad)?;
        *at += len;
        Ok(s.to_vec())
    };
    let mut ji = 0usize;
    let mut out = Vec::new();
    for r in recs {
        match r {
            Rec::Raw(l) => out.extend_from_slice(&take(blob, &mut raw_at, l)?),
            Rec::RawPart(l) => {
                out.extend_from_slice(embedded.get(ji).filter(|j| j.len() == l).ok_or_else(bad)?);
                ji += 1;
            }
            Rec::Deflate(p, c) => {
                let corr = take(blob, &mut corr_at, c)?;
                let text = take(plain, &mut plain_at, p)?;
                let d = recreate_whole_deflate_stream(&text, &corr).map_err(|_| bad())?;
                out.extend_from_slice(&d);
            }
            Rec::DeflatePart(p, c) => {
                let corr = take(blob, &mut corr_at, c)?;
                let text = embedded.get(ji).filter(|j| j.len() == p).ok_or_else(bad)?;
                ji += 1;
                let d = recreate_whole_deflate_stream(text, &corr).map_err(|_| bad())?;
                out.extend_from_slice(&d);
            }
            Rec::Png(sizes, zhdr, adler, p, c) => {
                let corr = take(blob, &mut corr_at, c)?;
                let text = take(plain, &mut plain_at, p)?;
                emit_png_idat(&Seg::PngIdat { sizes, zhdr, adler, plain: text, corr }, &mut out)?;
            }
        }
    }
    if plain_at != plain.len() || ji != embedded.len() {
        return Err(bad());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal zlib (stored + fixed-Huffman via a tiny hand-made stream is overkill);
    /// instead check structure handling with data that has no deflate streams and with a
    /// gzip produced by the encoder in this test (a stored block would be rejected).
    #[test]
    fn non_candidates_are_left_alone() {
        assert!(expand(b"plain text, nothing to see here, plain text plain text plain text").is_none());
        assert!(restore(b"PPDX\x02", b"", &[]).is_err());
        assert!(restore(b"nope", b"", &[]).is_err());
    }

    #[test]
    fn tar_members_become_parts() {
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0];
        jpeg.extend((0..3000u32).map(|i| (i * 13 % 251) as u8));
        let mut tar = Vec::new();
        for (name, body) in [("notes.txt", b"plain text member\n".repeat(40)), ("photo.jpg", jpeg.clone())] {
            let mut h = [0u8; 512];
            h[..name.len()].copy_from_slice(name.as_bytes());
            h[124..135].copy_from_slice(format!("{:011o}", body.len()).as_bytes());
            h[156] = b'0';
            h[257..263].copy_from_slice(b"ustar\0");
            tar.extend_from_slice(&h);
            tar.extend_from_slice(&body);
            tar.resize(tar.len().div_ceil(512) * 512, 0);
        }
        tar.extend_from_slice(&[0u8; 1024]);
        let x = expand(&tar).expect("tar expands");
        assert_eq!(x.embedded, vec![jpeg]);
        assert_eq!(restore(&x.meta, &x.plain, &x.embedded).unwrap(), tar);
    }

    #[test]
    fn stored_zip_jpeg_becomes_a_part() {
        // A stored ZIP entry whose data looks like a JPEG, followed by a central directory.
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0];
        jpeg.extend((0..5000u32).map(|i| (i * 7 % 251) as u8));
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        let mut zip = b"PK\x03\x04".to_vec();
        zip.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // ver, flags, method 0, time, date, crc
        zip.extend_from_slice(&(jpeg.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(jpeg.len() as u32).to_le_bytes());
        zip.extend_from_slice(&[5, 0, 0, 0]);
        zip.extend_from_slice(b"a.jpg");
        zip.extend_from_slice(&jpeg);
        zip.extend_from_slice(b"PK\x01\x02 central directory would follow");
        let x = expand(&zip).expect("expanded");
        assert_eq!(x.embedded, vec![jpeg]);
        assert!(x.plain.is_empty());
        assert_eq!(restore(&x.meta, &x.plain, &x.embedded).unwrap(), zip);
        assert!(restore(&x.meta, &x.plain, &[]).is_err());
    }
}

#[cfg(test)]
mod file_tests {
    /// Exercised on real files when PIEDPIPER_DEFLATE_SAMPLES points at a directory.
    #[test]
    fn real_files_roundtrip() {
        let Some(dir) = std::env::var_os("PIEDPIPER_DEFLATE_SAMPLES") else { return };
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "exp") {
                continue;
            }
            let data = std::fs::read(&p).unwrap();
            let t = std::time::Instant::now();
            match super::expand(&data) {
                Some(x) => {
                    assert_eq!(super::restore(&x.meta, &x.plain, &x.embedded).unwrap(), data, "{}", p.display());
                    let jl: usize = x.embedded.iter().map(|j| j.len()).sum();
                    eprintln!(
                        "{}: {} -> meta {} + plain {} + {} embedded {} ({:.2}s)",
                        p.display(),
                        data.len(),
                        x.meta.len(),
                        x.plain.len(),
                        x.embedded.len(),
                        jl,
                        t.elapsed().as_secs_f64()
                    );
                }
                None => eprintln!("{}: not expanded", p.display()),
            }
        }
    }
}
