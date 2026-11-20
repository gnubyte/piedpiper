//! Recognises already-compressed archives and media that no byte-level model can shrink.
//!
//! The neural model needs minutes per megabyte, and on an `.xz`, `.7z`, `.mp3` or `.webp` it
//! saves next to nothing. When a file both starts with such a format's signature and its
//! bytes look uniformly random, `compress` stores it right away. Both tests are integer-only,
//! so the decision is the same on every platform (and only affects the encoder anyway).
//!
//! Formats PiedPiper can recompress (JPEG, PNG, ZIP, gzip, PDF, H.264 video) are not listed:
//! their own modes run first.

/// Files smaller than this are left to the model: it is quick on them, and their headers
/// and metadata are a larger share.
const MIN_LEN: usize = 64 << 10;
const BLOCK: usize = 4096;
const BLOCKS: usize = 32;

/// Name of the compressed format `data` starts with, if any.
pub fn compressed_format(data: &[u8]) -> Option<&'static str> {
    let at = |off: usize, sig: &[u8]| data.len() >= off + sig.len() && &data[off..off + sig.len()] == sig;
    let name = if at(0, &[0xFD, b'7', b'z', b'X', b'Z', 0]) {
        "xz"
    } else if at(0, &[0x28, 0xB5, 0x2F, 0xFD]) {
        "zstd"
    } else if at(0, &[b'7', b'z', 0xBC, 0xAF, 0x27, 0x1C]) {
        "7z"
    } else if at(0, b"Rar!\x1a\x07") {
        "rar"
    } else if at(0, b"BZh") && data.len() > 3 && (b'1'..=b'9').contains(&data[3]) {
        "bzip2"
    } else if at(0, &[0x04, 0x22, 0x4D, 0x18]) {
        "lz4"
    } else if at(0, b"LZIP") {
        "lzip"
    } else if at(0, b"ID3") || (data.len() > 2 && data[0] == 0xFF && matches!(data[1], 0xFB | 0xFA | 0xF3 | 0xF2 | 0xE3)) {
        "mp3"
    } else if data.len() > 2 && data[0] == 0xFF && matches!(data[1], 0xF1 | 0xF9) {
        "aac"
    } else if at(0, b"OggS") {
        "ogg"
    } else if at(0, b"fLaC") {
        "flac"
    } else if at(0, b"RIFF") && at(8, b"WEBP") {
        "webp"
    } else if at(0, b"wOF2") {
        "woff2"
    } else if at(4, b"ftyp") && (at(8, b"M4A ") || at(8, b"M4B ") || at(8, b"heic") || at(8, b"heix") || at(8, b"avif") || at(8, b"mif1")) {
        "isobmff-audio-image"
    } else {
        return None;
    };
    Some(name)
}

/// True when at least 7/8 of evenly spaced 4 KiB samples have a nearly flat byte histogram
/// (chi-square below 1500 at 255 degrees of freedom: noise measures ~255, xz/brotli output
/// 210-360, bzip2's Huffman output up to ~2000 in rare blocks, text and tables tens of
/// thousands).
pub fn looks_random(data: &[u8]) -> bool {
    if data.len() < BLOCK * 4 {
        return false;
    }
    let n = BLOCKS.min(data.len() / BLOCK);
    let step = (data.len() - BLOCK) / (n - 1).max(1);
    let mut uniform = 0;
    for i in 0..n {
        let block = &data[i * step..i * step + BLOCK];
        let mut hist = [0u32; 256];
        for &b in block {
            hist[b as usize] += 1;
        }
        // chi² = Σ (c − 16)² / 16 for 4096 bytes; compare 16·chi² against 16·1500.
        let s: u32 = hist.iter().map(|&c| (c as i32 - 16).unsigned_abs().pow(2)).sum();
        if s < 1500 * 16 {
            uniform += 1;
        }
    }
    uniform * 8 >= n * 7
}

/// Already-compressed data that `compress` should store immediately.
pub fn is_opaque(data: &[u8]) -> bool {
    data.len() >= MIN_LEN && compressed_format(data).is_some() && looks_random(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn detects_compressed_noise_only() {
        let mut rng = Rng::new(5);
        let mut noise: Vec<u8> = (0..200_000).map(|_| rng.next_u64() as u8).collect();
        assert!(looks_random(&noise));
        assert!(!is_opaque(&noise), "no signature");
        noise[..6].copy_from_slice(&[0xFD, b'7', b'z', b'X', b'Z', 0]);
        assert!(is_opaque(&noise));
        assert!(!is_opaque(&noise[..30_000]), "too small");
        // Same signature, but text-like content: leave it to the model.
        let mut text = b"The quick brown fox jumps over the lazy dog. ".repeat(5000);
        text[..6].copy_from_slice(&[0xFD, b'7', b'z', b'X', b'Z', 0]);
        assert!(!is_opaque(&text));
        assert_eq!(compressed_format(b"fLaC\0\0\0\x22"), Some("flac"));
        assert_eq!(compressed_format(b"plain"), None);
    }
}
