//! Format mode 3: lossless recompression of JPEG images with Lepton.
//!
//! A JPEG's entropy-coded data looks random to any byte model, so the Transformer gains only
//! a few percent on it. Lepton (Dropbox's design, Rust port by Microsoft, vendored in
//! `third_party/lepton_jpeg`) instead decodes the Huffman-coded DCT coefficients, codes them
//! with a context model and a binary arithmetic coder, and on decompression re-encodes them
//! into the original file bit for bit, typically about 22% smaller (see docs/DESIGN.md).
//!
//! Lepton uses integer arithmetic only, so its output is the same on every platform. The
//! number of partitions (which shapes the stream) is fixed, so the thread count only changes
//! the speed, never the bytes.

use crate::codec::ProgressFn;
use crate::error::{Error, Result};

/// Name reported by [`crate::inspect`] for mode-3 streams.
pub const NAME: &str = "jpeg-lepton";

/// Largest JPEG that is recompressed (also caps what a decoder will allocate).
#[cfg(feature = "jpeg")]
pub const MAX_JPEG_BYTES: u64 = 128 << 20;

/// True if `data` starts like a JPEG file (SOI marker followed by another marker).
#[cfg(feature = "jpeg")]
pub fn looks_like_jpeg(data: &[u8]) -> bool {
    data.len() >= 4 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF
}

#[cfg(feature = "jpeg")]
mod lepton {
    use super::MAX_JPEG_BYTES;
    use lepton_jpeg::{EnabledFeatures, LeptonError, LeptonThreadPool, SingleThreadPool};
    use std::io::Cursor;

    /// Lepton codes the image as this many independent row bands. Fixed, so the output
    /// never depends on the machine or the thread count.
    const PARTITIONS: u32 = 8;

    fn features(threads: usize) -> EnabledFeatures {
        let mut f = EnabledFeatures::compat_lepton_vector_write();
        f.max_partitions = PARTITIONS;
        f.max_jpeg_file_size = MAX_JPEG_BYTES as u32;
        f.max_processor_threads = if threads == 0 { PARTITIONS } else { (threads as u32).min(PARTITIONS) };
        f
    }

    fn pool(threads: usize) -> &'static dyn LeptonThreadPool {
        static SINGLE: SingleThreadPool = SingleThreadPool {};
        #[cfg(feature = "parallel")]
        if threads != 1 {
            return &lepton_jpeg::DEFAULT_THREAD_POOL;
        }
        let _ = threads;
        &SINGLE
    }

    pub fn encode(input: &[u8], threads: usize) -> Result<Vec<u8>, LeptonError> {
        let f = features(threads);
        lepton_jpeg::catch_unwind_result(|| lepton_jpeg::encode_lepton_verify(input, &f, pool(threads)))
            .map(|(out, _metrics)| out)
    }

    pub fn decode(payload: &[u8], orig_len: usize, threads: usize) -> Result<Vec<u8>, LeptonError> {
        let f = features(threads);
        let mut out = Vec::with_capacity(orig_len);
        lepton_jpeg::catch_unwind_result(|| {
            lepton_jpeg::decode_lepton(&mut Cursor::new(payload), &mut out, &f, pool(threads))
        })?;
        Ok(out)
    }
}

/// Recompresses `input` if it is a JPEG that Lepton supports and the result is smaller.
/// Returns `Ok(None)` otherwise. Lepton decodes its output and compares it with the input
/// before it is returned, so a parser bug can only cost compression, never data.
#[cfg(feature = "jpeg")]
pub fn compress(input: &[u8], crc: u32, threads: usize, progress: &mut ProgressFn<'_>) -> Result<Option<Vec<u8>>> {
    use crate::format::{self, MODE_JPEG};
    if !looks_like_jpeg(input) || input.len() as u64 > MAX_JPEG_BYTES {
        return Ok(None);
    }
    let total = input.len() as u64;
    if !progress(0, total) {
        return Err(Error::Cancelled);
    }
    let packed = match lepton::encode(input, threads) {
        Ok(p) => p,
        Err(_) => return Ok(None), // unsupported or malformed JPEG: let the neural path try
    };
    if packed.len() + 16 >= input.len() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(packed.len() + 16);
    format::write_prefix(&mut out, MODE_JPEG, total, crc);
    out.extend_from_slice(&packed);
    progress(total, total);
    Ok(Some(out))
}

#[cfg(feature = "jpeg")]
pub fn decompress(payload: &[u8], orig_len: u64, threads: usize, progress: &mut ProgressFn<'_>) -> Result<Vec<u8>> {
    if orig_len > MAX_JPEG_BYTES {
        return Err(Error::Format("JPEG recompression stream is larger than the format allows".into()));
    }
    if !progress(0, orig_len) {
        return Err(Error::Cancelled);
    }
    let out = lepton::decode(payload, orig_len as usize, threads)
        .map_err(|e| Error::Format(format!("corrupt JPEG recompression stream: {e}")))?;
    if out.len() as u64 != orig_len {
        return Err(Error::Format("JPEG recompression stream has the wrong length".into()));
    }
    progress(orig_len, orig_len);
    Ok(out)
}

#[cfg(not(feature = "jpeg"))]
pub fn compress(_input: &[u8], _crc: u32, _threads: usize, _progress: &mut ProgressFn<'_>) -> Result<Option<Vec<u8>>> {
    Ok(None)
}

#[cfg(not(feature = "jpeg"))]
pub fn decompress(_payload: &[u8], _orig_len: u64, _threads: usize, _progress: &mut ProgressFn<'_>) -> Result<Vec<u8>> {
    Err(Error::Format("this build cannot decompress JPEG recompression streams (built without the `jpeg` feature)".into()))
}

#[cfg(all(test, feature = "jpeg"))]
mod tests {
    use crate::format::MODE_JPEG;
    use crate::{compress, decompress, decompress_with_progress, inspect, CompressOptions, DecompressOptions};

    /// Baseline JPEG synthesised by scripts/make_golden.py (integer-only encoder).
    const JPEG: &[u8] = include_bytes!("../../../testdata/golden/jpeg.bin");

    #[test]
    fn roundtrip_in_mode_3() {
        let c = compress(JPEG, &CompressOptions::default()).unwrap();
        assert_eq!(c[5], MODE_JPEG);
        assert!(c.len() < JPEG.len(), "{} vs {}", c.len(), JPEG.len());
        assert_eq!(inspect(&c).unwrap().tokenizer, super::NAME);
        assert!(inspect(&c).unwrap().model.is_none());
        assert_eq!(decompress(&c).unwrap(), JPEG);
    }

    #[test]
    fn thread_count_does_not_change_output() {
        let a = compress(JPEG, &CompressOptions { threads: 1, ..Default::default() }).unwrap();
        let b = compress(JPEG, &CompressOptions { threads: 4, ..Default::default() }).unwrap();
        assert_eq!(a, b);
        let back = decompress_with_progress(&a, &DecompressOptions { threads: 3, prior: None }, &mut |_, _| true).unwrap();
        assert_eq!(back, JPEG);
    }

    #[test]
    fn trailing_data_fallback_and_opt_out() {
        let mut t = JPEG.to_vec();
        t.extend_from_slice(b"bytes after the end-of-image marker");
        let c = compress(&t, &CompressOptions::default()).unwrap();
        assert_eq!(c[5], MODE_JPEG);
        assert_eq!(decompress(&c).unwrap(), t);
        // Starts like a JPEG but is not one: the neural path takes it.
        let fake: Vec<u8> = [0xFF, 0xD8, 0xFF, 0xE0].into_iter().chain(b"no image here ".repeat(30)).collect();
        let c = compress(&fake, &CompressOptions::level(1)).unwrap();
        assert_ne!(c[5], MODE_JPEG);
        assert_eq!(decompress(&c).unwrap(), fake);
        let c = compress(JPEG, &CompressOptions { jpeg: false, ..CompressOptions::level(1) }).unwrap();
        assert_ne!(c[5], MODE_JPEG);
        assert_eq!(decompress(&c).unwrap(), JPEG);
    }

    #[test]
    fn corruption_is_an_error_not_a_panic() {
        let c = compress(JPEG, &CompressOptions::default()).unwrap();
        let mut bad = c.clone();
        let k = bad.len() / 2;
        bad[k] ^= 0x55;
        assert!(decompress(&bad).is_err());
        assert!(decompress(&c[..c.len() / 2]).is_err());
        let mut bad_len = c.clone();
        bad_len[6] ^= 0x01; // original length (first varint byte)
        assert!(decompress(&bad_len).is_err());
    }
}
