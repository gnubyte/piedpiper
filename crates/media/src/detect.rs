//! Recognising already-compressed video that cannot be recoded.
//!
//! Byte-level models (PiedPiper's Transformer, LZ, ...) save next to nothing on compressed
//! video, so spending hours on it is pointless. [`is_opaque_video`] spots such files from
//! the container signature plus a cheap, integer-only randomness probe, so the decision is
//! identical on every platform.

use crate::mp4;

/// Block size and count of the randomness probe.
const BLOCK: usize = 16 * 1024;
const BLOCKS: usize = 32;
/// A block whose byte histogram is this close to uniform (chi-square, 255 degrees of
/// freedom; random data averages 255) is treated as entropy-coded data. Compressed video
/// measures 200-400, PCM audio and text tens of thousands.
const CHI2_RANDOM: u64 = 2000;

/// True for a video container (MP4/MOV/3GP, Matroska/WebM, AVI, MPEG-TS/M2TS, MPEG-PS, FLV)
/// whose bytes look incompressible: at least three quarters of the sampled blocks have a
/// near-uniform byte distribution. Files with large uncompressed parts (PCM audio,
/// padding) do not qualify.
pub fn is_opaque_video(input: &[u8]) -> bool {
    input.len() >= 4 * BLOCK && is_video_container(input) && looks_random(input)
}

fn is_video_container(b: &[u8]) -> bool {
    let iso = b.len() >= 12 && &b[4..8] == b"ftyp" && mp4::has_video_track(b);
    let mkv = b.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]);
    let avi = b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"AVI ";
    let ts = (0..5).all(|k| b.get(k * 188) == Some(&0x47));
    let m2ts = (0..5).all(|k| b.get(4 + k * 192) == Some(&0x47));
    let ps = b.starts_with(&[0, 0, 1, 0xBA]);
    let flv = b.len() >= 4 && &b[..3] == b"FLV" && b[3] == 1;
    iso || mkv || avi || ts || m2ts || ps || flv
}

fn looks_random(b: &[u8]) -> bool {
    let n = BLOCKS.min(b.len() / BLOCK);
    if n == 0 {
        return false;
    }
    let step = if n > 1 { (b.len() - BLOCK) / (n - 1) } else { 0 };
    let expected = (BLOCK / 256) as u64;
    let mut random = 0;
    for k in 0..n {
        let block = &b[k * step..k * step + BLOCK];
        let mut counts = [0u32; 256];
        for &x in block {
            counts[x as usize] += 1;
        }
        let chi2: u64 = counts.iter().map(|&c| (c as u64).abs_diff(expected).pow(2)).sum::<u64>() / expected;
        if chi2 < CHI2_RANDOM {
            random += 1;
        }
    }
    random * 4 >= n * 3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize) -> Vec<u8> {
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn matroska_noise_is_opaque() {
        let mut f = vec![0x1A, 0x45, 0xDF, 0xA3];
        f.extend(noise(600_000));
        assert!(is_opaque_video(&f));
    }

    #[test]
    fn compressible_or_unknown_is_not() {
        let mut pcm = vec![0x1A, 0x45, 0xDF, 0xA3];
        pcm.extend((0..600_000u32).map(|i| ((i as f32 * 0.01).sin() * 60.0) as i8 as u8));
        assert!(!is_opaque_video(&pcm));
        assert!(!is_opaque_video(&noise(600_000))); // random, but not a video container
        let text = b"plain text, not a video ".repeat(30_000);
        assert!(!is_opaque_video(&text));
    }
}
