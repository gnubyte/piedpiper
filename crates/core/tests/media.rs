//! End-to-end checks of format mode 2 (lossless video recompression) through the public API.
#![cfg(feature = "media")]

use piedpiper::{compress, decompress, inspect, CompressOptions};

fn sample() -> Vec<u8> {
    // A synthetic H.264 MP4: CABAC I/P/B slices with noise content (no third-party video).
    piedpiper_media::testgen::synthetic_mp4(11, 10, 12, 8)
}

#[test]
fn h264_mp4_is_recoded_and_roundtrips() {
    let file = sample();
    let packed = compress(&file, &CompressOptions::default()).unwrap();
    let info = inspect(&packed).unwrap();
    assert_eq!(info.tokenizer, "h264-recode");
    assert!(!info.stored);
    assert!(info.model.is_none());
    assert_eq!(info.original_size, file.len() as u64);
    assert!(packed.len() < file.len(), "{} >= {}", packed.len(), file.len());
    assert_eq!(decompress(&packed).unwrap(), file);
}

#[test]
fn h264_cavlc_mp4_is_recoded_and_roundtrips() {
    // Baseline (CAVLC) and High-with-CAVLC streams: slices coded bit by bit, not via CABAC.
    for high in [false, true] {
        let file = piedpiper_media::testgen::synthetic_mp4_cavlc(12, 10, 9, 6, high);
        let packed = compress(&file, &CompressOptions::default()).unwrap();
        let info = inspect(&packed).unwrap();
        assert_eq!(info.tokenizer, "h264-recode");
        assert!(!info.stored);
        assert!(packed.len() < file.len(), "{} >= {}", packed.len(), file.len());
        assert_eq!(decompress(&packed).unwrap(), file);
        let single = compress(&file, &CompressOptions { threads: 1, ..CompressOptions::default() }).unwrap();
        assert_eq!(packed, single);
    }
}

#[test]
fn h264_transport_stream_is_recoded_and_roundtrips() {
    let file = piedpiper_media::ts::synth::from_mp4(14, 8, 9, 6);
    let packed = compress(&file, &CompressOptions::default()).unwrap();
    let info = inspect(&packed).unwrap();
    assert_eq!(info.tokenizer, "h264-ts-recode");
    assert!(!info.stored);
    assert!(packed.len() < file.len(), "{} >= {}", packed.len(), file.len());
    assert_eq!(decompress(&packed).unwrap(), file);
}

#[test]
fn recoding_is_deterministic() {
    let file = sample();
    let a = compress(&file, &CompressOptions::default()).unwrap();
    let b = compress(&file, &CompressOptions { threads: 1, ..CompressOptions::default() }).unwrap();
    assert_eq!(a, b);
}

#[test]
fn corrupt_payload_is_an_error() {
    let file = sample();
    let packed = compress(&file, &CompressOptions::default()).unwrap();
    let mut bad = packed.clone();
    let i = bad.len() * 2 / 3;
    bad[i] ^= 0x10;
    assert!(decompress(&bad).is_err());
    assert!(decompress(&packed[..packed.len() / 2]).is_err());
}

#[test]
fn non_video_input_is_unaffected() {
    let text = b"not a video file, just some text that repeats. ".repeat(40);
    let packed = compress(&text, &CompressOptions::level(1)).unwrap();
    assert_ne!(inspect(&packed).unwrap().tokenizer, "h264-recode");
    assert_eq!(decompress(&packed).unwrap(), text);
}

fn vp9_sample() -> Vec<u8> {
    // A synthetic VP9 WebM: key, inter, hidden, compound, show-existing, superframe and
    // lossless frames with random decisions (no third-party video).
    piedpiper_media::vp9::synth::synthetic_webm(5, 96, 64, 0, 0)
}

#[test]
fn vp9_webm_is_recoded_and_roundtrips() {
    let file = vp9_sample();
    let packed = compress(&file, &CompressOptions::default()).unwrap();
    let info = inspect(&packed).unwrap();
    assert_eq!(info.tokenizer, "vp9-recode");
    assert!(!info.stored);
    assert_eq!(info.original_size, file.len() as u64);
    assert!(packed.len() < file.len(), "{} >= {}", packed.len(), file.len());
    assert_eq!(decompress(&packed).unwrap(), file);
    let one = compress(&file, &CompressOptions { threads: 1, ..CompressOptions::default() }).unwrap();
    assert_eq!(one, packed);
    let mut bad = packed.clone();
    let i = bad.len() * 2 / 3;
    bad[i] ^= 0x10;
    assert!(decompress(&bad).is_err());
}

fn hevc_sample() -> Vec<u8> {
    // A synthetic HEVC MP4 with wavefronts and dependent slice segments: random decisions,
    // noise content (no third-party video).
    let cfg = piedpiper_media::hevc::testgen::config("wpp").unwrap();
    piedpiper_media::hevc::testgen::synthetic_mp4(13, 8, &cfg)
}

#[test]
fn hevc_mp4_is_recoded_and_roundtrips() {
    let file = hevc_sample();
    let packed = compress(&file, &CompressOptions::default()).unwrap();
    let info = inspect(&packed).unwrap();
    assert_eq!(info.tokenizer, "hevc-recode");
    assert!(!info.stored);
    assert_eq!(info.original_size, file.len() as u64);
    assert!(packed.len() < file.len(), "{} >= {}", packed.len(), file.len());
    assert_eq!(decompress(&packed).unwrap(), file);
    let one = compress(&file, &CompressOptions { threads: 1, ..CompressOptions::default() }).unwrap();
    assert_eq!(one, packed);
    let mut bad = packed.clone();
    let i = bad.len() * 2 / 3;
    bad[i] ^= 0x10;
    assert!(decompress(&bad).is_err());
}

#[test]
fn vp8_webm_is_recoded_and_roundtrips() {
    let file = piedpiper_media::vp8::stream::synth::synthetic_webm(4, 64, 48, 8);
    let packed = compress(&file, &CompressOptions::default()).unwrap();
    assert_eq!(inspect(&packed).unwrap().tokenizer, "vp8-recode");
    assert!(packed.len() < file.len(), "{} >= {}", packed.len(), file.len());
    assert_eq!(decompress(&packed).unwrap(), file);
}

#[test]
fn mpeg2_mkv_is_recoded_and_roundtrips() {
    let file = piedpiper_media::mpeg2::synth::synthetic_mkv(6, 5, 3, 8);
    let packed = compress(&file, &CompressOptions::default()).unwrap();
    assert_eq!(inspect(&packed).unwrap().tokenizer, "mpeg2-recode");
    assert!(packed.len() < file.len(), "{} >= {}", packed.len(), file.len());
    assert_eq!(decompress(&packed).unwrap(), file);
}
