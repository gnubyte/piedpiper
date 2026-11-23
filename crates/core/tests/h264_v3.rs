//! H.264 streams from before the picture-reconstruction model (format mode 2, payload magic
//! `PPM3`) must keep decoding.
#![cfg(feature = "media")]

use piedpiper::{decompress, inspect};

#[test]
fn ppm3_h264_stream_decodes() {
    // The H.264 golden vector as PiedPiper 0.1.x wrote it (scripts/make_golden.py).
    let mp4 = include_bytes!("data/h264-v3.mp4");
    let packed = include_bytes!("data/h264-v3.mp4.pp");
    assert_eq!(inspect(packed).unwrap().tokenizer, "h264-recode");
    assert_eq!(decompress(packed).unwrap(), mp4);
}
