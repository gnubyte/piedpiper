//! Audio streams from before the context-mixing coder (format mode 5, version 1: one fixed
//! predictor, residuals coded by the network) must keep decoding.
#![cfg(feature = "audio")]

use piedpiper::{decompress, inspect};

#[test]
fn version_1_audio_stream_decodes() {
    // The audio golden vector as PiedPiper 0.1.x wrote it (scripts/make_golden.py).
    let wav = include_bytes!("data/audio-v1.wav");
    let packed = include_bytes!("data/audio-v1.wav.pp");
    let info = inspect(packed).unwrap();
    assert_eq!(info.tokenizer, "pcm-audio");
    assert_eq!(decompress(packed).unwrap(), wav);
}
