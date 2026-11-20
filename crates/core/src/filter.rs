//! Reversible pre-transforms applied to the input before tokenizing (neural mode only).
//!
//! A filter rewrites structure that a byte-level model cannot see into a form it predicts
//! better: relative branch targets in machine code become absolute addresses, so calls to
//! the same function repeat byte-for-byte. The filter id and its parameters are stored in
//! the neural header right after the match-model byte. The tokenizer, prior and model all
//! see the filtered bytes; the decoder inverts the filter after detokenizing and then checks
//! the CRC of the original data.
//!
//! Rules for every filter: `decode(encode(x)) == x` exactly, and the output is never longer
//! than the input. `compress` also verifies the round trip before using a filter.

use crate::bcj::{self, Arch};
use crate::error::{Error, Result};
use crate::image::{self, ImageLayout};

/// A reversible transform of the input bytes (stored in the neural header).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Filter {
    /// No transform (id 0).
    None,
    /// x86 / x86-64 `CALL`/`JMP rel32` targets made absolute (id 1).
    X86,
    /// AArch64 `BL` targets made absolute (id 2).
    Arm64,
    /// Colour decorrelation and 2-D prediction of uncompressed pixels (id 3); the layout
    /// (header length, size, channels, row stride, predictor) follows the id.
    Image(ImageLayout),
}

/// Which filter [`crate::compress`] applies.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum FilterChoice {
    /// Pick one from the content (executables get a branch filter). Default.
    #[default]
    Auto,
    /// Never filter.
    Off,
    /// Use this filter (skipped when it does not apply to the input).
    Force(Filter),
}

impl Filter {
    pub fn id(&self) -> u8 {
        match self {
            Filter::None => 0,
            Filter::X86 => 1,
            Filter::Arm64 => 2,
            Filter::Image(_) => 3,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Filter::None => "none",
            Filter::X86 => "x86",
            Filter::Arm64 => "arm64",
            Filter::Image(_) => "image",
        }
    }

    /// Serializes the id followed by the filter's parameters.
    pub fn write(&self, out: &mut Vec<u8>) {
        out.push(self.id());
        if let Filter::Image(l) = self {
            l.write(out);
        }
    }

    pub fn read(buf: &[u8], pos: &mut usize) -> Result<Filter> {
        let id = *buf.get(*pos).ok_or_else(|| Error::Format("truncated header".into()))?;
        *pos += 1;
        match id {
            0 => Ok(Filter::None),
            1 => Ok(Filter::X86),
            2 => Ok(Filter::Arm64),
            3 => Ok(Filter::Image(ImageLayout::read(buf, pos)?)),
            k => Err(Error::Format(format!("unknown filter {k} (newer PiedPiper version?)"))),
        }
    }

    /// The filter that suits `data`, judged from its content.
    pub fn detect(data: &[u8]) -> Filter {
        match bcj::detect(data) {
            Some(Arch::X86) => Filter::X86,
            Some(Arch::Arm64) => Filter::Arm64,
            None => image::detect(data).map_or(Filter::None, Filter::Image),
        }
    }

    /// The filtered bytes, or `None` when the filter does not apply to `data`.
    pub fn encode(&self, data: &[u8]) -> Option<Vec<u8>> {
        let arch = match self {
            Filter::None => return None,
            Filter::X86 => Arch::X86,
            Filter::Arm64 => Arch::Arm64,
            Filter::Image(l) => return image::encode(data, l).ok(),
        };
        let mut v = data.to_vec();
        bcj::encode(arch, &mut v);
        Some(v)
    }

    /// Inverts [`Filter::encode`].
    pub fn decode(&self, mut data: Vec<u8>) -> Result<Vec<u8>> {
        match self {
            Filter::None => {}
            Filter::X86 => bcj::decode(Arch::X86, &mut data),
            Filter::Arm64 => bcj::decode(Arch::Arm64, &mut data),
            Filter::Image(l) => return image::decode(&data, l),
        }
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let img = ImageLayout { header_len: 15, width: 640, height: 480, channels: 3, stride: 1920, predictor: image::Predictor::ColorMed, png: false };
        for f in [Filter::None, Filter::X86, Filter::Arm64, Filter::Image(img), Filter::Image(ImageLayout { png: true, stride: 1921, ..img })] {
            let mut out = vec![];
            f.write(&mut out);
            let mut pos = 0;
            assert_eq!(Filter::read(&out, &mut pos).unwrap(), f);
            assert_eq!(pos, out.len());
        }
        assert!(Filter::read(&[200], &mut 0).is_err());
        assert!(Filter::read(&[], &mut 0).is_err());
    }
}
