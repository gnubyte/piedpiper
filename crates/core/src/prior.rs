//! Pretrained priors (TufaZip v2 starts its online model from an offline-trained prior).
//!
//! A prior is a network shape + tokenizer + initial weights. Compressing with a prior
//! starts online learning from those weights instead of random ones, which helps most on
//! small inputs where a from-scratch model has little data to learn from. The compressed
//! stream records the prior's 64-bit id; decompression needs the same prior.
//!
//! File layout (`.ppp`):
//! ```text
//! "PPRI" | version u8 | tokenizer | d_model n_layer n_head d_ff window (varints)
//! | n_params varint | dtype u8 (1 = f16) | weights (little-endian) | crc32 u32le
//! ```

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use std::sync::{Arc, OnceLock};
use crate::format::{crc32, get_varint, put_varint};
use crate::tokenizer::Tokenizer;

pub const PRIOR_MAGIC: [u8; 4] = *b"PPRI";
const PRIOR_VERSION: u8 = 1;
const DTYPE_F16: u8 = 1;

/// A loaded prior model.
#[derive(Clone)]
pub struct Prior {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model: ModelConfig,
    pub(crate) weights: Vec<f32>,
    id: u64,
    size: usize,
}

impl std::fmt::Debug for Prior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prior {{ id: {:016x}, tokenizer: {}, d_model: {}, n_layer: {}, params: {} }}",
            self.id,
            self.tokenizer.name(),
            self.model.d_model,
            self.model.n_layer,
            self.weights.len()
        )
    }
}

/// FNV-1a 64 — identifies a prior file (not a security hash).
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    if h == 0 {
        1
    } else {
        h
    }
}

impl Prior {
    /// Parses a prior file.
    pub fn from_bytes(data: &[u8]) -> Result<Prior> {
        let bad = |m: &str| Error::Prior(m.to_string());
        if data.len() < 9 || data[..4] != PRIOR_MAGIC {
            return Err(bad("not a PiedPiper prior (bad magic)"));
        }
        let body = &data[..data.len() - 4];
        let want = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
        if crc32(body) != want {
            return Err(bad("prior file is corrupt (crc mismatch)"));
        }
        if data[4] != PRIOR_VERSION {
            return Err(bad("unsupported prior version"));
        }
        let mut pos = 5;
        let tokenizer = Tokenizer::read(body, &mut pos)?;
        let mut field = || -> Result<u32> {
            let v = get_varint(body, &mut pos)?;
            u32::try_from(v).map_err(|_| bad("field out of range"))
        };
        let model = ModelConfig {
            d_model: field()?,
            n_layer: field()?,
            n_head: field()?,
            d_ff: field()?,
            window: field()?,
            seg_len: 1,
            n_streams: 1,
        };
        model.validate(tokenizer.vocab_size()).map_err(|e| bad(&e.to_string()))?;
        let n = get_varint(body, &mut pos)? as usize;
        if n != model.n_params(tokenizer.vocab_size()) {
            return Err(bad("parameter count does not match the network shape"));
        }
        let dtype = *body.get(pos).ok_or_else(|| bad("truncated"))?;
        pos += 1;
        if dtype != DTYPE_F16 {
            return Err(bad("unsupported weight type"));
        }
        let raw = body.get(pos..pos + 2 * n).ok_or_else(|| bad("truncated weights"))?;
        if pos + 2 * n != body.len() {
            return Err(bad("trailing bytes"));
        }
        let weights = raw.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect();
        Ok(Prior { tokenizer, model, weights, id: fnv1a64(data), size: data.len() })
    }

    /// Serialises shape, tokenizer and weights (weights rounded to f16).
    pub(crate) fn encode(tokenizer: &Tokenizer, model: &ModelConfig, weights: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(weights.len() * 2 + 64);
        out.extend_from_slice(&PRIOR_MAGIC);
        out.push(PRIOR_VERSION);
        tokenizer.write(&mut out);
        for v in [model.d_model, model.n_layer, model.n_head, model.d_ff, model.window] {
            put_varint(&mut out, v as u64);
        }
        put_varint(&mut out, weights.len() as u64);
        out.push(DTYPE_F16);
        for &w in weights {
            out.extend_from_slice(&f32_to_f16(w).to_le_bytes());
        }
        let c = crc32(&out);
        out.extend_from_slice(&c.to_le_bytes());
        out
    }

    /// 64-bit identifier recorded in streams compressed with this prior.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Size of the prior file in bytes.
    pub fn file_size(&self) -> usize {
        self.size
    }

    pub fn model_config(&self) -> ModelConfig {
        self.model
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    pub fn tokenizer_name(&self) -> String {
        self.tokenizer.name()
    }

    pub fn n_params(&self) -> usize {
        self.weights.len()
    }
}

/// Priors compiled into the library (feature `builtin-priors`): `(level, name, file)`.
/// They were trained on a mixed corpus of code, prose and markup and are used
/// automatically for text-like input at levels 1–3. Their ids never change once released,
/// so old streams keep decoding.
#[cfg(feature = "builtin-priors")]
static BUILTIN_FILES: &[(u32, &str, &[u8])] = &[
    (1, "text-s", include_bytes!("../priors/text-s.ppp")),
    (2, "text-m", include_bytes!("../priors/text-m.ppp")),
    (3, "text-l", include_bytes!("../priors/text-l.ppp")),
    // Alternates for levels 2 and 1, trained further on the whole corpus: better on source
    // code and HTML, worse on XML. The encoder picks between a level's priors per file.
    (2, "text-m-code", include_bytes!("../priors/text-m-code.ppp")),
    (1, "text-s-code", include_bytes!("../priors/text-s-code.ppp")),
];
#[cfg(not(feature = "builtin-priors"))]
static BUILTIN_FILES: &[(u32, &str, &[u8])] = &[];

/// A built-in prior.
#[derive(Clone, Debug)]
pub struct BuiltinPrior {
    pub level: u32,
    pub name: &'static str,
    pub prior: Arc<Prior>,
}

/// All priors compiled into this build (parsed once, lazily).
pub fn builtin_priors() -> &'static [BuiltinPrior] {
    static CELL: OnceLock<Vec<BuiltinPrior>> = OnceLock::new();
    CELL.get_or_init(|| {
        BUILTIN_FILES
            .iter()
            .filter_map(|&(level, name, bytes)| {
                Prior::from_bytes(bytes).ok().map(|p| BuiltinPrior { level, name, prior: Arc::new(p) })
            })
            .collect()
    })
}

/// The built-in prior used by compression level `level`, if any.
pub fn builtin_prior(level: u32) -> Option<Arc<Prior>> {
    builtin_priors().iter().find(|b| b.level == level).map(|b| b.prior.clone())
}

/// Further built-in priors for `level` (same network shape as [`builtin_prior`]), which the
/// encoder may choose instead of the default one.
pub fn builtin_alternates(level: u32) -> Vec<Arc<Prior>> {
    builtin_priors().iter().filter(|b| b.level == level).skip(1).map(|b| b.prior.clone()).collect()
}

/// Looks a built-in prior up by the id recorded in a stream.
pub fn builtin_prior_by_id(id: u64) -> Option<&'static BuiltinPrior> {
    builtin_priors().iter().find(|b| b.prior.id() == id)
}

/// f32 → IEEE half, round-to-nearest-even (values clamped to the finite f16 range).
pub fn f32_to_f16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    let x = x.clamp(-65504.0, 65504.0);
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    let e = exp - 112;
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let rem = m & ((1u32 << shift) - 1);
        let mut q = m >> shift;
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (q & 1) == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut q = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (q & 1) == 1) {
        q += 1;
    }
    sign | q as u16
}

/// IEEE half → f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            let mut e2 = 113u32;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m2 & 0x3FF) << 13)
        }
    } else if e == 31 {
        sign | 0x7F80_0000 | (m << 13)
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_all_values() {
        for h in 0..=u16::MAX {
            let e = (h >> 10) & 0x1F;
            if e == 31 {
                continue; // inf / nan
            }
            let f = f16_to_f32(h);
            assert_eq!(f32_to_f16(f), h, "h={h:04x} f={f}");
        }
        assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(0.1)), 0.099975586);
        assert_eq!(f32_to_f16(1e9), 0x7BFF);
    }
}
