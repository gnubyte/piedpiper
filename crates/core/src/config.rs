//! Model / training configuration and the compression-level presets.

use crate::error::{Error, Result};

pub const MIN_LEVEL: u32 = 1;
pub const MAX_LEVEL: u32 = 9;
pub const DEFAULT_LEVEL: u32 = 2;

/// Transformer + streaming configuration. Stored verbatim in every compressed file, so the
/// decoder rebuilds exactly the same network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    pub d_model: u32,
    pub n_layer: u32,
    pub n_head: u32,
    /// Hidden width of the SwiGLU feed-forward block.
    pub d_ff: u32,
    /// Attention span in tokens (the current token plus `window - 1` predecessors).
    pub window: u32,
    /// Tokens per stream between two weight updates.
    pub seg_len: u32,
    /// Number of parallel streams the token sequence is split into (TufaZip uses 64).
    pub n_streams: u32,
}

/// Online-training hyper-parameters (also stored in every file).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainConfig {
    /// Peak learning rate (reached after `warmup` updates).
    pub lr: f32,
    /// Learning-rate floor; the schedule decays hyperbolically towards it.
    pub lr_min: f32,
    pub warmup: u32,
    /// Updates after warm-up at which the decay is half-way from `lr` to `lr_min`.
    pub decay: u32,
    /// Adam β1. TufaZip (and NNCP) use 0: no momentum, which also saves memory.
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Global gradient-norm clip (0 = off).
    pub clip: f32,
    /// Multiplier on the default `1/sqrt(fan_in)` weight-init scale.
    pub init_scale: f32,
    pub seed: u64,
}

impl ModelConfig {
    pub fn head_dim(&self) -> u32 {
        self.d_model / self.n_head.max(1)
    }

    /// Number of trainable parameters for a given output vocabulary.
    pub fn n_params(&self, vocab: usize) -> usize {
        let d = self.d_model as usize;
        let hd = self.head_dim() as usize;
        let ff = self.d_ff as usize;
        let per_layer = d + 3 * d * d + 2 * hd + d * d + d + d + 2 * d * ff + ff * d + d;
        (vocab + 1) * d + self.n_layer as usize * per_layer + d + d * vocab
    }

    /// Approximate bytes of RAM the codec needs for this configuration.
    pub fn memory_estimate(&self, vocab: usize) -> u64 {
        let d = self.d_model as u64;
        let nh = self.n_head as u64;
        let ff = self.d_ff as u64;
        let win = self.window as u64;
        let rows = self.seg_len as u64 * self.n_streams as u64;
        let nl = self.n_layer as u64;
        let params = self.n_params(vocab) as u64;
        let per_row_layer = 13 * d + 3 * ff + nh * win + 3 + 2 * nh;
        let acts = rows * (nl * per_row_layer + 2 * d + 1 + vocab as u64);
        let scratch = rows * (6 * d + 3 * ff) + rows * d.max(ff);
        let hist = 2 * nl * self.n_streams as u64 * win * d;
        4 * (5 * params + acts + scratch + hist)
    }

    pub fn validate(&self, vocab: usize) -> Result<()> {
        let bad = |m: &str| Err(Error::Options(m.to_string()));
        if !(1..=8192).contains(&self.d_model) {
            return bad("d_model must be in 1..=8192");
        }
        if self.n_head == 0 || !self.d_model.is_multiple_of(self.n_head) {
            return bad("d_model must be a multiple of n_head");
        }
        let hd = self.head_dim();
        if !hd.is_multiple_of(2) || hd > 1024 {
            return bad("head dimension (d_model / n_head) must be even and <= 1024");
        }
        if !(1..=128).contains(&self.n_layer) {
            return bad("n_layer must be in 1..=128");
        }
        if !(1..=65536).contains(&self.d_ff) {
            return bad("d_ff must be in 1..=65536");
        }
        if !(1..=65536).contains(&self.window) {
            return bad("window must be in 1..=65536");
        }
        if !(1..=65536).contains(&self.seg_len) {
            return bad("seg_len must be in 1..=65536");
        }
        if !(1..=4096).contains(&self.n_streams) {
            return bad("n_streams must be in 1..=4096");
        }
        if !(2..=65535).contains(&vocab) {
            return bad("vocabulary size must be in 2..=65535");
        }
        let mem = self.memory_estimate(vocab);
        if mem > max_memory() {
            return Err(Error::TooLarge(format!(
                "configuration needs ~{} MiB, limit is {} MiB",
                mem >> 20,
                max_memory() >> 20
            )));
        }
        Ok(())
    }
}

fn max_memory() -> u64 {
    if usize::BITS >= 64 {
        64 << 30
    } else {
        3 << 30
    }
}

impl TrainConfig {
    pub fn validate(&self) -> Result<()> {
        let ok = |v: f32| v.is_finite();
        if !(ok(self.lr) && self.lr >= 0.0 && ok(self.lr_min) && self.lr_min >= 0.0) {
            return Err(Error::Options("learning rates must be finite and >= 0".into()));
        }
        if !(ok(self.beta1) && (0.0..1.0).contains(&self.beta1)) {
            return Err(Error::Options("beta1 must be in [0, 1)".into()));
        }
        if !(ok(self.beta2) && (0.0..1.0).contains(&self.beta2)) {
            return Err(Error::Options("beta2 must be in [0, 1)".into()));
        }
        if !(ok(self.eps) && self.eps > 0.0) {
            return Err(Error::Options("eps must be > 0".into()));
        }
        if !(ok(self.clip) && self.clip >= 0.0) || !(ok(self.init_scale) && self.init_scale >= 0.0) {
            return Err(Error::Options("clip and init_scale must be finite and >= 0".into()));
        }
        if self.warmup == 0 {
            return Err(Error::Options("warmup must be >= 1".into()));
        }
        Ok(())
    }
}

/// A compression level: network shape, training recipe and stream policy.
#[derive(Clone, Copy, Debug)]
pub struct Preset {
    /// `n_streams` here is the *maximum*; the encoder lowers it for short inputs.
    pub model: ModelConfig,
    pub train: TrainConfig,
    /// Minimum tokens per stream before another stream is added.
    pub min_stream_tokens: u32,
    /// Largest BPE vocabulary the automatic tokenizer may choose.
    pub max_vocab: u32,
    /// Peak learning rate when starting from a pretrained prior (fine-tuning).
    pub prior_lr: f32,
}

/// Streams and segment length the encoder picks for `n_tokens` tokens. Small inputs get
/// small batches (many updates — learning speed is what limits them); large inputs get
/// more parallel streams for throughput.
pub fn batch_policy(n_tokens: usize, max_streams: u32) -> (u32, u32) {
    let n = n_tokens as u64;
    let seg = if n < 256 * 1024 {
        4
    } else if n < 4 << 20 {
        8
    } else {
        16
    };
    let ns = (n / (256 * 1024)).clamp(2, max_streams.max(2) as u64) as u32;
    (ns.min(n_tokens.max(1) as u32).max(1), seg)
}

const SEED: u64 = 0x5049_4544_5049_5045; // "PIEDPIPE"

/// Returns the preset for `level` (clamped to `MIN_LEVEL..=MAX_LEVEL`).
pub fn preset(level: u32) -> Preset {
    let level = level.clamp(MIN_LEVEL, MAX_LEVEL);
    // Levels 1-3 match the built-in priors' shapes (text-s / text-m / text-l); levels 4-9
    // are larger networks trained from scratch, worthwhile for large inputs.
    //            d   layers heads  ff  window seg streams   lr   vocab
    let (d, nl, nh, ff, win, seg, ns, lr, vocab): (u32, u32, u32, u32, u32, u32, u32, f32, u32) =
        match level {
            1 => (64, 2, 2, 192, 64, 16, 16, 2.0e-3, 1024),
            2 => (96, 3, 3, 256, 128, 16, 24, 1.5e-3, 2048),
            3 => (128, 4, 4, 352, 128, 16, 32, 1.0e-3, 4096),
            4 => (160, 5, 5, 448, 160, 16, 32, 8.0e-4, 4096),
            5 => (192, 6, 6, 512, 192, 16, 48, 6.0e-4, 4096),
            6 => (256, 6, 8, 704, 256, 16, 64, 5.0e-4, 4096),
            7 => (320, 8, 10, 864, 256, 16, 64, 4.0e-4, 4096),
            8 => (384, 10, 12, 1024, 256, 16, 64, 3.0e-4, 4096),
            _ => (512, 12, 16, 1408, 320, 16, 64, 2.5e-4, 4096),
        };
    Preset {
        model: ModelConfig {
            d_model: d,
            n_layer: nl,
            n_head: nh,
            d_ff: ff,
            window: win,
            seg_len: seg,
            n_streams: ns,
        },
        train: TrainConfig {
            lr,
            lr_min: lr * 0.25,
            warmup: 64,
            decay: 2048,
            beta1: 0.0,
            beta2: 0.9999,
            eps: 1e-8,
            clip: 0.0,
            init_scale: 1.0,
            seed: SEED,
        },
        min_stream_tokens: 8192,
        max_vocab: vocab,
        prior_lr: lr * 0.4,
    }
}
