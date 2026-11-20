//! GPU acceleration of the neural model: the same Transformer, trained online the same way, run
//! as Vulkan compute shaders. Tested on AMD Radeon GPUs; any Vulkan 1.2 GPU whose driver
//! guarantees the IEEE behaviour below can run it.
//!
//! The GPU computes **exactly** the CPU's bits (every probability, every weight), so output is
//! byte-identical either way and a file compressed on a GPU decompresses on any CPU. The rules
//! that make the CPU deterministic (docs/DESIGN.md, *Determinism*) carry over:
//!
//! * every float multiply and add is marked `precise` (SPIR-V `NoContraction`), so no driver can
//!   fuse them into an FMA; a test checks every operation in the embedded SPIR-V;
//! * division and square root are computed in integer arithmetic, correctly rounded (Vulkan only
//!   guarantees `+ - *`; native `sqrt` differed on 10% of inputs on the test GPU);
//! * the float-control execution modes keep subnormals, round to nearest even and preserve
//!   signed zeros, infinities and NaNs; a device that cannot promise them is not used;
//! * `exp` is `fpmath::exp` transcribed, and every sum keeps the CPU's order.
//!
//! On top of that, the first use in a process runs a small network on both the CPU and the GPU
//! and compares every probability and weight; any difference and the GPU is not used.
//!
//! Shaders live in `gpu/shaders`; `scripts/gpu_shaders.py` compiles them to the SPIR-V in
//! `gpu/spv`, which is checked in, so building PiedPiper needs no shader compiler. The Vulkan
//! loader is opened at run time, so machines without a Vulkan driver simply report no GPU.

mod model;
mod vkrt;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::config::{ModelConfig, TrainConfig};
use crate::model::Model;
use crate::rng::Rng;
pub(crate) use model::{Ctx, GpuModel, SegIo, KMAX};

/// Devices kept open between models (the kernels stay compiled).
static IDLE: Mutex<Vec<Ctx>> = Mutex::new(Vec::new());
/// Result of opening and self-testing the GPU, once per process: device and driver names.
static PROBE: OnceLock<Result<(String, String), String>> = OnceLock::new();
/// Set when the GPU failed during a run (lost device, timeout): it is not used again in this
/// process, and its device is left alone rather than reused or torn down.
static BROKEN: AtomicBool = AtomicBool::new(false);

fn mark_broken() {
    BROKEN.store(true, Ordering::Relaxed);
}

fn open_ctx() -> Result<Ctx, String> {
    let mut gpu = vkrt::Gpu::new()?;
    let k = model::Kernels::new(&mut gpu)?;
    Ok(Ctx { gpu, k })
}

/// Called by a dropped model: frees its memory and keeps the device for the next one.
pub(crate) fn recycle(mut ctx: Ctx) {
    if BROKEN.load(Ordering::Relaxed) {
        // A device that failed may still be busy or lost; touching it again risks a hang.
        std::mem::forget(ctx);
        return;
    }
    ctx.gpu.free_buffers();
    if let Ok(mut idle) = IDLE.lock() {
        if idle.len() < 2 {
            idle.push(ctx);
        }
    }
}

fn take_ctx() -> Result<Ctx, String> {
    if let Some(c) = IDLE.lock().ok().and_then(|mut v| v.pop()) {
        return Ok(c);
    }
    open_ctx()
}

/// Opens the GPU and checks it against the CPU model, once per process.
pub(crate) fn probe() -> &'static Result<(String, String), String> {
    PROBE.get_or_init(|| {
        let ctx = open_ctx()?;
        let names = (ctx.gpu.name.clone(), ctx.gpu.driver.clone());
        self_test(ctx).map_err(|e| format!("{}: {e}", names.0))?;
        Ok(names)
    })
}

/// Device and driver of a usable GPU, or why there is none.
pub(crate) fn status() -> Result<(String, String), String> {
    if BROKEN.load(Ordering::Relaxed) {
        return Err("the GPU failed earlier in this process".into());
    }
    probe().clone()
}

/// A GPU model for a stream, or why there is none. `init` is the CPU model's starting weights.
pub(crate) fn model_for(vocab: usize, mc: &ModelConfig, tc: &TrainConfig, init: &[f32]) -> Result<GpuModel, String> {
    status()?;
    GpuModel::supports(vocab, mc, tc)?;
    let ctx = take_ctx()?;
    GpuModel::new(ctx, vocab, mc, tc, init).map_err(|(e, ctx)| {
        recycle(*ctx);
        e
    })
}

/// The per-position interface `codec::run` drives a decoder with (the encoder uses whole
/// segments instead, see `codec::run_gpu_encoder`).
pub(crate) struct GpuNet {
    pub model: GpuModel,
    targets: Vec<u32>,
    train: bool,
    /// Set when the GPU failed during this stream: the codec then codes it again on the CPU.
    pub failed: bool,
}

impl GpuNet {
    pub fn new(model: GpuModel) -> GpuNet {
        let rows = model.n_streams() * model.segment_len();
        GpuNet { model, targets: vec![0; rows], train: false, failed: false }
    }

    /// Records a GPU failure: this stream is redone on the CPU and the GPU is not used again.
    pub fn fail(&mut self, e: String) -> crate::error::Error {
        self.failed = true;
        mark_broken();
        crate::error::Error::Format(format!("GPU: {e}"))
    }

    pub fn begin_segment(&mut self, seg_start: u64) {
        self.model.begin_segment(seg_start);
        self.train = false;
    }

    pub fn forward_step(&mut self, t: usize, inputs: &[u32], active: &[bool]) -> crate::error::Result<()> {
        self.model.forward_position(t, inputs, active).map_err(|e| self.fail(e))
    }

    pub fn probs_row(&self, t: usize, b: usize) -> &[f32] {
        self.model.probs_row(t, b)
    }

    pub fn set_target(&mut self, t: usize, b: usize, sym: u32) {
        self.targets[t * self.model.n_streams() + b] = sym;
    }

    /// Training happens with the memory update in `end_segment` (one submission).
    pub fn train_step(&mut self, _lseg: usize) {
        self.train = true;
    }

    pub fn end_segment(&mut self, lseg: usize) -> crate::error::Result<()> {
        let train = std::mem::replace(&mut self.train, false);
        self.model.finish_segment(lseg, &self.targets, train).map_err(|e| self.fail(e))
    }
}

/// Inputs, active flags and targets of the segment at `pos` (rows t*ns+b), as `codec::run`
/// builds them.
fn segment_io(tokens: &[u32], starts: &[usize], lens: &[usize], bos: u32, pos: usize, lseg: usize) -> (Vec<u32>, Vec<bool>, Vec<u32>) {
    let ns = starts.len();
    let mut inputs = vec![bos; lseg * ns];
    let mut active = vec![false; lseg * ns];
    let mut targets = vec![0u32; lseg * ns];
    for t in 0..lseg {
        let p = pos + t;
        for b in 0..ns {
            let r = t * ns + b;
            active[r] = p < lens[b];
            if active[r] {
                inputs[r] = if p == 0 { bos } else { tokens[starts[b] + p - 1] };
                targets[r] = tokens[starts[b] + p];
            }
        }
    }
    (inputs, active, targets)
}

/// Runs a small network on the CPU and the GPU and compares every probability row and, at the
/// end, every weight. Uneven streams (inactive rows), a window the history wraps around, a
/// vocabulary and a feed-forward width that are not multiples of 8 (the tails of the 8-lane
/// sums), and both the encoder (several segments per submission) and the decoder path.
fn self_test(ctx: Ctx) -> Result<(), String> {
    let _fp = crate::fpenv::FpGuard::new();
    let mc = ModelConfig { d_model: 16, n_layer: 2, n_head: 2, d_ff: 20, window: 5, seg_len: 4, n_streams: 3 };
    let tc = crate::config::preset(1).train;
    let vocab = 11;
    let mut cpu = Model::new(vocab, &mc, &tc);
    // A non-zero output layer, so the first step already sends gradients through every layer.
    let mut rng = Rng::new(0x5E1F_7E57);
    let np = cpu.params().len();
    for x in &mut cpu.params_mut()[np - mc.d_model as usize * vocab..] {
        *x = rng.uniform() * 0.5;
    }
    let mut gpu = GpuModel::new(ctx, vocab, &mc, &tc, cpu.params()).map_err(|(e, ctx)| {
        recycle(*ctx);
        e
    })?;
    let n = 3 * 4 * 9 + 2;
    let tokens: Vec<u32> = (0..n).map(|_| (rng.next_u64() % vocab as u64) as u32).collect();
    let (ns, seg) = (3usize, 4usize);
    let (starts, lens) = crate::codec::split_streams(n, ns);
    let max_len = lens[0];
    let bos = vocab as u32;
    let mut pos = 0;
    let mut batch = 0;
    while pos < max_len {
        // Alternate between three-segment encoder submissions and decoder segments.
        let mut segs = Vec::new();
        let mut p = pos;
        let want = if batch % 2 == 0 { 3 } else { 1 };
        while p < max_len && segs.len() < want {
            let lseg = seg.min(max_len - p);
            segs.push((p, lseg, segment_io(&tokens, &starts, &lens, bos, p, lseg)));
            p += lseg;
        }
        if want == 3 {
            let sio: Vec<SegIo> = segs
                .iter()
                .map(|(p, l, io)| SegIo { pos: *p as u64, lseg: *l, inputs: &io.0, active: &io.1, targets: &io.2, train: p + l < max_len })
                .collect();
            gpu.run_segments(&sio)?;
        }
        for (k, (sp, lseg, io)) in segs.iter().enumerate() {
            let (sp, lseg) = (*sp, *lseg);
            cpu.begin_segment(sp as u64);
            if want == 1 {
                gpu.begin_segment(sp as u64);
            }
            for t in 0..lseg {
                let rows = t * ns..(t + 1) * ns;
                cpu.forward_step(t, &io.0[rows.clone()], &io.1[rows.clone()]);
                if want == 1 {
                    gpu.forward_position(t, &io.0[rows.clone()], &io.1[rows])?;
                }
                for b in 0..ns {
                    let g = if want == 1 { gpu.probs_row(t, b) } else { gpu.probs_row_k(k, t, b) };
                    if cpu.probs_row(t, b).iter().zip(g).any(|(x, y)| x.to_bits() != y.to_bits()) {
                        return Err(format!("probabilities differ from the CPU at position {}", sp + t));
                    }
                    if io.1[t * ns + b] {
                        cpu.set_target(t, b, io.2[t * ns + b]);
                    }
                }
            }
            let train = sp + lseg < max_len;
            if train {
                cpu.train_step(lseg);
            }
            cpu.end_segment(lseg);
            if want == 1 {
                gpu.finish_segment(lseg, &io.2, train)?;
            }
        }
        pos = p;
        batch += 1;
    }
    if gpu.read_params()?.iter().zip(cpu.params()).any(|(x, y)| x.to_bits() != y.to_bits()) {
        return Err("weights differ from the CPU after training".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPV: &[(&str, &[u8])] = &[
        ("adam", include_bytes!("spv/adam.spv")),
        ("attn_bwd_kv", include_bytes!("spv/attn_bwd_kv.spv")),
        ("attn_bwd_q", include_bytes!("spv/attn_bwd_q.spv")),
        ("attn_fwd", include_bytes!("spv/attn_fwd.spv")),
        ("dlogits", include_bytes!("spv/dlogits.spv")),
        ("embed", include_bytes!("spv/embed.spv")),
        ("embed_grad", include_bytes!("spv/embed_grad.spv")),
        ("end_segment", include_bytes!("spv/end_segment.spv")),
        ("fill", include_bytes!("spv/fill.spv")),
        ("gainsum", include_bytes!("spv/gainsum.spv")),
        ("matmul", include_bytes!("spv/matmul.spv")),
        ("matmul16", include_bytes!("spv/matmul16.spv")),
        ("matmul_nt", include_bytes!("spv/matmul_nt.spv")),
        ("matmul_nt16", include_bytes!("spv/matmul_nt16.spv")),
        ("postnorm", include_bytes!("spv/postnorm.spv")),
        ("qknorm_bwd", include_bytes!("spv/qknorm_bwd.spv")),
        ("qknorm_rope", include_bytes!("spv/qknorm_rope.spv")),
        ("rmsnorm", include_bytes!("spv/rmsnorm.spv")),
        ("rmsnorm_bwd", include_bytes!("spv/rmsnorm_bwd.spv")),
        ("softmax", include_bytes!("spv/softmax.spv")),
        ("swiglu", include_bytes!("spv/swiglu.spv")),
        ("swiglu_bwd", include_bytes!("spv/swiglu_bwd.spv")),
    ];

    /// Every float multiply, add and subtract in every kernel carries NoContraction (no driver
    /// may fuse it into an FMA), and every kernel declares the float-control modes.
    #[test]
    fn spirv_is_contraction_free_and_ieee() {
        const OP_EXT_INST: u32 = 12;
        const OP_EXECUTION_MODE: u32 = 16;
        const OP_DECORATE: u32 = 71;
        const DEC_NO_CONTRACTION: u32 = 42;
        // OpFAdd, OpFSub, OpFMul, OpVectorTimesScalar, OpMatrixTimesVector, OpDot
        const FLOAT_OPS: [u32; 6] = [129, 131, 133, 142, 145, 148];
        // OpFDiv, OpFRem, OpFMod: not correctly rounded in Vulkan (f32_div is used instead).
        const BANNED_OPS: [u32; 3] = [136, 140, 141];
        // GLSL.std.450 Pow, Exp, Log, Exp2, Log2, Sqrt, InverseSqrt, FMin, FMax, FClamp, Fma,
        // NMin, NMax, NClamp: precision or NaN handling is left to the driver.
        const BANNED_EXT: [u32; 14] = [26, 27, 28, 29, 30, 31, 32, 37, 40, 43, 50, 79, 80, 81];
        // DenormPreserve, SignedZeroInfNanPreserve, RoundingModeRTE
        const MODES: [u32; 3] = [4459, 4461, 4462];
        assert_eq!(SPV.len(), std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/spv")).unwrap().count());
        for (name, bytes) in SPV {
            assert_eq!(bytes.len() % 4, 0, "{name}");
            let w: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            assert_eq!(w[0], 0x0723_0203, "{name}: not SPIR-V");
            let mut protected = std::collections::HashSet::new();
            let mut results = Vec::new();
            let mut modes = std::collections::HashSet::new();
            let mut i = 5;
            while i < w.len() {
                let (count, op) = ((w[i] >> 16) as usize, w[i] & 0xFFFF);
                assert!(count > 0, "{name}: malformed instruction");
                match op {
                    OP_DECORATE if w[i + 2] == DEC_NO_CONTRACTION => {
                        protected.insert(w[i + 1]);
                    }
                    OP_EXECUTION_MODE if count >= 4 && w[i + 3] == 32 => {
                        modes.insert(w[i + 2]);
                    }
                    OP_EXT_INST => assert!(!BANNED_EXT.contains(&w[i + 4]), "{name}: GLSL.std.450 instruction {} is not bit-exact", w[i + 4]),
                    _ if BANNED_OPS.contains(&op) => panic!("{name}: opcode {op} is not correctly rounded in Vulkan"),
                    _ if FLOAT_OPS.contains(&op) => results.push(w[i + 2]),
                    _ => {}
                }
                i += count;
            }
            assert!(!results.is_empty() || *name == "embed" || *name == "end_segment" || *name == "fill", "{name}: no float ops?");
            for r in results {
                assert!(protected.contains(&r), "{name}: float operation %{r} lacks NoContraction");
            }
            for m in MODES {
                assert!(modes.contains(&m), "{name}: float-control execution mode {m} missing");
            }
        }
    }

    /// The constants in shaders/consts.glsl are the core's f32 literals, bit for bit.
    // Spelled out like fpmath.rs does: they are part of the format, not std's constants.
    #[allow(clippy::approx_constant, clippy::excessive_precision)]
    #[test]
    fn constants_match_the_core() {
        let text = include_str!("shaders/consts.glsl");
        let want: &[(&str, f32)] = &[
            ("C_LOG2E", 1.442_695_f32),
            ("C_LN2_HI", 0.693_145_75_f32),
            ("C_LN2_LO", 1.428_606_8e-6_f32),
            ("C_P3", 0.166_666_67_f32),
            ("C_P4", 0.041_666_668_f32),
            ("C_P5", 0.008_333_334_f32),
            ("C_P6", 0.001_388_888_9_f32),
            ("C_EXP_FLOOR", crate::fpmath::EXP_FLOOR),
            ("C_EXP_LO", -80.0_f32),
            ("C_EXP_HI", 80.0_f32),
            ("C_NORM_EPS", 1e-6_f32),
            ("C_ADAM_ZERO", 1e-30_f32),
        ];
        for (name, v) in want {
            let line = text.lines().find(|l| l.starts_with(&format!("#define {name} "))).unwrap_or_else(|| panic!("{name} missing"));
            let hex = line.split("0x").nth(1).and_then(|h| h.get(..8)).unwrap();
            assert_eq!(u32::from_str_radix(hex, 16).unwrap(), v.to_bits(), "{name}");
        }
        // The same literals the CPU uses (spot checks against fpmath).
        assert_eq!(crate::fpmath::exp(0.0), 1.0);
    }

    /// With a usable GPU: the self-test passes and compression is byte-identical to the CPU's,
    /// in both directions. Without one the test only reports why.
    #[test]
    fn gpu_matches_cpu_when_present() {
        match probe() {
            Err(e) => eprintln!("no usable GPU, skipped: {e}"),
            Ok((name, _)) => {
                eprintln!("GPU: {name}");
                let mut data = Vec::new();
                for i in 0..800u32 {
                    data.extend_from_slice(format!("record {i}: value {} flag {}\n", i * 37 % 1001, i % 7 == 0).as_bytes());
                }
                let level = 2;
                let mut o = crate::CompressOptions::level(level);
                o.gpu = crate::GpuMode::Off;
                let cpu = crate::compress(&data, &o).unwrap();
                o.gpu = crate::GpuMode::On;
                let gpu = crate::compress(&data, &o).unwrap();
                assert_eq!(cpu, gpu, "level {level}: GPU output differs from the CPU's");
                let d = crate::DecompressOptions { gpu: crate::GpuMode::On, ..Default::default() };
                assert_eq!(crate::decompress_with_progress(&gpu, &d, &mut |_, _| true).unwrap(), data);
            }
        }
    }
}
