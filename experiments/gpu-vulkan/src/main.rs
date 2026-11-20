//! GPU experiment for PiedPiper: can the online-learning Transformer run on a Vulkan GPU and
//! still produce exactly the CPU's bits (so a file compressed on a GPU decompresses on any CPU,
//! and the other way round), and how much faster is it?
//!
//!   piedpiper-gpu-experiment info
//!   piedpiper-gpu-experiment arith [N]                 soft-float div/sqrt, exp, sigmoid vs CPU
//!   piedpiper-gpu-experiment kernels                   matmul kernels vs the CPU kernels
//!   piedpiper-gpu-experiment check FILE [options]      whole model, CPU vs GPU, bit for bit
//!   piedpiper-gpu-experiment bench FILE [options]      tokens/s: CPU (1 / all threads) vs GPU
//!
//! options: --level L  --streams N  --seg S  --tokens T  --vocab V  --mode enc|dec|batch|both
//!          --batch K (segments per submission)  --params-every K  --cpu-threads N,M
//!          --skip-cpu  --profile

mod shaders {
    include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
}

// The CPU reference is the core's own model, compiled from its sources unchanged.
#[allow(dead_code)]
#[path = "../../../crates/core/src/config.rs"]
mod config;
#[allow(dead_code)]
#[path = "../../../crates/core/src/error.rs"]
mod error;
#[allow(dead_code)]
#[path = "../../../crates/core/src/fpenv.rs"]
mod fpenv;
#[allow(dead_code)]
#[path = "../../../crates/core/src/fpmath.rs"]
mod fpmath;
#[allow(dead_code)]
#[path = "../../../crates/core/src/kernels.rs"]
mod kernels;
#[allow(dead_code)]
#[path = "../../../crates/core/src/model.rs"]
mod model;
#[allow(dead_code)]
#[path = "../../../crates/core/src/par.rs"]
mod par;
#[allow(dead_code)]
#[path = "../../../crates/core/src/rng.rs"]
mod rng;
#[allow(dead_code)]
#[path = "../../../crates/core/src/simd.rs"]
mod simd;

mod gpu_model;
mod vkrt;

use gpu_model::{GpuModel, SegIo, KMAX};
use model::Model;
use rng::Rng;
use std::time::Instant;
use vkrt::{Gpu, Res};

// ---------------------------------------------------------------------------------------------
// options and inputs
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Opts {
    file: Option<String>,
    level: u32,
    streams: Option<u32>,
    seg: Option<u32>,
    tokens: usize,
    vocab: usize,
    mode: String,
    params_every: usize,
    cpu_threads: Vec<usize>,
    skip_cpu: bool,
    profile: bool,
    batch: usize,
}

fn parse(args: &[String]) -> Res<Opts> {
    let mut o = Opts {
        file: None,
        level: 2,
        streams: None,
        seg: None,
        tokens: 65536,
        vocab: 256,
        mode: "both".into(),
        params_every: 16,
        cpu_threads: vec![1, 0],
        skip_cpu: false,
        profile: false,
        batch: 1,
    };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut val = || -> Res<String> {
            i += 1;
            args.get(i).cloned().ok_or(format!("{a} needs a value"))
        };
        match a {
            "--level" => o.level = val()?.parse().map_err(|_| "bad --level")?,
            "--streams" => o.streams = Some(val()?.parse().map_err(|_| "bad --streams")?),
            "--seg" => o.seg = Some(val()?.parse().map_err(|_| "bad --seg")?),
            "--tokens" => o.tokens = val()?.parse().map_err(|_| "bad --tokens")?,
            "--vocab" => o.vocab = val()?.parse().map_err(|_| "bad --vocab")?,
            "--mode" => o.mode = val()?,
            "--params-every" => o.params_every = val()?.parse().map_err(|_| "bad --params-every")?,
            "--cpu-threads" => {
                o.cpu_threads = val()?.split(',').map(|s| s.parse().map_err(|_| "bad --cpu-threads".to_string())).collect::<Res<_>>()?
            }
            "--skip-cpu" => o.skip_cpu = true,
            "--profile" => o.profile = true,
            "--batch" => o.batch = val()?.parse().map_err(|_| "bad --batch")?,
            s if !s.starts_with("--") && o.file.is_none() => o.file = Some(s.to_string()),
            s => return Err(format!("unknown option {s}")),
        }
        i += 1;
    }
    Ok(o)
}

/// Bytes, or (vocab > 256) bytes plus the most frequent byte pairs as extra tokens, merged
/// greedily. The model only cares about the vocabulary size, not what tokens mean.
fn tokenize(data: &[u8], vocab: usize) -> Vec<u32> {
    if vocab <= 256 {
        return data.iter().map(|&b| b as u32).collect();
    }
    let mut counts = vec![0u32; 65536];
    for w in data.windows(2) {
        counts[(w[0] as usize) << 8 | w[1] as usize] += 1;
    }
    let mut idx: Vec<usize> = (0..65536).collect();
    idx.sort_by_key(|&i| (std::cmp::Reverse(counts[i]), i));
    let mut map = vec![u32::MAX; 65536];
    for (k, &i) in idx.iter().take(vocab - 256).enumerate() {
        if counts[i] > 0 {
            map[i] = 256 + k as u32;
        }
    }
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 1 < data.len() {
            let p = (data[i] as usize) << 8 | data[i + 1] as usize;
            if map[p] != u32::MAX {
                out.push(map[p]);
                i += 2;
                continue;
            }
        }
        out.push(data[i] as u32);
        i += 1;
    }
    out
}

/// codec.rs split_streams: stream b covers starts[b] .. starts[b] + lens[b].
fn split_streams(n: usize, ns: usize) -> (Vec<usize>, Vec<usize>) {
    let base = n / ns;
    let extra = n % ns;
    let mut starts = Vec::with_capacity(ns);
    let mut lens = Vec::with_capacity(ns);
    let mut s = 0;
    for b in 0..ns {
        let l = base + usize::from(b < extra);
        starts.push(s);
        lens.push(l);
        s += l;
    }
    (starts, lens)
}

struct Job {
    tokens: Vec<u32>,
    vocab: usize,
    mc: config::ModelConfig,
    tc: config::TrainConfig,
    starts: Vec<usize>,
    lens: Vec<usize>,
}

impl Job {
    fn new(o: &Opts) -> Res<Job> {
        let path = o.file.as_ref().ok_or("missing FILE")?;
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let mut tokens = tokenize(&data, o.vocab);
        tokens.truncate(o.tokens);
        let vocab = o.vocab.max(256);
        let p = config::preset(o.level);
        let mut mc = p.model;
        // Default batch shape: the codec's own policy for this many tokens.
        let (ns, seg) = config::batch_policy(tokens.len(), mc.n_streams);
        mc.n_streams = o.streams.unwrap_or(ns);
        mc.seg_len = o.seg.unwrap_or(seg);
        let tc = p.train;
        mc.validate(vocab).map_err(|e| e.to_string())?;
        let (starts, lens) = split_streams(tokens.len(), mc.n_streams as usize);
        Ok(Job { tokens, vocab, mc, tc, starts, lens })
    }

    fn ns(&self) -> usize {
        self.mc.n_streams as usize
    }
    fn seg(&self) -> usize {
        self.mc.seg_len as usize
    }
    fn max_len(&self) -> usize {
        self.lens[0]
    }

    /// Inputs, active flags and targets of rows t*ns+b for the segment starting at `pos`.
    fn segment(&self, pos: usize, lseg: usize) -> (Vec<u32>, Vec<bool>, Vec<u32>) {
        let ns = self.ns();
        let bos = self.vocab as u32;
        let mut inputs = vec![bos; lseg * ns];
        let mut active = vec![false; lseg * ns];
        let mut targets = vec![0u32; lseg * ns];
        for t in 0..lseg {
            let p = pos + t;
            for b in 0..ns {
                let r = t * ns + b;
                active[r] = p < self.lens[b];
                inputs[r] = if p == 0 || !active[r] { bos } else { self.tokens[self.starts[b] + p - 1] };
                if active[r] {
                    targets[r] = self.tokens[self.starts[b] + p];
                }
            }
        }
        (inputs, active, targets)
    }

    fn describe(&self) -> String {
        let m = &self.mc;
        format!(
            "{} tokens, vocab {}, d{} x {} layers ({} heads, ff {}), window {}, {} streams x seg {} ({} rows/update), {:.2}M params",
            self.tokens.len(),
            self.vocab,
            m.d_model,
            m.n_layer,
            m.n_head,
            m.d_ff,
            m.window,
            m.n_streams,
            m.seg_len,
            m.n_streams * m.seg_len,
            m.n_params(self.vocab) as f64 / 1e6
        )
    }
}

/// One CPU segment through the real model: forward every position, set targets.
fn cpu_forward_segment(cpu: &mut Model, job: &Job, pos: usize, lseg: usize, io: &(Vec<u32>, Vec<bool>, Vec<u32>)) {
    let ns = job.ns();
    cpu.begin_segment(pos as u64);
    for t in 0..lseg {
        cpu.forward_step(t, &io.0[t * ns..(t + 1) * ns], &io.1[t * ns..(t + 1) * ns]);
        for b in 0..ns {
            if io.1[t * ns + b] {
                cpu.set_target(t, b, io.2[t * ns + b]);
            }
        }
    }
}

fn cpu_finish_segment(cpu: &mut Model, job: &Job, pos: usize, lseg: usize) {
    if pos + lseg < job.max_len() {
        cpu.train_step(lseg);
    }
    cpu.end_segment(lseg);
}

fn bits_equal(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits())
}

// ---------------------------------------------------------------------------------------------
// info / arith / kernels
// ---------------------------------------------------------------------------------------------

fn cmd_info() -> Res<()> {
    let g = Gpu::new()?;
    println!("device   {}", g.name);
    println!("driver   {}", g.driver);
    println!("float controls: denormals preserved, round-to-nearest-even, signed zero/Inf/NaN preserved (required, present)");
    println!("max push constants {} B, max shared memory {} B", g.limits.max_push_constants_size, g.limits.max_compute_shared_memory_size);
    Ok(())
}

/// Test inputs: special values, uniformly random bit patterns (every exponent, subnormals,
/// NaNs) and ordinary magnitudes.
fn test_values(rng: &mut Rng, n: usize) -> Vec<f32> {
    let special = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        3.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x007F_FFFF),
        f32::from_bits(0x0040_0000),
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-30,
        1e-38,
        1e38,
        1.0 + f32::EPSILON,
        1.0 - f32::EPSILON / 2.0,
    ];
    let mut v: Vec<f32> = special.to_vec();
    while v.len() < n {
        let r = rng.next_u64();
        let x = match r % 4 {
            0 | 1 => f32::from_bits((r >> 32) as u32),
            2 => (rng.uniform() * 4.0).exp2_det(),
            _ => rng.uniform() * 1000.0,
        };
        v.push(x);
    }
    v
}

trait Exp2Det {
    fn exp2_det(self) -> f32;
}
impl Exp2Det for f32 {
    /// A spread of magnitudes without libm: 2^(round(16x)) scaled by a mantissa.
    fn exp2_det(self) -> f32 {
        let e = (self * 16.0) as i32;
        f32::from_bits((((e + 127).clamp(1, 254)) as u32) << 23) * (1.0 + self.abs() / 8.0)
    }
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn cmd_arith(n: usize) -> Res<()> {
    let _fp = fpenv::FpGuard::new();
    let mut g = Gpu::new()?;
    let k = g.kernel("t_arith", shaders::T_ARITH)?;
    let mut rng = Rng::new(2026);
    let a = test_values(&mut rng, n);
    let mut b = test_values(&mut rng, n);
    // Pair every special value with every other one at the front.
    let ns = 22;
    for i in 0..ns * ns {
        if i < n {
            b[i] = a[i % ns];
        }
    }
    let a: Vec<f32> = (0..n).map(|i| if i < ns * ns { a[i / ns] } else { a[i] }).collect();
    let pool = g.buffer((3 * n * 4) as u64, false)?;
    let host = g.buffer((3 * n * 4) as u64, true)?;
    g.bind_pool(&pool);
    host.write_f32(0, &a);
    host.write_f32(n, &b);
    let names = ["soft a/b", "native a/b", "soft sqrt", "native sqrt", "exp (fpmath)", "sigmoid (fpmath)"];
    println!("{n} inputs per operation on {}", g.name);
    for mode in 0..6u32 {
        g.begin();
        g.copy(&host, 0, &pool, 0, 2 * n);
        g.dispatch_1d(&k, n, &[n as u32, mode, 0, n as u32, 2 * n as u32]);
        g.copy(&pool, 2 * n, &host, 2 * n, n);
        g.submit();
        let out = host.f32s(2 * n, n);
        let mut bad = 0usize;
        let mut first = None;
        for i in 0..n {
            // exp/sigmoid are defined on clamped inputs; feed them moderate values.
            let want = match mode {
                0 | 1 => a[i] / b[i],
                2 | 3 => a[i].sqrt(),
                4 => fpmath::exp(a[i]),
                _ => fpmath::sigmoid(a[i]),
            };
            if !same(out[i], want) {
                bad += 1;
                if first.is_none() {
                    first = Some((i, want, out[i]));
                }
            }
        }
        let note = match first {
            None => "bit-identical to the CPU".to_string(),
            Some((i, w, o)) => format!(
                "first: a={:e} b={:e}: CPU {:08x}, GPU {:08x}",
                a[i],
                b[i],
                w.to_bits(),
                o.to_bits()
            ),
        };
        println!("  {:<17} {:>9} mismatches  ({:.4}%)  {note}", names[mode as usize], bad, 100.0 * bad as f64 / n as f64);
    }
    Ok(())
}

/// About half a second of large matmuls, so the GPU leaves its idle clocks before timing.
fn warm_up(g: &mut Gpu, mm: &vkrt::Kernel, pool_words: usize) {
    assert!(pool_words >= 3 << 20);
    for _ in 0..10 {
        g.begin();
        for _ in 0..40 {
            g.dispatch(mm, [32, 32, 1], &[1024, 1024, 1024, 0, 1024, 1, 1 << 20, 1024, 1, 2 << 20, 1024]);
        }
        g.submit();
    }
}

fn cmd_overhead() -> Res<()> {
    let mut g = Gpu::new()?;
    let fill = g.kernel("fill", shaders::FILL)?;
    let mm = g.kernel("matmul", shaders::MATMUL)?;
    let words = 4 << 20;
    let pool = g.buffer((words * 4) as u64, false)?;
    let host = g.buffer(4096, true)?;
    g.bind_pool(&pool);
    warm_up(&mut g, &mm, words);
    println!("dispatch overhead on {} (after warm-up):", g.name);
    for mode in [vkrt::Barriers::Full, vkrt::Barriers::Compute, vkrt::Barriers::Off] {
        g.barriers = mode;
        g.begin();
        for _ in 0..2000 {
            g.dispatch(&fill, [1, 1, 1], &[0, 64]);
        }
        let t = g.submit();
        println!("  tiny dispatch followed by {:<8} barrier: {:6.2} us of GPU time each", format!("{mode:?}"), t / 2000.0 * 1e6);
    }
    g.barriers = vkrt::Barriers::Full;
    for spin in [false, true] {
        g.spin = spin;
        let mut rt: Vec<f64> = (0..300).map(|_| g.roundtrip()).collect();
        rt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  empty submit + {} round trip: median {:.1} us, p90 {:.1} us",
            if spin { "spin-wait" } else { "vkWaitForFences" },
            rt[150] * 1e6,
            rt[270] * 1e6
        );
    }
    let mut rt2: Vec<f64> = (0..300)
        .map(|_| {
            let t0 = Instant::now();
            host.write_u32(0, &[1, 2, 3, 4]);
            g.begin();
            g.copy(&host, 0, &pool, 0, 64);
            g.dispatch(&fill, [1, 1, 1], &[64, 64]);
            g.copy(&pool, 0, &host, 64, 256);
            g.submit();
            t0.elapsed().as_secs_f64()
        })
        .collect();
    rt2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  upload + 1 dispatch + readback round trip: median {:.1} us, p90 {:.1} us", rt2[150] * 1e6, rt2[270] * 1e6);
    Ok(())
}

fn cmd_kernels() -> Res<()> {
    let _fp = fpenv::FpGuard::new();
    let mut g = Gpu::new()?;
    let mm = g.kernel("matmul", shaders::MATMUL)?;
    let mm_fast = g.kernel("matmul_fast", shaders::MATMUL_FAST)?;
    let mm_nofc = g.kernel("matmul_fast_nofc", shaders::MATMUL_FAST_NOFC)?;
    let nt = g.kernel("matmul_nt", shaders::MATMUL_NT)?;
    // (rows, k, n): forward projections per position and per segment, and the output head,
    // at levels 2, 5 and 9 (rows = streams for one position, streams * seg for a segment).
    let shapes: &[(usize, usize, usize)] = &[
        (24, 96, 288),
        (384, 96, 288),
        (384, 96, 512),
        (384, 256, 96),
        (384, 96, 2048),
        (768, 192, 576),
        (768, 192, 1024),
        (1024, 512, 1536),
        (1024, 512, 2816),
        (1024, 1408, 512),
        (1024, 512, 4096),
        (7, 33, 70),
    ];
    let maxw = shapes.iter().map(|&(r, k, n)| r * k + k * n + r * n + n * k + r.max(k) * n).max().unwrap().max(3 << 20);
    let pool = g.buffer((maxw * 4) as u64, false)?;
    let host = g.buffer((maxw * 4) as u64, true)?;
    g.bind_pool(&pool);
    warm_up(&mut g, &mm, maxw);
    let mut rng = Rng::new(77);
    println!("matmul kernels on {}: GPU (precise) vs crates/core/src/kernels.rs; elements that differ without `precise`", g.name);
    println!(
        "{:>18}  {:>10} {:>10} {:>10}  {:>9} {:>9} {:>9}  {:>9} {:>9}",
        "rows x k x n", "matmul", "tn", "nt", "GPU us", "CPU1 us", "GF/s GPU", "no-prec", "no-prec+fc"
    );
    for &(rows, k, n) in shapes {
        // x: rows x k, w: k x n, plus a rows x n "b" for tn/nt.
        let x: Vec<f32> = (0..rows * k).map(|_| rng.uniform()).collect();
        let w: Vec<f32> = (0..k * n).map(|_| rng.uniform() * 0.1).collect();
        let bm: Vec<f32> = (0..rows * n).map(|_| rng.uniform()).collect();
        let wt: Vec<f32> = (0..n * k).map(|_| rng.uniform() * 0.1).collect(); // n x k (rows of length k) for nt
        let (ox, ow, ob, owt) = (0, rows * k, rows * k + k * n, rows * k + k * n + rows * n);
        let oy = owt + n * k;
        host.write_f32(ox, &x);
        host.write_f32(ow, &w);
        host.write_f32(ob, &bm);
        host.write_f32(owt, &wt);
        let total = oy + rows * n.max(k);
        let up = |g: &mut Gpu| {
            g.begin();
            g.copy(&host, 0, &pool, 0, oy);
            g.submit();
        };
        up(&mut g);
        let run = |g: &mut Gpu, kern: &vkrt::Kernel, push: &[u32], gx: usize, gy: usize, ylen: usize| -> (Vec<f32>, f64) {
            // warm-up, then time 20 dispatches in one submission
            g.begin();
            g.dispatch(kern, [gx as u32, gy as u32, 1], push);
            g.submit();
            g.begin();
            for _ in 0..20 {
                g.dispatch(kern, [gx as u32, gy as u32, 1], push);
            }
            let t = g.submit() / 20.0;
            g.begin();
            g.copy(&pool, oy, &host, oy, ylen);
            g.submit();
            (host.f32s(oy, ylen).to_vec(), t)
        };
        let u = |v: usize| v as u32;
        // matmul: y[rows x n] = x · w
        let push = [u(rows), u(n), u(k), u(ox), u(k), 1, u(ow), u(n), 1, u(oy), u(n)];
        let (gy, t_mm) = run(&mut g, &mm, &push, n.div_ceil(32), rows.div_ceil(32), rows * n);
        let mut cy = vec![0f32; rows * n];
        let t0 = Instant::now();
        let reps = (2e8 / (rows * k * n) as f64).clamp(1.0, 50.0) as usize;
        rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| {
            for _ in 0..reps {
                kernels::matmul(&mut cy, &x, &w, rows, k, n);
            }
        });
        let t_cpu = t0.elapsed().as_secs_f64() / reps as f64;
        let ok_mm = bits_equal(&gy, &cy).is_none();
        let (gyf, _) = run(&mut g, &mm_fast, &push, n.div_ceil(32), rows.div_ceil(32), rows * n);
        let fma_bad = gyf.iter().zip(&cy).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let (gyn, _) = run(&mut g, &mm_nofc, &push, n.div_ceil(32), rows.div_ceil(32), rows * n);
        let nofc_bad = gyn.iter().zip(&cy).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        // tn: y[k x n] = xᵀ · b   (x is rows x k, b is rows x n)
        let push = [u(k), u(n), u(rows), u(ox), 1, u(k), u(ob), u(n), 1, u(oy), u(n)];
        let (gt, _) = run(&mut g, &mm, &push, n.div_ceil(32), k.div_ceil(32), k * n);
        let mut ct = vec![0f32; k * n];
        kernels::matmul_tn(&mut ct, &x, &bm, rows, k, n);
        let ok_tn = bits_equal(&gt, &ct).is_none();
        // nt: y[rows x n] = x · wtᵀ  (wt is n x k)
        let push = [u(rows), u(n), u(k), u(ox), u(owt), u(oy)];
        let (gn, _) = run(&mut g, &nt, &push, n.div_ceil(32), rows.div_ceil(32), rows * n);
        let mut cn = vec![0f32; rows * n];
        kernels::matmul_nt(&mut cn, &x, &wt, rows, k, n);
        let ok_nt = bits_equal(&gn, &cn).is_none();
        let _ = total;
        let flops = 2.0 * (rows * k * n) as f64;
        let mark = |b: bool| if b { "identical" } else { "DIFFERENT" };
        println!(
            "{:>18}  {:>10} {:>10} {:>10}  {:>9.1} {:>9.1} {:>9.0}  {:>8.1}% {:>8.1}%",
            format!("{rows}x{k}x{n}"),
            mark(ok_mm),
            mark(ok_tn),
            mark(ok_nt),
            t_mm * 1e6,
            t_cpu * 1e6,
            flops / t_mm / 1e9,
            100.0 * nofc_bad as f64 / (rows * n) as f64,
            100.0 * fma_bad as f64 / (rows * n) as f64
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// check: the whole model, CPU vs GPU
// ---------------------------------------------------------------------------------------------

fn cmd_check(o: &Opts) -> Res<()> {
    let _fp = fpenv::FpGuard::new();
    let job = Job::new(o)?;
    println!("{}", job.describe());
    let modes: Vec<&str> = match o.mode.as_str() {
        "both" => vec!["enc", "dec", "batch"],
        m => vec![m],
    };
    let kb = o.batch.clamp(2, KMAX);
    for mode in modes {
        let mut cpu = Model::new(job.vocab, &job.mc, &job.tc);
        let gpu = Gpu::new()?;
        let name = gpu.name.clone();
        let mut gm = GpuModel::new(gpu, job.vocab, &job.mc, &job.tc, cpu.params())?;
        let (ns, seg, max_len) = (job.ns(), job.seg(), job.max_len());
        let (mut rows_cmp, mut rows_bad, mut segs, mut param_checks) = (0usize, 0usize, 0usize, 0usize);
        let mut first_bad: Option<String> = None;
        let mut params_ok = true;
        let mut pos = 0;
        let t0 = Instant::now();
        let mut group_start = 0usize;
        while pos < max_len {
            let lseg = seg.min(max_len - pos);
            let io = job.segment(pos, lseg);
            let train = pos + lseg < max_len;
            if mode == "batch" && segs % kb == 0 {
                // The next kb segments go to the GPU in one submission.
                let mut ios = Vec::new();
                let mut p = pos;
                while p < max_len && ios.len() < kb {
                    let l = seg.min(max_len - p);
                    ios.push((p, l, job.segment(p, l), p + l < max_len));
                    p += l;
                }
                let sio: Vec<SegIo> = ios
                    .iter()
                    .map(|(p, l, io, tr)| SegIo { pos: *p as u64, lseg: *l, inputs: &io.0, active: &io.1, targets: &io.2, train: *tr })
                    .collect();
                gm.run_segments(&sio);
                group_start = segs;
            }
            cpu_forward_segment(&mut cpu, &job, pos, lseg, &io);
            gm.begin_segment(pos as u64);
            if mode == "enc" {
                gm.run_segment(lseg, &io.0, &io.1, &io.2, train);
            }
            let kslot = if mode == "batch" { segs - group_start } else { 0 };
            for t in 0..lseg {
                if mode == "dec" {
                    gm.forward_position(t, &io.0[t * ns..(t + 1) * ns], &io.1[t * ns..(t + 1) * ns]);
                }
                for b in 0..ns {
                    rows_cmp += 1;
                    if let Some(i) = bits_equal(cpu.probs_row(t, b), gm.probs_row_k(kslot, t, b)) {
                        rows_bad += 1;
                        if first_bad.is_none() {
                            first_bad = Some(format!(
                                "segment at {pos}, t {t}, stream {b}, symbol {i}: CPU {:08x} GPU {:08x}",
                                cpu.probs_row(t, b)[i].to_bits(),
                                gm.probs_row_k(kslot, t, b)[i].to_bits()
                            ));
                        }
                    }
                }
            }
            if mode == "dec" {
                gm.finish_segment(lseg, &io.2, train);
            }
            cpu_finish_segment(&mut cpu, &job, pos, lseg);
            segs += 1;
            pos += lseg;
            let group_done = mode != "batch" || segs % kb == 0 || pos >= max_len;
            if o.params_every > 0 && group_done && (segs % o.params_every == 0 || pos >= max_len || mode == "batch") {
                param_checks += 1;
                let gp = gm.read_params();
                if let Some(i) = bits_equal(cpu.params(), &gp) {
                    params_ok = false;
                    if first_bad.is_none() {
                        first_bad = Some(format!(
                            "after segment {segs}: parameter {i} CPU {:08x} GPU {:08x}",
                            cpu.params()[i].to_bits(),
                            gp[i].to_bits()
                        ));
                    }
                }
            }
            if first_bad.is_some() {
                break;
            }
        }
        let mode = if mode == "batch" { format!("batch{kb}") } else { mode.to_string() };
        println!(
            "[{mode}] {name}: {segs} segments ({} Adam steps), {rows_cmp} probability rows of {} symbols, {param_checks} full weight comparisons in {:.1}s",
            if segs > 0 { segs - 1 } else { 0 },
            job.vocab,
            t0.elapsed().as_secs_f64()
        );
        match first_bad {
            None => println!("[{mode}] RESULT: every probability and every weight is bit-identical to the CPU model"),
            Some(s) => println!("[{mode}] RESULT: MISMATCH ({rows_bad} rows differ, weights equal: {params_ok}); first: {s}"),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------------------------

fn cmd_bench(o: &Opts) -> Res<()> {
    let _fp = fpenv::FpGuard::new();
    let job = Job::new(o)?;
    println!("{}", job.describe());
    let (ns, seg, max_len) = (job.ns(), job.seg(), job.max_len());
    let n_tok = job.tokens.len() as f64;
    if !o.skip_cpu {
        for &th in &o.cpu_threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(th).build().unwrap();
            let threads = pool.current_num_threads();
            let secs = pool.install(|| {
                let _fp = fpenv::FpGuard::new();
                let mut cpu = Model::new(job.vocab, &job.mc, &job.tc);
                let t0 = Instant::now();
                let mut pos = 0;
                while pos < max_len {
                    let lseg = seg.min(max_len - pos);
                    let io = job.segment(pos, lseg);
                    cpu_forward_segment(&mut cpu, &job, pos, lseg, &io);
                    cpu_finish_segment(&mut cpu, &job, pos, lseg);
                    pos += lseg;
                }
                t0.elapsed().as_secs_f64()
            });
            println!("CPU  {threads:>2} thread(s): {:>8.2}s  {:>9.0} tokens/s", secs, n_tok / secs);
        }
    }
    for mode in ["enc", "dec"] {
        if o.mode != "both" && o.mode != mode {
            continue;
        }
        let cpu_init = Model::new(job.vocab, &job.mc, &job.tc);
        let gpu = Gpu::new()?;
        let mut gm = GpuModel::new(gpu, job.vocab, &job.mc, &job.tc, cpu_init.params())?;
        if o.profile {
            gm.gpu.profile = Some(Default::default());
        }
        let (d0, s0, gs0, ws0) = (gm.gpu.dispatches, gm.gpu.submits, gm.gpu.gpu_seconds, gm.gpu.wait_seconds);
        let t0 = Instant::now();
        let mut pos = 0;
        let mut sink = 0f32;
        while pos < max_len {
            let lseg = seg.min(max_len - pos);
            let io = job.segment(pos, lseg);
            let train = pos + lseg < max_len;
            gm.begin_segment(pos as u64);
            if mode == "enc" && o.batch > 1 {
                let kb = o.batch.min(KMAX);
                let mut ios = vec![(pos, lseg, io, train)];
                let mut p = pos + lseg;
                while p < max_len && ios.len() < kb {
                    let l = seg.min(max_len - p);
                    ios.push((p, l, job.segment(p, l), p + l < max_len));
                    p += l;
                }
                let sio: Vec<SegIo> = ios
                    .iter()
                    .map(|(p, l, io, tr)| SegIo { pos: *p as u64, lseg: *l, inputs: &io.0, active: &io.1, targets: &io.2, train: *tr })
                    .collect();
                gm.run_segments(&sio);
                for k in 0..sio.len() {
                    sink += gm.probs_row_k(k, 0, 0)[0];
                }
                pos = p;
                continue;
            } else if mode == "enc" {
                gm.run_segment(lseg, &io.0, &io.1, &io.2, train);
                sink += gm.probs_row(0, 0)[0];
            } else {
                for t in 0..lseg {
                    gm.forward_position(t, &io.0[t * ns..(t + 1) * ns], &io.1[t * ns..(t + 1) * ns]);
                    // A decoder reads every active row here before it can build the next input.
                    for b in 0..ns {
                        sink += gm.probs_row(t, b)[0];
                    }
                }
                gm.finish_segment(lseg, &io.2, train);
            }
            pos += lseg;
        }
        let secs = t0.elapsed().as_secs_f64();
        let (d, s) = (gm.gpu.dispatches - d0, gm.gpu.submits - s0);
        let (gs, ws) = (gm.gpu.gpu_seconds - gs0, gm.gpu.wait_seconds - ws0);
        println!(
            "GPU  {}: {:>8.2}s  {:>9.0} tokens/s   ({} submits, {} dispatches = {:.1} us/dispatch of GPU time; GPU busy {:.2}s, submit+wait {:.2}s){}",
            if mode == "dec" {
                "decoder".to_string()
            } else if o.batch > 1 {
                format!("encoder x{}", o.batch.min(KMAX))
            } else {
                "encoder".to_string()
            },
            secs,
            n_tok / secs,
            s,
            d,
            gs / d.max(1) as f64 * 1e6,
            gs,
            ws,
            if sink.is_nan() { " nan" } else { "" }
        );
        if let Some(p) = &gm.gpu.profile {
            let total: f64 = p.values().map(|v| v.0).sum();
            let mut v: Vec<_> = p.iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            for (name, (t, n)) in v {
                println!("      {:<12} {:>7.3}s {:>5.1}%  {:>8} calls  {:>8.1} us/call", name, t, 100.0 * t / total, n, t / *n as f64 * 1e6);
            }
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    let rest = if args.is_empty() { &args[..] } else { &args[1..] };
    let r = match cmd {
        "info" => cmd_info(),
        "overhead" => cmd_overhead(),
        "arith" => cmd_arith(rest.first().and_then(|s| s.parse().ok()).unwrap_or(1 << 22)),
        "kernels" => cmd_kernels(),
        "check" => parse(rest).and_then(|o| cmd_check(&o)),
        "bench" => parse(rest).and_then(|o| cmd_bench(&o)),
        _ => {
            println!("{}", include_str!("main.rs").lines().take(14).map(|l| l.trim_start_matches("//!")).collect::<Vec<_>>().join("\n"));
            Ok(())
        }
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
