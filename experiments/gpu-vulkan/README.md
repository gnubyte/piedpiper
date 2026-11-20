# Experiment: PiedPiper's model on a GPU (Vulkan), bit for bit

> **Shipped.** The GPU path from this experiment is now part of PiedPiper itself:
> `crates/core/src/gpu` (cargo feature `gpu`), `piedpiper --gpu` / `--gpu-mode auto|on|off`,
> `piedpiper gpu`, and **Use the graphics card** in the Windows app. This crate remains the
> research harness (CPU-vs-GPU `check`, `bench`, `arith`, `kernels`).

**Question.** Can a GPU (Vulkan or CUDA) speed up compression?

**Short answer.** Yes for large inputs, and without changing the file format. This crate runs
PiedPiper's whole online-learning Transformer (forward pass, backward pass, Adam, attention
memory) as Vulkan compute shaders. On an AMD Radeon RX 6800 XT it produces **exactly the same
bits as the CPU model**: every probability and every weight, over 25 to 1,667 training steps
per run at levels 1, 2, 5 and 9. So a file compressed on a GPU decompresses on any CPU, and the other way
round. On large inputs the GPU encoder is 6x faster than all 16 CPU threads at level 2,
13x at level 5 and 21x at level 9 (24x, 48x and 98x faster than one core). Small files gain
about 2x when compressing and get slower when decompressing, and decoding generally gains less
than encoding (see *What limits the GPU*).

Nothing here touches the product: this is a standalone crate (own `[workspace]`) that compiles
the core's `model.rs`, `kernels.rs`, `simd.rs`, `fpmath.rs`, `config.rs`, `par.rs`, `rng.rs`
and `fpenv.rs` unchanged as its CPU reference.

## 1. Where compression time goes

`piedpiper -2 -t 1` on 256 KB of source code and docs (built with `--features piedpiper/profile`):

| part | seconds | share |
|---|---:|---:|
| forward pass (matmuls 10.5 s, attention 8.5 s, output head 1.2 s) | 22.3 | 40% |
| backward pass (matmuls 14.0 s, attention 6.1 s) | 25.6 | 46% |
| Adam | 7.4 | 13% |
| match model + range coder | 0.4 | 0.7% |

The neural network is 99.3% of the time, so it is the only thing worth moving. Sixteen threads
only got this file from 4.7 to 5.5 KB/s: inputs under 1 MB get 2 streams of 8 tokens per
update, so the CPU has almost nothing to parallelise.

## 2. Same bits on a GPU: what it takes

The codec is bit-exact across CPUs because every float operation is an IEEE-754 basic operation
in a fixed order (docs/DESIGN.md, *Determinism*). A GPU can follow the same rules, but by
default it breaks three of them:

| rule | what a GPU does by default | what the shaders do |
|---|---|---|
| no FMA contraction | drivers fuse `a*b+c` into one FMA (one rounding instead of two) | every multiply and add is `precise` (SPIR-V `NoContraction`); a build step checks each `OpFMul/OpFAdd/OpFSub` carries it |
| IEEE `÷` and `√` | Vulkan only guarantees `+ − ×`; division may be 2.5 ulp, sqrt derives from inversesqrt | `f32_div` and `f32_sqrt` in integer arithmetic (restoring division, digit-by-digit root), exactly rounded by construction |
| IEEE environment | flush-to-zero and NaN/sign shortcuts are allowed | `SPV_KHR_float_controls` execution modes: DenormPreserve, RoundingModeRTE, SignedZeroInfNanPreserve (the device must report support) |
| no libm | GPU `exp` is approximate and vendor-specific | `fpmath::exp` transcribed operation for operation |
| fixed summation order | tree reductions, split-K, tensor cores | every element is summed in the CPU's order: ascending sums, or the 8-lane pattern `((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7))` plus the tail |

Measured on this GPU (`arith`, `kernels`):

| test | result |
|---|---|
| `f32_div` vs CPU `/`, 4.2M inputs (all exponents, subnormals, Inf, NaN) | 0 differences |
| GPU native `/` | 0 differences on this AMD driver (Vulkan does not guarantee it) |
| `f32_sqrt` vs CPU `sqrt`, 4.2M inputs | 0 differences |
| GPU native `sqrt` | **9.96% of results differ** (e.g. subnormal inputs flushed) |
| `exp`, `sigmoid` vs `fpmath` | 0 differences |
| matmul / matmul_tn / matmul_nt vs `kernels.rs`, 12 shapes up to 1024x512x4096 | identical |
| the same matmul without `precise` and without float controls | **64-88% of outputs differ** (FMA contraction) |
| without `precise` but with float controls | identical on this driver (other drivers contract anyway, so `precise` stays) |

`build.rs` compiles every bit-exact shader to SPIR-V assembly as well and fails the build if any
`OpFMul`, `OpFAdd` or `OpFSub` lacks `NoContraction` (tested by adding a shader without it).
Note that glslang marks *every* float operation of a shader once any variable in it is
`precise`; that only adds safety, but it means a test shader must avoid `precise` entirely to
show the unprotected case.

Whole model (`check`): the CPU model and the GPU model are driven side by side over the same
tokens; every probability row is compared bitwise at every position, and the full weight vector
after training steps.

| configuration | Adam steps | result |
|---|---:|---|
| level 1, 3 streams x 4, 20,011 tokens (uneven streams: inactive tail rows) | 1,667 | identical (encoder, decoder, batched) |
| level 2, 2 streams x 8 (the codec's shape for a 256 KB file) | 511 | identical (encoder, decoder) |
| level 2, 8 streams x 8, vocabulary 1,003 (tails in every 8-lane sum) | 468 | identical (encoder, decoder, batched) |
| level 2, 24 streams x 16, vocabulary 2,048 | 260 | identical (encoder, decoder, batched) |
| level 5, 48 streams x 16, vocabulary 4,096 (4.2M parameters) | 52 | identical |
| level 9, 16 streams x 16, vocabulary 4,096 (42.8M parameters, attention memory wraps) | 25 | identical |

The full weight comparison also caught a staging bug in this harness (a table overwrote initial
weights in a row that had not been used yet) before any probability differed; checking only the
outputs would have missed it for a while.

## 3. How the work maps onto a GPU

* **Encoder:** the inputs of a whole segment are known in advance, and the network never sees
  the match model or the coder (it trains on its own unmixed prediction). So all positions of a
  segment are computed at once, layer by layer; causal attention makes this equal to the CPU's
  position-by-position loop. Forward, backward, Adam and the memory update of a segment (about
  95 dispatches at level 2) go out in one submission, or several segments per submission.
* **Decoder:** the next input exists only after the previous symbol has been decoded, so every
  position is one submission plus a readback, then one more submission per segment for training.
* Fixed orders that look sequential stay sequential, but their memory traffic is restructured:
  the gain gradients (a running sum over rows on the CPU) become stored products plus one
  ordered column reduction for all gains at once; attention backward splits into a per-row part
  and a per-element part that replays each `dk`/`dv` element's additions in ascending position
  order; the embedding gradient is grouped by token on the host. A max is computed in parallel
  where order cannot matter (`x > m` never lets a NaN win; the sign of a zero max cancels in
  `x - m`).

## 4. Speed

Tokens per second for the model alone (no coder), same tokens on both sides. CPU: Ryzen 7 5800
(Zen 3, 8 cores / 16 threads), the core's own AVX2 kernels. GPU: Radeon RX 6800 XT. "Encoder"
records 8 segments per submission; "decoder" does one submission per position, as decompression
must. The 256 KB shape uses bytes; the others use a 2,048- or 4,096-token vocabulary (about 2
bytes per token with this harness's pair tokenizer).

| shape (codec batch policy) | CPU 1 thread | CPU 16 threads | GPU encoder | GPU decoder |
|---|---:|---:|---:|---:|
| level 2, 2 streams x 8: a file of a few hundred KB | 9,864 | 8,812 | **18,447** (2.1x) | 3,436 (0.4x) |
| level 2, 24 streams x 16: 6M+ tokens (roughly 20 MB of text) | 6,292 | 24,301 | **149,435** (6.1x) | 37,149 (1.5x) |
| level 5, 48 streams x 16 | 856 | 3,227 | **41,331** (12.8x) | 22,035 (6.8x) |
| level 9, 64 streams x 16 (42.8M parameters) | 108 | 496 | **10,625** (21x) | 6,562 (13x) |

Factors are against 16 CPU threads. At level 2 that is roughly 48 KB/s on the CPU against
300 KB/s on the GPU; at level 9 about 1 KB/s against 21 KB/s. The level-9 run is short (128
positions per stream, so the 320-token attention window was never full) and the CPU numbers come
from a machine that was busy with other jobs, so treat the factors as indicative.

Where the GPU's time goes at level 2, 24 x 16 (`--profile`): matmuls 34% (no FMA, small
matrices), attention forward and backward 31% (sequential sums over the keys), reading back
3 MB of probabilities per segment 10%, all gain gradients 6%, everything else under 3% each.

Optimisations that kept every bit identical and took level 2 from 61,000 to 149,000 tokens/s:
gain gradients reduced in one batched dispatch off the critical path, attention backward split so
rows run in parallel, smaller shared-memory footprints (more workgroups resident per compute
unit), 16x16 matmul tiles for small grids, full instead of compute-only barriers (2 us instead of
4 us per dispatch on this driver), and 8 segments per submission. What made things slower:
staging everything in shared memory, because the occupancy loss outweighed the latency saved.

## 5. What limits the GPU

1. **Fixed costs.** A dispatch with its barrier costs about 2 us of GPU time; a submit-and-wait
   round trip about 150 us on Windows (0.5 ms when the GPU has clocked down). A training step is
   about 95 dispatches whatever the batch, so a 16-row batch spends most of its time on
   overhead.
2. **The CPU's summation orders.** Bit-exactness forbids tree reductions, split-K and tensor
   cores. The sums over attention keys, over vocabulary lanes and over rows (weight and gain
   gradients) stay dependent chains, so those kernels are latency-bound. Every matmul also runs
   without FMA, at half the GPU's peak rate.
3. **Decoding.** One round trip per position. With 2 streams that is 2 tokens per round trip.
4. **Small inputs.** The batch policy (`config::batch_policy`) gives inputs under 1 MB 2 streams
   of 4-8 tokens per update, because small inputs learn better with frequent updates. That is
   the worst case for a GPU, and changing the policy would change the compressed bytes.

## 6. CUDA or Vulkan

This machine has an AMD GPU, so CUDA could not be measured here. CUDA would be easier to make
exact: `__fadd_rn`, `__fmul_rn`, `__fdiv_rn` and `__fsqrt_rn` are documented as IEEE
round-to-nearest and are never contracted, and denormals are kept unless `-ftz=true`. But it
runs only on NVIDIA and needs the CUDA driver API at run time. Vulkan runs on NVIDIA, AMD and
Intel (and on Apple through MoltenVK, untested), loads from the system driver (`vulkan-1.dll`)
with no toolkit, and the integer `f32_div`/`f32_sqrt` remove the one guarantee it lacks.
Vulkan is the better fit for a codec whose point is to work everywhere. The device must report
`shaderDenormPreserveFloat32`, `shaderRoundingModeRTEFloat32` and
`shaderSignedZeroInfNanPreserveFloat32`; devices without them would keep the CPU path.

## 7. Not pursued, and why

* **GPU deflate / LZ (nvCOMP, GDeflate):** fast, but they implement other, much weaker
  compressors. They would not speed up PiedPiper's format, and they would not beat its ratio.
* **Tensor cores / fp16 / bf16:** unspecified internal accumulation order and precision, so
  outputs could not match the CPU.
* **The other modes:** Lepton (JPEG), the video recoders and preflate are long chains of
  adaptive binary decisions. They parallelise across segments, which the CPU threads already
  do, not within one.

## 8. If this were to ship

* A predictor trait in `codec::run` so the CPU `Model` or a `GpuModel` can drive the coder.
  The format would not change, and files would decode either way.
* GPU behind an opt-in cargo feature and a run-time check: Vulkan present, the three float
  controls supported, and a short self-test against the CPU (the `arith` and `kernels` checks
  take well under a second). Anything else stays on the CPU.
* Use it for large inputs only: about 1 MB and up, where the codec chooses many streams.
  Double-buffer segments so the CPU's coding of one segment overlaps the GPU's next.
* CI on more than one GPU vendor before trusting it. Driver compilers are the new risk: one
  that ignored `NoContraction` or a float-control mode would silently change bits. The
  encoder's existing verify step (decompress and compare) would catch that, at the cost of
  a CPU decode.
* The two biggest speed-ups not tried here: fusing each layer into one persistent kernel
  (removes most of the 95 dispatches, which is what small inputs need), and compressing several
  independent files or container parts concurrently on one GPU.

## Caveats

* One GPU and one driver (RX 6800 XT, AMD 26.8.1, Windows 11). Other vendors are untested.
* The machine was shared with other jobs (CPU load 50-80% before runs), so the CPU baselines
  are on the low side. The GPU numbers are less affected.
* Covers the default optimiser (beta1 = 0, no clipping), head dimension <= 64, window <= 512 and
  vocabulary <= 4,096: every preset fits. Priors only change initial weights, which this harness
  copies from the CPU model, so they take the same code path.
* NaN payloads are not reproduced (generated NaNs are 0x7FC00000). x86 and ARM already differ
  there, so the codec cannot depend on them.
* Throughput excludes the match model and range coder: 0.7% of CPU time today, but at 100K+
  tokens/s they would need their own thread.

## Running it

Needs a Vulkan driver and `glslc` (found in `GLSLC`, the Vulkan SDK, or an Android NDK's
`shader-tools`).

```bash
cargo run --release -- info
cargo run --release -- arith
cargo run --release -- kernels
cargo run --release -- overhead
cargo run --release -- check FILE --level 2 --streams 24 --seg 16 --vocab 2048 --tokens 100000
cargo run --release -- bench FILE --level 2 --streams 24 --seg 16 --vocab 2048 --tokens 200000 --batch 8
```

`check` and `bench` take `--mode enc|dec|batch|both`, `--batch K` (segments per submission),
`--cpu-threads 1,16`, `--skip-cpu` and `--profile` (per-kernel GPU time).
