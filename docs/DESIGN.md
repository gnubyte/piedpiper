# Design notes

## The method (TufaZip)

[TufaZip](https://tufalabs.ai/research/tufazip/), by [Tommy He](https://x.com/tommyhe41) at
[Tufa Labs](https://tufalabs.ai/) (2026), placed #2 on Matt
Mahoney's Large Text Compression Benchmark with a single predictor: a decoder-only
Transformer that **trains online while it compresses**.

```
for each token:
    p = model(context)            # next-token distribution
    arithmetic_code(token, p)     # costs -log2 p(token) bits
    model.adam_step(token)        # decoder repeats this exactly
```

The decoder runs the same loop, so the weights never have to be sent: it decodes a token
with `p`, then takes the same training step. TufaZip v2 also starts from a quantised
offline prior, trains on replayed history, and uses a 163M-parameter network (12 layers,
d=1024, Peri-LN, QK-norm, RoPE, SwiGLU, 4K word vocabulary).

The original needs two NVIDIA B200 GPUs, takes about 7 days per GB, and decodes only on
the exact CUDA/cuDNN/PyTorch build that encoded. PiedPiper keeps the method and
re-engineers it for portability.

| | TufaZip v2 | PiedPiper |
|---|---|---|
| Hardware | 2× B200, CUDA 13 | any CPU: x86-64 (SSE2/AVX2), ARM64/ARMv7 (NEON/VFP), WebAssembly SIMD |
| Determinism | same software stack only | **bit-exact across OS, CPU, SIMD width, thread count and language binding** |
| Network | 163M params, d=1024, 12 layers | 0.14–40M params (levels 1–9), same block design |
| Context | 192 tokens, recomputed per token | `window` tokens, Transformer-XL key/value memory |
| Update | every token, batch of 64 streams | every `seg_len` tokens per stream, `n_streams` streams |
| Tokenizer | 4K word vocabulary (enwik9-specific) | bytes, or word-aware BPE learned per file (stored in the header) |
| Prior | quantised offline prior (4.3 MB) | optional `.ppp` priors (f16), trainable with `piedpiper train` |
| Optimizer | Adam β=(0, 0.9999), clip 0.25 | Adam (β1=0 by default, β2=0.9999), optional clip |
| Scope | enwik9 | any file |

## Determinism

The encoder and decoder must compute **identical** probabilities, or decoding diverges
after the first difference. A file compressed on an x86 laptop has to decode on an ARM
phone and in a browser. Neural networks are usually not reproducible across hardware.
These rules make PiedPiper reproducible:

1. **Only IEEE-754 basic operations.** `+ − × ÷` and `sqrt` are correctly rounded on every
   target (SSE2/AVX2, NEON, VFP, WebAssembly). Rust never fuses `a*b+c` into an FMA unless
   asked, so the code's operation order is the computation.
2. **No libm.** `exp`, `sin` and `cos` differ between C libraries and JS engines, so
   `fpmath.rs` builds them from basic operations. RoPE frequencies use bisection with
   multiplications only, not `powf`.
3. **Fixed summation order.** Every matrix element is accumulated in ascending inner index
   from `0.0`. Tiling, SIMD width and threads only change which elements are computed
   together, never one element's arithmetic. Dot products use a fixed 8-lane pattern and
   reduction tree. Gradients are reduced in a fixed row order.
4. **Explicit, IEEE-exact SIMD.** Kernels are written once against an 8-lane vector type
   (`simd.rs`) with one implementation per instruction set: AVX2 (runtime-detected), SSE2,
   AArch64 NEON and WebAssembly SIMD128. Each lane does a plain IEEE `+ − × ÷ √`, so every
   backend gives the same bits. 32-bit ARM NEON is avoided because it flushes subnormals;
   ARMv7 uses the scalar backend.
5. **Thread-count invariance.** Work is partitioned by output region, never by splitting a
   sum.
6. **No subnormal surprises.** Softmax exponents are clamped so probabilities stay normal
   floats. On top of that, every thread doing model math pins the FP control register
   (MXCSR / FPCR / FPSCR) to IEEE defaults, because a host process may have enabled
   flush-to-zero (shared libraries built with `-ffast-math` do this).
7. **Integer coding.** Probabilities become integer frequencies `max(1, floor(p·2²⁴))`, so
   the range coder is pure integer arithmetic.
8. **One loop.** Encoder and decoder share `codec::run`; the encoder processes positions
   one at a time exactly as the decoder must.

`testdata/golden` holds streams made on Windows x86-64. CI decodes and re-encodes them on
Linux (x86-64 and ARM64), macOS (Intel and Apple Silicon), Windows ARM64, WebAssembly
(Node), the Python wheel, the C ABI and Android.

## Network

Peri-LN blocks: `h += N_out(F(N_in(h)))` for attention and feed-forward, all RMSNorm with
learned gains. Attention has per-head QK-RMSNorm, then RoPE (split-half pairing,
base 10000), with the scale fixed at `1/sqrt(head_dim)`. The feed-forward is SwiGLU. There
are no biases. Embeddings are untied. The output matrix starts at zero, so the first
prediction is uniform.

## Online training

Tokens are split into `n_streams` contiguous streams processed in lockstep. One step
codes a segment of `seg_len` positions in every stream. After it, one Adam step goes on
the mean cross-entropy of what was just coded. Gradients flow through the current segment;
keys and values of earlier segments stay in a per-stream ring buffer as stop-gradient
memory.

Small inputs are learning-limited and get small batches (2 streams × 4 tokens). Large
inputs get more streams for throughput (`config::batch_policy`).

Adam with β1 = 0 (TufaZip/NNCP) needs no first-moment buffer. The learning rate warms up
linearly, then decays hyperbolically to `lr_min`. That schedule needs no knowledge of the
input length.

## Priors

Learning from scratch is what limits small inputs, so levels 1–3 ship with **built-in
priors** (`crates/core/priors/text-{s,m,l}.ppp`, 0.3 / 0.8 / 1.7 MB in f16). They are
byte-level networks trained offline with `piedpiper train` on a mixed corpus: Python, Java
and C sources, technical English prose, HTML and XML, all permissively licensed (all three were
trained further on a second 12 MB slice mixed half-and-half with the first, at a
gentler rate; continuing on the new slice alone had cost 2-5% on markup). Text-like
input up to 8 MB starts from them, fine-tuned at 0.4× the level's learning rate, or at
half that when it pays: a lower rate keeps more of what the prior knows, which helps on text
like its training data (Java −3.6%, prose −2.2%, XML −1.1% at level 2) and hurts on text
less like it (the synthetic server log +1.1%). For inputs of 8 KB and more the encoder compresses a
16K-token prefix both ways and keeps the rate that produced fewer bytes. That is an integer
comparison of deterministic coder output, so every platform picks the same rate, and the
rate travels in the header like any other setting.
Levels 2 and 1 also ship alternates, `text-m-code` and `text-s-code`, trained further on the
whole corpus: better on source code and HTML (at level 2: Java −4.0%, HTML −3.8%, Python
−1.9%, C −1.8%; at level 1: Java −3.1%, HTML −2.5%) but worse on XML. The
same prefix trial picks the prior per file (the default at both rates, the alternate at the
lower one), so every file gets the better of the two; it costs one more trial and 1.05 MB in
each binary. Text detection counts letters and digits, so number-heavy records such as CSV and JSON also
start from a prior (5% and 10% smaller at level 2 than training from scratch). Binary input
trains from scratch, because a text prior hurts there. A stream records the
prior's 64-bit id; decoders find built-in priors by that id, and those ids never change
once released. Custom priors (`.ppp`) can be trained on your own data and passed explicitly.

## Match model

The network attends to the last `window` tokens (64–320), but logs, CSV, JSON and source
code repeat themselves over far longer distances. Like the match models of PAQ and cmix,
each stream keeps hash tables of earlier contexts, follows the most recent earlier
occurrence of the current context, and predicts the token that came next there. Its
reliability is learned online from integer hit/miss counts, and the predicted token's
probability is raised to that estimate when it beats the network's.

There are two match models per stream: a long one (12-byte context) that locks onto
repeated records and lines, and a short one (3-byte context) that acts like a
recency-weighted order-3 context model, a strong complement to a small network that has
only just started learning. BPE streams use a single 4-token model. A sweep over context
lengths at level 2 on held-out text found 3 + 12 best or within 0.2% of the best on every
file type, against the single 8-byte model it replaced: XML −6%, C −2%, HTML −2%, CSV −4%,
executables −3%. Photo residuals from the image filter are the exception (the short model
only adds noise there), so they get the long model alone.

The counts are kept per match length *and* per bin of the network's own probability for
the predicted token. Without the second index the match model hurt structured text: the
network already predicts the repetitive parts of logs and JSON almost perfectly, so the
match model mattered mostly where the network was unsure — random ids, numbers, timestamps
— exactly where an old occurrence is least reliable. Conditioning on the network's
confidence learns that.

Level 2, bits per byte, both match models against none (measured with the earlier `text-m` prior and fixed fine-tuning rate, so the absolute numbers are a little above the README table):

| file | without | with | change |
|---|---:|---:|---:|
| English prose | 1.635 | 1.471 | -10.0% |
| Java source | 0.994 | 0.830 | -16.4% |
| C source | 1.251 | 1.065 | -14.9% |
| XML | 0.779 | 0.662 | -15.0% |
| HTML | 0.699 | 0.622 | -11.0% |
| server log (synthetic) | 0.861 | 0.857 | -0.5% |
| CSV orders (synthetic) | 0.983 | 0.866 | -11.9% |
| JSON records (synthetic) | 0.367 | 0.351 | -4.4% |

The models' state is integers and their mixing a fixed sequence of f32 operations, so they
are as bit-exact as the rest. The network keeps training on its own, unmixed prediction.

## Deflate recompression (format mode 4)

PNG, the ZIP family (docx, xlsx, pptx, odt, epub, jar, apk), gzip, PDF and zlib files are
already compressed, so no model gains anything on their bytes. `crates/core/src/deflate.rs`
finds their DEFLATE streams and inflates them with
[preflate-rs](https://github.com/microsoft/preflate-rs), which records the small amount of
information needed to recreate the *exact* original compressed bytes. The container bytes
plus those corrections go into one PiedPiper stream and the inflated content (the XML,
class files, pixels or text) into another, so text gets the text priors and the match model
while headers do not dilute it. Every stream is verified to recreate byte-exactly before it
is used and the whole file is verified to restore; streams that preflate cannot model
(for example zlib-ng's fast levels) stay raw. Measured against `xz -9e`: an XLSX workbook
61% smaller, a JAR 35%, a PNG 12%.

Files inside those containers often deserve a mode of their own, so mode 4 carries each
*embedded file* as a separate part that is compressed like a file on its own: an inflated
ZIP entry (Word, PowerPoint, EPUB), a stored ZIP entry (APKs store media uncompressed) or a
PDF `/DCTDecode` image. Photos go to the Lepton recoder: a PDF, a DOCX and an APK holding a
1 MB photo each came out 29–30% smaller this way, where xz saves 0–3%. Nested containers
(a ZIP of Office documents, the classes.jar inside an Android AAR, PNGs stored in an APK)
are expanded again, up to three levels deep, and MP4 videos reach the video recoder. Raw
images inside PDFs (FlateDecode XObjects: 8-bit gray, RGB or CMYK rows, optionally with PNG
predictors) become parts too, with the image filter set up from the image dictionary; on a
PDF holding a photo as raw RGB rows that took the result from 74% to 51% of the file. Tar
archives have no DEFLATE of their own but use the same machinery: members that are files of
their own become parts and the headers stay raw, and a `.tar.gz` reaches it through its
inflated tar. A 256 KB tarball of a photo, two Office files and Java source went from
199,562 to 98,994 bytes (xz: 200,672).

preflate-rs reproduces streams written by zlib-compatible encoders. Streams from encoders
with different match finders (zlib-ng at higher levels, miniz) can fail its prediction;
those stay raw and the file falls back to plain neural coding.

preflate-rs depends on the LGPL-3.0 `cabac` crate. Because PiedPiper links statically into
every SDK, the workspace patches in `crates/cabac-shim`, its own MIT implementation of the
few `cabac` interfaces preflate-rs calls (an LZMA-style binary range coder with adaptive
probabilities). See THIRD_PARTY_NOTICES.md.

## Executable filters

Machine code stores call and jump targets relative to the instruction, so calls to one
function look different at every call site. Before tokenizing, executables (detected from
their PE, ELF or Mach-O header) get the branch filters known from xz and 7-Zip: x86
`CALL`/`JMP rel32` and AArch64 `BL` targets become absolute addresses, which repeat. The
filter id is stored in the neural header, the decoder inverts it after detokenizing, and the
encoder verifies the round trip. Measured at level 2 it saves 1–5% (a Windows EXE 2.6%, a
DLL 4.2%, an ARM64 Android library 4.7%; it costs about 1% on a slice that is mostly data,
as xz's filter does). With the match models, PiedPiper ends up 4–12% smaller than `xz -9e`
and level with xz using the same filter (0.4% larger on the ARM64 library, 0.7% and 8%
smaller on the x86-64 files); executables remain the hardest file type for a byte-level model
that learns from scratch. The filter
field has room for parameters, so further transforms plug in the same way; raw image pixels
use it (filter 3).

## Image filter

Uncompressed pixels are what a 1-D byte model handles worst: at level 2 the attention
window covers 128 bytes, about 42 RGB pixels, so the pixel directly above the one being
predicted is out of sight (a 512-pixel row is 1,536 bytes back). Filter id 3 rewrites the
pixel bytes before tokenizing, the way WebP lossless and JPEG-LS do:

* **Colour decorrelation**: with three or more channels, `c0 − c1` and `c2 − c1` (+128).
  Channel 1 is green in both RGB and BGR order.
* **Spatial prediction** (photographic images): each sample becomes its residual against
  the LOCO-I / JPEG-LS median edge predictor of its left, upper and upper-left neighbours.
  Flat images, where at least half of the pixels repeat their left neighbour (screenshots,
  diagrams), skip this step: the model and its match models do better on the colour values
  themselves.

Headers, row padding and trailing bytes pass through unchanged, so the data keeps its
length. Raw formats are recognised from their headers: BMP (24/32-bit), binary PPM/PGM,
PAM and uncompressed TGA, all with 8-bit samples. For PNG, deflate mode (format 4) passes
the layout from the IHDR chunk when the inflated content is exactly the image's scanlines.
The filter then unfilters the rows to pixels, predicts those, and re-applies each row's
original PNG filter type on decode, which gives back the identical inflated bytes for
preflate. Palette, 16-bit and interlaced PNGs keep their scanlines as they are.
`--no-filter` turns the filter off.

Measured at level 2 (bytes; the pixels of three test images, coded as they are and with
the filter):

| image (raw size) | pixels as they are | with the filter | change |
|---|---:|---:|---:|
| photo crop 512×320 (491,520) | 317,318 | 249,652 | −21% |
| photo collage 280×175 (147,000) | 79,051 | 54,377 | −31% |
| UI screenshot 1080×360 (1,166,400) | 9,345 | 7,618 | −18% (colour only) |

On the photo crop that is 4% smaller than WebP lossless (261,080), and on the screenshot
smaller than `xz -9e` (8,728), though WebP lossless (6,282) still wins there. For PNG files,
deflate mode used to code the inflated scanlines, i.e. the PNG's own per-row filter
residuals: 281,337 bytes for the photo crop and 11,611 for the screenshot. With the filter
the model sees the pixels as above, plus one filter-type byte per row.

## JPEG recompression

JPEG data is already entropy-coded, so a byte-level model, PiedPiper's included, saves only
a few percent on it. Format mode 3 recompresses JPEG files with Lepton instead: it parses
the file, Huffman-decodes the DCT coefficients, codes them with a context model that
predicts each coefficient from its neighbours and a binary arithmetic coder, and on
decompression re-encodes the identical Huffman bitstream. Typical photos come out about 22%
smaller, losslessly and bit for bit, at several MB/s.

* **Engine.** `lepton_jpeg` 0.5.8 (Microsoft's Rust port of Dropbox's Lepton, Apache-2.0),
  vendored in `third_party/lepton_jpeg` with two patches: no clock calls on
  `wasm32-unknown-unknown` (they panic there), and a fixed git-revision header field
  (upstream records the build tree's `git describe`, so output depended on the build
  machine). See `third_party/lepton_jpeg/PIEDPIPER-PATCHES.md`.
* **Determinism.** Lepton uses integer arithmetic only, and the header records which of its
  two arithmetic variants was used. The partition count is fixed at 8, so threads change
  only the speed. Compressing therefore gives the same bytes on every platform, and the
  `jpeg` golden vector checks that.
* **Safety.** The encoder decodes its own output and compares it with the input before
  using it. Files Lepton rejects (arithmetic-coded or 12-bit JPEGs, damaged files, or
  anything that only starts like a JPEG) go to the neural path as usual.
* **Scope.** Whole-file JPEGs up to 128 MB. `--no-jpeg` (CLI) or `CompressOptions::jpeg =
  false` turns the mode off. JPEGs inside other containers (ZIP, PDF, MP4) are not handled
  yet.

## Tokenizers

`bytes` works for anything. `bpe` first splits text into word-like pieces (optional leading
space + letters, digit groups of up to 3, short punctuation runs, whitespace runs), then
learns byte-pair merges over piece counts. It uses an indexed pair-count heap, so training
on 100 MB takes seconds. Only the merge list is stored. `auto` uses BPE for text of 1 MB
or more, growing the vocabulary with the input, and bytes otherwise.

## Video: H.264 and HEVC recompression (format mode 2)

A byte-level model cannot compress video files: H.264, HEVC, VP9 and AV1 end with an
arithmetic coder, so their output looks like noise at the byte level (PiedPiper's neural
codec and xz both measure 8.00 bits per byte on real clips). The redundancy that is left
lives one layer down, in the decisions the codec's entropy coder made. `crates/media`
recompresses that layer losslessly, the way Lepton does for JPEG and Dropbox's avrecode
did for H.264:

1. Find the H.264 track of an MP4/MOV file (`mp4.rs`; in fragmented files, such as some
   screen recorders write, the samples come from the `moof`/`traf`/`trun` boxes) or a
   Matroska/WebM file (`mkv.rs`) and walk its NAL units.
2. For every CABAC slice, run a complete macroblock-layer parser (`h264/mb.rs`: I, P and B
   slices, all partitions, direct modes, 8x8 transforms, weighted prediction) that computes
   each bin's CABAC context exactly as the standard does.
3. Code each bin with a context-mixing model instead of CABAC. The model sees CABAC's own
   probability as one input, plus context CABAC ignores: the coefficient levels and signs
   of the left and upper blocks at the same scan position, neighbour coefficient counts,
   the co-located macroblock of the previous picture, the macroblock's QP and type, and
   the other motion-vector component.
4. On decompression, run the same parser with bins coming from the model, re-encode them
   with a standard CABAC encoder, and rebuild every NAL unit byte for byte (slice header
   bits, alignment, final-byte fix-up for encoders such as x264 that put padding bits after
   the stop bit, cabac_zero_words, emulation prevention).

Safety: a slice is recoded only after its re-encoded CABAC payload has matched the original
exactly (CAVLC slices: see below); everything else (interlaced or 4:2:2/4:4:4 video, I_PCM
macroblocks in CABAC slices, slice groups, non-video NAL units, audio, container boxes) is
coded as raw bytes. The core library then
decompresses the result and compares it with the input before using mode 2, so a parser bug
can cost compression but never data.

State that the model uses across slices (the previous picture's macroblock summaries) is
committed only after a slice has been recoded, so compressor and decompressor always see the
same history even when some slices are stored raw. All arithmetic is integer, and
`cargo test -p piedpiper-media` pins the exact output bytes of a synthetic file.

Large files are cut into segments of about 3 MB of video, chosen from the content alone,
each coded with its own model; segments are compressed and decompressed in parallel (up to 8
threads, sequentially in WebAssembly) without the thread count changing a single output
byte. A cut costs about 0.2 percentage points of ratio on a 1080p clip.

Compressed video the recoder cannot handle (AV1, other Matroska/WebM, AVI, MPEG-PS/TS, FLV,
and HEVC or VP9 it cannot parse) is recognised by its container signature plus an integer-only chi-square test on 32
sampled 16 KB blocks: when at least three quarters look uniformly distributed (compressed
video scores 200-400, PCM audio and text tens of thousands), the encoder stores the file
immediately rather than spending hours in the neural model for a gain near zero.

Typical results are 7% smaller for camera-like or rendered footage, 10-25% for screen
recordings, and more for animations and padded files. That is far less than on text,
because H.264 already removed nearly all of the redundancy; the remaining gain is the gap
between CABAC's simple adaptive contexts and a stronger model.

Speed is about 0.3-0.5 MB/s per core to compress and 0.2-0.4 MB/s to decompress. Most of
the time used to go into waiting for memory: the model's hash tables (16-32 MB per segment)
are far larger than the caches, and every bin reads several random slots. A bin's table
positions depend only on its context, so they can be fetched early. The encoder records a
slice's bins while verifying it, then codes them with the slots of the bin six places ahead
already being prefetched. The decoder cannot see ahead, so before each decision the parser
announces the bins that can come next (both outcomes for coefficient, skip, end-of-slice,
motion-vector and coded-block-flag bins), and the model prefetches them. Macroblock-level
syntax elements, which have few distinct contexts, use a small table that stays in the
caches. Model updates are applied at the next prediction, after its loads have started.
Container bytes that look random (compressed audio) are coded without the model, since the
header stream runs before the parallel segments. An ablation also trimmed the model from
eight to six hashed contexts and found a better set. Together these made compression about
2.4 times and decompression about 1.5 times faster per core than the first version, while
files got 0.2-0.6 percentage points smaller.

### H.264 CAVLC (Baseline)

Baseline-profile H.264, common in messaging apps, video calls, webcams and older phones, has no
arithmetic coder: CAVLC writes macroblock syntax as Exp-Golomb codes and residual blocks with
fixed code tables (coeff_token picked by the neighbouring blocks' average coefficient count
nC, levels with a growing suffix length, total_zeros, run_before). Like the MPEG-2 recoder,
`h264/cavlc.rs` re-encodes nothing. It parses every slice-data bit (macroblock and
sub-macroblock types of I, P and, for High-profile CAVLC streams, B slices, reference indices,
motion vector differences, intra modes, coded block patterns, transform-size flags, QP deltas,
I_PCM samples, residual blocks of every kind) and the model codes each bit with its syntax
element, its place in the code (the Exp-Golomb prefix or suffix position, or the node of the
code table's tree), and context the tables ignore. That context includes the left and upper
blocks' coefficient counts separately rather than their average, the block index, how many
blocks of the same 8x8 already had coefficients, the co-located block of the previous picture,
the neighbours' last scan position and largest level, a coefficient's scan position for
run_before, the QP and the macroblock class.

Two elements are coded as values instead of bits: `mb_skip_run`, as one "skipped?" decision
per macroblock with the left, upper and co-located macroblocks' skip state as context, and
`coded_block_pattern`, binarized as CABAC does it (a decision per 8x8 block in the context of
the neighbouring blocks' bits, then two chroma decisions). The decoder writes their Exp-Golomb
codes back. The end of the slice data (`more_rbsp_data()`) is a decision too, and costs
almost nothing because slices end at the end of the picture or after as many macroblocks as
the previous slice. A CAVLC slice is recoded only when its parse ends exactly at the stop bit.
Neighbours count only inside the current slice, and cross-slice context changes only when a
slice is committed, on both sides in the same order.

On the 18 Baseline-profile files found on the test machine (1.83 GB) every slice recoded and
every file came back byte-identical: 17.3% smaller overall, 14.6-31.1% per file,
where these files used to gain 2-5% from raw-byte modelling (xz -9e: 2.3% on a 90 MB subset).
Both directions run at about 2-4 MB/s on 6 threads. Residual syntax is about two thirds of the slice
bits: coeff_token codes became 17% cheaper, total_zeros and run_before 25%, level prefixes 10%;
the signs of trailing ones stay incompressible. Coding the coded block pattern as decisions
rather than through its code number gained 0.15-0.23 points, the per-8x8 counts 0.12 more.

Tried and dropped: coding coeff_token, total_zeros and run_before as values with one
binarization each (TotalCoeff and total_zeros counted up a decision at a time, run_before as a
"zero here?" decision per scan position with the neighbours' coefficients at that position as
context, as CABAC's significance map does). It came out 0.1-0.2 points worse than coding the
bits of the original codes with the tree node as context, and slower: each nC table's tree
already is a reasonable binarization, and its nodes hand the model the table for free.

### HEVC

HEVC (`crates/media/src/hevc/`) goes through the same pipeline. `hevc/mod.rs` reads VPS, SPS
and PPS (with VUI, HRD, scaling lists and predicted reference picture sets, as far as finding
the slice data needs) and slice segment headers. `hevc/ctu.rs` parses coding tree units: SAO
parameters, the coding quadtree, CU and PU syntax (skip, merge and AMVP with motion vector
differences, AMP partitions), intra mode derivation (the most-probable-mode list decides the
coefficient scan order, so it must be exact), transform trees, and residual coding with
coded sub-block flags, significance maps, greater-1/greater-2 flags, Rice-coded remaining
levels, sign data hiding and transform skip. Tiles, wavefront rows and dependent slice
segments are followed exactly, including the substream restarts and the context storage and
synchronisation they imply. Context initialisation values come from the H.265 tables,
checked against the HM reference software.

The model's HEVC hints follow the H.264 ones, adapted to variable block sizes. A coefficient
sees the already-decoded coefficients around it in its transform block (the five-neighbour
template VVC later standardised, as a count and a level sum), its position and diagonal band,
how many coefficients its block and sub-block have so far, the coefficient counts of the left
and upper transform blocks and of the co-located block in the previous picture, and, for
chroma, the counts of the same transform unit's luma and Cb blocks. The last significant
position sees that of the previous block of the same size and component, its best single
predictor. Signs of the lowest-frequency coefficients see the same coefficients' signs in
the neighbouring blocks, which made sign bits 3% cheaper. Split, skip, merge and motion
decisions see the neighbouring and co-located blocks' depth, class and motion-difference
size. Bypass bins (signs, remaining levels, motion-vector suffixes, intra modes, merge
indices) are modelled too; CABAC spends exactly one bit on each. Many richer hints made
things worse: every field multiplies the number of hashed contexts, which then learn slower.

Supported: everything the Main, Main 10 and Main Still Picture profiles allow, and other
4:2:0 or monochrome streams of 8 to 12 bits that avoid the range-extension tools that change
CABAC parsing and the SCC and 3-D extensions. Slices with PCM coding units, other layers of
multi-layer streams (the second view of MV-HEVC), 4:2:2 and 4:4:4, and pictures using tiles
and wavefronts together are stored raw; a quick check of the first slice headers declines
files where nothing can be recoded, so they take the fast store path.

On phone footage (Pixel, 1080p and 4K) HEVC comes out 6-10% smaller and on rendered footage
from a GPU encoder about 10%, with every slice recoded. That is less than for H.264 because
HEVC's own CABAC contexts already capture more (position classes, coded neighbouring
sub-blocks, earlier greater-1 flags). The parser was checked against 102 JCT-VC conformance
bitstreams covering B slices, wavefronts, tiles, dependent slices, entry points, sign hiding,
AMP, weighted prediction, scaling lists, lossless coding, reference picture set variants and
odd picture sizes: all round-trip byte-identical, and every slice recodes (15,832 slices,
12,786 of them B) except the 6 that contain PCM coding units. `scripts/hevc_conformance.py`
repeats the check on downloaded streams.

### VP9

VP9 (`crates/media/src/vp9/`, Matroska/WebM with `V_VP9`) has no CABAC: every decision is
coded by a boolean arithmetic coder with an 8-bit probability taken from tables that the
frame's compressed header updates. `vp9/mod.rs` parses both frame headers and keeps what VP9
carries from frame to frame (four saved probability contexts, reference frame sizes, the
previous frame's motion vectors and libvpx's rules for when they apply). `vp9/tile.rs` parses
tiles: the partition tree, mode info with libvpx's neighbour contexts, transform sizes,
motion-vector prediction exactly as `dec_find_mv_refs` does it (the predicted vector decides
whether a difference carries a high-precision bit, so it must be exact), and coefficient
tokens with their bands, energy contexts, scan orders and frame-edge context rules. Tiles are
re-encoded with a port of libvpx's `vpx_writer` and compared with the original before a frame
is recoded. A useful check that a parse is right: the cost of its decisions under libvpx's own
probabilities comes within about 1% of the tile sizes.

The model gets libvpx's probability (mapped to the nearest CABAC state) as its direct input.
VP9 streams normally fix their probabilities per frame, and the encoder chose them for that
frame, so the gain comes from context VP9 does not use: the co-located block of the last
shown frame (partition size, skip flag, mode, reference, motion vector, and the end of block
at the same place; on camera footage this took partition decisions from 11% to 18% below
libvpx's own cost), a five-neighbour coefficient template, the exact scan position,
coefficients so far, neighbouring blocks' end-of-block counts and DC signs, and for motion
vectors what the co-located vector would code, field by field. Frames that repeat an earlier
frame byte for byte (static scenes, test patterns) are stored as a reference to it.

Parsing needs state from earlier frames, so both sides parse every frame, raw and copied ones
included, from its original bytes with the same parser; a frame that fails to parse
invalidates the state until the next key frame. Segments start at key frames, where VP9 resets
that state.

Supported: profiles 0 and 2 (4:2:0, 8 to 12 bit), key, inter, hidden and show-existing frames,
superframes, tiles, compound prediction and lossless frames. Stored raw: intra-only frames,
segmentation, profiles 1 and 3 (4:4:4, 4:2:2), and streams that adapt their probabilities
backwards from the symbol counts of each frame (`frame_parallel_decoding_mode` off without
error resilience): no such file was at hand to validate that path, while every VP9 file tested
was frame-parallel or error-resilient and recoded completely.

Results: about 5.5% smaller for 1080p camera-like footage (Windows' SDR and 10-bit HDR display
samples), 9-25% for the Steam client's UI animations, and 75.7% for a 4K HDR calibration clip
whose 1,046 frames include 887 exact repeats (xz: 73.6%).

### VP8

VP8 (`crates/media/src/vp8/`) uses the same boolean coder as VP9, with a first partition
(frame header, probability updates and every macroblock's modes and motion vectors) and one
to eight token partitions that alternate by macroblock row. `vp8/parse.rs` follows libvpx's
decoder: the header fields, the near-motion-vector search with its counts, sign-bias flips
and clamping (they choose the probabilities of the mode tree), split motion vectors with
their left/above contexts, and the token loop with its band, previous-token context and
the 25-block order (Y2 first when present). Every frame of the VP8 files tested re-encodes
exactly, with no tail fix-up at all.

VP8's contexts are much simpler than VP9's (one probability per mode decision, the previous
token instead of neighbouring coefficients, no adaptation within a frame), so the model's
extra context pays more: the macroblock's neighbours and co-located macroblock in the last
frame for skip, reference and mode decisions, the coefficient's raster neighbours inside its
block and the same coefficient's level and sign in the blocks above and to the left, its exact
position and the number of coefficients so far, the neighbouring blocks' ends of block, the
end of block at the same place in the last frame, and for the sub-block intra modes of inter
frames the neighbouring sub-blocks' modes (which libvpx uses only in key frames). The
neighbouring signs alone made sign bits 9-26% cheaper. Old WebM clips came out 12-23%
smaller, 15.7% overall, where xz saves nothing.

### MPEG-2

MPEG-2 video (DVD rips and broadcast recordings in Matroska) codes its syntax with fixed
variable-length codes from annex B of ISO/IEC 13818-2 instead of an adaptive arithmetic
coder, so far more redundancy is left than in H.264, HEVC or VP9. `crates/media/src/mpeg2/`
does not re-encode anything. `parse.rs` walks each slice bit by bit (macroblock address
increments and escapes, macroblock types for I/P/B pictures, frame and field motion types,
DCT types, quantiser changes, motion codes and residuals, dual-prime vectors, coded block
patterns, intra DC sizes and differences, run/level codes from tables B.14 and B.15 with
escapes and the special first-coefficient code), and the model codes every bit in the
context of where it is: the syntax element, the node of its code tree, and for coefficients
the scan position, the number of coefficients so far, the previous run and level, the block,
the same block of the previous macroblock, the quantiser and the picture type. The decoder
runs the same parser on the bits the model returns, so the output is exact by construction;
slices that do not parse completely are stored raw.

Slice parsing needs only the current sequence and picture headers (MPEG-2 resets motion and
DC prediction at each slice and never refers to other pictures while parsing), so frames are
independent and segments can start at any GOP.

On a 1.52 GB DVD rip (720x480 interlaced, 42,231 frames, 1.27 million slices of which 32 did
not parse) the video came out 19.6% smaller and the whole file 15.7%, or 18.7% once its AC-3
audio (a fifth of the file) is recoded as well (next section); xz saves 1.7%. Coefficient
codes are 69% of the video bits. The first version of the contexts made them 14.5% cheaper,
coded block patterns 40%, macroblock types 45% and quantiser codes 68%, for 18.4% on the video.
Adding the co-located blocks of the row above and of the previous picture to the coefficient
contexts made results worse: the extra fields multiplied the hashed contexts faster than they
added information, so only macroblock types and patterns use them.

Where a value goes in the hint turned out to matter as much as what it is. The model hashes
fixed combinations of a hint's four fields (`a` with `d`, `a` alone, `c` alone, `a` with `b`,
`a` with `c` and `d`, and all four without the context number), and the context number picks
the mixer, so a value is only ever seen together with the values it shares those combinations
with. A second round of tuning took the video from 18.4% to 19.6% without parsing anything new.
On the first 40 MB of the rip, where the whole file went from 13.6% to 14.7%, most of the gain
came from moving two values:

- The code-tree node of each VLC bit was only in `a`, so the slot that hashes `c` alone mixed
  the statistics of different nodes of a code. With the node in `c` as well, that slot
  predicts per node (+0.38 points).
- The neighbour counts (how many coefficients the same block of the previous macroblock had)
  moved from `b`, where they were hashed together with the scan position and the number of
  coefficients so far, to `d`, next to the quantiser (+0.41 points).

The rest added 0.28 points: the macroblock type left the coefficient contexts, the previous
level is bucketed logarithmically, the coefficient context number follows the scan position
and the code-tree depth instead of the depth alone, intra blocks get separate sets for tables
B.14 and B.15, sign bits see the previous sign and the block, and motion codes and signs see
the other component's code. The above block's coefficient count, the intra DC size and a
coarser quantiser made things worse in whichever field they went.

### AC-3 audio

DVD rips carry their sound as AC-3 (Dolby Digital, ATSC A/52), often several tracks. Its bits
are packed tightly, and coded as plain container bytes they shrank only 1.7%.
`crates/media/src/ac3.rs` recodes the AC-3 frames it finds in the container bytes between the
video frames (bsid 8 or lower, so not E-AC-3) the same way the video parsers work: it reads
every field of a frame through the model, and the decoder runs the same parser on the bits the
model returns.

- Header and block fields (bit stream information, block switching, dither, dynamic range,
  coupling, rematrixing, exponent strategies, bit-allocation parameters) are coded in the
  context of the same field in the same block of the previous frame, which they nearly always
  repeat.
- Exponents come as 7-bit codes that each pack three differences. Each code is coded in the
  context of the differences the previous block of the same channel had at these frequencies,
  and those of the channel coded just before it, which usually follows the same envelope.
- To know how many bits each mantissa takes, the parser runs AC-3's own bit allocation (power
  spectral density, band integration, the masking curve with its decay, gain and floor tables,
  delta bit allocation, the BAP table). Grouped mantissas (three in 5 bits, three in 7, two in
  7) are split into their levels, and every mantissa is coded in the context of its allocation
  and exponent, the size of the mantissa before it, and the sizes at the same frequency in the
  previous block and in the previous channel.
- The two CRCs are not stored. The second is the CRC of the frame; the first sits at the start
  of the region it protects (the first five eighths of the frame) and is solved for, a small
  linear system over GF(2). Only frames whose CRCs check are recoded.

On the rip above the audio shrinks 13.7% and the whole file 18.7%.

### H.264 in MPEG transport streams

Camcorders (AVCHD `.mts`), TV recordings and capture tools write H.264 into MPEG transport
streams: 188-byte packets with 4-byte headers, some with adaptation fields (timing, padding),
the video interleaved with audio and tables. The slices are cut across packets, so the slice
parsers cannot read them in place. `crates/media/src/ts.rs` finds the H.264 stream through the
program tables (PAT, then a PMT entry of stream type `0x1B`), splits the file into the video's
elementary stream and a skeleton of everything else (packet headers, adaptation fields, PES
headers, other streams' packets), and cuts the elementary stream at its start codes into NAL
units, with a new sample at each access unit delimiter. The units go through the same track
recoder as an MP4 track (with in-band parameter sets); the skeleton and the zero bytes in
front of each start code go into a small side stream with its own model. The decoder pours the
rebuilt elementary stream back into the packets, using the packet and PES headers in the
skeleton to find where each payload goes.

On a 1.05 GB game capture (H.264 High, CABAC) all 79,546 slices were recoded and came out 8.1%
smaller; the 62 MB around them (packet and PES headers, padding, other streams, other NAL
units) packed into 25 MB, and the whole file came out 11.1% smaller, where it used to be
stored as is. xz saves 2.6%.

## Audio: PCM prediction (format mode 5)

On uncompressed audio a byte-level model loses to classical codecs: 16-bit WAV files came
out 8–45% larger than with FLAC. Byte order is not the cause (storing the samples most
significant byte first changed nothing). The redundancy in audio is linear: each sample is
close to a weighted sum of the ones before it, with weights that drift as the sound
changes, and a network that learns from one file does not discover linear prediction over
pairs of bytes. Lossless audio codecs are built around such a predictor, so mode 5
(`crates/core/src/audio.rs`) puts one in front of the network:

1. Parse the RIFF/WAVE file (16-bit PCM, mono or stereo). Every byte that is not a sample
   (the header, trailing chunks such as `LIST` metadata, a dangling odd byte) goes into a
   separate *meta* stream, so text metadata does not disturb the statistics of the samples.
2. Turn stereo into mid and side with an integer lifting step (`s = L − R`,
   `m = R + (s >> 1)`), which inverts exactly.
3. Predict each channel's next sample with a cascade of three normalised-LMS filters, the
   structure Monkey's Audio uses: 32 taps over the channel's own history plus 16 over the
   other channel's, then 16 and 4 taps over what the earlier stages missed. All three adapt
   after every sample, so the predictor follows the sound without side information.
4. Code the prediction errors with the network, as a second stream: zigzag-mapped, two
   bytes per sample, most significant first, with an escape for the rare large ones.

Measured at level 2 (size as % of the WAV file, lower is better):

| file | FLAC -8 | raw samples (`--no-audio`) | mode 5 | mode 5 + audio prior |
|---|---:|---:|---:|---:|
| speech (synthesised), 22 kHz mono, 31 s | 46.4% | 53.0% | **42.7%** | 42.3% |
| music (decoded MP3), 44.1 kHz stereo, 8 s | 23.9% | 34.6% | **22.6%** | 22.2% |
| ringtone, 22 kHz stereo, 13 s | 49.0% | 53.0% | 49.3% | 48.6% |

The FLAC -8 sizes are estimated with a reimplementation of FLAC's LPC and Rice-partition
search; the FLAC encoder that ships with Windows came out 4–19% larger than that estimate.

- **The predictor does most of the work.** A plain adaptive Laplace coder on the same
  residuals comes within 1% of the network (42.8 / 22.6 / 48.9%), and level 3 gains less
  than 0.2% over level 2. The match models change the result by at most 0.2% (the default
  short + long pair is marginally best, so the residual stream keeps it). What is left after
  prediction is mostly noise, where a larger model has little to find.
- **Short clips** of 1–3 seconds land within 3.5% of FLAC either way: the online model has
  too little audio to learn from. A prior trained on residuals of similar material (other
  tracks in the same style, the same synthetic voice, other system sounds; none of the test
  files) saved a further 1–3%, most on short clips. None ships yet, because a general audio
  prior needs a licensed training corpus. `piedpiper train` turns WAV inputs into their
  residuals, so you can train your own.
- **No text-like ratios.** The low bits of any recording (microphone, room, converter,
  dither) are noise, and noise does not compress, so lossless audio codecs all end up within
  a few percent of each other. The music sample compresses as well as it does because it
  was decoded from an MP3, which drops the top of the spectrum; a CD rip comes out larger
  with every codec.
- **Determinism.** The predictor computes in binary64 with only `+ − × ÷` in a fixed order
  and rounds by truncating and comparing the exact remainder, under the same pinned FP
  environment as the network, so it predicts bit-identically everywhere. The `audio-wav`
  golden vector checks it, and the encoder runs the inverse over its own residuals before
  it uses mode 5.
- **A stream mode, not a filter.** The residuals can be longer than the samples (large
  errors are escaped), which the filter contract forbids, and the WAV header and metadata
  are better coded apart from them.
- **Scope.** 16-bit PCM, mono or stereo, which covers CD audio and most recordings. 8-bit,
  24-bit and float samples, more than two channels and AIFF take the plain neural path;
  compressed audio (MP3, AAC, Ogg, FLAC) is recognised and stored as it is. WAV files
  inside ZIP archives, Office documents and APKs reach mode 5 as embedded parts of mode 4
  (a ZIP holding a one-second clip came out 1.3% larger than the bare WAV). `--no-audio`
  (CLI) or `CompressOptions::audio = false` turns the mode off. Unlike the JPEG and video
  modes it stays on with `--prior` or explicit model settings, which then apply to the
  residual stream.
- **Speed** is that of the neural path on a file of the same size, since the network codes
  two bytes per sample; the predictor and its check add about 1.5 seconds per minute of
  CD-quality stereo, under 1% of the network's time.

## Performance notes

- The decoder is inherently sequential within a stream, so parallelism comes from
  batching streams (matrix-matrix kernels) and from threads over output tiles. Thread
  dispatch only pays off for large matrices, so small inputs mostly run on one core.
- `matmul` (forward, axpy order) reaches about 100 GFLOP/s on one Zen 3 core without FMA.
  Weight gradients use `matmul_tn`, which reads activations transposed in place.
  Input gradients use `matmul_nt` (dot order), so no transposed weight copies are kept.
- Attention walks the key ring buffer without per-key division and runs in the AVX2
  backend when available. Adam is vectorised the same way.
- Each update still costs a couple of passes over the parameters (zeroing and Adam), so
  very small batches trade speed for the faster learning that small inputs need.
