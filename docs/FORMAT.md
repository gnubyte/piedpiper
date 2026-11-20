# PiedPiper file formats (version 1)

All integers are little-endian. `varint` is unsigned LEB128: 7 bits per byte, low bits
first, high bit set on every byte except the last. `f32` values are stored as their
IEEE-754 bit patterns, so the decoder rebuilds bit-identical hyper-parameters.

## Compressed stream (`.pp`)

| field | type | notes |
|---|---|---|
| magic | 4 bytes | `PIED` (`50 49 45 44`) |
| version | u8 | `1` |
| mode | u8 | `0` = stored, `1` = neural, `2` = video recode (H.264, HEVC, VP9, VP8, MPEG-2), `3` = JPEG recode, `4` = deflate recompression, `5` = PCM audio |
| original length | varint | bytes |
| crc32 | u32 | CRC-32/IEEE (zip/gzip polynomial) of the original data |

**Stored** (`mode 0`): the original bytes follow verbatim. Encoders use this for empty input
and whenever the model cannot beat storing.

**Neural** (`mode 1`):

| field | type | notes |
|---|---|---|
| tokenizer | see below | |
| d_model, n_layer, n_head, d_ff, window, seg_len, n_streams | 7 × varint | network shape and streaming layout |
| lr, lr_min | 2 × f32 | peak learning rate, floor |
| warmup, decay | 2 × varint | in updates |
| beta1, beta2, eps, clip, init_scale | 5 × f32 | Adam and initialisation |
| seed | u64 | weight-initialisation seed |
| n_tokens | varint | length of the token sequence |
| prior id | u64 | `0` = random initialisation, otherwise the FNV-1a-64 of the prior file |
| match models | 2 × u8 | context lengths (at most 64) of the short and the long match model; `0` = that model is off |
| filter | u8, then its parameters | transform applied before tokenizing: `0` none, `1` x86, `2` ARM64 (no parameters), `3` image (parameters below) |
| payload | bytes | range-coder output, until the end of the stream |

### Tokenizer

| byte | meaning |
|---|---|
| `0` | bytes: 256 symbols, token = byte |
| `1` | BPE: `n_merges` varint, then `n_merges` pairs (`left` varint, `right` varint). Merge *i* creates token `256 + i` = bytes(left) ‖ bytes(right); both ids must be `< 256 + i`. |

### Decoding procedure

1. Build the network: token embedding `(V+1) × d` (row `V` is the begin-of-stream token),
   `n_layer` Peri-LN blocks (RMSNorm before and after attention and SwiGLU; per-head
   QK-RMSNorm; RoPE with base 10000 on split halves; no biases), final RMSNorm, untied
   output `d × V`. Initialise from `seed` (SplitMix64, uniform with the stated scales; the
   output matrix is zero; gains are one) or copy the prior's weights.
2. Split the `n_tokens` tokens into `n_streams` contiguous streams. The first
   `n_tokens mod n_streams` streams get one extra token.
3. Process all streams in lockstep, `seg_len` positions per segment. At each position,
   each active stream feeds its previous token (or BOS) and attends to the last `window`
   positions of its own stream. The keys and values of earlier segments are kept as they
   were computed. Every active stream's next symbol is then decoded in stream order with
   frequencies `max(1, floor(p · 2²⁴))`, where `p` is the network's distribution, mixed
   with the match model's prediction when that is on (see below). The network trains on
   its own, unmixed distribution.
4. After each segment except the last, take one Adam step on the mean cross-entropy of
   the segment's coded symbols. The step uses the learning rate `lr · (s+1)/warmup`
   during warm-up, then `lr_min + (lr − lr_min) · decay / (decay + s − warmup)`.
5. Detokenize, invert the filter, then check the length and the CRC-32.

The exact arithmetic is specified by the reference implementation (`crates/core`). Every
port must reproduce it bit-for-bit; `testdata/golden` checks that.

### Match model

Each stream has up to two match models, one per non-zero match-model byte: the *short*
model (first byte) and the *long* model (second byte). They work identically, with their own
tables and counters; below, `m` is the model's context length. Each model keeps a hash table of `2^k` u32 positions, `k = clamp(bitlen(stream length), 10, B)`
with `B = clamp(22 − floor(log2(n_streams)), 12, 20)`, all zero at the start. The hash of
the last `m` tokens `t₁…tₘ` is `h = 0x9E3779B1`, then for each token
`h = rotl₅((h ⊕ t) · 0x01000193)` (u32 arithmetic), and the index is
`((h · 0x9E3779B1) >> 7) & (2^k − 1)`.

State: a match pointer `ptr` and length `len` (no match while `len = 0`), and hit / try
counters per (length bucket, probability bin): 24 buckets (`len` below 16 is its own
bucket, otherwise `12 + floor(log2(len))`, capped at 23) and 10 bins split at
`2⁻⁸, 2⁻⁶, 2⁻⁴, 1/8, 1/4, 1/2, 3/4, 0.9, 0.97` (`p ≥ edge` moves up a bin).

Before coding position `i` of a stream, if `len > 0` and `ptr < i`, a model predicts
token `t = stream[ptr]` with bucket `b = bucket(len)`. With `pₙ = p[t]` from the current
distribution and `c = bin(pₙ)`, the estimate is `pₕ = (hits[b][c] + 0.5) / (tries[b][c] + 1)`
in f32. Only if `pₕ > pₙ`, the distribution becomes `p ← p · (1 − λ)` (every entry), then
`p[t] += λ`, with `λ = (pₕ − pₙ) / (1 − pₙ)`, all in f32 in that order. The long model
mixes first, into a copy of the network's distribution; the short model then mixes into
the result.

After coding `i`, each model (long first) updates: if it offered a prediction, `tries[b][c] += 1` and, if it was right,
`hits[b][c] += 1`; a counter pair is halved (integer shift) when `tries` reaches 1024. A
right prediction extends the match (`len = min(len + 1, 65535)`, `ptr += 1`); a wrong one
ends it (`len = 0`). Then, once `i + 1 ≥ m`, the context of the last `m` tokens is hashed:
if there is no match, the table entry `j` is a candidate when `m ≤ j < i + 1` and the `m`
tokens before `j` equal the context; the match length is then extended backwards while
earlier tokens agree, up to 32 (`ptr = j`, `len` = that length). Finally the entry is set
to `i + 1`.

### Filters

A filter rewrites the input before tokenizing; decoders apply the inverse after
detokenizing. Encoders choose one from the content (PE, ELF or Mach-O headers) and verify
the round trip.

- `1` x86: the x86 branch converter of the LZMA SDK (`x86_Convert`, encoding direction),
  run over the whole input with start position 0: the 32-bit operand of `E8`/`E9`
  instructions whose top byte is `00` or `FF` becomes absolute (`operand + position + 5`).
- `2` ARM64: every 4-byte-aligned little-endian word whose top six bits are `100101`
  (`BL`) gets `imm26 ← (imm26 + offset / 4) mod 2²⁶`.
- `3` image: parameters `header_len` varint, `width` varint, `height` varint, `channels`
  u8 (1–4, 8-bit samples), `stride` varint, `flags` u8 (bit 0: MED prediction, bit 1: PNG
  scanlines; other bits must be 0). Row `y` starts at `header_len + y·stride` and holds
  `width·channels` samples (after one filter-type byte with the PNG flag); all other bytes
  are unchanged. Encoding: (1) PNG rows are unfiltered to raw samples with their filter type
  (0–4, as in the PNG specification, bytes per pixel = `channels`); (2) with 3 or 4
  channels, `s0 = p0 − p1 + 128` and `s2 = p2 − p1 + 128` (mod 256), other channels as
  they are; (3) with the MED flag, each sample becomes `s − pred` (mod 256), where `pred`
  is 128 for the first sample of the image, the left neighbour (same channel) in the first
  row, the upper neighbour in the first column, and otherwise `MED(a, b, c)` of left `a`,
  upper `b` and upper-left `c`: `min(a, b)` if `c ≥ max(a, b)`, `max(a, b)` if
  `c ≤ min(a, b)`, else `a + b − c`. Decoding reverses the steps in raster order and
  re-filters PNG rows with their stored filter type.

### Range coder

LZMA-style carry propagation over a 48-bit `low` and a range kept in `[2⁴⁰, 2⁴⁸)`. To
encode `(start, size, total)`: `r = range / total; low += r·start; range = r·size`, then
renormalise by bytes. The first output byte is always 0. Flushing writes 7 bytes. The
decoder treats bytes past the end as 0.

## H.264 video recode (`mode 2`)

For MP4/MOV (plain or fragmented) and Matroska/WebM files whose video track is H.264 (CABAC or CAVLC). The PiedPiper header (magic to
crc32) is followed directly by this payload:

| field | type | notes |
|---|---|---|
| magic | 4 bytes | `PPM3` |
| header length | varint | bytes of the header stream |
| header stream | bytes | binary arithmetic-coder stream (model with 2^20 slots) |
| per segment: table bits | u8 | log2 of that segment model's hash-table size, `16..=24` (4 bytes per slot) |
| per segment: length | varint | bytes of the segment stream |
| per segment: stream | bytes | binary arithmetic-coder stream |

The header stream holds the file length, the NAL length-field size and the number of video
samples; each sample's offset (as the gap after the previous sample) and size; the number of
segments and the first sample of each segment after the first (as deltas); and every byte
that is not inside a video sample (container boxes, audio, padding), in file order. A run of
such bytes (between two samples, before the first or after the last) that is at least 512
bytes long starts with a flag: 1 means the run is coded at exactly one bit per bit
(probability 1/2, no model). The encoder sets it when the run's byte histogram is close to
uniform (chi-square below 255 + n/9 for n bytes), which is typical of compressed audio.

Segments are runs of consecutive samples, cut once a run holds about 3 MB of video. The cut
depends only on the content, never on the thread count, and segments may start at any sample.
Each segment is coded with its own fresh model and parser state, so they are encoded and
decoded in parallel. A segment stream holds:

1. the in-band SPS/PPS NAL units in effect where the segment starts (count, then length and
   bytes of each), applied after the `avcC` parameter sets;
2. for each NAL unit of each of its samples: the NAL header byte and a flag. Flag 0: the NAL
   length and the remaining raw bytes. Flag 1 (a recoded slice): the slice-header bit count
   and bits, then, if the slice's PPS selects CABAC, every CABAC bin of the slice data, the
   number of bytes to drop from the end of the re-encoded payload and the replacement bytes
   (encoders may put arbitrary bits after the final stop bit), and the number of trailing
   `cabac_zero_word` bytes; if it selects CAVLC (`entropy_coding_mode_flag` 0), the slice
   data as decisions (below) and the number of zero bytes after the byte that holds the stop
   bit.

To decode a recoded slice, parse its header, then run the CABAC macroblock-layer parser
(H.264 clause 9.3) with bins taken from the model instead of a bitstream; feed every bin to
a standard CABAC encoder to rebuild the slice data, apply the tail fix-up, append the
alignment bits and zero bytes, and re-insert emulation-prevention bytes.

A CAVLC slice's data starts right after its header bits. To decode it, run the CAVLC
macroblock-layer parser (H.264 clauses 7.3.4, 7.3.5 and 9.2) on decisions from the model.
Every slice-data bit is one decision, in stream order, except:

- `mb_skip_run`: one decision per macroblock, 1 = skipped, until a 0 or the end of the
  picture (the run cannot pass it);
- `coded_block_pattern` (macroblocks other than Intra_16x16): four decisions for the luma
  8x8 blocks in order, then "chroma is not 0" and, if 1, "chroma is 2";
- `more_rbsp_data()`: a decision after each macroblock and after a non-zero skip run.

The decoder writes the Exp-Golomb codes of skip runs and coded block patterns back where they
belong, then the stop bit, zero bits up to the next byte, the trailing zero bytes, and the
emulation-prevention bytes. The encoder recodes a CAVLC slice only when its parse ends exactly
at the stop bit (the last set bit of the RBSP).

Parameter sets come from the `avcC` record (the MP4 `avcC` box or the Matroska `CodecPrivate`, read from the
reconstructed container bytes) and from in-band SPS/PPS NAL units.

For CAVLC decisions the model's direct input is a fixed probability of 1/2 (CABAC state 0),
and its mixer weights are selected per syntax element and place in the code instead of by
CABAC context index.

The bit model mixes, in the logistic domain, CABAC's own probability for the bin's context
with six adaptive estimates selected by hashes of the context and side information the
parser computes (neighbouring coefficient levels, the co-located macroblock of the previous
picture, positions, QP, ...), using integer arithmetic only. Estimates for macroblock-level
syntax elements (skip, macroblock and sub-macroblock types, reference indices, QP deltas,
intra modes, coded block pattern and flags, transform size, motion-vector signs) come from a
second table with 1/64 of the segment's slots (2^12 to 2^18). Its exact definition is the
reference implementation in `crates/media`; `testdata/golden/h264-mp4.bin.pp` (CABAC) and
`h264-cavlc-mp4.bin.pp` (CAVLC) check it.

### HEVC (`PPH3`)

For MP4/MOV (`hvc1`/`hev1` sample entries with an `hvcC` box) and Matroska/WebM
(`V_MPEGH/ISO/HEVC`) files whose video track is HEVC. The payload has the layout above with
the magic `PPH3`, and these differences:

- Each NAL unit starts with both of its header bytes (the second coded in the context of the
  first), then the flag; a raw NAL unit's remaining bytes start after the second byte.
- The segment preamble holds the in-band SPS and PPS NAL units (types 33 and 34) of the base
  layer (`nuh_layer_id` 0), applied after the parameter sets of the `hvcC` record (the MP4 box
  or the Matroska `CodecPrivate`, read from the reconstructed container bytes). VPS, SEI and
  every NAL unit of another layer are ordinary NAL units.
- Recoded NAL units are base-layer slice segments (types 0-9 and 16-21). Their coded header
  bits run to the end of the slice segment header's `byte_alignment()`, so slice data starts
  on a byte boundary.
- To decode a recoded slice segment, parse its header (a dependent slice segment inherits
  fields from the last independent segment, which both sides track over every slice NAL
  unit, raw or recoded), then run the HEVC slice-segment-data parser (H.265 clause 9.3) with
  bins from the model, feeding them to a standard CABAC encoder. At each tile or
  wavefront-row boundary inside a segment the parser's `end_of_subset_one_bit` is not
  stored (it is always 1): the encoder codes it, pads with zero bits to a byte and restarts.
  Context variables follow the standard: initialised at tile starts, taken from the stored
  wavefront state at row starts, and resumed from the end of the previous slice segment for
  dependent slice segments.
- A dependent slice segment is recoded only when every earlier segment of its slice was, so
  the contexts and neighbour data it resumes from exist on both sides.

The bit model is the same, with HEVC's own side information (see `docs/DESIGN.md`);
`testdata/golden/hevc-mp4.bin.pp` checks it.

### VP9 (`PPV9`)

For Matroska/WebM files whose video track is VP9 (`V_VP9`). The payload has the same outer
layout (magic `PPV9`, header stream, segments), but the streams hold:

- Header stream: the file length; the number of video blocks (Matroska `SimpleBlock` or
  `Block` frames of the VP9 track, unlaced) with each block's offset (as the gap after the
  previous one) and size; the number of segments and the first block of each segment after
  the first (as deltas); then the bytes outside the blocks, coded like the H.264 ones (gaps
  of at least 512 bytes that look random are stored at one bit per bit).
- Segments start at key frames (blocks whose first frame is a profile-0/2 key frame), cut
  once a run holds about 3 MB. Each starts with a fresh model and fresh VP9 parse state.
- For each block: the length of its superframe index (0 if there is none) and the index
  bytes; the frame sizes follow from the index, or the block is one frame.
- For each frame, a copy flag. 1: the frame is byte-identical to an earlier frame of the
  segment, given as the number of frames between them minus one. 0: a recode flag. Recode
  flag 0: the frame's bytes, raw. Recode flag 1: the length of the frame's two headers and
  their bytes (uncompressed header plus the bool-coded compressed header), then for each
  tile in tile-row, tile-column order every boolean decision of the tile, the number of
  bytes to drop from the end of the re-encoded tile and the replacement bytes. Tile sizes
  are not stored: each tile but the last gets its 4-byte big-endian size from the rebuilt
  tile.

Parsing a frame needs state from earlier frames: the four saved probability contexts,
reference frame sizes, the previous frame's motion vectors and the bookkeeping that decides
when those are used, as libvpx keeps it. Both sides therefore parse every frame from its
original bytes, frames stored raw or copied included, with the same parser, and update the
state the same way. A frame that fails to parse makes the state invalid; until the next key
frame parses, frames are not recoded.

To decode a recoded frame, parse its headers, then run the tile parser (partition, mode
info, motion-vector prediction, coefficient tokens, following libvpx's
`vp9_decodeframe.c`/`vp9_decodemv.c`/`vp9_detokenize.c`) with decisions from the model,
writing each decision with the probability the parse gives into a libvpx-compatible boolean
encoder (`vpx_writer`, including its 32-bit flush and padding byte); then apply the tail
fix-up. The decisions are coded with the same bit model, given libvpx's probability mapped to
the nearest CABAC state as its direct input. The encoder recodes a frame only when every
re-encoded tile matched (with a tail fix-up of at most 32 bytes).

Recoded: profiles 0 and 2 (4:2:0, 8, 10 or 12 bits), key and inter frames, hidden and
show-existing frames, superframes, tiles, compound prediction, lossless frames.
Stored raw: intra-only frames, segmentation, 4:4:4 and 4:2:2 (profiles 1 and 3), and
streams that adapt probabilities backwards (`frame_parallel_decoding_mode` 0 without error
resilience).

### VP8 (`PPV8`)

For Matroska/WebM files whose video track is VP8 (`V_VP8`). The outer layout, header stream,
segments (cut at key frames, whose tag has bit 0 clear and the start code `9d 01 2a`) and
copy flags are those of `PPV9`; VP8 has no superframes, so every block is one frame. A
recoded frame holds the length of its tag (3 bytes, or 10 for key frames with the start
code and dimensions) and the tag bytes, then every boolean decision of the frame in parse
order (the first partition's header fields, probability updates and macroblock modes, then
the tokens of each macroblock row, which come from token partition `row % count`), and
finally a tail fix-up (bytes to drop, replacement bytes) for each partition in order: the
first partition, then the token partitions. The decoder rebuilds the partitions with
libvpx-compatible boolean encoders (`vp8_stop_encode`: 32 zero bits, no padding byte),
checks the first partition's size against the tag, and writes the 3-byte little-endian size
of every token partition but the last after it.

State carried between frames is the probability context (coefficient, mode and motion-vector
probabilities, reset by key frames and restored after a frame whose header clears
`refresh_entropy_probs`) and the golden/altref sign biases, tracked as libvpx does.
The parser covers everything VP8 allows (segmentation, multiple token partitions, split
motion vectors); a frame that does not re-encode exactly is stored raw.

### MPEG-2 (`PPMV`)

For Matroska files whose video track is MPEG-2 (`V_MPEG2`, e.g. DVD rips). The outer layout
and header stream are those of `PPV9`; segments start at frames that begin with a sequence
or GOP header. A frame is coded as the length and bytes of anything before its first start
code, then for each start-code unit a flag 1 (0 ends the frame), the start code value
(`00 00 01` is implied), and the payload: for a slice (values `01`-`AF`) a flag says whether
it was parsed; if so, every bit of the slice follows, each coded with the model in the
context the slice parser gives it, plus after each macroblock a decision whether another one
follows (the bitstream signals this by 23 zero bits), then the number and values of the bits
after the last macroblock (alignment and stuffing). Other units, and slices that do not parse,
are their length and raw bytes.

No bits are re-encoded: the decoder runs the same parser over the bits the model returns
and writes them out, so a parsed slice is reproduced exactly by construction. The parser
needs the sequence header and extension (chroma format, picture height) and the picture
header and picture coding extension (picture type, f_codes, picture structure,
frame_pred_frame_dct, concealment vectors, intra_vlc_format); both sides read them from the
units as they pass, starting from the track's CodecPrivate at each segment. MPEG-1 streams
(no sequence extension), scalable extensions and D-pictures are stored raw.

AC-3 audio frames outside the video frames (DVD rips carry several AC-3 tracks) are recoded
too. The encoder scans the container bytes for AC-3 sync frames (`0B 77`, bsid 8 or lower)
whose two CRCs check out and that the frame parser handles; their bytes are left out of the
header stream's container bytes. When there are any, the last segment is followed by: the
number of AC-3 runs (varint), then per run its table bits (u8), length (varint) and stream.
A run is consecutive frames (about 4 MB) coded with a fresh model: the frame count, the
offset of the first frame and, for each later frame, the gap since the previous frame's end
(numbers), then each frame's bits from `fscod` on (the sync word, `crc1` and `crc2` are
not coded), each coded in its syntax context by `crate::ac3`'s parser, which runs the A/52
bit allocation to know every mantissa's length, followed by the bits between the last audio
block and `crc2` (auxiliary data). The decoder decodes the runs first, writes the sync word,
recomputes `crc1` (so that the CRC of the first 5/8 of the frame is zero) and `crc2`, and
decodes the container bytes around the frames. A stream without AC-3 frames has no runs and
ends after its last segment, as before.

### H.264 in MPEG transport streams (`PPTS`)

For 188-byte-packet transport streams whose first program (PAT) has an H.264 stream (PMT
stream type `0x1B`). The payload is the magic `PPTS`, the length of a side stream (varint), the
side stream, and then a complete `PPM3` payload (as above) whose "file" is the stream's NAL
units, each as a 4-byte big-endian length and the unit, a new sample at each access unit
delimiter, with no container bytes and no parameter sets outside the samples (in-band SPS
and PPS work as usual).

The encoder splits the file into the video elementary stream (the payload bytes of the
H.264 PID's packets after any PES header) and a skeleton (every other byte, in file order:
packet headers, adaptation fields, PES headers, other PIDs' packets, and any bytes after the
last whole packet). The side stream, coded with its own model (2^20 slots), holds: the file
length, the video PID, the skeleton's length and bytes, the bytes of the elementary stream
before its first start code, the number of NAL units and for each the number of zero bytes
before its start code beyond the two of `00 00 01`, and the number of zero bytes at the end.
A unit runs from its start code to the zero bytes in front of the next start code.

The decoder decodes the `PPM3` payload, rebuilds the elementary stream from the units and
the zero counts, and walks the skeleton packet by packet: a packet's header and adaptation
field come from the skeleton; for the video PID with a payload, a payload-unit start also
takes the PES header (9 bytes plus its header-data length) from the skeleton, and the rest of
the packet comes from the elementary stream; every other packet comes whole from the skeleton.
The encoder recodes a file only when this rebuilds it exactly.

## JPEG recode (`mode 3`)

For JPEG files. The PiedPiper header (magic to crc32) is followed directly by a Lepton
stream, until the end of the stream: the format of Dropbox's lepton and Microsoft's
`lepton_jpeg`, file version 1. It stores the JPEG's headers zlib-compressed and its DCT
coefficients arithmetic-coded, plus whatever is needed to rebuild the original
Huffman-coded bytes exactly (padding bits, restart markers, data after the end-of-image
marker). Encoders write 8 partitions and a git-revision field of zero, so the bytes depend
only on the input. Decoding must give exactly `original length` bytes with the stored
CRC-32. The reference is the vendored `third_party/lepton_jpeg` (0.5.8, see
`PIEDPIPER-PATCHES.md` there); any conforming Lepton decoder reads these streams, and
`testdata/golden/jpeg.bin.pp` checks them.

## Deflate recompression (`mode 4`)

For PNG, ZIP-family (docx, xlsx, pptx, odt, epub, jar, apk), gzip, PDF, zlib and tar files. After
the common header comes `n` (varint, at least 2), then `n` parts, each a `length` (varint)
followed by a complete PiedPiper stream of that length. Part 0 is the *meta* stream and
part 1 the *plain* stream (mode `0` or `1`); parts 2 and up are *embedded files* (photos,
nested containers, videos) in any mode, including mode `4` again. Mode-4 streams nest at
most three deep (the outermost counts as one). Decode all parts, then rebuild the file from
the meta bytes:

| field | type | notes |
|---|---|---|
| magic | 4 bytes | `PPDX` |
| version | u8 | `2` |
| n | varint | number of segments |
| segments | n records | see below |
| raw bytes | bytes | the container bytes kept as they are, in segment order |
| corrections | bytes | the preflate reconstruction data of every stream, in segment order (to the end) |

Segment records: `0` raw (`length` varint) copies the next `length` raw bytes. `1` deflate
(`plain length`, `correction length`, varints) recreates a DEFLATE stream from the next
bytes of the plain stream and of the corrections. `2` PNG image data (`k` varint, `k` chunk
payload sizes as varints, the 2-byte zlib header, the 4-byte Adler-32, `plain length`,
`correction length`) recreates the zlib stream and writes it back as `k` IDAT chunks (length
big-endian, `IDAT`, payload, CRC-32 big-endian). `3` deflate JPEG (`plain length`,
`correction length`) is like `1`, but takes its inflated bytes from the next embedded
part. `4` raw embedded file (`length`) copies the next embedded part, which must have that
length. All parts and bytes must be used exactly.

The corrections are those of [preflate-rs](https://github.com/microsoft/preflate-rs) 0.7.6,
with its binary arithmetic coder replaced by PiedPiper's (`crates/cabac-shim`), so they
are only readable by PiedPiper. Decoders must reproduce the original bytes exactly.

## PCM audio (`mode 5`)

For RIFF/WAVE files with 16-bit PCM samples, mono or stereo (`WAVE_FORMAT_PCM`, or
`WAVE_FORMAT_EXTENSIBLE` with the PCM sub-format). The PiedPiper header (magic to crc32) is
followed by:

| field | type | notes |
|---|---|---|
| audio version | u8 | `1`: the predictor and residual layout below |
| channels | u8 | `1` or `2` |
| bits | u8 | `16` |
| frames | varint | sample frames (one sample per channel) |
| header length | varint | bytes of the original file before the first sample |
| meta length | varint | length of the meta stream |
| meta stream | bytes | complete PiedPiper stream (mode 0 or 1) of `meta` |
| residual stream | bytes | complete PiedPiper stream (mode 0 or 1) of the residuals, until the end |

`meta` holds every byte of the original file that is not a sample: the first `header
length` bytes (everything up to and including the `data` chunk header), then everything
after the last whole frame (trailing chunks, a dangling odd byte). The original file is
`meta[..header length] ‖ samples ‖ meta[header length..]`, the samples being `frames ×
channels` interleaved little-endian i16.

**Residuals.** One integer residual `r` per sample, in frame order, channel 0 first. Each
is written as its zigzag value `u = (r << 1) ^ (r >> 63)` in two bytes, most significant
first; `u ≥ 0xFFFF` is written as `FF FF` followed by `u` in three bytes, most significant
first.

**Reconstruction.** For each frame and each channel `c`: read `r`, compute the prediction
`p` (below), then `e0 = r + p` and `x = prev + e0`, where `prev` is the channel's previous
`x` (0 before the first frame). Mono `x` is the sample. Stereo channels are mid and side:
`R = x0 − (x1 >> 1)`, `L = x1 + R` (arithmetic shift). Mid and mono values must lie in
`[−32768, 32767]`, side values in `[−65535, 65535]`, and L and R in the i16 range.

**Prediction.** Each channel has a cascade of three normalised-LMS filters over IEEE-754
binary64 values. Weights and histories start at zero; histories are most recent first.

| stage | input history `h` | output | step size `mu` |
|---|---|---|---|
| 1 | 32 past `e0` of this channel, then (stereo only) 16 `e0` of the other channel | `p1` | 0.003 |
| 2 | 16 past `e1 = e0 − p1` | `p2` | 0.006 |
| 3 | 4 past `e2 = e1 − p2` | `p3` | 0.02 |

In stage 1, channel 1 sees channel 0's `e0` of the current frame first; channel 0 sees
channel 1's values up to the previous frame. Each `pk = Σ w[i]·h[i]`, summed for ascending
`i` starting from `0.0`. The prediction is `p = round(clamp((p1 + p2) + p3, −2²⁰, 2²⁰))`,
rounding half away from zero. After the sample is known, each stage updates with its own
error (`e1`, `e2`, `e3 = e2 − p3`): `g = (mu · e) / pw`, where `pw = 1.0 + h[0]² + h[1]² +
…` is summed in order, then `w[i] += g · h[i]`. Then `e0`, `e1` and `e2` are pushed onto the
histories of stages 1, 2 and 3. The reference implementation is `crates/core/src/audio.rs`;
`testdata/golden/audio-wav.bin.pp` checks it.

## Prior (`.ppp`)

| field | type | notes |
|---|---|---|
| magic | 4 bytes | `PPRI` |
| version | u8 | `1` |
| tokenizer | as above | |
| d_model, n_layer, n_head, d_ff, window | 5 × varint | |
| n_params | varint | must equal the count implied by the shape |
| dtype | u8 | `1` = IEEE half precision |
| weights | n_params × u16 | flat parameter vector, in the reference layout |
| crc32 | u32 | of everything before it |

The parameter layout is: embedding, then for each layer `g_attn_in, W_qkv (d×3d), g_q,
g_k (head dim each), W_o, g_attn_out, g_ffn_in, W_gate‖up (d×2·d_ff), W_down, g_ffn_out`,
then `g_final`, then `W_out (d×V)`. All matrices are row-major `[in][out]`.

### Built-in priors

Builds with the `builtin-priors` feature (the default for every package) embed three
priors. A stream whose prior id matches one of them decodes without extra files:

| name | level | network | id | file |
|---|---|---|---|---|
| `text-s` | 1 | 2 × 64, window 64 | `0c6ce1767ee9c37e` | `crates/core/priors/text-s.ppp` |
| `text-m` | 2 | 3 × 96, window 128 | `ade35393a57d769b` | `crates/core/priors/text-m.ppp` |
| `text-l` | 3 | 4 × 128, window 128 | `4fcc37f8511a1a6d` | `crates/core/priors/text-l.ppp` |
| `text-m-code` | 2 (alternate) | 3 × 96, window 128 | `99f6aedee12a3454` | `crates/core/priors/text-m-code.ppp` |
| `text-s-code` | 1 (alternate) | 2 × 64, window 64 | `360f862cc7addfd8` | `crates/core/priors/text-s-code.ppp` |

Released priors are never modified. A future version that ships better priors adds new
files, with new ids, and keeps the old ones so that existing streams stay decodable.
