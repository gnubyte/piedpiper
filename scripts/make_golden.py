#!/usr/bin/env python3
"""Generate the cross-platform golden vectors in testdata/golden/.

Inputs are synthesised deterministically (original text, no third-party content).
Outputs are produced by the native `piedpiper` CLI. Every port (Linux, macOS, Android,
WebAssembly, Python) must decompress each `.pp` file back to its input bit-exactly and,
given the same options, re-compress to identical bytes.

usage: python scripts/make_golden.py [path/to/piedpiper]
"""
import hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "testdata", "golden")
EXE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "target", "release", "piedpiper" + (".exe" if os.name == "nt" else ""))


class Rng:  # splitmix64, so the inputs are reproducible anywhere
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & (2**64 - 1)
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
        return z ^ (z >> 31)

    def pick(self, seq):
        return seq[self.next() % len(seq)]


def text_input(n):
    r = Rng(2026)
    nouns = "model coder stream token weight signal river engine garden window letter number pattern system piper melody village bridge".split()
    verbs = "predicts encodes learns follows carries builds remembers compresses shapes guides watches repeats".split()
    adjs = "small quiet careful bright ancient online lossless steady curious patient random hidden".split()
    advs = "slowly again always rarely gently precisely".split()
    out = []
    while sum(map(len, out)) < n:
        s = f"The {r.pick(adjs)} {r.pick(nouns)} {r.pick(verbs)} the {r.pick(nouns)}"
        if r.next() % 3 == 0:
            s += f" {r.pick(advs)}"
        if r.next() % 4 == 0:
            s += f", and every {r.pick(nouns)} {r.pick(verbs)} {r.next() % 1000}"
        out.append(s + (".\n" if r.next() % 5 == 0 else ". "))
    return "".join(out).encode()[:n]


def h264_input(_n):
    # A synthetic H.264 MP4 (noise pictures, CABAC I/P/B slices) made by the media crate's
    # test generator; regenerate with
    #   cargo run --release -p piedpiper-media --bin ppmedia -- synth testdata/golden/h264-mp4.bin 2026 12 20 12
    return open(os.path.join(OUT, "h264-mp4.bin"), "rb").read()


def h264_cavlc_input(_n):
    # A synthetic H.264 MP4 with CAVLC slices (High profile with entropy_coding_mode_flag 0:
    # I/P/B slices, skip runs, sub-partitions, 8x8 transforms, I_PCM, several slices per
    # picture) made by the media crate's generator:
    #   cargo run --release -p piedpiper-media --bin ppmedia -- synth-cavlc testdata/golden/h264-cavlc-mp4.bin 2026 12 20 12 high
    return open(os.path.join(OUT, "h264-cavlc-mp4.bin"), "rb").read()


def hevc_input(_n):
    # A synthetic HEVC MP4 (noise pictures, I/P/B slices, wavefront rows and dependent slice
    # segments) made by the media crate's test generator; regenerate with
    #   cargo run --release -p piedpiper-media --bin ppmedia -- synth-hevc testdata/golden/hevc-mp4.bin 2026 10 wpp
    return open(os.path.join(OUT, "hevc-mp4.bin"), "rb").read()


def vp9_input(_n):
    # A synthetic VP9 WebM (key, inter, hidden, compound, show-existing, superframe and
    # lossless frames with pseudo-random decisions) made by the media crate's generator:
    #   cargo run --release -p piedpiper-media --bin ppmedia -- synth-vp9 testdata/golden/vp9-webm.bin 5 96 64
    return open(os.path.join(OUT, "vp9-webm.bin"), "rb").read()


def vp8_input(_n):
    # A synthetic VP8 WebM (key and inter frames, segmentation, several token partitions,
    # pseudo-random decisions) made by the media crate's generator:
    #   cargo run --release -p piedpiper-media --bin ppmedia -- synth-vp8 testdata/golden/vp8-webm.bin 8 96 64 10
    return open(os.path.join(OUT, "vp8-webm.bin"), "rb").read()


def mpeg2_input(_n):
    # A synthetic MPEG-2 Matroska file (I/P/B pictures, frame and field motion, escapes,
    # both coefficient tables) made by the media crate's generator:
    #   cargo run --release -p piedpiper-media --bin ppmedia -- synth-mpeg2 testdata/golden/mpeg2-mkv.bin 7 6 4 8
    return open(os.path.join(OUT, "mpeg2-mkv.bin"), "rb").read()


def binary_input(n):
    r = Rng(7)
    b = bytearray()
    x = 0
    while len(b) < n:
        x = (x + 3 + (r.next() % 5)) & 0xFFFF
        b += x.to_bytes(2, "little") + bytes([r.next() % 7, 0xAA])
    return bytes(b[:n])


def trial_input(_n):
    """Original project prose: the start of docs/DESIGN.md, frozen in the golden input file on
    first use. At level 1 the encoder's prefix trial picks the lower prior fine-tuning rate for
    it, so every port must make that same choice."""
    path = os.path.join(OUT, "text-prior-trial.bin")
    if not os.path.exists(path):
        return open(os.path.join(ROOT, "docs", "DESIGN.md"), "rb").read()[:12000]
    return open(path, "rb").read()


def exe_input(n):
    """A fake x86-64 PE: headers, then code that calls a few functions over and over. It
    exercises the x86 branch filter (neural header filter id 1)."""
    r = Rng(86)
    b = bytearray(0x200)
    b[0:2] = b"MZ"
    b[0x3C] = 0x80
    b[0x80:0x84] = b"PE" + bytes(2)
    b[0x84:0x86] = (0x8664).to_bytes(2, "little")
    targets = [0x1000 + 0x340 * k for k in range(6)]
    while len(b) < n:
        pos = len(b)
        b += bytes([0x48, 0x89, 0xC1 + r.next() % 4, 0xE8])
        rel = (r.pick(targets) - (pos + 8)) & 0xFFFFFFFF
        b += rel.to_bytes(4, "little")
    return bytes(b[:n])


def audio_input(frames):
    """A 16-bit stereo PCM WAV (16 kHz) built with integer arithmetic only: two "magic circle"
    oscillators plus splitmix noise, the right channel a scaled copy of the left with its own
    noise, and a LIST chunk after the samples (kept verbatim by mode 5)."""
    r = Rng(44)
    a, b, c, d = 9000, 0, 4000, 0
    pcm = bytearray()
    for _ in range(frames):
        a -= (b * 301) >> 12
        b += (a * 301) >> 12
        c -= (d * 53) >> 10
        d += (c * 53) >> 10
        left = max(-32768, min(32767, a + c + r.next() % 41 - 20))
        right = max(-32768, min(32767, (left * 3) // 4 + r.next() % 31 - 15))
        pcm += left.to_bytes(2, "little", signed=True) + right.to_bytes(2, "little", signed=True)
    u16 = lambda v: v.to_bytes(2, "little")
    u32 = lambda v: v.to_bytes(4, "little")
    fmt = u16(1) + u16(2) + u32(16000) + u32(16000 * 4) + u16(4) + u16(16)
    info = b"INFOISFT" + u32(10) + b"piedpiper\x00"
    body = (b"WAVEfmt " + u32(len(fmt)) + fmt + b"data" + u32(len(pcm)) + bytes(pcm)
            + b"LIST" + u32(len(info)) + info)
    return b"RIFF" + u32(len(body)) + body


def jpeg_input(width):
    """A baseline JPEG (YCbCr 4:4:4) built with integer arithmetic only, so the bytes are the
    same on every machine and no third-party encoder or image is involved. It exercises the
    JPEG recompression mode (format mode 3)."""
    w, h = width, width * 2 // 3 // 8 * 8
    r = Rng(31)
    rgb = []
    for y in range(h):
        for x in range(w):
            disc = (x - w // 3) ** 2 + (y - h // 2) ** 2 < (h // 3) ** 2
            px = (x * 255 // (w - 1), y * 255 // (h - 1), 200 if disc else 60)
            rgb.append([min(255, max(0, c + r.next() % 17 - 8)) for c in px])
    # JFIF RGB -> YCbCr in fixed point.
    planes = [[], [], []]
    for R, G, B in rgb:
        planes[0].append((77 * R + 150 * G + 29 * B + 128) >> 8)
        planes[1].append(((-43 * R - 85 * G + 128 * B + 128) >> 8) + 128)
        planes[2].append(((128 * R - 107 * G - 21 * B + 128) >> 8) + 128)
    zz = []  # zigzag order
    for s in range(15):
        rows = list(range(max(0, s - 7), min(s, 7) + 1))
        for row in (rows[::-1] if s % 2 == 0 else rows):
            zz.append(row * 8 + s - row)
    # 4096 * cos(k*pi/16), k = 0..8, written out so no libm is involved.
    cos_t = [4096, 4017, 3784, 3406, 2896, 2276, 1567, 799, 0]

    def cosine(u, x):
        if u == 0:
            return 2896  # 4096 / sqrt(2)
        m = (2 * x + 1) * u % 32
        return cos_t[m] if m <= 8 else -cos_t[16 - m] if m <= 16 else -cos_t[m - 16] if m <= 24 else cos_t[32 - m]

    cmat = [[cosine(u, x) for x in range(8)] for u in range(8)]
    quant = [[4 + 2 * (i // 8 + i % 8) for i in range(64)], [6 + 3 * (i // 8 + i % 8) for i in range(64)]]

    def block(plane, bx, by, q):
        s = [[plane[(by + y) * w + bx + x] - 128 for x in range(8)] for y in range(8)]
        tmp = [[sum(cmat[u][x] * s[y][x] for x in range(8)) for u in range(8)] for y in range(8)]
        out = [0] * 64
        for v in range(8):
            for u in range(8):
                f = (sum(cmat[v][y] * tmp[y][u] for y in range(8)) + (1 << 25)) >> 26
                qq = q[v * 8 + u]
                out[v * 8 + u] = (abs(f) + qq // 2) // qq * (1 if f >= 0 else -1)
        return [out[i] for i in zz]

    # One fixed-length Huffman table per class: 12 DC categories x 4 bits, 162 AC symbols x 8 bits.
    ac_syms = [0x00, 0xF0] + [(run << 4) | size for run in range(16) for size in range(1, 11)]
    dc_code = {s: (s, 4) for s in range(12)}
    ac_code = {s: (i, 8) for i, s in enumerate(ac_syms)}
    bits, nbits, scan = 0, 0, bytearray()

    def put(v, n):
        nonlocal bits, nbits
        bits, nbits = (bits << n) | (v & ((1 << n) - 1)), nbits + n
        while nbits >= 8:
            nbits -= 8
            byte = (bits >> nbits) & 0xFF
            scan.append(byte)
            if byte == 0xFF:
                scan.append(0)

    def magnitude(v):
        n = abs(v).bit_length()
        return n, (v if v >= 0 else v + (1 << n) - 1)

    prev = [0, 0, 0]
    for by in range(0, h, 8):
        for bx in range(0, w, 8):
            for c in range(3):
                coef = block(planes[c], bx, by, quant[min(c, 1)])
                n, v = magnitude(coef[0] - prev[c])
                prev[c] = coef[0]
                put(*dc_code[n])
                put(v, n)
                run = 0
                for a in coef[1:]:
                    if a == 0:
                        run += 1
                        continue
                    while run > 15:
                        put(*ac_code[0xF0])
                        run -= 16
                    n, v = magnitude(a)
                    put(*ac_code[(run << 4) | n])
                    put(v, n)
                    run = 0
                if run:
                    put(*ac_code[0x00])
    if nbits:
        put(0x7F, 8 - nbits)  # pad with one-bits

    def seg(marker, body):
        return bytes([0xFF, marker]) + (len(body) + 2).to_bytes(2, "big") + bytes(body)

    dqt = bytes([0]) + bytes(quant[0][i] for i in zz) + bytes([1]) + bytes(quant[1][i] for i in zz)
    dht = (bytes([0x00]) + bytes([0, 0, 0, 12] + [0] * 12) + bytes(range(12))
           + bytes([0x10]) + bytes([0] * 7 + [162] + [0] * 8) + bytes(ac_syms))
    sof = bytes([8]) + h.to_bytes(2, "big") + w.to_bytes(2, "big") + bytes([3, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1])
    sos = bytes([3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0])
    return (b"\xFF\xD8" + seg(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00") + seg(0xDB, dqt)
            + seg(0xC0, sof) + seg(0xC4, dht) + seg(0xDA, sos) + bytes(scan) + b"\xFF\xD9")


def ppm_input(width):
    """A binary PPM photo-like image (gradients, a disc, noise). It exercises the image
    filter of the neural codec (colour decorrelation + MED prediction, filter id 3)."""
    w, h = width, width * 2 // 3
    r = Rng(47)
    px = bytearray()
    for y in range(h):
        for x in range(w):
            disc = (x - w // 3) ** 2 + (y - h // 2) ** 2 < (h // 3) ** 2
            base = (x * 255 // (w - 1), y * 255 // (h - 1), 200 if disc else 60)
            px += bytes(min(255, max(0, c + r.next() % 13 - 6)) for c in base)
    return f"P6\n{w} {h}\n255\n".encode() + bytes(px)


CASES = [
    # name, input generator, size, CLI flags (explicit shapes keep goldens stable
    # when level presets are retuned)
    ("text-bytes", text_input, 24000,
     "--d-model 32 --layers 2 --heads 2 --d-ff 64 --window 32 --seg 4 --streams 3 -T bytes"),
    ("text-bpe", text_input, 40000,
     "--d-model 48 --layers 2 --heads 4 --d-ff 96 --window 48 --seg 8 --streams 5 -T bpe:600 --beta1 0.9 --clip 1.0"),
    ("binary", binary_input, 16000,
     "--d-model 32 --layers 1 --heads 2 --d-ff 48 --window 20 --seg 6 --streams 2 -T bytes --lr 3e-3"),
    # Default settings: text starts from the built-in priors (text-s / text-m).
    ("text-prior-s", text_input, 12000, "-1"),
    ("text-prior-m", text_input, 6000, "-2"),
    ("text-prior-l", text_input, 4000, "-3"),
    # Default settings where the prior-rate trial chooses the lower rate.
    ("text-prior-trial", trial_input, 0, "-1"),
    ("exe-x86", exe_input, 12000, "-1"),
    # Default settings on an H.264 MP4: lossless video recompression (format mode 2).
    ("h264-mp4", h264_input, 0, ""),
    # The same with CAVLC slices, which are coded bit by bit instead of through CABAC.
    ("h264-cavlc-mp4", h264_cavlc_input, 0, ""),
    # Default settings on an HEVC MP4: lossless video recompression (format mode 2, `PPH3`).
    ("hevc-mp4", hevc_input, 0, ""),
    # Default settings on a VP9 WebM: lossless video recompression (format mode 2, `PPV9`).
    ("vp9-webm", vp9_input, 0, ""),
    # Default settings on a VP8 WebM: lossless video recompression (format mode 2, `PPV8`).
    ("vp8-webm", vp8_input, 0, ""),
    # Default settings on an MPEG-2 MKV: lossless video recompression (format mode 2, `PPMV`).
    ("mpeg2-mkv", mpeg2_input, 0, ""),
    # A JPEG: recompressed losslessly by Lepton (format mode 3), not by the neural model.
    ("jpeg", jpeg_input, 96, "-2"),
    # An uncompressed PPM image: neural codec with the image filter (id 3).
    ("ppm-image", ppm_input, 96, "-2"),
    # A 16-bit PCM WAV: stereo decorrelation + LMS prediction, residuals coded by a small
    # network (format mode 5).
    ("audio-wav", audio_input, 6000,
     "--d-model 32 --layers 1 --heads 2 --d-ff 48 --window 24 --seg 6 --streams 2 --lr 3e-3"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for name, gen, size, flags in CASES:
        data = gen(size)
        src = os.path.join(OUT, name + ".bin")
        dst = src + ".pp"
        open(src, "wb").write(data)
        subprocess.run([EXE, "-q", "-f", "-o", dst, *flags.split(), src], check=True)
        packed = open(dst, "rb").read()
        manifest.append({
            "name": name,
            "input": name + ".bin",
            "compressed": name + ".bin.pp",
            "flags": flags,
            "input_sha256": hashlib.sha256(data).hexdigest(),
            "compressed_sha256": hashlib.sha256(packed).hexdigest(),
            "input_size": len(data),
            "compressed_size": len(packed),
        })
        print(f"{name}: {len(data)} -> {len(packed)}")
    with open(os.path.join(OUT, "manifest.json"), "w", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
