#!/usr/bin/env python3
"""Checks the H.264 picture reconstruction of crates/media (h264::recon) against FFmpeg.

The PPM4 stream format rebuilds every decoded picture to give the model context. Exactness is
not needed for correctness (both sides run the same code), but it is needed for the best
ratio, and this script checks it: it decodes an MP4's H.264 track with FFmpeg's libavcodec
(through ctypes) and compares a hash of every picture with the reconstruction's.

Requirements: a `ppmedia` built with the `recon-debug` feature
    cargo build --release -p piedpiper-media --bin ppmedia --features recon-debug
and a directory with FFmpeg 7.x shared libraries (avcodec-61, avutil-59; the AVFrame and
AVPacket field offsets below are those of that ABI).

usage: python scripts/h264_recon_check.py FILE.mp4 FFMPEG_LIB_DIR [path/to/ppmedia]
Exit status 1 when any picture differs.
"""
import ctypes, glob, os, struct, subprocess, sys, tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ffmpeg(libdir):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(libdir)

    def lib(stem):
        pats = [f'{stem}-*.dll', f'lib{stem}.so.*', f'lib{stem}.*.dylib']
        for p in pats:
            hits = sorted(glob.glob(os.path.join(libdir, p)))
            if hits:
                return ctypes.CDLL(hits[0])
        raise SystemExit(f'no {stem} library in {libdir}')

    avutil, avcodec = lib('avutil'), lib('avcodec')
    vp = ctypes.c_void_p
    for name, res, args in [('avcodec_find_decoder_by_name', vp, [ctypes.c_char_p]), ('avcodec_alloc_context3', vp, [vp]),
                            ('avcodec_open2', ctypes.c_int, [vp, vp, vp]), ('av_packet_alloc', vp, []),
                            ('av_new_packet', ctypes.c_int, [vp, ctypes.c_int]), ('av_packet_unref', None, [vp]),
                            ('avcodec_send_packet', ctypes.c_int, [vp, vp]), ('avcodec_receive_frame', ctypes.c_int, [vp, vp])]:
        f = getattr(avcodec, name)
        f.restype, f.argtypes = res, args
    avutil.av_frame_alloc.restype = vp
    avutil.av_frame_unref.argtypes = [vp]
    return avutil, avcodec


def picture_hash(planes):
    # Same position-weighted sums as h264::recon::hash_line (wrapping 64-bit).
    v = np.concatenate([a.reshape(-1) for a in planes]).astype(np.uint64)
    i = np.arange(1, v.size + 1, dtype=np.uint64)
    with np.errstate(over='ignore'):
        s1 = int((i * v).sum(dtype=np.uint64))
        s2 = int((v * v * (i % np.uint64(65521) + np.uint64(1))).sum(dtype=np.uint64))
    return '%016x%016x' % (s1, s2)


def ffmpeg_pictures(avutil, avcodec, aus):
    """Yields (decode index, [Y, U, V]) for every decoded picture (the decode index travels in pts)."""
    codec = avcodec.avcodec_find_decoder_by_name(b'h264')
    ctx = avcodec.avcodec_alloc_context3(codec)
    assert avcodec.avcodec_open2(ctx, codec, None) == 0
    pkt, frm = avcodec.av_packet_alloc(), avutil.av_frame_alloc()
    base = 1 << 40
    pts_off = None

    def drain():
        nonlocal pts_off
        while avcodec.avcodec_receive_frame(ctx, frm) >= 0:
            raw = ctypes.string_at(frm, 512)
            ptrs, ls = struct.unpack_from('<8Q', raw, 0), struct.unpack_from('<8i', raw, 64)
            w, h = struct.unpack_from('<2i', raw, 104)
            if pts_off is None:
                # The first int64 field holding one of our pts values.
                pts_off = next(o for o in range(0, 504, 8) if 0 <= struct.unpack_from('<q', raw, o)[0] - base < len(aus))
            seq = struct.unpack_from('<q', raw, pts_off)[0] - base
            planes = []
            for k, (pw, ph) in enumerate([(w, h), (w // 2, h // 2), (w // 2, h // 2)]):
                buf = ctypes.string_at(ptrs[k], ls[k] * ph)
                planes.append(np.frombuffer(buf, np.uint8).reshape(ph, ls[k])[:, :pw])
            yield seq, planes
            avutil.av_frame_unref(frm)

    for i, au in enumerate(aus):
        assert avcodec.av_new_packet(pkt, len(au)) == 0
        dptr = struct.unpack_from('<Q', ctypes.string_at(pkt, 32), 24)[0]
        ctypes.memmove(dptr, au, len(au))
        ctypes.memmove(pkt + 8, struct.pack('<q', base + i), 8)
        avcodec.avcodec_send_packet(ctx, pkt)
        avcodec.av_packet_unref(pkt)
        yield from drain()
    avcodec.avcodec_send_packet(ctx, None)
    yield from drain()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    path, libdir = sys.argv[1], sys.argv[2]
    exe = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 'target', 'release', 'ppmedia' + ('.exe' if os.name == 'nt' else ''))
    avutil, avcodec = load_ffmpeg(libdir)
    with tempfile.TemporaryDirectory() as tmp:
        aus_path, hash_path = os.path.join(tmp, 'aus.bin'), os.path.join(tmp, 'hash.txt')
        subprocess.run([exe, 'aus', path, aus_path], check=True)
        data, aus, p = open(aus_path, 'rb').read(), [], 0
        while p < len(data):
            (n,) = struct.unpack_from('<I', data, p)
            aus.append(data[p + 4:p + 4 + n])
            p += 4 + n
        theirs = {}
        size = None
        for seq, planes in ffmpeg_pictures(avutil, avcodec, aus):
            size = size or (planes[0].shape[1], planes[0].shape[0])
            theirs[seq] = picture_hash(planes)
        env = dict(os.environ, PPM_THREADS='1', PPM_NODEC='1', PPM_HASH=hash_path, PPM_HASH_W=str(size[0]), PPM_HASH_H=str(size[1]))
        subprocess.run([exe, 'rt', path], check=True, env=env, stdout=subprocess.DEVNULL)
        ours = {int(n): h for n, _, h in (line.split() for line in open(hash_path))}
    bad = sorted(s for s in ours if s in theirs and ours[s] != theirs[s])
    checked = sum(1 for s in ours if s in theirs)
    print(f'{path}: {checked} pictures compared, {len(bad)} differ' + (f' (first: picture {bad[0]} in decoding order)' if bad else ''))
    return 1 if bad or checked == 0 else 0


if __name__ == '__main__':
    sys.exit(main())
