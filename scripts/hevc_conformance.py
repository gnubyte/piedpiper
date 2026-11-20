"""Check the HEVC recoder against the JCT-VC conformance bitstreams.

usage: python scripts/hevc_conformance.py ZIPDIR [path/to/ppmedia]

ZIPDIR holds conformance zips downloaded from
https://www.itu.int/wftp3/av-arch/jctvc-site/bitstream_exchange/draft_conformance/HEVC_v1/
(this script downloads nothing; never commit the streams). Each zip's bitstream is wrapped into
an MP4 (`hev1`, access units of 4-byte length-prefixed NAL units, parameter sets in band and
in `hvcC`) in a temporary folder, then `ppmedia diag` reports how many slices recode and
`ppmedia rt` checks the full compress/decompress round trip. Exit status 1 if any round trip
fails. 2026-11-20: 102 streams, all round trips byte-identical, every slice recoded except
the 6 that hold PCM coding units (stored raw by design).
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    ROOT, "target", "release", "ppmedia" + (".exe" if os.name == "nt" else ""))


def nal_units(b):
    starts, i, n = [], 0, len(b)
    while i + 3 <= n:
        if b[i] == 0 and b[i + 1] == 0 and b[i + 2] == 1:
            starts.append(i + 3)
            i += 3
        else:
            i += 1
    out = []
    for k, s in enumerate(starts):
        nal = b[s:starts[k + 1] - 3 if k + 1 < len(starts) else n].rstrip(b"\x00")
        if len(nal) >= 2:
            out.append(nal)
    return out


def access_units(nals):
    aus, cur, pending, has_vcl = [], [], [], False
    for nal in nals:
        t = (nal[0] >> 1) & 63
        if t < 32:
            if len(nal) > 2 and nal[2] & 0x80 and has_vcl:  # first_slice_segment_in_pic_flag
                aus.append(cur)
                cur, has_vcl = [], False
            cur += pending + [nal]
            pending, has_vcl = [], True
        elif t in (36, 37, 38, 40):  # EOS, EOB, filler data, suffix SEI stay with the picture
            (cur if has_vcl else pending).append(nal)
        else:
            if t == 35 and has_vcl:  # an access unit delimiter starts a new access unit
                aus.append(cur)
                cur, has_vcl = [], False
            pending.append(nal)
    cur += pending
    return [a for a in aus + [cur] if any(((n[0] >> 1) & 63) < 32 for n in a)]


def box(typ, body):
    return struct.pack(">I", len(body) + 8) + typ + body


def to_mp4(nals):
    first = {}
    for nal in nals:
        first.setdefault((nal[0] >> 1) & 63, nal)
    samples = [b"".join(struct.pack(">I", len(n)) + n for n in au) for au in access_units(nals)]
    arrays = [first[t] for t in (32, 33, 34) if t in first]
    hvcc = bytes([1, 1, 0x60, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 93, 0xF0, 0, 0xFC, 0xFD, 0xF8, 0xF8, 0, 0, 0x0F, len(arrays)])
    for nal in arrays:
        hvcc += bytes([0x80 | ((nal[0] >> 1) & 63)]) + struct.pack(">HH", 1, len(nal)) + nal
    ftyp = box(b"ftyp", b"isom\0\0\x02\0isomiso2")
    entry = bytes(7) + b"\x01" + bytes(70) + box(b"hvcC", hvcc)
    stbl = box(b"stbl", box(b"stsd", bytes(4) + struct.pack(">I", 1) + box(b"hev1", entry))
               + box(b"stsz", bytes(4) + struct.pack(">II", 0, len(samples)) + b"".join(struct.pack(">I", len(s)) for s in samples))
               + box(b"stsc", bytes(4) + struct.pack(">IIII", 1, 1, len(samples), 1))
               + box(b"stco", bytes(4) + struct.pack(">II", 1, len(ftyp) + 8)))
    hdlr = box(b"hdlr", bytes(8) + b"vide" + bytes(13))
    moov = box(b"moov", box(b"trak", box(b"mdia", hdlr + box(b"minf", stbl))))
    return ftyp + box(b"mdat", b"".join(samples)) + moov


def main():
    zipdir = sys.argv[1]
    failures, totals = 0, [0, 0, 0]
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(os.listdir(zipdir)):
            if not name.endswith(".zip"):
                continue
            with zipfile.ZipFile(os.path.join(zipdir, name)) as z:
                members = [m for m in z.namelist() if m.lower().endswith((".bit", ".bin"))]
                if not members:
                    print(f"SKIP {name}: no bitstream")
                    continue
                data = z.read(max(members, key=lambda m: z.getinfo(m).file_size))
            mp4 = os.path.join(tmp, name[:-4] + ".mp4")
            open(mp4, "wb").write(to_mp4(nal_units(data)))
            diag = subprocess.run([EXE, "diag", mp4, "0"], capture_output=True, text=True).stdout
            m = re.search(r"B \[(\d+), (\d+)\] P \[(\d+), (\d+)\] I \[(\d+), (\d+)\]", diag)
            ok = sum(int(m.group(i)) for i in (1, 3, 5)) if m else 0
            raw = sum(int(m.group(i)) for i in (2, 4, 6)) if m else -1
            rt = subprocess.run([EXE, "rt", mp4], capture_output=True, text=True).stdout
            good = " OK" in rt and "MISMATCH" not in rt
            failures += not good
            totals[0] += ok
            totals[1] += max(raw, 0)
            totals[2] += int(m.group(1)) if m else 0
            print(f"{'PASS' if good else 'FAIL'} {name[:-4]}: {ok} slices recoded, {raw} raw")
    print(f"{totals[0]} slices recoded ({totals[2]} B slices), {totals[1]} raw; "
          f"{'ALL ROUND TRIPS OK' if not failures else f'{failures} ROUND TRIPS FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
