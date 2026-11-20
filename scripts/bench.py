#!/usr/bin/env python3
"""Compare PiedPiper levels with zlib / bz2 / xz on some files; print a markdown table.

usage: python scripts/bench.py [--levels 1,2,3] [--prior FILE] [--threads N] FILE...
Uses the native CLI (target/release/piedpiper) and verifies every roundtrip.
"""
import argparse, bz2, lzma, os, re, subprocess, sys, time, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "target", "release", "piedpiper" + (".exe" if os.name == "nt" else ""))


def human(n):
    return f"{n / 1e6:.2f} MB" if n >= 1e6 else f"{n / 1e3:.1f} KB" if n >= 1e3 else f"{n} B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--exe", default=EXE)
    ap.add_argument("--prior")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    cols = ["gzip -9", "bzip2 -9", "xz -9e"] + [f"pp -{l}" for l in levels]
    print("| file | size | " + " | ".join(cols) + " |")
    print("|---|---:|" + "---:|" * len(cols))
    for path in a.files:
        data = open(path, "rb").read()
        n = len(data)
        cells = []
        for c in (zlib.compress(data, 9), bz2.compress(data, 9), lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)):
            cells.append(f"{len(c) * 8 / n:.3f}")
        for l in levels:
            flags = ["bench", "-q", f"-{l}", "-t", str(a.threads)] + (["--prior", a.prior] if a.prior else []) + [path]
            t0 = time.time()
            out = subprocess.run([a.exe] + flags, capture_output=True, text=True)
            m = re.search(r"-> (\d+) bytes .*comp ([\d.]+)s \(([^)]*)\).*decomp ([\d.]+)s.*(OK|MISMATCH)", out.stdout)
            if not m or m.group(5) != "OK":
                cells.append("ERR")
                print(out.stdout, out.stderr, file=sys.stderr)
                continue
            cells.append(f"{int(m.group(1)) * 8 / n:.3f} ({m.group(3)})")
        print(f"| {os.path.basename(path)} | {human(n)} | " + " | ".join(cells) + " |", flush=True)
    print("\nValues are bits per byte (lower is better); PiedPiper cells show compression speed.")


if __name__ == "__main__":
    main()
