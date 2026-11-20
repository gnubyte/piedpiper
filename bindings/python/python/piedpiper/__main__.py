"""``python -m piedpiper`` — small command-line front end.

    python -m piedpiper [-1..-9] FILE...      compress FILE -> FILE.pp
    python -m piedpiper -d FILE.pp...         decompress
    python -m piedpiper -l FILE.pp...         show header
"""
from __future__ import annotations

import argparse
import sys
import time

import piedpiper


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m piedpiper", description="PiedPiper neural compressor (TufaZip method).")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-d", "--decompress", action="store_true")
    ap.add_argument("-l", "--list", action="store_true", help="show stream headers")
    for lvl in range(piedpiper.MIN_LEVEL, piedpiper.MAX_LEVEL + 1):
        ap.add_argument(f"-{lvl}", dest="level", action="store_const", const=lvl)
    ap.add_argument("-T", "--tokenizer", default="auto", choices=["auto", "bytes", "bpe"])
    ap.add_argument("-t", "--threads", type=int, default=0)
    ap.add_argument("-p", "--prior", help="prior file")
    ap.add_argument("-f", "--force", action="store_true", help="overwrite outputs")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    prior = piedpiper.Prior.from_file(a.prior) if a.prior else None
    status = 0
    for path in a.files:
        try:
            if a.list:
                with open(path, "rb") as f:
                    print(path, piedpiper.info(f.read()))
                continue
            show = not a.quiet and sys.stderr.isatty()

            def progress(done, total):
                if show:
                    print(f"\r{path}: {100.0 * done / max(total, 1):5.1f}%", end="", file=sys.stderr, flush=True)

            t0 = time.time()
            if a.decompress:
                out = piedpiper.decompress_file(path, overwrite=a.force, threads=a.threads, prior=prior, progress=progress)
            else:
                out = piedpiper.compress_file(
                    path, overwrite=a.force, level=a.level or piedpiper.DEFAULT_LEVEL, tokenizer=a.tokenizer,
                    threads=a.threads, prior=prior, progress=progress,
                )
            if show:
                print("\r", end="", file=sys.stderr)
            if not a.quiet:
                import os
                print(f"{path} -> {out}: {os.path.getsize(path)} -> {os.path.getsize(out)} bytes in {time.time() - t0:.1f}s",
                      file=sys.stderr)
        except (OSError, piedpiper.PiedPiperError) as e:
            print(f"piedpiper: {path}: {e}", file=sys.stderr)
            status = 1
        except KeyboardInterrupt:
            print(f"\npiedpiper: {path}: interrupted", file=sys.stderr)
            return 130
    return status


if __name__ == "__main__":
    sys.exit(main())
