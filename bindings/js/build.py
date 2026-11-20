#!/usr/bin/env python3
"""Build the WebAssembly module and copy it into the npm package (bindings/js/src).

Also refreshes the package's copies of LICENSE and the third-party notices (see
bindings/sync_notices.py); npm only packs files inside bindings/js."""
import os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

sys.path.insert(0, os.path.dirname(HERE))
from sync_notices import copy_notices  # noqa: E402

copy_notices("js")  # first, so a missing notice file fails before the long cargo build

subprocess.run(
    ["cargo", "build", "-p", "piedpiper-wasm", "--target", "wasm32-unknown-unknown", "--release"],
    cwd=ROOT, check=True)
src = os.path.join(ROOT, "target", "wasm32-unknown-unknown", "release", "piedpiper_wasm.wasm")
dst = os.path.join(HERE, "src", "piedpiper.wasm")
shutil.copyfile(src, dst)
# Optional size optimisation if binaryen is installed (never changes semantics).
if shutil.which("wasm-opt") and "--no-opt" not in sys.argv:
    subprocess.run(["wasm-opt", "-O3", "--enable-simd", "--enable-bulk-memory", "--enable-nontrapping-float-to-int",
                    "--enable-sign-ext", "--enable-mutable-globals", dst, "-o", dst], check=True)
print(f"wrote {dst} ({os.path.getsize(dst)} bytes)")
