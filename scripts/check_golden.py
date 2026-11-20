#!/usr/bin/env python3
"""Cross-platform determinism check with the native CLI.

For every case in testdata/golden/manifest.json (produced on Windows x86_64):
  1. decompress the committed .pp file and compare with the input (SHA-256);
  2. re-compress the input with the recorded flags and require byte-identical output.

usage: python scripts/check_golden.py [path/to/piedpiper]   (exit status 1 on any mismatch)
"""
import hashlib, json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "testdata", "golden")
EXE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "target", "release", "piedpiper" + (".exe" if os.name == "nt" else ""))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    failures = 0
    manifest = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    with tempfile.TemporaryDirectory() as tmp:
        for c in manifest:
            src = os.path.join(GOLDEN, c["input"])
            packed = os.path.join(GOLDEN, c["compressed"])
            out = subprocess.run([EXE, "-d", "-c", packed], capture_output=True)
            ok1 = out.returncode == 0 and sha(out.stdout) == c["input_sha256"]
            dst = os.path.join(tmp, c["name"] + ".pp")
            r = subprocess.run([EXE, "-q", "-f", "-o", dst, *c["flags"].split(), src], capture_output=True)
            ok2 = r.returncode == 0 and sha(open(dst, "rb").read()) == c["compressed_sha256"]
            print(f"{'PASS' if ok1 else 'FAIL'} {c['name']}: decompress golden")
            print(f"{'PASS' if ok2 else 'FAIL'} {c['name']}: compress -> identical bytes")
            if not ok1 and out.stderr:
                print("   ", out.stderr.decode(errors="replace").strip())
            failures += (not ok1) + (not ok2)
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
